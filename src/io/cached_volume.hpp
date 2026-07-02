// io/cached_volume.hpp — the on-demand training chunk cache: a .fxvol archive lazily filled
// from a remote (or local) OME-Zarr. gather() checks the archive's coverage tri-state for the
// requested box; ABSENT chunks are fetched from the zarr ONCE (bounding-region fetch through
// the hardened reader — a transient fetch error hard-fails, never becomes air), appended into
// the archive (crash-safe: a killed run just refetches the in-flight chunks), then every later
// query — this run or the next — is served locally. The archive is created at the FULL remote
// volume dims: the sparse mmap page table + coverage sentinels make an almost-empty
// whole-scroll archive cost ~nothing on disk.
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/zarr.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace fenix::io {

class CachedVolume {
  public:
    // `zarr_level_root` = the pyramid-level dir holding .zarray (local path or URL).
    // Creates `cache_path` sized to the remote dims if absent; validates dims if present.
    static Expected<CachedVolume>
    open(const std::string& cache_path, const std::string& zarr_level_root, f32 q = 2.0f) {
        auto meta = read_zarray(zarr_level_root);
        if (!meta) return std::unexpected(meta.error());
        CachedVolume cv;
        cv.lroot_ = zarr_level_root;
        cv.dims_ = meta->shape;
        if (std::filesystem::exists(cache_path)) {
            auto a = codec::VolumeArchive::open(cache_path);
            if (!a) return std::unexpected(a.error());
            if (a->dims().z != cv.dims_.z || a->dims().y != cv.dims_.y || a->dims().x != cv.dims_.x)
                return err(Errc::invalid_argument,
                           "cached-volume: cache dims don't match the zarr (stale cache?): " + cache_path);
            cv.arch_ = std::move(*a);
        } else {
            auto a = codec::VolumeArchive::create(cache_path, cv.dims_, codec::DctParams{.q = q});
            if (!a) return std::unexpected(a.error());
            cv.arch_ = std::move(*a);
        }
        return cv;
    }

    [[nodiscard]] Extent3 dims() const { return dims_; }
    [[nodiscard]] codec::VolumeArchive& archive() { return arch_; }

    // Gather a dense f32 box (same semantics as VolumeArchive::gather_box_f32), fetching any
    // ABSENT chunks from the zarr first. Thread-safe: fills take the exclusive lock, gathers
    // the shared one — readers never observe a half-written page table.
    Expected<void> gather_box_f32(s64 oz, s64 oy, s64 ox, s64 D, s64 H, s64 W, f32* out) {
        if (auto r = ensure(Index3{oz, oy, ox}, Extent3{D, H, W}); !r) return r;
        std::shared_lock lk(*mu_);
        return arch_.gather_box_f32(0, oz, oy, ox, D, H, W, out);
    }

    // Make every 64³ chunk overlapping the (clamped) box present in the cache.
    Expected<void> ensure(Index3 org, Extent3 ext) {
        constexpr s64 C = codec::fxvol_chunk_side;
        const s64 z0 = std::clamp<s64>(org.z, 0, dims_.z - 1) / C;
        const s64 y0 = std::clamp<s64>(org.y, 0, dims_.y - 1) / C;
        const s64 x0 = std::clamp<s64>(org.x, 0, dims_.x - 1) / C;
        const s64 z1 = std::clamp<s64>(org.z + ext.z - 1, 0, dims_.z - 1) / C;
        const s64 y1 = std::clamp<s64>(org.y + ext.y - 1, 0, dims_.y - 1) / C;
        const s64 x1 = std::clamp<s64>(org.x + ext.x - 1, 0, dims_.x - 1) / C;

        // fast path: shared-lock scan for anything missing
        {
            std::shared_lock lk(*mu_);
            if (!any_absent_(z0, z1, y0, y1, x0, x1)) return {};
        }
        std::unique_lock lk(*mu_);
        if (!any_absent_(z0, z1, y0, y1, x0, x1)) return {};  // raced: someone else filled it

        // Bounding box of the still-missing chunks -> ONE zarr region fetch (typical training
        // gathers are dense boxes, so the bbox is tight; the fetch itself parallelizes inside).
        s64 mz0 = INT64_MAX, my0 = INT64_MAX, mx0 = INT64_MAX, mz1 = -1, my1 = -1, mx1 = -1;
        for (s64 cz = z0; cz <= z1; ++cz)
            for (s64 cy = y0; cy <= y1; ++cy)
                for (s64 cx = x0; cx <= x1; ++cx)
                    if (arch_.coverage({cz, cy, cx}) == codec::Coverage::Absent) {
                        mz0 = std::min(mz0, cz);
                        mz1 = std::max(mz1, cz);
                        my0 = std::min(my0, cy);
                        my1 = std::max(my1, cy);
                        mx0 = std::min(mx0, cx);
                        mx1 = std::max(mx1, cx);
                    }
        const Index3 forg{mz0 * C, my0 * C, mx0 * C};
        const Extent3 fext{std::min(dims_.z, (mz1 + 1) * C) - forg.z,
                           std::min(dims_.y, (my1 + 1) * C) - forg.y,
                           std::min(dims_.x, (mx1 + 1) * C) - forg.x};
        auto v = read_zarr_region<u8>(lroot_, forg, fext);
        if (!v) return std::unexpected(v.error());

        // Write each missing chunk (edge chunks are C-clipped; write_chunk edge-replicates).
        std::vector<u8> block(static_cast<usize>(C * C * C));
        auto vv = v->view();
        for (s64 cz = mz0; cz <= mz1; ++cz)
            for (s64 cy = my0; cy <= my1; ++cy)
                for (s64 cx = mx0; cx <= mx1; ++cx) {
                    if (arch_.coverage({cz, cy, cx}) != codec::Coverage::Absent) continue;
                    const s64 bz = cz * C - forg.z, by = cy * C - forg.y, bx = cx * C - forg.x;
                    for (s64 z = 0; z < C; ++z)
                        for (s64 y = 0; y < C; ++y)
                            for (s64 x = 0; x < C; ++x) {
                                const s64 sz = std::min(bz + z, fext.z - 1);
                                const s64 sy = std::min(by + y, fext.y - 1);
                                const s64 sx = std::min(bx + x, fext.x - 1);
                                block[static_cast<usize>((z * C + y) * C + x)] = vv(sz, sy, sx);
                            }
                    if (auto w = arch_.write_chunk(0, ChunkCoord{cz, cy, cx}, std::span<const u8>(block), u8{0}); !w)
                        return w;
                }
        return arch_.commit();
    }

  private:
    [[nodiscard]] bool any_absent_(s64 z0, s64 z1, s64 y0, s64 y1, s64 x0, s64 x1) const {
        for (s64 cz = z0; cz <= z1; ++cz)
            for (s64 cy = y0; cy <= y1; ++cy)
                for (s64 cx = x0; cx <= x1; ++cx)
                    if (arch_.coverage({cz, cy, cx}) == codec::Coverage::Absent) return true;
        return false;
    }

    codec::VolumeArchive arch_;
    std::string lroot_;
    Extent3 dims_{};
    // unique_ptr so CachedVolume stays movable (Expected<CachedVolume> needs it)
    std::unique_ptr<std::shared_mutex> mu_ = std::make_unique<std::shared_mutex>();
};

}  // namespace fenix::io
