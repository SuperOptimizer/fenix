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
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
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
            auto a = codec::VolumeArchive::open(cache_path, /*writable=*/true);  // cache keeps growing
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
        std::shared_lock lk(sync_->mu);
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
            std::shared_lock lk(sync_->mu);
            if (!any_absent_(z0, z1, y0, y1, x0, x1, false)) return {};
        }
        // Claim the missing chunks (in-flight set) so fills of DIFFERENT regions run in
        // parallel — the slow S3 fetch happens OUTSIDE any lock; only the coverage scan/claim
        // and the fast write_chunk appends serialize. (v1 held the exclusive lock across the
        // fetch: 8 feeder workers collapsed to ~0.3 draws/s on a cold cache.)
        s64 mz0 = INT64_MAX, my0 = INT64_MAX, mx0 = INT64_MAX, mz1 = -1, my1 = -1, mx1 = -1;
        std::vector<u64> claimed;
        {
            std::unique_lock lk(sync_->mu);
            for (s64 cz = z0; cz <= z1; ++cz)
                for (s64 cy = y0; cy <= y1; ++cy)
                    for (s64 cx = x0; cx <= x1; ++cx) {
                        if (arch_.coverage({cz, cy, cx}) != codec::Coverage::Absent) continue;
                        const u64 key = chunk_key_(cz, cy, cx);
                        if (sync_->inflight.count(key)) continue;  // a peer is already fetching it
                        sync_->inflight.insert(key);
                        claimed.push_back(key);
                        mz0 = std::min(mz0, cz);
                        mz1 = std::max(mz1, cz);
                        my0 = std::min(my0, cy);
                        my1 = std::max(my1, cy);
                        mx0 = std::min(mx0, cx);
                        mx1 = std::max(mx1, cx);
                    }
        }
        auto release = [&] {
            std::unique_lock lk(sync_->mu);
            for (u64 k : claimed) sync_->inflight.erase(k);
            sync_->cv.notify_all();
        };
        if (claimed.empty()) {  // all missing chunks are being fetched by peers: wait for them
            std::unique_lock lk(sync_->mu);
            sync_->cv.wait(lk, [&] { return !any_absent_(z0, z1, y0, y1, x0, x1, true); });
            if (any_absent_(z0, z1, y0, y1, x0, x1, false))
                return err(Errc::fetch_failed, "cached-volume: a peer's fill of this region failed");
            return {};
        }
        const Index3 forg{mz0 * C, my0 * C, mx0 * C};
        const Extent3 fext{std::min(dims_.z, (mz1 + 1) * C) - forg.z,
                           std::min(dims_.y, (my1 + 1) * C) - forg.y,
                           std::min(dims_.x, (mx1 + 1) * C) - forg.x};
        auto v = read_zarr_region<u8>(lroot_, forg, fext);  // the SLOW part: no lock held
        if (!v) {
            release();
            return std::unexpected(v.error());
        }

        // Append the CLAIMED chunks: fast (DCT encode + bump-alloc memcpy), exclusive lock held.
        std::unique_lock lk(sync_->mu);
        std::vector<u8> block(static_cast<usize>(C * C * C));
        auto vv = v->view();
        for (s64 cz = mz0; cz <= mz1; ++cz)
            for (s64 cy = my0; cy <= my1; ++cy)
                for (s64 cx = mx0; cx <= mx1; ++cx) {
                    const u64 key = chunk_key_(cz, cy, cx);
                    if (std::find(claimed.begin(), claimed.end(), key) == claimed.end()) continue;
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
                    if (auto w = arch_.write_chunk(0, ChunkCoord{cz, cy, cx}, std::span<const u8>(block), u8{0}); !w) {
                        lk.unlock();
                        release();
                        return w;
                    }
                }
        auto c = arch_.commit();
        lk.unlock();
        release();
        return c;
    }

  private:
    [[nodiscard]] static u64 chunk_key_(s64 cz, s64 cy, s64 cx) {
        return (static_cast<u64>(cz) << 42) | (static_cast<u64>(cy) << 21) | static_cast<u64>(cx);
    }
    // any chunk in the range absent AND (ignore_inflight ? not currently claimed by a peer : at all)
    [[nodiscard]] bool any_absent_(s64 z0, s64 z1, s64 y0, s64 y1, s64 x0, s64 x1, bool ignore_inflight) const {
        for (s64 cz = z0; cz <= z1; ++cz)
            for (s64 cy = y0; cy <= y1; ++cy)
                for (s64 cx = x0; cx <= x1; ++cx) {
                    if (arch_.coverage({cz, cy, cx}) != codec::Coverage::Absent) continue;
                    if (ignore_inflight && sync_->inflight.count(chunk_key_(cz, cy, cx))) continue;
                    return true;
                }
        return false;
    }

    codec::VolumeArchive arch_;
    std::string lroot_;
    Extent3 dims_{};
    // heap-held so CachedVolume stays movable (Expected<CachedVolume> needs it)
    struct Sync {
        std::shared_mutex mu;
        std::set<u64> inflight;  // chunk keys a worker is currently fetching (guarded by mu)
        std::condition_variable_any cv;
    };
    std::unique_ptr<Sync> sync_ = std::make_unique<Sync>();
};

}  // namespace fenix::io
