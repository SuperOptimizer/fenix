// io/cache.hpp — the local artifact cache: fetch remote open-data ONCE, recompress into
// fenix-native containers on disk, serve every later access locally.
//  - default_cache_dir(): $FENIX_CACHE > $XDG_CACHE_HOME/fenix > ~/.cache/fenix.
//  - cached_surface(): a (remote or local) VC tifxyz segment dir -> a cached .fxsurf,
//    transcoded on first touch, source-identity-bound via a `.src` sidecar.
//  - CachedPyramid: a codec::VolumeSource over a remote OME-zarr multiscale — one
//    io::CachedVolume (lazily-filled .fxvol) per pyramid level, so the viewer engine
//    streams chunks on first view and hits disk thereafter.
#pragma once

#include "codec/source.hpp"
#include "core/core.hpp"
#include "io/cached_volume.hpp"
#include "io/surface.hpp"
#include "io/tifxyz.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace fenix::io {

inline std::string default_cache_dir() {
    if (const char* e = std::getenv("FENIX_CACHE"); e && *e) return e;
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) return std::string(x) + "/fenix";
    if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.cache/fenix";
    return "fenix-cache";
}

// mkdir -p (POSIX, not std::filesystem — see cached_volume.hpp for the macOS ABI rationale).
inline Expected<void> make_dirs(const std::string& path) {
    for (usize i = 1; i <= path.size(); ++i) {
        if (i != path.size() && path[i] != '/') continue;
        const std::string part = path.substr(0, i);
        if (::mkdir(part.c_str(), 0755) != 0 && errno != EEXIST)
            return err(Errc::io_error, "mkdir " + part + ": " + std::strerror(errno));
    }
    return {};
}

// Filesystem-safe cache key for a source URL/path: a readable tail + a hash of the full
// string (two sources with the same tail must never collide onto one cache entry).
inline std::string cache_key(const std::string& src) {
    std::string norm = src;  // trailing-slash-insensitive: .../x.zarr/ == .../x.zarr
    while (!norm.empty() && norm.back() == '/') norm.pop_back();
    std::string tail = norm;
    if (const auto cut = tail.size() > 64 ? tail.size() - 64 : 0; cut) tail = tail.substr(cut);
    for (char& c : tail)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '-'))
            c = '_';
    const u64 h = hash64(std::span<const u8>(reinterpret_cast<const u8*>(norm.data()), norm.size()));
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h));
    return tail + "-" + hex;
}

namespace detail {
// Bind a cache entry to its source string, exactly like CachedVolume's `.src` sidecar:
// a cache entry re-pointed at a different source must fail loudly, never serve stale data.
inline Expected<void> check_src_sidecar(const std::string& entry, const std::string& src) {
    const std::string sp = entry + ".src";
    std::ifstream sf(sp);
    std::string recorded;
    if (sf && std::getline(sf, recorded) && !recorded.empty()) {
        if (recorded != src)
            return err(Errc::invalid_argument,
                       "cache: " + entry + " was built from a DIFFERENT source (" + recorded +
                           ") — delete the cache entry or fix the pairing");
        return {};
    }
    std::ofstream of(sp, std::ios::trunc);
    of << src << "\n";
    return {};
}
}  // namespace detail

// A (remote or local) tifxyz segment dir -> the path of its cached .fxsurf. First call
// fetches + transcodes (atomic write-temp-rename via write_fxsurf); later calls are pure
// local. Concurrent transcodes of the same segment are benign (same content, atomic swap).
inline Expected<std::string> cached_surface(const std::string& tifxyz_dir,
                                            const std::string& cache_dir = default_cache_dir()) {
    const std::string dir = cache_dir + "/surf";
    if (auto d = make_dirs(dir); !d) return std::unexpected(d.error());
    const std::string path = dir + "/" + cache_key(tifxyz_dir) + ".fxsurf";
    if (::access(path.c_str(), F_OK) == 0) {
        if (auto s = detail::check_src_sidecar(path, tifxyz_dir); !s) return std::unexpected(s.error());
        return path;
    }
    auto s = read_tifxyz(tifxyz_dir);
    if (!s) return std::unexpected(s.error());
    if (auto w = write_fxsurf(path, *s); !w) return std::unexpected(w.error());
    if (auto sc = detail::check_src_sidecar(path, tifxyz_dir); !sc) return std::unexpected(sc.error());
    log(LogLevel::info, "cache: transcoded {} -> {} ({}x{} grid)", tifxyz_dir, path, s->nu, s->nv);
    return path;
}

// A streaming multi-LOD source over an OME-zarr multiscale root (local dir or URL):
// level k of the zarr pyramid is served through its own CachedVolume at
// <cache_dir>/vol/<key>_l<k>.fxvol. Offline mode per level (an unreachable source is
// fine wherever the cache already covers). Level discovery probes <root>/<k>/.zarray
// (falling back to an existing cache file when offline).
class CachedPyramid final : public codec::VolumeSource {
  public:
    static Expected<CachedPyramid> open(const std::string& zarr_root,
                                        const std::string& cache_dir = default_cache_dir(),
                                        f32 q = 2.0f) {
        const std::string dir = cache_dir + "/vol";
        if (auto d = make_dirs(dir); !d) return std::unexpected(d.error());
        const std::string prefix = dir + "/" + cache_key(zarr_root);
        CachedPyramid p;
        for (int k = 0; k < kMaxLevels; ++k) {
            const std::string lroot = zarr_root + "/" + std::to_string(k);
            const std::string cpath = prefix + "_l" + std::to_string(k) + ".fxvol";
            const bool have_cache = ::access(cpath.c_str(), F_OK) == 0;
            if (!have_cache && !read_zarray(lroot)) break;  // level absent (or offline w/o cache)
            auto cv = CachedVolume::open(cpath, lroot, q);
            if (!cv) return std::unexpected(cv.error());
            p.lv_.push_back(std::make_unique<CachedVolume>(std::move(*cv)));
        }
        if (p.lv_.empty())
            return err(Errc::not_found, "cached-pyramid: no zarr levels under " + zarr_root +
                                            " and no existing cache at " + prefix + "_l*.fxvol");
        // Deeper levels must actually halve (a flat multiscale would corrupt pick_lod math).
        // Both floor and ceil halving occur in the wild — accept either.
        const auto halves = [](s64 a, s64 b) { return b == a / 2 || b == (a + 1) / 2; };
        for (usize k = 1; k < p.lv_.size(); ++k) {
            const Extent3 a = p.lv_[k - 1]->dims(), b = p.lv_[k]->dims();
            if (!halves(a.z, b.z) || !halves(a.y, b.y) || !halves(a.x, b.x))
                return err(Errc::invalid_argument,
                           "cached-pyramid: level " + std::to_string(k) + " is not a 2x downsample");
        }
        return p;
    }

    [[nodiscard]] u32 nlods() const override { return static_cast<u32>(lv_.size()); }
    [[nodiscard]] Extent3 dims_at(s64 lod) const override { return lv_[static_cast<usize>(lod)]->dims(); }
    [[nodiscard]] codec::DType src_dtype() const override {
        return lv_[0]->archive().src_dtype();
    }
    [[nodiscard]] ChunkCoord chunk_extent(s64 lod) const override {
        return lv_[static_cast<usize>(lod)]->archive().chunk_extent(0);
    }
    [[nodiscard]] Expected<codec::BlockCache::Ref> block16(s64 lod, ChunkCoord bc) override {
        return lv_[static_cast<usize>(lod)]->block16(bc);
    }
    Expected<void>
    gather_box_f32(s64 lod, s64 oz, s64 oy, s64 ox, s64 D, s64 H, s64 W, f32* out) override {
        return lv_[static_cast<usize>(lod)]->gather_box_f32(oz, oy, ox, D, H, W, out);
    }
    // Halve the budget per level: level 0 dominates the working set; deeper levels are
    // tiny but must never be starved to zero (min 16 MiB each).
    void reserve_cache(u64 bytes) override {
        u64 b = bytes / 2;
        for (auto& cv : lv_) {
            cv->reserve_cache(std::max<u64>(b, 16ull << 20));
            b /= 2;
        }
    }
    // Bound the on-disk cache: the budget applies to level 0; deeper levels get the
    // matching 8x-smaller share (their voxel counts shrink 8x per level).
    void disk_budget(u64 bytes) {
        for (auto& cv : lv_) {
            cv->disk_budget(bytes);
            bytes = bytes ? std::max<u64>(bytes / 8, 64ull << 20) : 0;
        }
    }

  private:
    static constexpr int kMaxLevels = 10;
    std::vector<std::unique_ptr<CachedVolume>> lv_;  // stable addresses; VolumeSource is moved by value
};

}  // namespace fenix::io
