// view/slice_engine.hpp — the Qt-free slice renderer over a .fxvol archive: xy/xz/yz
// axis-aligned panes (one gather per view + bilinear resample) and oblique planes
// (per-pixel trilinear through the block cache), at the zoom-appropriate LOD from the
// archive's explicit pyramid, with a viewport prefetcher for scroll readahead. Pure
// compute — the GUI blits SliceImage::pix; the CLI can render headless. All positions
// are LOD-0 voxel coordinates (ZYX); the engine handles LOD scaling internally.
// See view/CLAUDE.md, docs/design/viewer-annotation.md.
#pragma once

#include "codec/archive.hpp"
#include "codec/source.hpp"
#include "core/core.hpp"
#include "view/prefetch.hpp"
#include "view/sampler.hpp"

#include <atomic>
#include <cmath>
#include <optional>
#include <vector>

namespace fenix::view {

enum class SliceAxis : u8 { z, y, x };  // the slice NORMAL: z = the xy pane, y = xz, x = yz

struct SliceSpec {
    SliceAxis axis = SliceAxis::z;
    f32 slice = 0;                 // position along the normal axis (LOD-0 voxels)
    f32 center_u = 0, center_v = 0;  // view center on the in-plane axes (u = fast/horizontal)
    f32 zoom = 1.0f;               // output pixels per LOD-0 voxel
    s64 width = 0, height = 0;     // output pixels
};

// An arbitrary plane: `origin` at the image center, unit in-plane directions du (right)
// and dv (down) in LOD-0 voxel space.
struct ObliqueSpec {
    Vec3f origin;
    Vec3f du{0, 0, 1}, dv{0, 1, 0};
    f32 zoom = 1.0f;
    s64 width = 0, height = 0;
};

namespace detail {
// In-plane axis indices (0=z,1=y,2=x) for each pane; u is the faster-varying axis.
inline void plane_axes(SliceAxis a, int& ax_u, int& ax_v, int& ax_n) {
    switch (a) {
        case SliceAxis::z: ax_u = 2; ax_v = 1; ax_n = 0; break;
        case SliceAxis::y: ax_u = 2; ax_v = 0; ax_n = 1; break;
        default:           ax_u = 1; ax_v = 0; ax_n = 2; break;
    }
}
inline s64 axis_of(Extent3 d, int ax) { return ax == 0 ? d.z : (ax == 1 ? d.y : d.x); }
inline f32& comp_of(Vec3f& p, int ax) { return ax == 0 ? p.z : (ax == 1 ? p.y : p.x); }
}  // namespace detail

struct SliceImage {
    s64 width = 0, height = 0;
    s64 lod = 0;
    f32 scale = 1;  // LOD-0 voxels per LOD voxel (= 2^lod)
    SliceSpec spec;
    std::vector<f32> pix;  // height*width, row-major

    // Center of output pixel (px,py) in LOD-0 voxel coordinates.
    [[nodiscard]] Vec3f pixel_to_volume(f32 px, f32 py) const {
        int au, av, an;
        detail::plane_axes(spec.axis, au, av, an);
        Vec3f p{};
        detail::comp_of(p, an) = spec.slice;
        detail::comp_of(p, au) = spec.center_u + (px + 0.5f - static_cast<f32>(spec.width) * 0.5f) / spec.zoom;
        detail::comp_of(p, av) = spec.center_v + (py + 0.5f - static_cast<f32>(spec.height) * 0.5f) / spec.zoom;
        return p;
    }
    // Inverse of pixel_to_volume (the normal-axis component is ignored).
    void volume_to_pixel(Vec3f p, f32& px, f32& py) const {
        int au, av, an;
        detail::plane_axes(spec.axis, au, av, an);
        px = (detail::comp_of(p, au) - spec.center_u) * spec.zoom - 0.5f + static_cast<f32>(spec.width) * 0.5f;
        py = (detail::comp_of(p, av) - spec.center_v) * spec.zoom - 0.5f + static_cast<f32>(spec.height) * 0.5f;
    }

    // Window [lo,hi] -> u8 for display.
    [[nodiscard]] std::vector<u8> to_u8(f32 lo, f32 hi) const {
        std::vector<u8> out(pix.size());
        const f32 s = hi > lo ? 255.0f / (hi - lo) : 0.0f;
        for (usize i = 0; i < pix.size(); ++i)
            out[i] = static_cast<u8>(std::clamp((pix[i] - lo) * s, 0.0f, 255.0f));
        return out;
    }
};

class SliceEngine {
public:
    // The engine owns the source's cache sizing + a prefetcher; the source must outlive it.
    // A source may be a local archive or a streaming stack (io::CachedPyramid over a
    // remote zarr) — render() is the same either way.
    explicit SliceEngine(codec::VolumeSource& src, u64 cache_bytes = 0, int prefetch_threads = 2)
        : src_(src), prefetch_(src_, prefetch_threads) {
        src_.reserve_cache(cache_bytes ? cache_bytes : 256ull << 20);
    }
    explicit SliceEngine(codec::VolumeArchive& arch, u64 cache_bytes = 0, int prefetch_threads = 2)
        : own_(std::in_place, arch), src_(*own_), prefetch_(src_, prefetch_threads) {
        src_.reserve_cache(cache_bytes ? cache_bytes : 256ull << 20);
    }
    SliceEngine(const SliceEngine&) = delete;
    SliceEngine& operator=(const SliceEngine&) = delete;

    [[nodiscard]] Prefetcher& prefetcher() { return prefetch_; }

    // LOD so that one LOD voxel is >= one output pixel (2^lod <= voxels-per-pixel).
    [[nodiscard]] s64 pick_lod(f32 zoom) const {
        const f32 vpp = 1.0f / std::max(zoom, 1e-6f);
        s64 lod = 0;
        while (lod + 1 < static_cast<s64>(src_.nlods()) && vpp >= static_cast<f32>(1 << (lod + 1))) ++lod;
        return lod;
    }

    // Axis-aligned pane: ONE edge-clamped gather of the covering LOD rect, then bilinear.
    [[nodiscard]] Expected<SliceImage> render(const SliceSpec& s) const {
        if (s.width <= 0 || s.height <= 0 || s.width * s.height > kMaxPixels)
            return err(Errc::invalid_argument, "bad slice size");
        if (!(s.zoom > 0)) return err(Errc::invalid_argument, "zoom must be > 0");
        int au, av, an;
        detail::plane_axes(s.axis, au, av, an);

        SliceImage img;
        img.width = s.width;
        img.height = s.height;
        img.lod = pick_lod(s.zoom);
        img.scale = static_cast<f32>(1 << img.lod);
        img.spec = s;
        img.pix.assign(static_cast<usize>(s.width * s.height), 0.0f);

        const Extent3 vd = src_.dims_at(img.lod);
        const f32 inv = 1.0f / img.scale;
        // Covering rect in LOD voxels (+1 for the bilinear neighbour, +1 margin).
        const f32 half_u = static_cast<f32>(s.width) * 0.5f / s.zoom;
        const f32 half_v = static_cast<f32>(s.height) * 0.5f / s.zoom;
        const s64 gu0 = static_cast<s64>(std::floor((s.center_u - half_u) * inv)) - 1;
        const s64 gv0 = static_cast<s64>(std::floor((s.center_v - half_v) * inv)) - 1;
        const s64 gue = static_cast<s64>(std::ceil((s.center_u + half_u) * inv)) + 2 - gu0;
        const s64 gve = static_cast<s64>(std::ceil((s.center_v + half_v) * inv)) + 2 - gv0;
        const s64 sn = std::clamp<s64>(static_cast<s64>(std::llround(s.slice * inv)), 0,
                                       detail::axis_of(vd, an) - 1);

        s64 o[3] = {0, 0, 0}, e[3] = {1, 1, 1};
        o[an] = sn;
        o[au] = gu0;
        e[au] = gue;
        o[av] = gv0;
        e[av] = gve;
        std::vector<f32> buf(static_cast<usize>(gue * gve));
        if (auto g = src_.gather_box_f32(img.lod, o[0], o[1], o[2], e[0], e[1], e[2], buf.data()); !g)
            return std::unexpected(g.error());

        // In every axis layout the gathered slab reads as buf[vi*gue + ui].
        parallel_for(0, s.height, [&](s64 py) {
            f32* row = img.pix.data() + py * s.width;
            const f32 v0 = (s.center_v + (static_cast<f32>(py) + 0.5f - static_cast<f32>(s.height) * 0.5f) / s.zoom) * inv -
                           static_cast<f32>(gv0);
            const f32 vf = std::clamp(v0, 0.0f, static_cast<f32>(gve - 1) - 1e-3f);
            const s64 vi = static_cast<s64>(vf);
            const f32 tv = vf - static_cast<f32>(vi);
            const f32* r0 = buf.data() + vi * gue;
            const f32* r1 = buf.data() + std::min(vi + 1, gve - 1) * gue;
            for (s64 px = 0; px < s.width; ++px) {
                const f32 u0 = (s.center_u + (static_cast<f32>(px) + 0.5f - static_cast<f32>(s.width) * 0.5f) / s.zoom) * inv -
                               static_cast<f32>(gu0);
                const f32 uf = std::clamp(u0, 0.0f, static_cast<f32>(gue - 1) - 1e-3f);
                const s64 ui = static_cast<s64>(uf);
                const f32 tu = uf - static_cast<f32>(ui);
                const s64 ui1 = std::min(ui + 1, gue - 1);
                row[px] = (r0[ui] * (1 - tu) + r0[ui1] * tu) * (1 - tv) +
                          (r1[ui] * (1 - tu) + r1[ui1] * tu) * tv;
            }
        });
        return img;
    }

    // Oblique plane: per-pixel trilinear through the block cache (per-row sampler memo).
    [[nodiscard]] Expected<SliceImage> render_oblique(const ObliqueSpec& s) const {
        if (s.width <= 0 || s.height <= 0 || s.width * s.height > kMaxPixels)
            return err(Errc::invalid_argument, "bad slice size");
        if (!(s.zoom > 0)) return err(Errc::invalid_argument, "zoom must be > 0");
        SliceImage img;
        img.width = s.width;
        img.height = s.height;
        img.lod = pick_lod(s.zoom);
        img.scale = static_cast<f32>(1 << img.lod);
        img.pix.assign(static_cast<usize>(s.width * s.height), 0.0f);
        const f32 inv = 1.0f / img.scale;
        std::atomic<bool> failed{false};
        parallel_for(0, s.height, [&](s64 py) {
            BlockSampler smp(src_, img.lod);
            f32* row = img.pix.data() + py * s.width;
            const f32 dvs = (static_cast<f32>(py) + 0.5f - static_cast<f32>(s.height) * 0.5f) / s.zoom;
            for (s64 px = 0; px < s.width; ++px) {
                const f32 dus = (static_cast<f32>(px) + 0.5f - static_cast<f32>(s.width) * 0.5f) / s.zoom;
                const Vec3f p{s.origin.z + s.du.z * dus + s.dv.z * dvs,
                              s.origin.y + s.du.y * dus + s.dv.y * dvs,
                              s.origin.x + s.du.x * dus + s.dv.x * dvs};
                row[px] = smp.trilinear(p.z * inv, p.y * inv, p.x * inv);
            }
            if (smp.failed()) failed.store(true, std::memory_order_relaxed);
        });
        if (failed.load()) return err(Errc::decode_error, "oblique render: chunk decode failed");
        return img;
    }

    // Queue readahead around a pane: this slice's viewport ring + the next `slices_ahead`
    // slices each way. Call after render(); drops any stale queue first.
    void prefetch_around(const SliceSpec& s, int slices_ahead = 4) {
        int au, av, an;
        detail::plane_axes(s.axis, au, av, an);
        const s64 lod = pick_lod(s.zoom);
        const f32 inv = 1.0f / static_cast<f32>(1 << lod);
        const Extent3 vd = src_.dims_at(lod);
        constexpr s64 T = codec::fxvol_chunk_side;
        const f32 half_u = static_cast<f32>(s.width) * 0.5f / s.zoom;
        const f32 half_v = static_cast<f32>(s.height) * 0.5f / s.zoom;
        const s64 tu0 = std::max<s64>(0, static_cast<s64>(std::floor((s.center_u - half_u) * inv)) / T - 1);
        const s64 tv0 = std::max<s64>(0, static_cast<s64>(std::floor((s.center_v - half_v) * inv)) / T - 1);
        const s64 tu1 = static_cast<s64>(std::ceil((s.center_u + half_u) * inv)) / T + 1;
        const s64 tv1 = static_cast<s64>(std::ceil((s.center_v + half_v) * inv)) / T + 1;
        const s64 sn = std::clamp<s64>(static_cast<s64>(std::llround(s.slice * inv)), 0,
                                       detail::axis_of(vd, an) - 1);
        prefetch_.begin_batch();
        for (int off = -slices_ahead; off <= slices_ahead; ++off) {
            const s64 n = sn + off * T;  // one tile step along the normal covers T slices
            const s64 tn = n / T;
            const f32 pri = 1.0f / (1.0f + static_cast<f32>(off < 0 ? -off : off));
            for (s64 tv = tv0; tv <= tv1; ++tv)
                for (s64 tu = tu0; tu <= tu1; ++tu) {
                    s64 t[3] = {0, 0, 0};
                    t[an] = tn;
                    t[au] = tu;
                    t[av] = tv;
                    prefetch_.request_tile(lod, {t[0], t[1], t[2]}, pri);
                }
        }
    }

private:
    static constexpr s64 kMaxPixels = 64ll << 20;  // 64 Mpx guard
    std::optional<codec::ArchiveSource> own_;  // set by the archive-convenience ctor only
    codec::VolumeSource& src_;
    Prefetcher prefetch_;
};

}  // namespace fenix::view
