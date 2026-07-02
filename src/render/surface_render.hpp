// render/surface_render.hpp — sample the volume along a parametric Surface at +/-N offsets
// of the local surface normal, producing the flattened layered texture stack (the
// ink-detection input). Normals are estimated from the surface coord grid. See render/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <cmath>
#include <optional>

namespace fenix::render {

struct LayerParams {
    s64 num_layers = 16;  // +/- this many -> (2*num_layers+1) layers
    f32 step = 1.0f;      // voxels between layers along the normal
};

// render_surface's result: the {2*num_layers+1, nv, nu} layer stack (depth x v x u) plus a {nv,nu}
// per-pixel validity mask. A pixel is invalid (mask=0, stack left at 0) if the surface marks it
// invalid, if neither u- nor v-neighbour pair yields a usable tangent (isolated valid cell), or if
// the resulting normal is degenerate — never silently sampled from garbage/zero neighbour coords.
struct RenderResult {
    Volume<f32> stack;
    Volume<u8> mask;  // {nv, nu}, 1 = layers at (v,u) are real samples
};

// Samples +/-N layers along the local surface normal, estimated from one-sided differences that
// skip invalid neighbours (never differencing against an invalid cell's zero-initialized coord —
// core/surface.hpp value-inits invalid cells to {0,0,0}, and validity masks, not NaN, are the
// project's sentinel convention).
inline RenderResult render_surface(const Surface& surf, VolumeView<const f32> vol, LayerParams p) {
    const s64 nl = 2 * p.num_layers + 1;
    RenderResult res{Volume<f32>::zeros({nl, surf.nv, surf.nu}), Volume<u8>::zeros({1, surf.nv, surf.nu})};
    VolumeView<f32> ov = res.stack.view();
    VolumeView<u8> mv = res.mask.view();

    // One-sided difference against a neighbour only if it is valid; returns {false,{}} if neither
    // side of the pair is usable.
    auto tangent = [&](s64 u0, s64 v0, s64 du, s64 dv, s64 n, s64 t) -> std::optional<Vec3f> {
        const s64 lo = t - (du != 0 ? du : dv), hi = t + (du != 0 ? du : dv);
        (void)n;
        const bool has_lo = lo >= 0 && (du != 0 ? surf.is_valid(lo, v0) : surf.is_valid(u0, lo));
        const bool has_hi = hi < n && (du != 0 ? surf.is_valid(hi, v0) : surf.is_valid(u0, hi));
        const Vec3f c0 = surf.at(u0, v0);
        if (has_lo && has_hi) {
            const Vec3f a = du != 0 ? surf.at(lo, v0) : surf.at(u0, lo);
            const Vec3f b = du != 0 ? surf.at(hi, v0) : surf.at(u0, hi);
            return (b - a) * 0.5f;
        }
        if (has_hi) return (du != 0 ? surf.at(hi, v0) : surf.at(u0, hi)) - c0;
        if (has_lo) return c0 - (du != 0 ? surf.at(lo, v0) : surf.at(u0, lo));
        return std::nullopt;
    };

    parallel_for(0, surf.nv, [&](s64 v) {
        for (s64 u = 0; u < surf.nu; ++u) {
            if (!surf.is_valid(u, v)) continue;
            const Vec3f c = surf.at(u, v);
            const auto tu = tangent(u, v, 1, 0, surf.nu, u);
            const auto tv = tangent(u, v, 0, 1, surf.nv, v);
            if (!tu || !tv) continue;
            Vec3f nrm = cross(*tu, *tv);
            const f32 nl2 = norm(nrm);
            if (nl2 < tol::dir_eps) continue;
            nrm = nrm / nl2;
            for (s64 l = 0; l < nl; ++l) {
                const f32 off = static_cast<f32>(l - p.num_layers) * p.step;
                const Vec3f s = c + nrm * off;
                ov(l, v, u) = sample_trilinear(vol, s);
            }
            mv(0, v, u) = 1;
        }
    });
    return res;
}

}  // namespace fenix::render
