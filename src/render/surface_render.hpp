// render/surface_render.hpp — sample the volume along a parametric Surface at +/-N offsets
// of the local surface normal, producing the flattened layered texture stack (the
// ink-detection input). Normals are estimated from the surface coord grid. See render/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <cmath>

namespace fenix::render {

struct LayerParams {
    s64 num_layers = 16;  // +/- this many -> (2*num_layers+1) layers
    f32 step = 1.0f;      // voxels between layers along the normal
};

// Returns a {2*num_layers+1, nv, nu} image stack: layer index (depth) x v x u.
inline Volume<f32> render_surface(const Surface& surf, VolumeView<const f32> vol, LayerParams p) {
    const s64 nl = 2 * p.num_layers + 1;
    Volume<f32> out = Volume<f32>::zeros({nl, surf.nv, surf.nu});
    VolumeView<f32> ov = out.view();

    auto clampu = [&](s64 u) { return std::clamp<s64>(u, 0, surf.nu - 1); };
    auto clampv = [&](s64 v) { return std::clamp<s64>(v, 0, surf.nv - 1); };

    parallel_for(0, surf.nv, [&](s64 v) {
        for (s64 u = 0; u < surf.nu; ++u) {
            if (!surf.is_valid(u, v)) continue;
            const Vec3f c = surf.at(u, v);
            // Tangents from neighbours (skip across invalid cells where possible).
            const Vec3f tu = surf.at(clampu(u + 1), v) - surf.at(clampu(u - 1), v);
            const Vec3f tv = surf.at(u, clampv(v + 1)) - surf.at(u, clampv(v - 1));
            Vec3f nrm = cross(tu, tv);
            const f32 nl2 = norm(nrm);
            if (nl2 < tol::dir_eps) continue;
            nrm = nrm / nl2;
            for (s64 l = 0; l < nl; ++l) {
                const f32 off = static_cast<f32>(l - p.num_layers) * p.step;
                const Vec3f s = c + nrm * off;
                ov(l, v, u) = sample_trilinear(vol, s);
            }
        }
    });
    return out;
}

}  // namespace fenix::render
