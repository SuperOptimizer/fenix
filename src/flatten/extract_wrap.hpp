// flatten/extract_wrap.hpp — extract one wrap (an iso-winding level set) as a parametric
// Surface, by ray-marching outward from the umbilicus at each (theta, z) to the crossing
// W == target. A simple, direct flatten; ABF/LSCM/SLIM refinement layers on top. See
// flatten/CLAUDE.md.
#pragma once

#include "annotate/umbilicus.hpp"
#include "core/core.hpp"

#include <cmath>
#include <numbers>

namespace fenix::flatten {

// Build the Surface for winding level `target`. nu angular samples, nv = z slices.
// At each (u=theta, v=z) march r in [1, r_max); where the winding field crosses `target`,
// place a surface point (linear interpolation in r). Cells with no crossing are invalid.
inline Surface extract_winding_surface(VolumeView<const f32> winding, const annotate::Umbilicus& umb,
                                       f32 target, s64 nu, s64 r_max) {
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    const Extent3 d = winding.dims();
    const s64 nv = d.z;
    Surface s(nu, nv);
    s.scale_u = two_pi / static_cast<f32>(nu);
    s.scale_v = 1.0f;

    for (s64 v = 0; v < nv; ++v) {
        const Vec3f c = umb.center(static_cast<f32>(v));
        for (s64 u = 0; u < nu; ++u) {
            const f32 ang = two_pi * static_cast<f32>(u) / static_cast<f32>(nu);
            const f32 cy = std::sin(ang), cx = std::cos(ang);
            f32 prev_w = 0;
            bool have_prev = false;
            for (s64 r = 1; r < r_max; ++r) {
                const s64 y = static_cast<s64>(std::lround(c.y + cy * static_cast<f32>(r)));
                const s64 x = static_cast<s64>(std::lround(c.x + cx * static_cast<f32>(r)));
                if (!winding.in_bounds(v, y, x)) break;
                const f32 w = winding(v, y, x);
                if (have_prev && ((prev_w - target) * (w - target) <= 0.0f) && (w != prev_w)) {
                    // crossing between r-1 and r: linear-interp the radius.
                    const f32 t = (target - prev_w) / (w - prev_w);
                    const f32 rr = static_cast<f32>(r - 1) + t;
                    s.set(u, v, Vec3f{static_cast<f32>(v), c.y + cy * rr, c.x + cx * rr});
                    break;
                }
                prev_w = w;
                have_prev = true;
            }
        }
    }
    return s;
}

}  // namespace fenix::flatten
