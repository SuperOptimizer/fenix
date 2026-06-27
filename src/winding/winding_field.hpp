// winding/winding_field.hpp — the Eulerian winding-number field (the continuous view of
// "which wrap"; spiral-space shifted_radius / dr). This is the analytic polar init
// (W = r/pitch + theta/2pi) + a GT-free monotonicity metric. The full masked relaxation
// solve and the diffeomorphic fit build on this (see winding/CLAUDE.md, spiral-v2.md).
#pragma once

#include "annotate/umbilicus.hpp"
#include "core/core.hpp"

#include <cmath>
#include <limits>
#include <numbers>

namespace fenix::winding {

struct WindingParams {
    f32 pitch = 8.0f;  // voxels per wrap (the Archimedean b·2pi); data-calibrated later
};

// Analytic winding field over the whole volume from the umbilicus polar frame.
// Level sets of W are sheets; along a radial ray W increases by 1 per `pitch` voxels.
inline Volume<f32> winding_init(Extent3 dims, const annotate::Umbilicus& umb, WindingParams p) {
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    Volume<f32> w = Volume<f32>::zeros(dims);
    VolumeView<f32> wv = w.view();
    parallel_for_z(dims, [&](s64 z) {
        for (s64 y = 0; y < dims.y; ++y)
            for (s64 x = 0; x < dims.x; ++x) {
                f32 r;
                Radian th;
                umb.polar(z, y, x, r, th);
                wv(z, y, x) = r / p.pitch + th.value / two_pi;
            }
    });
    return w;
}

// GT-free Monotonicity Violation Fraction: cast `nrays` rays from the axis at slice `zc`,
// step outward; the winding should strictly increase. Returns the fraction of radial steps
// (within `r_max`) that fail to increase. ~0 for a well-formed field.
inline f32 monotonicity_violation(VolumeView<const f32> field, const annotate::Umbilicus& umb,
                                  s64 zc, s64 r_max, int nrays = 180, s64 r_min = 1) {
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    const Vec3f c = umb.center(static_cast<f32>(zc));
    s64 total = 0, bad = 0;
    for (int a = 0; a < nrays; ++a) {
        const f32 ang = two_pi * static_cast<f32>(a) / static_cast<f32>(nrays);
        const f32 cy = std::sin(ang), cx = std::cos(ang);
        f32 prev = std::numeric_limits<f32>::lowest();  // finite (fast-math forbids -inf)
        for (s64 r = r_min; r < r_max; ++r) {
            const s64 y = static_cast<s64>(std::lround(c.y + cy * static_cast<f32>(r)));
            const s64 x = static_cast<s64>(std::lround(c.x + cx * static_cast<f32>(r)));
            if (!field.in_bounds(zc, y, x)) break;
            const f32 wv = field(zc, y, x);
            ++total;
            if (wv <= prev) ++bad;
            prev = wv;
        }
    }
    return total > 0 ? static_cast<f32>(bad) / static_cast<f32>(total) : 0.0f;
}

}  // namespace fenix::winding
