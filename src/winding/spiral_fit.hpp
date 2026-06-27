// winding/spiral_fit.hpp — closed-form Archimedean spiral fit r = a + b*theta_total
// (theta_total = 2*pi*winding), via ordinary least squares. The global radial pitch the
// diffeomorphic fit's gap-expander/dr_per_winding refines. See winding/CLAUDE.md, spiral-v2.
#pragma once

#include "annotate/umbilicus.hpp"
#include "core/core.hpp"

#include <cmath>
#include <numbers>
#include <span>

namespace fenix::winding {

struct SpiralParams {
    f32 a = 0;       // radial offset
    f32 b = 0;       // radial growth per radian (pitch = 2*pi*b)
    f32 rms = 0;     // fit residual RMS
    s64 nsamples = 0;
    [[nodiscard]] f32 pitch() const { return 2.0f * std::numbers::pi_v<f32> * b; }
};

// OLS fit of radius vs total angle. `winding[i]` is the winding number of sample i (so its
// total spiral angle is 2*pi*winding[i]); `radius[i]` its radius from the umbilicus.
inline SpiralParams spiral_fit_lsq(std::span<const f32> winding, std::span<const f32> radius) {
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    const usize n = winding.size();
    SpiralParams p;
    p.nsamples = static_cast<s64>(n);
    if (n < 2) return p;
    f64 st = 0, sr = 0, stt = 0, str = 0;
    for (usize i = 0; i < n; ++i) {
        const f64 t = static_cast<f64>(two_pi) * static_cast<f64>(winding[i]);
        const f64 r = static_cast<f64>(radius[i]);
        st += t;
        sr += r;
        stt += t * t;
        str += t * r;
    }
    const f64 nn = static_cast<f64>(n);
    const f64 denom = nn * stt - st * st;
    if (std::abs(denom) < 1e-12) return p;
    const f64 b = (nn * str - st * sr) / denom;
    const f64 a = (sr - b * st) / nn;
    p.a = static_cast<f32>(a);
    p.b = static_cast<f32>(b);
    f64 sse = 0;
    for (usize i = 0; i < n; ++i) {
        const f64 t = static_cast<f64>(two_pi) * static_cast<f64>(winding[i]);
        const f64 pred = a + b * t;
        const f64 e = pred - static_cast<f64>(radius[i]);
        sse += e * e;
    }
    p.rms = static_cast<f32>(std::sqrt(sse / nn));
    return p;
}

// Sample (winding, radius) over a field's material voxels (stride to subsample) and fit.
inline SpiralParams spiral_fit_from_field(VolumeView<const f32> winding,
                                          const annotate::Umbilicus& umb, s64 stride = 4) {
    std::vector<f32> ws, rs;
    const Extent3 d = winding.dims();
    for (s64 z = 0; z < d.z; z += stride)
        for (s64 y = 0; y < d.y; y += stride)
            for (s64 x = 0; x < d.x; x += stride) {
                f32 r;
                Radian th;
                umb.polar(z, y, x, r, th);
                if (r < 1.0f) continue;
                ws.push_back(winding(z, y, x));
                rs.push_back(r);
            }
    return spiral_fit_lsq(ws, rs);
}

}  // namespace fenix::winding
