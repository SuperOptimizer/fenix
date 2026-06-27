// preprocess/guided.hpp — He-Sun-Tang guided filter (self-guided), the fysics-recommended
// O(N) edge-preserving denoiser. out = a*I + b with a = var/(var+eps), box-averaged.
// eps (from the measured noise floor) sets the smoothing strength. See preprocess/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <vector>

namespace fenix::preprocess {

// Separable box mean over a (2r+1)^3 window, reflect boundary.
inline Volume<f32> box_mean(VolumeView<const f32> v, s64 r) {
    const Extent3 d = v.dims();
    Volume<f32> a(d);
    for (s64 i = 0; i < d.count(); ++i) a.flat()[static_cast<usize>(i)] = v.flat()[static_cast<usize>(i)];
    const f32 inv = 1.0f / static_cast<f32>(2 * r + 1);
    std::vector<f32> line, acc;
    auto reflect = [](s64 i, s64 n) { return i < 0 ? -i : (i >= n ? 2 * (n - 1) - i : i); };
    VolumeView<f32> av = a.view();
    auto pass = [&](int axis) {
        const s64 n = (axis == 0) ? d.z : (axis == 1) ? d.y : d.x;
        const s64 o1 = (axis == 0) ? d.y : d.z;
        const s64 o2 = (axis == 0) ? d.x : (axis == 1) ? d.x : d.y;
        line.resize(static_cast<usize>(n));
        for (s64 p = 0; p < o1; ++p)
            for (s64 q = 0; q < o2; ++q) {
                auto at = [&](s64 t) -> f32& {
                    if (axis == 0) return av(t, p, q);
                    if (axis == 1) return av(p, t, q);
                    return av(p, q, t);
                };
                for (s64 t = 0; t < n; ++t) line[static_cast<usize>(t)] = at(t);
                for (s64 t = 0; t < n; ++t) {
                    f32 s = 0;
                    for (s64 k = -r; k <= r; ++k) s += line[static_cast<usize>(reflect(t + k, n))];
                    at(t) = s * inv;
                }
            }
    };
    pass(2);
    pass(1);
    pass(0);
    return a;
}

// Self-guided filter. Larger eps -> more smoothing; eps ~ (noise_std)^2.
inline Volume<f32> guided_filter(VolumeView<const f32> img, s64 r, f32 eps) {
    const Extent3 d = img.dims();
    Volume<f32> sq(d);
    for (s64 i = 0; i < d.count(); ++i) {
        const f32 v = img.flat()[static_cast<usize>(i)];
        sq.flat()[static_cast<usize>(i)] = v * v;
    }
    Volume<f32> mean_i = box_mean(img, r);
    Volume<f32> mean_ii = box_mean(sq.view(), r);

    Volume<f32> a(d), b(d);
    for (s64 i = 0; i < d.count(); ++i) {
        const f32 mi = mean_i.flat()[static_cast<usize>(i)];
        const f32 var = mean_ii.flat()[static_cast<usize>(i)] - mi * mi;
        const f32 ai = var / (var + eps);
        a.flat()[static_cast<usize>(i)] = ai;
        b.flat()[static_cast<usize>(i)] = mi * (1.0f - ai);
    }
    Volume<f32> mean_a = box_mean(a.view(), r);
    Volume<f32> mean_b = box_mean(b.view(), r);

    Volume<f32> out(d);
    for (s64 i = 0; i < d.count(); ++i)
        out.flat()[static_cast<usize>(i)] =
            mean_a.flat()[static_cast<usize>(i)] * img.flat()[static_cast<usize>(i)] +
            mean_b.flat()[static_cast<usize>(i)];
    return out;
}

}  // namespace fenix::preprocess
