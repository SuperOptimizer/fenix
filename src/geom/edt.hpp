// geom/edt.hpp — exact squared Euclidean distance transform (Felzenszwalb-Huttenlocher),
// separable 1D-per-axis. Distance from each voxel to the nearest seed voxel. Used by
// NSD/eval, sheet repair, skeletonization, touching-sheet separation. One copy (geom).
#pragma once

#include "core/core.hpp"

#include <vector>

namespace fenix::geom {

namespace detail {
// Large finite stand-in for +inf (fast-math forbids real inf). Bigger than any squared
// distance in a 2^18/axis volume (max ~3*(2^18)^2 ~ 2^37) with headroom.
inline constexpr f32 edt_big = 1e18f;

// 1D squared distance transform of sampled function f (length n) -> d. Lower envelope of
// parabolas (FH 2004). v/z are scratch (size n and n+1).
inline void edt1d(const f32* f, s64 n, f32* d, s64* v, f32* z) {
    s64 k = 0;
    v[0] = 0;
    z[0] = -edt_big;
    z[1] = edt_big;
    for (s64 q = 1; q < n; ++q) {
        f32 s;
        for (;;) {
            const f32 vk = static_cast<f32>(v[k]);
            s = ((f[q] + static_cast<f32>(q) * static_cast<f32>(q)) - (f[v[k]] + vk * vk)) /
                (2.0f * static_cast<f32>(q) - 2.0f * vk);
            if (s > z[k]) break;
            --k;
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = edt_big;
    }
    k = 0;
    for (s64 q = 0; q < n; ++q) {
        while (z[k + 1] < static_cast<f32>(q)) ++k;
        const f32 dq = static_cast<f32>(q) - static_cast<f32>(v[k]);
        d[q] = dq * dq + f[v[k]];
    }
}
}  // namespace detail

// Squared Euclidean distance from each voxel to the nearest `seed` voxel (seed(z,y,x)!=0).
// Returns an f32 volume of squared distances (0 at seeds).
inline Volume<f32> edt_squared(VolumeView<const u8> seed) {
    const Extent3 d = seed.dims();
    Volume<f32> out(d);
    VolumeView<f32> ov = out.view();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x)
                ov(z, y, x) = seed(z, y, x) ? 0.0f : detail::edt_big;
    });

    const s64 maxlen = std::max({d.z, d.y, d.x});

    // Each 1D line is independent — parallelize over the first orthogonal axis, per-task scratch.
    auto run_axis = [&](int axis) {
        const s64 n = (axis == 0) ? d.z : (axis == 1) ? d.y : d.x;
        const s64 o1 = (axis == 0) ? d.y : d.z;
        const s64 o2 = (axis == 0) ? d.x : (axis == 1) ? d.x : d.y;
        parallel_for(0, o1, [&](s64 a) {
            std::vector<f32> f(static_cast<usize>(maxlen)), dd(static_cast<usize>(maxlen));
            std::vector<f32> zz(static_cast<usize>(maxlen + 1));
            std::vector<s64> vv(static_cast<usize>(maxlen));
            for (s64 b = 0; b < o2; ++b) {
                auto at = [&](s64 t) -> f32& {
                    if (axis == 0) return ov(t, a, b);
                    if (axis == 1) return ov(a, t, b);
                    return ov(a, b, t);
                };
                for (s64 t = 0; t < n; ++t) f[static_cast<usize>(t)] = at(t);
                detail::edt1d(f.data(), n, dd.data(), vv.data(), zz.data());
                for (s64 t = 0; t < n; ++t) at(t) = dd[static_cast<usize>(t)];
            }
        });
    };
    run_axis(2);
    run_axis(1);
    run_axis(0);
    return out;
}

}  // namespace fenix::geom
