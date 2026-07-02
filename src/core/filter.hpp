// core/filter.hpp — shared separable filters (Gaussian blur, central-difference gradient).
// One copy for the whole codebase (taberna duplicated the separable Gaussian repeatedly).
#pragma once

#include "core/parallel.hpp"
#include "core/types.hpp"
#include "core/vec.hpp"
#include "core/volume.hpp"

#include <cmath>
#include <vector>

namespace fenix {

// Separable Gaussian blur in place over a contiguous f32 volume. Reflect boundary.
inline void gaussian_blur(VolumeView<f32> v, f32 sigma) {
    if (sigma <= 0.0f) return;
    const s64 r = static_cast<s64>(std::ceil(3.0f * sigma));
    std::vector<f32> k(static_cast<usize>(2 * r + 1));
    f32 sum = 0;
    for (s64 i = -r; i <= r; ++i) {
        f32 w = std::exp(-0.5f * static_cast<f32>(i * i) / (sigma * sigma));
        k[static_cast<usize>(i + r)] = w;
        sum += w;
    }
    for (auto& w : k) w /= sum;

    const Extent3 d = v.dims();
    // Single-application reflect only maps [-(n-1), 2n-2] back in range; the small-line branch
    // below can see |offset| > n-1 (kernel radius >= line length), so iterate until in-range.
    // n==1 is a fixed point of the reflection (1 -> -1 -> 1 forever) and must short-circuit.
    auto reflect = [](s64 i, s64 n) {
        if (n <= 1) return s64{0};
        while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i;
        return i;
    };

    // Parallel over the outer index `a`: distinct `a` touch disjoint lines, so the in-place blur
    // is race-free. `line` is per-iteration (thread-local) scratch. (Nested inside an already-
    // parallel region OpenMP runs this serially — no oversubscription.)
    auto blur_axis = [&](int axis) {
        const s64 nz = d.z, ny = d.y, nx = d.x;
        const s64 len = (axis == 0) ? nz : (axis == 1) ? ny : nx;
        const s64 o1 = (axis == 0) ? ny : nz;
        const s64 o2 = (axis == 0) ? nx : (axis == 1) ? nx : ny;
        const s64 ksz = 2 * r + 1;
        const f32* kp = k.data();
        parallel_for(0, o1, [&](s64 a) {
            std::vector<f32> line(static_cast<usize>(len));
            for (s64 b = 0; b < o2; ++b) {
                auto idx = [&](s64 t) -> f32& {
                    if (axis == 0) return v(t, a, b);
                    if (axis == 1) return v(a, t, b);
                    return v(a, b, t);
                };
                for (s64 t = 0; t < len; ++t) line[static_cast<usize>(t)] = idx(t);
                const f32* lp = line.data();
                if (len >= ksz) {
                    // Interior [r, len-r): no boundary reflection -> branch-free, vectorizable.
                    for (s64 t = r; t < len - r; ++t) {
                        f32 acc = 0;
                        const f32* w = &lp[t - r];
                        for (s64 j = 0; j < ksz; ++j) acc += kp[j] * w[j];
                        idx(t) = acc;
                    }
                    // Two borders with reflection.
                    auto border = [&](s64 t) {
                        f32 acc = 0;
                        for (s64 j = -r; j <= r; ++j) acc += kp[j + r] * lp[reflect(t + j, len)];
                        idx(t) = acc;
                    };
                    for (s64 t = 0; t < r; ++t) border(t);
                    for (s64 t = len - r; t < len; ++t) border(t);
                } else {
                    for (s64 t = 0; t < len; ++t) {
                        f32 acc = 0;
                        for (s64 j = -r; j <= r; ++j) acc += kp[j + r] * lp[reflect(t + j, len)];
                        idx(t) = acc;
                    }
                }
            }
        });
    };
    blur_axis(2);
    blur_axis(1);
    blur_axis(0);
}

// Central-difference gradient at an interior-or-clamped voxel (ZYX).
inline Vec3f gradient_at(VolumeView<const f32> v, s64 z, s64 y, s64 x) {
    return {0.5f * (v.at_clamped(z + 1, y, x) - v.at_clamped(z - 1, y, x)),
            0.5f * (v.at_clamped(z, y + 1, x) - v.at_clamped(z, y - 1, x)),
            0.5f * (v.at_clamped(z, y, x + 1) - v.at_clamped(z, y, x - 1))};
}

}  // namespace fenix
