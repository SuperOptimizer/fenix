// preprocess/aircut.hpp — "air cut": zero the low-density background of a CT so only papyrus
// remains. A carbonized-scroll CT is weakly BIMODAL — a low background mode (inter-wrap
// low-density carbon / void) and a higher papyrus mode, separated by a valley. Otsu's method
// finds that valley (the threshold maximizing between-class variance); everything below it is
// set to 0. This turns the raw CT into a clean sheet signal: papyrus = high, everything else =
// 0, so a snap/threshold data term can no longer wander into the background. fysics lineage.
#pragma once

#include "core/core.hpp"

#include <vector>

namespace fenix::preprocess {

// Otsu threshold over [lo,hi) using a subsampled histogram (stride per axis — Otsu only needs
// the distribution shape, not every voxel). Returns the threshold in the volume's own units.
template <class T>
inline f32 otsu_threshold(VolumeView<const T> v, f32 lo, f32 hi, int nbins = 256, int stride = 4) {
    const Extent3 d = v.dims();
    std::vector<f64> hist(static_cast<usize>(nbins), 0.0);
    const f32 sc = static_cast<f32>(nbins) / (hi - lo);
    for (s64 z = 0; z < d.z; z += stride)
        for (s64 y = 0; y < d.y; y += stride)
            for (s64 x = 0; x < d.x; x += stride) {
                int b = static_cast<int>((static_cast<f32>(v(z, y, x)) - lo) * sc);
                b = std::clamp(b, 0, nbins - 1);
                hist[static_cast<usize>(b)] += 1.0;
            }
    f64 total = 0, sum = 0;
    for (int i = 0; i < nbins; ++i) { total += hist[static_cast<usize>(i)]; sum += i * hist[static_cast<usize>(i)]; }
    if (total == 0) return lo;
    f64 sumB = 0, wB = 0, maxVar = -1;
    int thr = 0;
    for (int i = 0; i < nbins; ++i) {
        wB += hist[static_cast<usize>(i)];
        if (wB == 0) continue;
        const f64 wF = total - wB;
        if (wF == 0) break;
        sumB += static_cast<f64>(i) * hist[static_cast<usize>(i)];
        const f64 mB = sumB / wB, mF = (sum - sumB) / wF;
        const f64 var = wB * wF * (mB - mF) * (mB - mF);
        if (var > maxVar) { maxVar = var; thr = i; }
    }
    return lo + (static_cast<f32>(thr) + 0.5f) * (hi - lo) / static_cast<f32>(nbins);
}

// Air-cut in place: zero every voxel below the Otsu valley. Returns the threshold used.
template <class T>
inline f32 air_cut(VolumeView<T> v, f32 lo, f32 hi) {
    const f32 thr = otsu_threshold<T>(VolumeView<const T>(v), lo, hi);
    FENIX_DEBUG("preprocess", "air-cut: Otsu threshold {:.0f}", static_cast<double>(thr));
    const Extent3 d = v.dims();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x)
                if (static_cast<f32>(v(z, y, x)) < thr) v(z, y, x) = T(0);
    });
    return thr;
}

}  // namespace fenix::preprocess
