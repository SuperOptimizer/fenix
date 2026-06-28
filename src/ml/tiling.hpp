// ml/tiling.hpp — torch-free sliding-window tiling math (patch start positions + Gaussian
// importance weights). Split out from infer.hpp so it is unit-testable in the core build
// (no libtorch). Pure functions over the ZYX conventions; used by ml/infer.hpp.
#pragma once

#include "core/core.hpp"

#include <cmath>
#include <vector>

namespace fenix::ml {

// 1-D Gaussian importance weights for a patch of length n, peak-normalized to 1 with a 0.1
// floor (nnU-Net-style; sigma = n/8). Heavier weight at the patch center so seams between
// overlapping tiles are smooth.
inline std::vector<float> gaussian1d(int n) {
    const double c = (n - 1) / 2.0, sigma = n / 8.0;
    std::vector<float> w(static_cast<std::size_t>(n));
    double mx = 0.0;
    for (int i = 0; i < n; ++i) {
        const double dd = (i - c) / sigma;
        w[static_cast<std::size_t>(i)] = static_cast<float>(std::exp(-0.5 * dd * dd));
        mx = std::max(mx, static_cast<double>(w[static_cast<std::size_t>(i)]));
    }
    for (auto& v : w) v = std::max(static_cast<float>(v / mx), 0.1f);
    return w;
}

// Patch start positions along one axis covering [0, dim): 0, step, ... , dim-patch. The last
// start is always clamped to dim-patch so the far edge is fully covered (overlap absorbs it).
// dim <= patch -> a single patch at 0 (caller edge-clamps the read).
inline std::vector<s64> tile_starts(s64 dim, int patch, double overlap) {
    std::vector<s64> starts;
    if (dim <= patch) { starts.push_back(0); return starts; }
    const s64 step = std::max<s64>(1, static_cast<s64>(patch * (1.0 - overlap)));
    for (s64 s = 0;; s += step) {
        const s64 clamped = std::min(s, dim - patch);
        if (starts.empty() || starts.back() != clamped) starts.push_back(clamped);
        if (clamped == dim - patch) break;
    }
    return starts;
}

}  // namespace fenix::ml
