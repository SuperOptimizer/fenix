// winding/relax.hpp — masked winding-field regularization: Tikhonov-smoothing via red-black
// Gauss-Seidel (minimize ∫|∇W|² + λ∫(W-W0)²). The isotropic core of the winding solve;
// anisotropic (sheet-tensor-weighted) + TV variants build on this. See winding/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <vector>

namespace fenix::winding {

struct RelaxParams {
    int iters = 50;     // Gauss-Seidel sweeps
    f32 lambda = 0.1f;  // data-fidelity weight (small => stronger smoothing)
};

// Smooth a (noisy) winding field. Each sweep: W <- (Σ 6-neighbours + λ·W0) / (6 + λ),
// red-black so each colour is parallel-safe. Reflect boundary (via clamp).
inline Volume<f32> relax(VolumeView<const f32> w0, RelaxParams p) {
    const Extent3 d = w0.dims();
    Volume<f32> w(d);
    for (s64 i = 0; i < d.count(); ++i) w.flat()[static_cast<usize>(i)] = w0.flat()[static_cast<usize>(i)];
    VolumeView<f32> wv = w.view();
    const f32 denom = 6.0f + p.lambda;
    for (int it = 0; it < p.iters; ++it)
        for (int color = 0; color < 2; ++color)
            parallel_for_z(d, [&](s64 z) {
                for (s64 y = 0; y < d.y; ++y)
                    for (s64 x = 0; x < d.x; ++x) {
                        if (((z + y + x) & 1) != color) continue;
                        const f32 sum = wv.at_clamped(z - 1, y, x) + wv.at_clamped(z + 1, y, x) +
                                        wv.at_clamped(z, y - 1, x) + wv.at_clamped(z, y + 1, x) +
                                        wv.at_clamped(z, y, x - 1) + wv.at_clamped(z, y, x + 1);
                        wv(z, y, x) = (sum + p.lambda * w0(z, y, x)) / denom;
                    }
            });
    return w;
}

}  // namespace fenix::winding
