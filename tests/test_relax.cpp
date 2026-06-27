// test_relax.cpp — Tikhonov Gauss-Seidel winding-field smoothing reduces noise.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "winding/relax.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::winding;

static f64 rmse(VolumeView<const f32> a, VolumeView<const f32> b) {
    f64 s = 0;
    const s64 n = a.size();
    for (s64 i = 0; i < n; ++i) {
        const f64 e = static_cast<f64>(a.flat()[static_cast<usize>(i)]) -
                      static_cast<f64>(b.flat()[static_cast<usize>(i)]);
        s += e * e;
    }
    return std::sqrt(s / static_cast<f64>(n));
}

TEST(relax_denoises_smooth_field) {
    const s64 s = 32;
    // Ground-truth smooth field (a radial ramp, no branch cut).
    Volume<f32> truth(Extent3{s, s, s});
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x)
                truth(z, y, x) = 0.1f * static_cast<f32>(z) + 0.2f * static_cast<f32>(y) +
                                 0.3f * static_cast<f32>(x);
    // Noisy observation.
    Volume<f32> noisy(Extent3{s, s, s});
    Pcg32 rng{4};
    for (s64 i = 0; i < truth.size(); ++i)
        noisy.flat()[static_cast<usize>(i)] =
            truth.flat()[static_cast<usize>(i)] + (rng.next_f32() - 0.5f) * 4.0f;

    f64 before = rmse(noisy.view(), truth.view());
    auto smoothed = relax(noisy.view(), {.iters = 80, .lambda = 0.05f});
    f64 after = rmse(smoothed.view(), truth.view());
    CHECK(after < before * 0.5);  // smoothing roughly halves the error (at least)
}

TEST(relax_preserves_constant_field) {
    const s64 s = 16;
    Volume<f32> c(Extent3{s, s, s});
    for (s64 i = 0; i < c.size(); ++i) c.flat()[static_cast<usize>(i)] = 7.0f;
    auto out = relax(c.view(), {.iters = 30, .lambda = 0.1f});
    CHECK(rmse(out.view(), c.view()) < 1e-3);  // constant is a fixed point
}
