// test_spiral_fit.cpp — closed-form Archimedean spiral OLS fit.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "winding/spiral_fit.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using namespace fenix;
using namespace fenix::winding;

TEST(spiral_fit_recovers_params) {
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    const f32 a0 = 5.0f, b0 = 1.3f;  // r = a0 + b0*theta
    std::vector<f32> wind, rad;
    for (int i = 0; i < 200; ++i) {
        f32 w = static_cast<f32>(i) * 0.05f;       // winding 0..10
        f32 theta = two_pi * w;
        wind.push_back(w);
        rad.push_back(a0 + b0 * theta);
    }
    auto p = spiral_fit_lsq(wind, rad);
    CHECK(std::abs(p.a - a0) < 1e-2f);
    CHECK(std::abs(p.b - b0) < 1e-3f);
    CHECK(p.rms < 1e-2f);
    CHECK(std::abs(p.pitch() - two_pi * b0) < 1e-2f);
}

TEST(spiral_fit_handles_noise) {
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    const f32 a0 = 2.0f, b0 = 0.8f;
    std::vector<f32> wind, rad;
    Pcg32 rng{9};
    for (int i = 0; i < 500; ++i) {
        f32 w = static_cast<f32>(i) * 0.02f;
        wind.push_back(w);
        rad.push_back(a0 + b0 * two_pi * w + (rng.next_f32() - 0.5f) * 2.0f);  // +-1 noise
    }
    auto p = spiral_fit_lsq(wind, rad);
    CHECK(std::abs(p.b - b0) < 0.05f);  // slope robust to zero-mean noise
}

TEST(spiral_fit_degenerate_returns_safely) {
    std::vector<f32> w{1.0f}, r{3.0f};
    auto p = spiral_fit_lsq(w, r);  // < 2 samples
    CHECK(p.nsamples == 1);
    CHECK(p.b == 0.0f);
}
