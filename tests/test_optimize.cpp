// test_optimize.cpp — AdamW converges on convex problems.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"

#include <array>
#include <cmath>
#include <vector>

using namespace fenix;

TEST(adamw_minimizes_quadratic) {
    // f(x) = sum (x_i - target_i)^2 ; grad = 2(x - target).
    std::array<f32, 4> target{3.0f, -1.5f, 7.0f, 0.25f};
    std::vector<f32> x{0, 0, 0, 0};
    minimize(
        x,
        [&](std::span<const f32> p, std::span<f32> g) {
            for (usize i = 0; i < p.size(); ++i) g[i] = 2.0f * (p[i] - target[i]);
        },
        2000, {.lr = 0.05f});
    for (usize i = 0; i < x.size(); ++i) CHECK(std::abs(x[i] - target[i]) < 1e-2f);
}

TEST(adamw_fits_line) {
    // Fit y = a*t + b to noiseless samples; recover a,b.
    const f32 a0 = 2.5f, b0 = -1.0f;
    std::vector<f32> t, y;
    for (int i = 0; i < 50; ++i) {
        f32 ti = static_cast<f32>(i) * 0.1f;
        t.push_back(ti);
        y.push_back(a0 * ti + b0);
    }
    std::vector<f32> p{0, 0};  // a, b
    minimize(
        p,
        [&](std::span<const f32> pr, std::span<f32> g) {
            g[0] = 0;
            g[1] = 0;
            for (usize i = 0; i < t.size(); ++i) {
                f32 e = (pr[0] * t[i] + pr[1]) - y[i];
                g[0] += 2.0f * e * t[i];
                g[1] += 2.0f * e;
            }
            g[0] /= static_cast<f32>(t.size());
            g[1] /= static_cast<f32>(t.size());
        },
        5000, {.lr = 0.05f});
    CHECK(std::abs(p[0] - a0) < 0.05f);
    CHECK(std::abs(p[1] - b0) < 0.05f);
}
