// test_guided.cpp — guided filter: reduces flat-region noise, preserves edges.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "preprocess/guided.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::preprocess;

TEST(box_mean_constant_is_unchanged) {
    Volume<f32> v(Extent3{8, 8, 8});
    for (s64 i = 0; i < v.size(); ++i) v.flat()[static_cast<usize>(i)] = 5.0f;
    auto m = box_mean(v.view(), 2);
    for (f32 x : m.flat()) CHECK(std::abs(x - 5.0f) < 1e-4f);
}

TEST(guided_filter_denoises_and_preserves_edge) {
    const s64 s = 32;
    // Step edge along x at x=16 (0 vs 100) + noise.
    Volume<f32> truth(Extent3{s, s, s}), noisy(Extent3{s, s, s});
    Pcg32 rng{5};
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x) {
                f32 t = (x < 16) ? 0.0f : 100.0f;
                truth(z, y, x) = t;
                noisy(z, y, x) = t + (rng.next_f32() - 0.5f) * 20.0f;  // +-10 noise
            }
    auto out = guided_filter(noisy.view(), 3, 50.0f);

    // Flat-region noise reduced: std over a far-from-edge slab drops.
    auto region_std = [&](VolumeView<const f32> v, s64 x0, s64 x1) {
        f64 sum = 0, sum2 = 0;
        s64 n = 0;
        for (s64 z = 4; z < s - 4; ++z)
            for (s64 y = 4; y < s - 4; ++y)
                for (s64 x = x0; x < x1; ++x) {
                    f64 val = static_cast<f64>(v(z, y, x));
                    sum += val;
                    sum2 += val * val;
                    ++n;
                }
        f64 mean = sum / static_cast<f64>(n);
        return std::sqrt(sum2 / static_cast<f64>(n) - mean * mean);
    };
    CHECK(region_std(out.view(), 2, 12) < region_std(noisy.view(), 2, 12) * 0.6);

    // Edge preserved: still a large jump across x=16.
    f32 left = out(16, 16, 8), right = out(16, 16, 24);
    CHECK((right - left) > 80.0f);
}
