// test_predictions.cpp — prediction-field normalization + thresholding.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "predictions/field.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::predictions;

TEST(minmax_normalizes_to_unit_range) {
    Volume<f32> v(Extent3{4, 4, 4});
    for (s64 i = 0; i < v.size(); ++i) v.flat()[static_cast<usize>(i)] = 10.0f + static_cast<f32>(i);
    auto n = normalize(v.view(), Norm::MinMax);
    f32 lo = 1, hi = 0;
    for (f32 x : n.flat()) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
    }
    CHECK(std::abs(lo - 0.0f) < 1e-6f);
    CHECK(std::abs(hi - 1.0f) < 1e-6f);
}

TEST(percentile_clips_outliers) {
    Volume<f32> v = Volume<f32>::zeros(Extent3{10, 10, 10});
    for (s64 i = 0; i < v.size(); ++i) v.flat()[static_cast<usize>(i)] = 0.5f;
    v.flat()[0] = 1000.0f;  // outlier
    auto n = normalize(v.view(), Norm::Percentile, 0.01f, 0.99f);
    // The outlier is clipped to 1; the bulk (0.5) maps near the middle/low, not crushed to 0.
    CHECK(n.flat()[0] <= 1.0f);
    CHECK(n.flat()[500] >= 0.0f && n.flat()[500] <= 1.0f);
}

TEST(threshold_makes_mask) {
    Volume<f32> v(Extent3{4, 4, 4});
    for (s64 i = 0; i < v.size(); ++i) v.flat()[static_cast<usize>(i)] = static_cast<f32>(i) / 63.0f;
    auto m = threshold(v.view(), 0.5f);
    CHECK(m.flat()[0] == 0);
    CHECK(m.flat()[63] == 1);
    s64 on = 0;
    for (u8 b : m.flat()) on += b;
    CHECK(on > 0 && on < 64);
}
