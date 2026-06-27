// test_unroll.cpp — winding-field unroll (forward-scatter flatten).
#define FENIX_TEST_MAIN
#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "core/test.hpp"
#include "render/unroll.hpp"
#include "winding/winding_field.hpp"

#include <cmath>

using namespace fenix;

TEST(unroll_straightens_constant_winding) {
    const s64 side = 64;
    const f32 pitch = 8.0f;
    annotate::Umbilicus u;
    for (s64 z = 0; z < side; ++z) {
        u.z.push_back(static_cast<f32>(z));
        u.y.push_back(32.0f);
        u.x.push_back(32.0f);
    }
    Volume<f32> w = winding::winding_init({side, side, side}, u, {.pitch = pitch});

    // Use the winding value itself as the "CT": after unrolling, every output column
    // (a fixed winding bin) should read back ~that winding, identically across all rows z.
    auto img = render::unroll(w.view(), w.view(), {.samp = 4.0f});
    REQUIRE(img.dims().z == 1);
    REQUIRE(img.dims().y == side);  // one row per z
    REQUIRE(img.dims().x >= 1);

    auto iv = img.view();
    const s64 width = img.dims().x;
    // Pick a populated middle column and check it's ~constant across rows (spiral straightened).
    s64 col = width / 2;
    f32 ref = iv(0, side / 2, col);
    if (ref != 0.0f) {  // ensure the column has data
        f32 max_dev = 0;
        int rows_with_data = 0;
        for (s64 z = 0; z < side; ++z) {
            f32 v = iv(0, z, col);
            if (v != 0.0f) {
                max_dev = std::max(max_dev, std::abs(v - ref));
                ++rows_with_data;
            }
        }
        CHECK(rows_with_data > side / 2);  // the column is well populated across z
        CHECK(max_dev < 0.5f);             // constant-winding column reads back ~constant
    }
}

TEST(unroll_columns_increase_with_winding) {
    const s64 side = 48;
    annotate::Umbilicus u;
    for (s64 z = 0; z < side; ++z) {
        u.z.push_back(static_cast<f32>(z));
        u.y.push_back(24.0f);
        u.x.push_back(24.0f);
    }
    Volume<f32> w = winding::winding_init({side, side, side}, u, {.pitch = 6.0f});
    auto img = render::unroll(w.view(), w.view(), {.samp = 4.0f});
    auto iv = img.view();
    const s64 width = img.dims().x;
    // Mean over rows for two columns: a later column = higher winding -> higher value.
    auto col_mean = [&](s64 c) {
        f32 s = 0;
        int n = 0;
        for (s64 z = 0; z < side; ++z)
            if (iv(0, z, c) != 0.0f) {
                s += iv(0, z, c);
                ++n;
            }
        return n ? s / static_cast<f32>(n) : 0.0f;
    };
    f32 lo = col_mean(width / 4), hi = col_mean(3 * width / 4);
    CHECK(hi > lo);
}
