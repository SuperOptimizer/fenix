// test_surface_render.cpp — flatten a wrap then render +/- normal layers from the volume.
#define FENIX_TEST_MAIN
#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "core/test.hpp"
#include "flatten/extract_wrap.hpp"
#include "render/surface_render.hpp"
#include "winding/winding_field.hpp"

#include <cmath>
#include <numbers>

using namespace fenix;

TEST(render_surface_layers_sample_volume) {
    const s64 side = 80;
    const f32 cy = 40.0f, cx = 40.0f, pitch = 8.0f;
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;

    (void)two_pi;
    annotate::Umbilicus u;
    for (s64 z = 0; z < side; ++z) {
        u.z.push_back(static_cast<f32>(z));
        u.y.push_back(cy);
        u.x.push_back(cx);
    }
    Volume<f32> w = winding::winding_init({side, side, side}, u, {.pitch = pitch});

    // Render the WINDING field itself: the winding=3 surface must sample ~3 at offset 0.
    const f32 target = 3.0f;
    Surface s = flatten::extract_winding_surface(w.view(), u, target, 120, 38);
    auto stack = render::render_surface(s, w.view(), {.num_layers = 4, .step = 1.0f});
    REQUIRE(stack.dims().z == 9);     // 2*4+1 layers
    REQUIRE(stack.dims().y == side);  // v
    REQUIRE(stack.dims().x == 120);   // u

    // Central layer (offset 0) lies on the iso-winding surface -> samples ~= target.
    f64 center_sum = 0;
    s64 cnt = 0;
    for (s64 v = 0; v < side; ++v)
        for (s64 uu = 0; uu < 120; ++uu) {
            f32 val = stack.view()(4, v, uu);  // layer 4 = offset 0
            if (s.is_valid(uu, v)) {
                center_sum += static_cast<f64>(val);
                ++cnt;
            }
        }
    REQUIRE(cnt > 0);
    CHECK(std::abs(center_sum / static_cast<f64>(cnt) - static_cast<f64>(target)) < 0.3);
}
