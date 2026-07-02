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
    auto res = render::render_surface(s, w.view(), {.num_layers = 4, .step = 1.0f});
    REQUIRE(res.stack.dims().z == 9);     // 2*4+1 layers
    REQUIRE(res.stack.dims().y == side);  // v
    REQUIRE(res.stack.dims().x == 120);   // u
    REQUIRE(res.mask.dims().y == side);
    REQUIRE(res.mask.dims().x == 120);

    // Central layer (offset 0) lies on the iso-winding surface -> samples ~= target.
    f64 center_sum = 0;
    s64 cnt = 0;
    for (s64 v = 0; v < side; ++v)
        for (s64 uu = 0; uu < 120; ++uu) {
            f32 val = res.stack.view()(4, v, uu);  // layer 4 = offset 0
            const bool m = res.mask.view()(0, v, uu) != 0;
            if (s.is_valid(uu, v)) CHECK(m == true || val == 0.0f);  // valid input doesn't force valid output
            if (m) {
                center_sum += static_cast<f64>(val);
                ++cnt;
            }
        }
    REQUIRE(cnt > 0);
    CHECK(std::abs(center_sum / static_cast<f64>(cnt) - static_cast<f64>(target)) < 0.3);
}

// render_surface must never fabricate tangents/normals from an invalid neighbour's zero coord: at a
// valid-region border (isolated valid strip with invalid neighbours on one side), the mask must be
// either unset or the sampled value must be consistent with a real one-sided tangent, never a huge
// garbage offset toward the world origin.
TEST(render_surface_border_no_garbage_normal) {
    // A flat sheet in the y=const plane occupying u in [0,10), with valid cells only for u<5 so
    // u=4's +1 neighbour (u=5) is invalid -> forces the one-sided-difference path.
    Surface s(10, 1);
    for (s64 u = 0; u < 5; ++u) s.set(u, 0, Vec3f{10.0f, 20.0f, static_cast<f32>(u)});

    Volume<f32> vol = Volume<f32>::zeros({64, 64, 64});
    for (s64 z = 0; z < 64; ++z)
        for (s64 y = 0; y < 64; ++y)
            for (s64 x = 0; x < 64; ++x) vol.view()(z, y, x) = static_cast<f32>(x);

    auto res = render::render_surface(s, vol.view(), {.num_layers = 2, .step = 1.0f});
    for (s64 u = 0; u < 5; ++u) {
        const bool m = res.mask.view()(0, 0, u) != 0;
        if (!m) continue;  // acceptable to mark invalid (isolated cell), never garbage
        for (s64 l = 0; l < 5; ++l) {
            const f32 v = res.stack.view()(l, 0, u);
            CHECK(std::abs(v - static_cast<f32>(u)) < 3.0f);  // near the sheet's own x, not a huge jump
        }
    }
}
