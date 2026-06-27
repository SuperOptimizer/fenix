// test_spiral_model.cpp — the unified compositional diffeomorphism is globally invertible
// and reads back a sensible winding number.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "winding/spiral_model.hpp"

#include <cmath>
#include <numbers>

using namespace fenix;
using namespace fenix::winding;

static SpiralModel make_model(s64 s) {
    SpiralModel m;
    for (s64 z = 0; z < s; ++z) {
        m.umbilicus.z.push_back(static_cast<f32>(z));
        m.umbilicus.y.push_back(static_cast<f32>(s) * 0.5f);
        m.umbilicus.x.push_back(static_cast<f32>(s) * 0.5f);
    }
    m.affine = AffineYX{.a = 0.05f, .b = 0.1f, .c = -0.08f, .d = 0.03f, .ty = 1.0f, .tx = -1.5f};
    // z-preserving swirl flow.
    m.flow = FlowField{Volume<f32>::zeros({s, s, s}), Volume<f32>::zeros({s, s, s}),
                       Volume<f32>::zeros({s, s, s})};
    const f32 c = static_cast<f32>(s) * 0.5f;
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x) {
                m.flow.vy(z, y, x) = -0.08f * (static_cast<f32>(x) - c);
                m.flow.vx(z, y, x) = 0.08f * (static_cast<f32>(y) - c);
            }
    m.has_flow = true;
    m.flow_steps = 10;
    m.dr_per_winding = 8.0f;
    return m;
}

TEST(spiral_model_is_globally_invertible) {
    const s64 s = 64;
    SpiralModel m = make_model(s);
    for (Vec3f p : {Vec3f{32, 20, 40}, Vec3f{10, 48, 30}, Vec3f{50, 35, 35}}) {
        Vec3f q = m.to_canonical(p);
        Vec3f back = m.to_scroll(q);
        CHECK(norm(back - p) < 0.5f);  // T^-1(T(p)) ~= p (no fold, globally invertible)
    }
}

TEST(spiral_model_winding_increases_outward) {
    const s64 s = 64;
    SpiralModel m = make_model(s);
    m.has_flow = false;  // pure affine+gap for a monotonic radial check along +x
    m.affine = AffineYX{};  // identity
    const f32 c = 32.0f;
    f32 prev = -1e9f;
    for (s64 r = 2; r < 28; ++r) {
        f32 w = m.winding_at(Vec3f{32.0f, c, c + static_cast<f32>(r)});  // +x ray, theta~0
        CHECK(w > prev);
        prev = w;
    }
    // one pitch out (8 voxels) ~ +1 winding
    f32 w0 = m.winding_at(Vec3f{32, c, c + 8});
    f32 w1 = m.winding_at(Vec3f{32, c, c + 16});
    CHECK(std::abs((w1 - w0) - 1.0f) < 0.2f);
}
