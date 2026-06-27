// test_flow.cpp — SVF/RK4 flow: the diffeomorphism is invertible (T then T^-1 ~= identity).
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "winding/flow.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::winding;

// A smooth swirling velocity field on a side^3 lattice (rotation about the volume centre +
// a gentle radial component) — a nontrivial deformation.
static FlowField swirl(s64 s) {
    FlowField f{Volume<f32>::zeros({s, s, s}), Volume<f32>::zeros({s, s, s}),
                Volume<f32>::zeros({s, s, s})};
    const f32 c = static_cast<f32>(s) * 0.5f;
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x) {
                const f32 dy = static_cast<f32>(y) - c, dx = static_cast<f32>(x) - c;
                f.vz(z, y, x) = 0.0f;
                f.vy(z, y, x) = -0.15f * dx;  // tangential -> rotation
                f.vx(z, y, x) = 0.15f * dy;
            }
    return f;
}

TEST(flow_forward_then_inverse_recovers_point) {
    const s64 s = 48;
    FlowField f = swirl(s);
    Vec3f p0{24.0f, 16.0f, 30.0f};
    Vec3f fwd = flow_point(f, p0, 12, +1.0f);
    Vec3f back = flow_point(f, fwd, 12, -1.0f);
    CHECK(norm(back - p0) < 0.5f);   // T^-1(T(p)) ~= p (invertibility / no fold)
    CHECK(norm(fwd - p0) > 1.0f);    // and the flow actually moved the point
}

TEST(zero_flow_is_identity) {
    const s64 s = 16;
    FlowField f{Volume<f32>::zeros({s, s, s}), Volume<f32>::zeros({s, s, s}),
                Volume<f32>::zeros({s, s, s})};
    Vec3f p{8.0f, 5.0f, 11.0f};
    Vec3f q = flow_point(f, p, 8, +1.0f);
    CHECK(norm(q - p) < 1e-4f);
}

TEST(flow_more_steps_more_accurate_inverse) {
    const s64 s = 48;
    FlowField f = swirl(s);
    Vec3f p0{24.0f, 14.0f, 28.0f};
    auto roundtrip_err = [&](int steps) {
        Vec3f fwd = flow_point(f, p0, steps, +1.0f);
        Vec3f back = flow_point(f, fwd, steps, -1.0f);
        return norm(back - p0);
    };
    CHECK(roundtrip_err(24) <= roundtrip_err(4) + 1e-4f);  // finer integration -> not worse
}
