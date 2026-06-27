// test_topo.cpp — Euler characteristic + Betti numbers on known shapes.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "topo/betti.hpp"

using namespace fenix;
using namespace fenix::topo;

static f32 dist2(s64 z, s64 y, s64 x, f32 cz, f32 cy, f32 cx) {
    const f32 dz = static_cast<f32>(z) - cz, dy = static_cast<f32>(y) - cy,
              dx = static_cast<f32>(x) - cx;
    return dz * dz + dy * dy + dx * dx;
}

TEST(solid_ball_is_b0_1) {
    const s64 s = 24;
    Volume<u8> m = Volume<u8>::zeros({s, s, s});
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x)
                if (dist2(z, y, x, 12, 12, 12) <= 49.0f) m(z, y, x) = 1;
    Betti b = betti_numbers(m.view());
    CHECK(b.b0 == 1);
    CHECK(b.b2 == 0);
    CHECK(b.b1 == 0);
    CHECK(b.chi == 1);  // ball: chi = 1
}

TEST(hollow_shell_has_one_cavity) {
    const s64 s = 28;
    Volume<u8> m = Volume<u8>::zeros({s, s, s});
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x) {
                f32 r2 = dist2(z, y, x, 14, 14, 14);
                if (r2 <= 121.0f && r2 >= 49.0f) m(z, y, x) = 1;  // thick shell
            }
    Betti b = betti_numbers(m.view());
    CHECK(b.b0 == 1);
    CHECK(b.b2 == 1);   // enclosed cavity
    CHECK(b.b1 == 0);
    CHECK(b.chi == 2);  // hollow sphere: chi = 2
}

TEST(two_balls_b0_2) {
    const s64 s = 32;
    Volume<u8> m = Volume<u8>::zeros({s, s, s});
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x)
                if (dist2(z, y, x, 8, 8, 8) <= 16.0f || dist2(z, y, x, 24, 24, 24) <= 16.0f)
                    m(z, y, x) = 1;
    Betti b = betti_numbers(m.view());
    CHECK(b.b0 == 2);
    CHECK(b.b2 == 0);
    CHECK(b.chi == 2);  // two balls -> chi = 2
}
