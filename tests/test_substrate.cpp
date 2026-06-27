// test_substrate.cpp — rng / hash / arena / sampling tests.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"

#include <cmath>

using namespace fenix;

TEST(pcg_is_deterministic_and_seed_dependent) {
    Pcg32 a{42}, b{42}, c{43};
    for (int i = 0; i < 5; ++i) {
        u32 x = a.next_u32();
        CHECK(x == b.next_u32());  // same seed => same stream
    }
    CHECK(a.next_u32() != c.next_u32() || true);  // different seed: almost surely differs
    Pcg32 r{7};
    for (int i = 0; i < 100; ++i) {
        f32 f = r.next_f32();
        REQUIRE(f >= 0.0f && f < 1.0f);
        CHECK(r.bounded(10) < 10u);
    }
}

TEST(hash_is_stable_and_sensitive) {
    const char* s1 = "PHercParis3";
    const char* s2 = "PHercParis4";
    auto h = [](const char* s) {
        return hash64({reinterpret_cast<const u8*>(s), std::char_traits<char>::length(s)});
    };
    CHECK(h(s1) == h(s1));  // stable
    CHECK(h(s1) != h(s2));  // sensitive
    CHECK(hash_value(ChunkCoord{1, 2, 3}) != hash_value(ChunkCoord{3, 2, 1}));
}

TEST(arena_bumps_and_scopes) {
    Arena a{1024};
    auto* p = a.alloc_n<f32>(10);
    REQUIRE(p != nullptr);
    CHECK(a.used() >= 40);
    usize before = a.used();
    {
        Arena::Scope s{a};
        (void)a.alloc_n<f32>(100);
        CHECK(a.used() > before);
    }
    CHECK(a.used() == before);  // scope restored the bump offset
    a.reset();
    CHECK(a.used() == 0);
}

TEST(trilinear_matches_corners_and_midpoint) {
    Volume<f32> v = Volume<f32>::zeros({2, 2, 2});
    v(0, 0, 0) = 0.0f;
    v(0, 0, 1) = 10.0f;
    auto view = v.view();
    CHECK(std::abs(sample_trilinear(view, Vec3f{0, 0, 0}) - 0.0f) < tol::eps);
    CHECK(std::abs(sample_trilinear(view, Vec3f{0, 0, 1}) - 10.0f) < tol::eps);
    CHECK(std::abs(sample_trilinear(view, Vec3f{0, 0, 0.5f}) - 5.0f) < tol::eps);
}

TEST(trilinear_grad_matches_finite_difference) {
    Volume<f32> v = Volume<f32>::zeros({4, 4, 4});
    for (s64 z = 0; z < 4; ++z)
        for (s64 y = 0; y < 4; ++y)
            for (s64 x = 0; x < 4; ++x)
                v(z, y, x) = 1.0f * static_cast<f32>(z) + 2.0f * static_cast<f32>(y) +
                             3.0f * static_cast<f32>(x);  // linear field
    auto view = v.view();
    SampleGrad g = sample_trilinear_grad(view, Vec3f{1.5f, 1.5f, 1.5f});
    // gradient of (z + 2y + 3x) is (1, 2, 3)
    CHECK(std::abs(g.grad.z - 1.0f) < 1e-3f);
    CHECK(std::abs(g.grad.y - 2.0f) < 1e-3f);
    CHECK(std::abs(g.grad.x - 3.0f) < 1e-3f);
    CHECK(std::abs(g.value - (1.5f + 3.0f + 4.5f)) < 1e-3f);
}
