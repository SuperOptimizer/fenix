// test_core.cpp — sample test (per-test-file binary). Define FENIX_TEST_MAIN for main().
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"

using namespace fenix;

TEST(error_carries_code_and_message) {
    Expected<int> e = err(Errc::not_found, "nope");
    REQUIRE(!e.has_value());
    CHECK(e.error().code == Errc::not_found);
    CHECK(e.error().message == "nope");
}

TEST(stage_registry_roundtrip) {
    int before = static_cast<int>(registry().size());
    register_stage(Stage{"__test_stage", "help", [](std::span<const std::string_view>,
                                                    Context&) -> Expected<int> {
                             return 0;
                         }});
    CHECK(static_cast<int>(registry().size()) == before + 1);
    REQUIRE(find_stage("__test_stage") != nullptr);
    CHECK(find_stage("does_not_exist") == nullptr);
}

TEST(version_is_set) { CHECK(!version.empty()); }

TEST(vec_cross_is_right_handed_and_bilinear) {
    // x-hat cross y-hat = z-hat (ZYX storage: {z,y,x})
    const Vec3f xh{0, 0, 1}, yh{0, 1, 0}, zh{1, 0, 0};
    const Vec3f c = cross(xh, yh);
    CHECK(std::abs(c.z - 1.0f) < 1e-6f);
    CHECK(std::abs(c.y) < 1e-6f);
    CHECK(std::abs(c.x) < 1e-6f);
    // anti-symmetry + orthogonality + Lagrange identity on arbitrary vectors
    const Vec3f a{0.3f, -1.2f, 2.1f}, b{1.7f, 0.4f, -0.6f};
    const Vec3f ab = cross(a, b), ba = cross(b, a);
    CHECK(std::abs(ab.z + ba.z) < 1e-5f);
    CHECK(std::abs(ab.y + ba.y) < 1e-5f);
    CHECK(std::abs(ab.x + ba.x) < 1e-5f);
    CHECK(std::abs(dot(ab, a)) < 1e-4f);
    CHECK(std::abs(dot(ab, b)) < 1e-4f);
    const f32 lagrange = dot(a, a) * dot(b, b) - dot(a, b) * dot(a, b);
    CHECK(std::abs(dot(ab, ab) - lagrange) < 1e-3f);
}
