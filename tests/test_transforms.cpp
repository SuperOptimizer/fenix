// test_transforms.cpp — invertible diffeomorphism factors: per-slice affine (expm) + gap-expander.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "winding/transforms.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::winding;

TEST(expm2_identity_and_determinant) {
    Mat2 i = expm2(0, 0, 0, 0);  // exp(0) = I
    CHECK(std::abs(i.m00 - 1.0f) < 1e-5f);
    CHECK(std::abs(i.m11 - 1.0f) < 1e-5f);
    CHECK(std::abs(i.m01) < 1e-5f);
    // det(expm(L)) = e^trace(L) > 0 always.
    Mat2 m = expm2(0.3f, 0.7f, -0.4f, 0.1f);
    CHECK(std::abs(m.det() - std::exp(0.3f + 0.1f)) < 1e-3f);
    CHECK(m.det() > 0.0f);
}

TEST(affine_apply_inverse_roundtrip) {
    AffineYX t{.a = 0.2f, .b = 0.5f, .c = -0.3f, .d = -0.1f, .ty = 4.0f, .tx = -2.0f};
    Vec3f p{10.0f, 7.0f, 13.0f};
    Vec3f q = t.apply(p);
    Vec3f r = t.inverse(q);
    CHECK(norm(r - p) < 1e-3f);   // invertible
    CHECK(std::abs(q.z - p.z) < 1e-6f);  // z untouched
    CHECK(norm(q - p) > 0.5f);    // actually transformed
}

TEST(gap_expander_monotonic_and_invertible) {
    GapExpander g;
    g.dr = 8.0f;
    g.logits = {0.0f, 0.3f, -0.2f, 0.1f, 0.0f};  // varying per-winding scale
    // Monotone increasing forward.
    f32 prev = -1e9f;
    for (f32 r = 0.0f; r < 39.0f; r += 0.5f) {
        f32 fr = g.forward(r);
        CHECK(fr >= prev);
        prev = fr;
    }
    // Inverse recovers the input.
    for (f32 r = 1.0f; r < 38.0f; r += 1.7f) {
        f32 back = g.inverse(g.forward(r));
        CHECK(std::abs(back - r) < 1e-2f);
    }
}
