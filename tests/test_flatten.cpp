// test_flatten.cpp — extract an iso-winding wrap as a parametric Surface.
#define FENIX_TEST_MAIN
#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "core/test.hpp"
#include "flatten/extract_wrap.hpp"
#include "winding/winding_field.hpp"

#include <cmath>
#include <numbers>

using namespace fenix;

TEST(extract_wrap_radius_matches_winding) {
    const s64 side = 80;
    const f32 cy = 40.0f, cx = 40.0f, pitch = 8.0f;
    annotate::Umbilicus u;
    for (s64 z = 0; z < side; ++z) {
        u.z.push_back(static_cast<f32>(z));
        u.y.push_back(cy);
        u.x.push_back(cx);
    }
    Volume<f32> w = winding::winding_init({side, side, side}, u, {.pitch = pitch});

    const f32 target = 3.0f;
    Surface s = flatten::extract_winding_surface(w.view(), u, target, 120, 38);
    CHECK(s.nu == 120);
    CHECK(s.nv == side);
    CHECK(s.valid_count() > 0);

    // At theta = 0 (u = 0): W = r/pitch + 0 -> r = target*pitch = 24.
    REQUIRE(s.is_valid(0, 40));
    Vec3f p = s.at(0, 40);
    f32 r = std::sqrt((p.y - cy) * (p.y - cy) + (p.x - cx) * (p.x - cx));
    CHECK(std::abs(r - target * pitch) < 1.5f);  // ~24, within ray-march quantization
    CHECK(std::abs(p.z - 40.0f) < 1e-3f);        // surface row v maps to z

    // The wrap is a closed loop: most (theta,z) cells found a crossing.
    CHECK(s.valid_count() > 120 * side / 2);
}
