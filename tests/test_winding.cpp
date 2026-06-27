// test_winding.cpp — umbilicus polar frame + analytic winding field + monotonicity.
#define FENIX_TEST_MAIN
#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "core/test.hpp"
#include "winding/winding_field.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::annotate;
using namespace fenix::winding;

// A synthetic scroll: concentric cylindrical shells (wraps) about the volume axis.
static Volume<f32> make_scroll(s64 side, f32 cy, f32 cx, f32 pitch) {
    Volume<f32> v = Volume<f32>::zeros({side, side, side});
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    for (s64 z = 0; z < side; ++z)
        for (s64 y = 0; y < side; ++y)
            for (s64 x = 0; x < side; ++x) {
                f32 r = std::sqrt((static_cast<f32>(y) - cy) * (static_cast<f32>(y) - cy) +
                                  (static_cast<f32>(x) - cx) * (static_cast<f32>(x) - cx));
                // bright papyrus shells every `pitch` voxels of radius
                v(z, y, x) = 0.5f + 0.5f * std::cos(two_pi * r / pitch);
            }
    return v;
}

TEST(umbilicus_estimate_finds_center) {
    const s64 side = 48;
    const f32 cy = 24.5f, cx = 23.0f;
    auto vol = make_scroll(side, cy, cx, 6.0f);
    // Threshold high so only bright shells (symmetric about the center) count.
    Umbilicus u = Umbilicus::estimate(vol.view(), 0.9f);
    REQUIRE(static_cast<s64>(u.z.size()) == side);
    Vec3f c = u.center(24.0f);
    CHECK(std::abs(c.y - cy) < 2.0f);
    CHECK(std::abs(c.x - cx) < 2.0f);
}

TEST(winding_field_is_radially_monotonic) {
    const s64 side = 64;
    const f32 cy = 32.0f, cx = 32.0f, pitch = 8.0f;
    Umbilicus u;
    for (s64 z = 0; z < side; ++z) {
        u.z.push_back(static_cast<f32>(z));
        u.y.push_back(cy);
        u.x.push_back(cx);
    }
    Volume<f32> w = winding_init({side, side, side}, u, {.pitch = pitch});
    // Off the theta branch cut, W = r/pitch + const is strictly increasing outward.
    for (s64 x = 40; x < 59; ++x) CHECK(w(32, 32, x + 1) > w(32, 32, x));  // +x ray (theta~0)
    for (s64 y = 40; y < 59; ++y) CHECK(w(32, y + 1, 32) > w(32, y, 32));  // +y ray (theta~pi/2)
    // One wrap out (pitch voxels) is ~+1 in winding number.
    CHECK(std::abs((w(32, 32, 48) - w(32, 32, 40)) - (8.0f / pitch)) < 0.2f);
    // Globally the analytic init is mostly monotonic; the residual (~5%) is the single
    // 2*pi branch-cut ray — the artifact the masked relaxation solve is built to remove.
    f32 mvf = monotonicity_violation(w.view(), u, 32, 28, 90, /*r_min=*/static_cast<s64>(pitch));
    CHECK(mvf < 0.10f);
}

TEST(umbilicus_polar_basics) {
    Umbilicus u;
    u.z = {0, 10};
    u.y = {5, 5};
    u.x = {5, 5};
    f32 r;
    Radian th;
    u.polar(5, 5, 9, r, th);  // 4 to the +x of center
    CHECK(std::abs(r - 4.0f) < 1e-4f);
    CHECK(std::abs(th.value) < 1e-4f);  // +x direction -> theta ~ 0
}
