// test_winding.cpp — umbilicus polar frame + analytic winding field + monotonicity.
#define FENIX_TEST_MAIN
#include "annotate/umbilicus.hpp"
#include "annotate/umbilicus_fit.hpp"
#include "core/core.hpp"
#include "core/test.hpp"
#include "winding/corpus_bridge.hpp"
#include "winding/spiral_model.hpp"
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

TEST(corpus_bridge_recovers_spacing_and_gauge) {
    // Two synthetic multi-wrap Archimedean segment meshes about a common axis: mesh A
    // spans turns [2,5], mesh B spans [4,7] — overlapping wrap ranges, one gauge.
    const f32 cy = 500.0f, cx = 500.0f, pitch = 30.0f;
    Umbilicus umb;
    umb.z = {0, 100};
    umb.y = {cy, cy};
    umb.x = {cx, cx};
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    auto make_seg = [&](f32 w_lo, f32 w_hi) {
        const s64 nu = 240, nv = 20;
        Surface s(nu, nv);
        for (s64 v = 0; v < nv; ++v)
            for (s64 u = 0; u < nu; ++u) {
                const f32 w = w_lo + (w_hi - w_lo) * static_cast<f32>(u) / static_cast<f32>(nu - 1);
                const f32 theta = two_pi * w;
                const f32 r = pitch * w;
                s.set(u, v, {static_cast<f32>(v) * 5.0f, cy + r * std::sin(theta), cx + r * std::cos(theta)});
            }
        return s;
    };
    std::vector<Surface> sheets;
    sheets.push_back(make_seg(2.0f, 5.0f));
    sheets.push_back(make_seg(4.0f, 7.0f));

    auto out = corpus_to_constraints(sheets, umb, {.stride = 3});
    CHECK(out.components == 2);
    CHECK(std::abs(out.spacing - pitch) < 1.5f);  // the meshes measure their own pitch
    // (bases differ by each seed's whole-turn anchor — gauge consistency is asserted on
    // the targets below, where the r/spacing anchoring cancels the seed offsets)
    REQUIRE(out.targets.size() > 500);

    // per-cell targets follow the true continuous winding (up to the shared gauge constant)
    f32 gauge = 0;
    s64 n = 0;
    for (const auto& t : out.targets) {
        const f32 dy = t.scroll_pt.y - cy, dx = t.scroll_pt.x - cx;
        const f32 w_true = std::sqrt(dy * dy + dx * dx) / pitch;  // exact for this spiral
        gauge += t.target_winding - w_true;
        ++n;
    }
    gauge /= static_cast<f32>(n);
    f32 max_err = 0;
    for (const auto& t : out.targets) {
        const f32 dy = t.scroll_pt.y - cy, dx = t.scroll_pt.x - cx;
        const f32 w_true = std::sqrt(dy * dy + dx * dx) / pitch;
        max_err = std::max(max_err, std::abs(t.target_winding - gauge - w_true));
    }
    CHECK(max_err < 0.15f);
}

TEST(umbilicus_from_sheets_tracks_drifting_axis) {
    // Spiral segment meshes about an axis that drifts +150 vox in y and -90 in x over z.
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    const f32 z_max = 4000.0f, pitch = 40.0f;
    auto axis_y = [&](f32 z) { return 800.0f + 150.0f * z / z_max; };
    auto axis_x = [&](f32 z) { return 800.0f - 90.0f * z / z_max; };
    auto make_seg = [&](f32 w_lo, f32 w_hi, f32 phase) {
        const s64 nu = 200, nv = 40;
        Surface s(nu, nv);
        for (s64 v = 0; v < nv; ++v) {
            const f32 z = z_max * static_cast<f32>(v) / static_cast<f32>(nv - 1);
            for (s64 u = 0; u < nu; ++u) {
                const f32 w = w_lo + (w_hi - w_lo) * static_cast<f32>(u) / static_cast<f32>(nu - 1);
                const f32 theta = two_pi * w + phase;
                const f32 r = pitch * w;
                s.set(u, v, {z, axis_y(z) + r * std::sin(theta), axis_x(z) + r * std::cos(theta)});
            }
        }
        return s;
    };
    std::vector<Surface> sheets;
    sheets.push_back(make_seg(3.0f, 6.0f, 0.0f));
    sheets.push_back(make_seg(5.0f, 9.0f, 2.1f));
    sheets.push_back(make_seg(8.0f, 12.0f, 4.0f));

    annotate::Umbilicus u =
        annotate::umbilicus_from_sheets(sheets, {.band = 500.0f, .stride = 1, .irls = 3});
    REQUIRE(!u.empty());
    REQUIRE(u.z.size() >= 6);
    f32 max_err = 0;
    for (f32 z : {500.0f, 1500.0f, 2500.0f, 3500.0f}) {
        const Vec3f c = u.center(z);
        max_err = std::max({max_err, std::abs(c.y - axis_y(z)), std::abs(c.x - axis_x(z))});
    }
    CHECK(max_err < 8.0f);  // tracks a 150-vox drift to within a few voxels

    // TOML round trip
    const std::string path = "test_umb.toml";
    REQUIRE(static_cast<bool>(annotate::save_umbilicus(u, path)));
    auto r = annotate::load_umbilicus(path);
    std::remove(path.c_str());
    REQUIRE(static_cast<bool>(r));
    REQUIRE(r->z.size() == u.z.size());
    const Vec3f a = u.center(1234.0f), b = r->center(1234.0f);
    CHECK(std::abs(a.y - b.y) < 1e-3f);
    CHECK(std::abs(a.x - b.x) < 1e-3f);
}

TEST(holdout_score_separates_good_from_bad_model) {
    const f32 cy = 500.0f, cx = 500.0f, pitch = 30.0f;
    Umbilicus umb;
    umb.z = {0, 100};
    umb.y = {cy, cy};
    umb.x = {cx, cx};
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    std::vector<Surface> sheets;
    {
        const s64 nu = 240, nv = 20;
        Surface s(nu, nv);
        for (s64 v = 0; v < nv; ++v)
            for (s64 u = 0; u < nu; ++u) {
                const f32 w = 3.0f + 4.0f * static_cast<f32>(u) / static_cast<f32>(nu - 1);
                const f32 theta = two_pi * w;
                s.set(u, v,
                      {static_cast<f32>(v) * 5.0f, cy + pitch * w * std::sin(theta),
                       cx + pitch * w * std::cos(theta)});
            }
        sheets.push_back(std::move(s));
    }
    SpiralModel good;
    good.umbilicus = umb;
    good.dr_per_winding = pitch;
    good.gap.dr = pitch;
    const auto gs = score_holdout(good, sheets, umb, 2);
    REQUIRE(gs.cells > 100);
    CHECK(gs.rmse < 0.05);  // exact spiral, exact model: gauge removal leaves ~nothing

    SpiralModel bad = good;
    bad.dr_per_winding = pitch * 1.4f;  // 40% pitch error
    bad.gap.dr = pitch * 1.4f;
    const auto bs = score_holdout(bad, sheets, umb, 2);
    CHECK(bs.rmse > 0.3);  // a wrong model cannot hide behind the gauge
}
