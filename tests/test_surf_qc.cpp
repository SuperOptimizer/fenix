// tests/test_surf_qc.cpp — surf-qc primitives: nearest-prominent-peak (the profile-mode
// offset estimator) + stencil normals. The old estimator (window global max) locked onto
// brighter NEIGHBOR wraps near tight winding — these pin the fixed behavior.
#define FENIX_TEST_MAIN
#include "core/test.hpp"
#include "ml/surf_qc.hpp"

#include <cmath>
#include <vector>

using namespace fenix;

namespace {
// profile over t=-W..W: gaussian bumps at given offsets/heights on a dark floor
std::vector<f32> bumps(s64 W, std::initializer_list<std::pair<f64, f64>> peaks, f32 floor_v = 20.0f) {
    std::vector<f32> p(static_cast<usize>(2 * W + 1), floor_v);
    for (s64 t = -W; t <= W; ++t)
        for (const auto& [c, h] : peaks)
            p[static_cast<usize>(t + W)] += static_cast<f32>(h * std::exp(-0.5 * (t - c) * (t - c) / 4.0));
    return p;
}
}  // namespace

TEST(nearest_peak_prefers_near_over_bright) {
    // the regression: a small on-sheet ridge at +1 must beat a BRIGHTER neighbor wrap at +9
    const s64 W = 12;
    const auto p = bumps(W, {{1.0, 60.0}, {9.0, 120.0}});
    const auto t = ml::detail::nearest_prominent_peak(p, W, /*search=*/8, /*prom=*/15.0f);
    REQUIRE(t.has_value());
    CHECK(std::abs(*t - 1) <= 1);
}

TEST(nearest_peak_respects_search_window_and_prominence) {
    const s64 W = 12;
    // only peak sits at +10, outside search=8 -> no peak
    CHECK(!ml::detail::nearest_prominent_peak(bumps(W, {{10.0, 80.0}}), W, 8, 15.0f).has_value());
    // low-prominence speckle (height 8 < prom 15) -> no peak
    CHECK(!ml::detail::nearest_prominent_peak(bumps(W, {{0.0, 8.0}}), W, 8, 15.0f).has_value());
    // flat profile -> no peak
    CHECK(!ml::detail::nearest_prominent_peak(std::vector<f32>(25, 100.0f), W, 8, 15.0f).has_value());
    // clean centered ridge -> offset ~0
    const auto t = ml::detail::nearest_prominent_peak(bumps(W, {{0.0, 80.0}}), W, 8, 15.0f);
    REQUIRE(t.has_value());
    CHECK(*t == 0);
}

TEST(stencil_normal_matches_plane) {
    // flat z=zc plane: normal must be ±(1,0,0) in ZYX regardless of uv scale
    Surface s(8, 8);
    s.scale_u = 20.0f;
    s.scale_v = 20.0f;
    for (s64 v = 0; v < 8; ++v)
        for (s64 u = 0; u < 8; ++u)
            s.set(u, v, Vec3f{50.0f, static_cast<f32>(v) * 20.0f, static_cast<f32>(u) * 20.0f});
    const auto n = ml::detail::stencil_normal(s, 4, 4);
    REQUIRE(n.has_value());
    CHECK(std::abs(std::abs(n->z) - 1.0f) < 1e-4f);
    CHECK(std::abs(n->y) < 1e-4f);
    CHECK(std::abs(n->x) < 1e-4f);
    // edge cell falls back to a 1-step stencil rather than failing
    CHECK(ml::detail::stencil_normal(s, 1, 1).has_value());
    // corner with no two-sided neighbors -> nullopt
    Surface tiny(2, 2);
    for (s64 v = 0; v < 2; ++v)
        for (s64 u = 0; u < 2; ++u) tiny.set(u, v, Vec3f{0, static_cast<f32>(v), static_cast<f32>(u)});
    CHECK(!ml::detail::stencil_normal(tiny, 0, 0).has_value());
}

#include "io/surface.hpp"
#include "ml/surf_consist.hpp"

#include <cstdio>

namespace {
fenix::Surface plane_at(fenix::f32 zc, fenix::f32 tilt_du = 0.0f) {
    fenix::Surface s(24, 24);
    s.scale_u = 8.0f;
    s.scale_v = 8.0f;
    for (fenix::s64 v = 0; v < 24; ++v)
        for (fenix::s64 u = 0; u < 24; ++u)
            s.set(u,
                  v,
                  fenix::Vec3f{zc + tilt_du * static_cast<fenix::f32>(u - 12),
                               static_cast<fenix::f32>(v) * 8.0f + 32.0f,
                               static_cast<fenix::f32>(u) * 8.0f + 32.0f});
    return s;
}
}  // namespace

TEST(surf_consist_runs_on_agree_offset_cross) {
    // plain string paths: fs::path in test TUs trips a pre-existing libc++ header/dylib
    // mismatch on the macOS dev build (test_feed has the same issue; Linux is fine)
    // A: z=64. B: z=64.6 (duplicate trace -> AGREE). C: z=68 (parallel offset 4 -> OFFSET).
    // D: tilted through z=64 (crosses A -> CROSS).
    const std::string pa = "/tmp/fenix_consist_a.fxsurf", pb = "/tmp/fenix_consist_b.fxsurf",
                      pc = "/tmp/fenix_consist_c.fxsurf", pd = "/tmp/fenix_consist_d.fxsurf";
    REQUIRE(io::write_fxsurf(pa, plane_at(64.0f)).has_value());
    REQUIRE(io::write_fxsurf(pb, plane_at(64.6f)).has_value());
    REQUIRE(io::write_fxsurf(pc, plane_at(68.0f)).has_value());
    REQUIRE(io::write_fxsurf(pd, plane_at(64.0f, 0.35f)).has_value());
    Context ctx;
    const std::string_view args[] = {pa, pb, pc, pd, "near=6", "k=2000"};
    const auto r = ml::run_surf_consist(args, ctx);
    REQUIRE(r.has_value());
    CHECK(*r == 0);
    for (const auto& p : {pa, pb, pc, pd}) std::remove(p.c_str());
}

TEST(mesh_cloud_nearest_and_key) {
    ml::detail::MeshCloud mc;
    mc.pts = {fenix::Vec3f{10, 10, 10}, fenix::Vec3f{10, 10, 20}, fenix::Vec3f{50, 50, 50}};
    mc.nrm.assign(3, fenix::Vec3f{1, 0, 0});
    mc.build_grid(6.0f);
    CHECK(mc.nearest(fenix::Vec3f{10, 10, 11}, 6.0f) == 0);
    CHECK(mc.nearest(fenix::Vec3f{10, 10, 18}, 6.0f) == 1);
    CHECK(mc.nearest(fenix::Vec3f{30, 30, 30}, 6.0f) == -1);  // nothing within reach
}
