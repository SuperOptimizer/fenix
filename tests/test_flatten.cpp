// test_flatten.cpp — extract an iso-winding wrap as a parametric Surface.
#define FENIX_TEST_MAIN
#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "core/test.hpp"
#include "flatten/extract_wrap.hpp"
#include "flatten/flatten.hpp"
#include "winding/model_io.hpp"
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

TEST(fxmodel_roundtrip) {
    winding::SpiralModel m;
    m.umbilicus.z = {0, 50, 100};
    m.umbilicus.y = {30, 31, 32};
    m.umbilicus.x = {40, 40, 41};
    m.affine = {.a = 0.01f, .b = -0.02f, .c = 0.03f, .d = 0.005f, .ty = 1.5f, .tx = -2.5f};
    m.gap.dr = 7.5f;
    m.gap.logits = {0.1f, -0.1f, 0.2f};
    m.dr_per_winding = 7.5f;
    m.winding_offset = 1.25f;
    m.flow_steps = 6;
    m.has_flow = true;
    m.flow.vz = Volume<f32>::zeros({3, 4, 5});
    m.flow.vy = Volume<f32>::zeros({3, 4, 5});
    m.flow.vx = Volume<f32>::zeros({3, 4, 5});
    m.flow.vy(1, 2, 3) = 0.75f;
    m.flow.lat_lo = {0, -20, -20};
    m.flow.lat_scale = {0.02f, 0.1f, 0.1f};

    const std::string path = "test_roundtrip.fxmodel";
    REQUIRE(static_cast<bool>(winding::write_fxmodel(path, m)));
    auto r = winding::read_fxmodel(path);
    REQUIRE(static_cast<bool>(r));
    std::remove(path.c_str());

    CHECK(r->umbilicus.z.size() == 3);
    CHECK(r->umbilicus.y[2] == 32.0f);
    CHECK(r->affine.b == -0.02f);
    CHECK(r->affine.tx == -2.5f);
    CHECK(r->gap.logits.size() == 3);
    CHECK(r->gap.logits[2] == 0.2f);
    CHECK(r->dr_per_winding == 7.5f);
    CHECK(r->winding_offset == 1.25f);
    CHECK(r->flow_steps == 6);
    CHECK(r->has_flow);
    CHECK(r->flow.vy.dims() == (Extent3{3, 4, 5}));
    CHECK(r->flow.vy(1, 2, 3) == 0.75f);
    CHECK(r->flow.lat_scale.y == 0.1f);

    // the round-tripped model computes the same winding
    const Vec3f probe{50, 55, 60};
    CHECK(std::abs(m.winding_at(probe) - r->winding_at(probe)) < 1e-6f);
}

TEST(wrap_surface_lies_on_its_winding_level) {
    // A non-trivial model: curved umbilicus, affine, gaps, gauge, flow — every factor live.
    winding::SpiralModel m;
    for (s64 z = 0; z <= 100; z += 10) {
        m.umbilicus.z.push_back(static_cast<f32>(z));
        m.umbilicus.y.push_back(60.0f + 0.05f * static_cast<f32>(z));
        m.umbilicus.x.push_back(60.0f - 0.03f * static_cast<f32>(z));
    }
    m.affine = {.a = 0.02f, .b = 0.01f, .c = -0.01f, .d = -0.015f, .ty = 0.8f, .tx = -0.4f};
    m.gap.dr = 8.0f;
    m.gap.logits = {0.05f, -0.05f, 0.1f, 0.0f, -0.1f, 0.05f};
    m.dr_per_winding = 8.0f;
    m.winding_offset = 0.5f;
    m.flow_steps = 8;
    m.has_flow = true;
    m.flow.vz = Volume<f32>::zeros({4, 6, 6});
    m.flow.vy = Volume<f32>::zeros({4, 6, 6});
    m.flow.vx = Volume<f32>::zeros({4, 6, 6});
    for (s64 z = 0; z < 4; ++z)
        for (s64 y = 0; y < 6; ++y)
            for (s64 x = 0; x < 6; ++x) {
                m.flow.vy(z, y, x) = 0.5f * std::sin(0.7f * static_cast<f32>(x));
                m.flow.vx(z, y, x) = 0.5f * std::cos(0.9f * static_cast<f32>(y));
            }
    m.flow.lat_lo = {0, -50, -50};
    m.flow.lat_scale = {3.0f / 100.0f, 5.0f / 100.0f, 5.0f / 100.0f};

    for (f32 wrap : {2.0f, 3.0f, 4.0f}) {
        Surface s = flatten::wrap_surface(m, wrap, 96, 0.0f, 100.0f, 10.0f, 1e9f);
        REQUIRE(s.valid_count() > 0);
        f32 max_err = 0;
        for (s64 v = 0; v < s.nv; ++v)
            for (s64 u = 0; u < s.nu; ++u)
                if (s.is_valid(u, v)) max_err = std::max(max_err, std::abs(m.winding_at(s.at(u, v)) - wrap));
        // to_scroll then winding_at round-trips through the RK4 flow both ways
        CHECK(max_err < 0.02f);
    }
}
