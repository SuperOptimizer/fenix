// test_patch_field.cpp — the coarse winding field + field-guided fill. Three concentric wraps with a
// hole punched in the MIDDLE wrap: the field (pinned by all three wraps) must place the hole's fill
// cells back onto the middle wrap's radius, reconstructed from its neighbours. This is the
// "fill the blank from the two neighbouring wraps" payoff.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/surface.hpp"
#include "core/test.hpp"
#include "segment/patch_graph.hpp"
#include "winding/patch_field.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using namespace fenix;

namespace {
constexpr f32 kTwoPi = 2.0f * std::numbers::pi_v<f32>;
constexpr f32 kCy = 200.0f, kCx = 200.0f, kZ0 = 50.0f;

Surface make_wrap(f32 radius, int nu, int nv, f32 step) {
    Surface S(nu, nv);
    S.alloc_channels();
    for (int v = 0; v < nv; ++v)
        for (int u = 0; u < nu; ++u) {
            const f32 a = kTwoPi * static_cast<f32>(u) / static_cast<f32>(nu);
            const f32 sy = std::sin(a), cx = std::cos(a);
            const usize id = S.idx(u, v);
            S.coord[id] = Vec3f{kZ0 + static_cast<f32>(v) * step, kCy + radius * sy, kCx + radius * cx};
            S.valid[id] = 1;
            S.normal[id] = Vec3f{0, sy, cx};
            S.conf[id] = 1.0f;
        }
    return S;
}
}  // namespace

TEST(winding_field_fills_middle_wrap_hole) {
    const f32 R0 = 40.0f, spacing = 8.0f, step = 2.0f;
    const int nu = 96, nv = 24;
    annotate::Umbilicus umb;
    umb.z = {kZ0 - 4, kZ0 + static_cast<f32>(nv) * step + 4};
    umb.y = {kCy, kCy};
    umb.x = {kCx, kCx};

    std::vector<Surface> sheets;
    sheets.push_back(make_wrap(R0, nu, nv, step));               // wrap 0
    sheets.push_back(make_wrap(R0 + spacing, nu, nv, step));     // wrap 1 (will get a hole)
    sheets.push_back(make_wrap(R0 + 2 * spacing, nu, nv, step)); // wrap 2

    // punch an enclosed hole in the middle wrap.
    s64 holed = 0;
    for (int v = 8; v <= 16; ++v)
        for (int u = 30; u <= 45; ++u) { sheets[1].valid[sheets[1].idx(u, v)] = 0; ++holed; }
    CHECK(holed > 0);

    segment::PatchGraphParams gp;
    gp.step = step;
    segment::PatchGraph g = segment::analyze_patches(sheets, umb, gp);
    CHECK(g.patches[1].wrap == 1);  // middle wrap

    const Extent3 full{112, 272, 272};
    winding::FieldParams fpar;
    fpar.ds = 2;
    fpar.iters = 150;
    winding::WindingField wf = winding::build_patch_winding_field(g.patches, full, g.spacing, fpar);

    // field sanity: increases ~1 per wrap radially at a fixed angle (x = cx + radius).
    const f32 w_at_r0 = wf.value(Vec3f{70, kCy, kCx + R0});
    const f32 w_at_r1 = wf.value(Vec3f{70, kCy, kCx + R0 + spacing});
    const f32 w_at_r2 = wf.value(Vec3f{70, kCy, kCx + R0 + 2 * spacing});
    CHECK(std::abs(w_at_r0 - 0.0f) < 0.35f);
    CHECK(std::abs(w_at_r1 - 1.0f) < 0.35f);
    CHECK(std::abs(w_at_r2 - 2.0f) < 0.35f);

    winding::FieldFillParams ffp;
    ffp.step = step;
    const s64 filled = winding::fill_surface_from_field(sheets[1], wf, g.patches[1].wrap, ffp);
    CHECK(filled > 0);

    // every newly filled cell must land on the middle wrap's radius (R0 + spacing = 48).
    int checked = 0, good = 0;
    for (int v = 8; v <= 16; ++v)
        for (int u = 30; u <= 45; ++u) {
            const usize id = sheets[1].idx(u, v);
            if (!sheets[1].valid[id]) continue;
            const Vec3f p = sheets[1].coord[id];
            const f32 r = std::sqrt((p.y - kCy) * (p.y - kCy) + (p.x - kCx) * (p.x - kCx));
            ++checked;
            if (std::abs(r - (R0 + spacing)) < 2.5f) ++good;
        }
    CHECK(checked > 0);
    CHECK(good >= checked * 9 / 10);  // >=90% of filled cells on the correct wrap
}
