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
#include <cstdio>
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

// An angular fragment [u0,u1) of a full nu_full-resolution wrap (radial outward normals) — mimics a
// tile-local trace fragment of one wrap.
Surface make_frag(f32 radius, int u0, int u1, int nu_full, int nv, f32 step) {
    const int nu = u1 - u0;
    Surface S(nu, nv);
    S.alloc_channels();
    for (int v = 0; v < nv; ++v)
        for (int uu = 0; uu < nu; ++uu) {
            const f32 a = kTwoPi * static_cast<f32>(u0 + uu) / static_cast<f32>(nu_full);
            const f32 sy = std::sin(a), cx = std::cos(a);
            const usize id = S.idx(uu, v);
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

// The Eulerian (normal-driven) winding solve must give COHERENT windings for FRAGMENTED wraps — the
// case the discrete patch-graph merge/winding fails on for tiled traces. Three concentric wraps, each
// cut into 3 disjoint angular fragments (9 patches): every fragment of a wrap must get the SAME winding,
// and the windings must increment by 1 per wrap — WITHOUT any merge/link decisions, straight from the
// integrated normal field.
TEST(eulerian_winding_stitches_fragmented_wraps) {
    const f32 R0 = 40.0f, spacing = 8.0f, step = 2.0f;
    const int nu_full = 96, nv = 24;
    annotate::Umbilicus umb;
    umb.z = {kZ0 - 4, kZ0 + static_cast<f32>(nv) * step + 4};
    umb.y = {kCy, kCy};
    umb.x = {kCx, kCx};

    // 3 wraps x 3 angular fragments = 9 patches. The fragments OVERLAP at their seams (like real
    // tile-local traces, which abut/overlap across tile boundaries) so adjacent same-wrap fragments
    // touch and MERGE into one cluster — that is what the per-cluster winding readout relies on.
    std::vector<Surface> sheets;
    std::vector<int> wrap_of;  // 0,1,2 ground-truth wrap index per fragment
    const int lo[3] = {0, 26, 58}, hi[3] = {38, 70, 96};
    for (int w = 0; w < 3; ++w) {
        const f32 r = R0 + static_cast<f32>(w) * spacing;
        for (int t = 0; t < 3; ++t) {
            sheets.push_back(make_frag(r, lo[t], hi[t], nu_full, nv, step));
            wrap_of.push_back(w);
        }
    }

    segment::PatchGraphParams gp;
    gp.step = step;
    segment::PatchGraph g = segment::build_patch_graph(sheets, umb, gp);  // orients normals + spacing
    segment::merge_same_sheet(g);  // same-sheet clusters -> per-cluster winding consistency

    const Extent3 full{112, 272, 272};
    winding::FieldParams fpar;
    fpar.ds = 2;
    fpar.iters = 300;     // band-restricted -> robust across the iteration count (no GS sweet-spot)
    fpar.band = 6;        // >= spacing/ds so adjacent wraps' bands connect
    const winding::WindingField wf =
        winding::build_eulerian_winding_field(g.patches, full, g.spacing, fpar);
    winding::assign_windings_from_field(g, wf);

    // all 3 fragments of a wrap share a winding; windings increment by exactly 1 across the 3 wraps.
    int w_by_wrap[3] = {-100, -100, -100};
    bool consistent = true;
    for (usize i = 0; i < g.patches.size(); ++i) {
        const int gtw = wrap_of[i], aw = g.patches[i].wrap;
        if (w_by_wrap[gtw] == -100) w_by_wrap[gtw] = aw;
        else if (w_by_wrap[gtw] != aw) consistent = false;  // a wrap's fragments disagree
    }
    CHECK(consistent);  // fragments of the same wrap got the SAME winding
    CHECK(w_by_wrap[1] - w_by_wrap[0] == 1);
    CHECK(w_by_wrap[2] - w_by_wrap[1] == 1);
}
