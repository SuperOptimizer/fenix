// test_patch_graph.cpp — the multi-scale patch graph on a synthetic concentric-cylinder "scroll":
// known wraps at a known spacing, with one wrap deliberately split into two patches. Checks the
// adjacency metrics (signed gap, co-normality), the wrap-spacing estimate, same-sheet MERGE, and the
// integer-winding assignment (monotone with radius, conflict-free).
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/surface.hpp"
#include "core/test.hpp"
#include "segment/patch_graph.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using namespace fenix;
using namespace fenix::segment;

namespace {
constexpr f32 kTwoPi = 2.0f * std::numbers::pi_v<f32>;
constexpr f32 kCy = 200.0f, kCx = 200.0f, kZ0 = 50.0f;

// One wrap (or angular slice of one) as a (u,v) Surface with normal/conf channels filled: a cylinder
// of the given radius, u -> angle in [ang0,ang1], v -> z.
Surface make_wrap(f32 radius, f32 ang0, f32 ang1, int nu, int nv, f32 step) {
    Surface S(nu, nv);
    S.alloc_channels();
    for (int v = 0; v < nv; ++v)
        for (int u = 0; u < nu; ++u) {
            const f32 a = ang0 + (ang1 - ang0) * static_cast<f32>(u) / static_cast<f32>(nu - 1);
            const f32 sy = std::sin(a), cx = std::cos(a);
            const Vec3f p{kZ0 + static_cast<f32>(v) * step, kCy + radius * sy, kCx + radius * cx};
            const usize id = S.idx(u, v);
            S.coord[id] = p;
            S.valid[id] = 1;
            S.normal[id] = Vec3f{0, sy, cx};  // radial outward
            S.conf[id] = 1.0f;
        }
    return S;
}

annotate::Umbilicus straight_axis(f32 zlo, f32 zhi) {
    annotate::Umbilicus u;
    u.z = {zlo, zhi};
    u.y = {kCy, kCy};
    u.x = {kCx, kCx};
    return u;
}
}  // namespace

TEST(patch_graph_concentric_wraps) {
    const f32 R0 = 40.0f, spacing = 8.0f, step = 2.0f;
    const int nw = 4, nu = 96, nv = 24;
    const annotate::Umbilicus umb = straight_axis(kZ0 - 4, kZ0 + static_cast<f32>(nv) * step + 4);

    std::vector<Surface> sheets;
    // wrap 0 split into two half-patches (must MERGE back); wraps 1..3 single patches.
    sheets.push_back(make_wrap(R0, 0.0f, std::numbers::pi_v<f32>, nu, nv, step));
    sheets.push_back(make_wrap(R0, std::numbers::pi_v<f32>, kTwoPi, nu, nv, step));
    for (int w = 1; w < nw; ++w)
        sheets.push_back(make_wrap(R0 + static_cast<f32>(w) * spacing, 0.0f, kTwoPi * (static_cast<f32>(nu) - 1) / static_cast<f32>(nu), nu, nv, step));

    PatchGraphParams gp;
    gp.step = step;
    PatchGraph g = analyze_patches(sheets, umb, gp);

    // wrap-spacing recovered from the adjacent-wrap gaps.
    CHECK(std::abs(g.spacing - spacing) < 2.0f);

    // the two halves of wrap 0 merged -> 4 clusters from 5 patches.
    CHECK(g.cluster_count == nw);
    CHECK(g.patches[0].cluster == g.patches[1].cluster);

    // winding assignment: conflict-free, range 0..nw-1, monotone with radius.
    CHECK(g.winding_conflicts == 0);
    CHECK(g.wrap_lo == 0);
    CHECK(g.wrap_hi == nw - 1);
    CHECK(g.patches[0].wrap == 0);          // R0 half a
    CHECK(g.patches[1].wrap == 0);          // R0 half b (merged, same winding)
    CHECK(g.patches[2].wrap == 1);          // R0 + 8
    CHECK(g.patches[3].wrap == 2);          // R0 + 16
    CHECK(g.patches[4].wrap == 3);          // R0 + 24

    // there is at least one same-sheet MERGE edge and several adjacent-wrap LINK edges.
    int merges = 0, links = 0;
    for (const PatchEdge& e : g.edges) {
        if (e.kind == EdgeKind::Merge) ++merges;
        if (e.kind == EdgeKind::Link) ++links;
    }
    CHECK(merges >= 1);
    CHECK(links >= 3);

    // a LINK edge between adjacent wraps must be co-normal (parallel) with |Δwrap| == 1.
    for (const PatchEdge& e : g.edges)
        if (e.kind == EdgeKind::Link) {
            CHECK(std::abs(e.conormal) > 0.7f);
            CHECK(std::abs(e.dwrap) == 1);
        }
}

TEST(patch_graph_orientation_propagation) {
    // make_patch keeps each patch's normals as-is (locally consistent, arbitrary global sign);
    // build_patch_graph propagates a consistent orientation over the manifold. Flip the MIDDLE wrap's
    // normals and check the winding assignment still comes out right (0,1,2, conflict-free).
    const f32 R0 = 40.0f, spacing = 8.0f, step = 2.0f;
    const int nu = 96, nv = 20;
    const annotate::Umbilicus umb = straight_axis(kZ0 - 4, kZ0 + static_cast<f32>(nv) * step + 4);
    std::vector<Surface> sheets;
    sheets.push_back(make_wrap(R0, 0.0f, kTwoPi * (static_cast<f32>(nu) - 1) / static_cast<f32>(nu), nu, nv, step));
    sheets.push_back(make_wrap(R0 + spacing, 0.0f, kTwoPi * (static_cast<f32>(nu) - 1) / static_cast<f32>(nu), nu, nv, step));
    sheets.push_back(make_wrap(R0 + 2 * spacing, 0.0f, kTwoPi * (static_cast<f32>(nu) - 1) / static_cast<f32>(nu), nu, nv, step));
    for (auto& n : sheets[1].normal) n = n * -1.0f;  // flip the middle wrap's normals

    PatchGraphParams gp;
    gp.step = step;
    PatchGraph g = analyze_patches(sheets, umb, gp);

    CHECK(std::abs(g.spacing - spacing) < 2.0f);
    CHECK(g.winding_conflicts == 0);          // propagation fixed the flipped patch
    CHECK(g.wrap_hi - g.wrap_lo == 2);        // three wraps still ordered
    CHECK(g.patches[1].wrap == g.patches[0].wrap + 1 || g.patches[1].wrap == g.patches[0].wrap - 1);
}
