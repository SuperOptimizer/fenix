// winding/patch_field.hpp — the coarse Eulerian winding field built FROM assigned patches, and the
// field-guided fill it enables. The patches (each tagged with an integer winding by segment's patch
// graph) pin a smooth scalar W on a downsampled grid: W = wrap on a patch's voxels, relaxed in
// between, so the level set {W = k} sits exactly on wrap k — pinned between wraps k-1 and k+1 where
// they exist. That is what lets a hole in one wrap be filled from its NEIGHBOURS: project the hole
// cell onto {W = its wrap} (a Newton step along ∇W). This is the segment→winding bridge: the tracer
// gives patch constraints, the field gives the dense interpolation that repairs weak-prediction gaps.
// See winding/CLAUDE.md (the Eulerian winding field is the continuous view of "which wrap").
#pragma once

#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "core/sampling.hpp"
#include "core/surface.hpp"
#include "geom/kdtree.hpp"
#include "segment/patch_graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace fenix::winding {

struct FieldParams {
    int ds = 4;          // downsample factor for the field grid vs full-res voxels
    int iters = 80;      // masked Gauss-Seidel sweeps (Dirichlet at patch voxels)
    f32 conf_min = 0.8f; // only cells with confidence >= this pin the field; weak cells are left to
                         // be DETERMINED by the relaxation from their good neighbours (so a bad
                         // prediction can't corrupt the field that is meant to correct it).
};

// A coarse winding-number field: W on a `ds`-downsampled grid, queryable at full-res coordinates.
struct WindingField {
    Volume<f32> w;        // coarse field (winding number per coarse voxel)
    int ds = 1;
    Extent3 full{0, 0, 0};
    f32 spacing = 1.0f;   // wrap spacing (full-res voxels); ~ 1/|∇W|

    [[nodiscard]] f32 value(Vec3f pf) const {
        const f32 s = static_cast<f32>(ds);
        return sample_trilinear(w.view(), Vec3f{pf.z / s, pf.y / s, pf.x / s});
    }
    // ∇W in per-full-voxel units (central difference on the coarse grid, rescaled by 1/ds).
    [[nodiscard]] Vec3f grad(Vec3f pf) const {
        const f32 s = static_cast<f32>(ds);
        const Vec3f pc{pf.z / s, pf.y / s, pf.x / s};
        auto smp = [&](f32 dz, f32 dy, f32 dx) { return sample_trilinear(w.view(), Vec3f{pc.z + dz, pc.y + dy, pc.x + dx}); };
        const f32 gz = 0.5f * (smp(1, 0, 0) - smp(-1, 0, 0));
        const f32 gy = 0.5f * (smp(0, 1, 0) - smp(0, -1, 0));
        const f32 gx = 0.5f * (smp(0, 0, 1) - smp(0, 0, -1));
        return Vec3f{gz, gy, gx} / s;
    }
};

// Build the coarse winding field from patches that already carry an integer `wrap`. Patch voxels are
// hard Dirichlet constraints (W = wrap); free voxels are warm-started to the nearest patch's wrap
// (a winding Voronoi) and then relaxed (red-black Gauss-Seidel) into smooth ramps between wraps.
inline WindingField build_patch_winding_field(std::span<const segment::Patch> patches, Extent3 full,
                                              f32 spacing, FieldParams fp = {}) {
    const int ds = std::max(1, fp.ds);
    const Extent3 dd{full.z / ds, full.y / ds, full.x / ds};
    WindingField wf;
    wf.ds = ds;
    wf.full = full;
    wf.spacing = spacing > 0 ? spacing : static_cast<f32>(ds);
    wf.w = Volume<f32>::zeros(dd);
    VolumeView<f32> W = wf.w.view();
    std::vector<u8> fixed(static_cast<usize>(dd.count()), 0);
    const f32 invds = 1.0f / static_cast<f32>(ds);

    std::vector<Vec3f> cc;  // patch cells in coarse units
    std::vector<f32> cw;    // their winding labels
    for (const segment::Patch& p : patches) {
        if (p.wrap == segment::kUnassignedWrap) continue;
        for (usize i = 0; i < p.pos.size(); ++i) {
            if (i < p.conf.size() && p.conf[i] < fp.conf_min) continue;  // weak cell: don't pin the field
            cc.push_back(p.pos[i] * invds);
            cw.push_back(static_cast<f32>(p.wrap));
        }
    }
    if (cc.empty()) return wf;

    for (usize i = 0; i < cc.size(); ++i) {
        const s64 z = std::lround(cc[i].z), y = std::lround(cc[i].y), x = std::lround(cc[i].x);
        if (z < 0 || y < 0 || x < 0 || z >= dd.z || y >= dd.y || x >= dd.x) continue;
        W(z, y, x) = cw[i];
        fixed[static_cast<usize>((z * dd.y + y) * dd.x + x)] = 1;
    }

    const geom::KdTree tree(cc);
    parallel_for_z(dd, [&](s64 z) {
        for (s64 y = 0; y < dd.y; ++y)
            for (s64 x = 0; x < dd.x; ++x) {
                if (fixed[static_cast<usize>((z * dd.y + y) * dd.x + x)]) continue;
                const s64 j = tree.nearest(Vec3f{static_cast<f32>(z) + 0.5f, static_cast<f32>(y) + 0.5f, static_cast<f32>(x) + 0.5f});
                W(z, y, x) = j >= 0 ? cw[static_cast<usize>(j)] : 0.0f;
            }
    });

    for (int it = 0; it < fp.iters; ++it)
        for (int color = 0; color < 2; ++color)
            parallel_for_z(dd, [&](s64 z) {
                for (s64 y = 0; y < dd.y; ++y)
                    for (s64 x = 0; x < dd.x; ++x) {
                        if (((z + y + x) & 1) != color) continue;
                        if (fixed[static_cast<usize>((z * dd.y + y) * dd.x + x)]) continue;
                        W(z, y, x) = (W.at_clamped(z - 1, y, x) + W.at_clamped(z + 1, y, x) +
                                      W.at_clamped(z, y - 1, x) + W.at_clamped(z, y + 1, x) +
                                      W.at_clamped(z, y, x - 1) + W.at_clamped(z, y, x + 1)) / 6.0f;
                    }
            });
    return wf;
}

// Solve the Eulerian winding field θ DIRECTLY from the patches' consistently-oriented normals — NOT
// from pre-assigned integer windings (which is what build_patch_winding_field needs and what the
// discrete patch-graph stitch fails to produce for many small tiled fragments). θ is the scalar whose
// gradient matches n/spacing (θ rises by 1 across each wrap): least-squares ∇θ ≈ n/spacing gives the
// Poisson equation ∇²θ = ∇·(n/spacing), solved by red-black Gauss-Seidel on the coarse grid. Because θ
// integrates the normal field GLOBALLY, every fragment of one wrap lands on the same θ no matter how the
// tracer (or tiling) cut it up — the robust stitch the brittle per-edge merge/winding can't give.
// Patches must already be orientation-consistent (segment::build_patch_graph propagates the sign via
// SignDSU over the co-normal graph, which spans across wraps). `seed`, if given, warm-starts θ (e.g. an
// analytic polar field) to speed convergence on large grids; empty -> zero start.
inline WindingField build_eulerian_winding_field(std::span<const segment::Patch> patches, Extent3 full,
                                                 f32 spacing, FieldParams fp = {},
                                                 const Volume<f32>* seed = nullptr) {
    const int ds = std::max(1, fp.ds);
    const Extent3 dd{full.z / ds, full.y / ds, full.x / ds};
    WindingField wf;
    wf.ds = ds;
    wf.full = full;
    wf.spacing = spacing > 0 ? spacing : static_cast<f32>(ds);
    wf.w = (seed && seed->dims() == dd) ? *seed : Volume<f32>::zeros(dd);
    if (dd.count() == 0) return wf;
    const f32 invds = 1.0f / static_cast<f32>(ds);

    std::vector<Vec3f> cc, cn;  // confident patch cells (coarse units) + their oriented unit normals
    for (const segment::Patch& p : patches)
        for (usize i = 0; i < p.pos.size(); ++i) {
            if (i < p.conf.size() && p.conf[i] < fp.conf_min) continue;
            cc.push_back(p.pos[i] * invds);
            cn.push_back(p.nrm[i]);
        }
    if (cc.empty()) return wf;

    // target gradient b = n/spacing (coarse-cell units), rasterized to every cell by nearest patch cell;
    // store only its divergence (the Poisson source).
    const geom::KdTree tree(cc);
    const f32 sp_c = std::max(1e-3f, wf.spacing * invds);
    Volume<f32> divb = Volume<f32>::zeros(dd);
    {
        Volume<f32> bz = Volume<f32>::zeros(dd), by = Volume<f32>::zeros(dd), bx = Volume<f32>::zeros(dd);
        VolumeView<f32> BZ = bz.view(), BY = by.view(), BX = bx.view();
        parallel_for_z(dd, [&](s64 z) {
            for (s64 y = 0; y < dd.y; ++y)
                for (s64 x = 0; x < dd.x; ++x) {
                    const s64 j = tree.nearest(Vec3f{static_cast<f32>(z) + 0.5f, static_cast<f32>(y) + 0.5f, static_cast<f32>(x) + 0.5f});
                    if (j < 0) continue;
                    const Vec3f n = cn[static_cast<usize>(j)];
                    BZ(z, y, x) = n.z / sp_c;
                    BY(z, y, x) = n.y / sp_c;
                    BX(z, y, x) = n.x / sp_c;
                }
        });
        VolumeView<f32> DBw = divb.view();
        parallel_for_z(dd, [&](s64 z) {
            for (s64 y = 0; y < dd.y; ++y)
                for (s64 x = 0; x < dd.x; ++x)
                    DBw(z, y, x) = 0.5f * ((BZ.at_clamped(z + 1, y, x) - BZ.at_clamped(z - 1, y, x)) +
                                           (BY.at_clamped(z, y + 1, x) - BY.at_clamped(z, y - 1, x)) +
                                           (BX.at_clamped(z, y, x + 1) - BX.at_clamped(z, y, x - 1)));
        });
    }
    // Poisson ∇²θ = ∇·b  ->  red-black GS  θ = (Σ neighbours θ − ∇·b)/6  (Neumann BC via at_clamped),
    // with a per-sweep RE-GAUGE: the Neumann problem is defined only up to a constant, and the radial
    // source is not discretely mean-zero, so the constant null-space mode drifts unboundedly (slowly
    // under GS, but it corrupts the field at scale/many-iters). Re-centring θ to mean 0 each sweep pins
    // the gauge; the differences the winding assignment uses are unaffected.
    VolumeView<f32> W = wf.w.view();
    VolumeView<const f32> DB = divb.view();
    const s64 ncell = dd.count();
    for (int it = 0; it < fp.iters; ++it) {
        for (int color = 0; color < 2; ++color)
            parallel_for_z(dd, [&](s64 z) {
                for (s64 y = 0; y < dd.y; ++y)
                    for (s64 x = 0; x < dd.x; ++x) {
                        if (((z + y + x) & 1) != color) continue;
                        W(z, y, x) = (W.at_clamped(z - 1, y, x) + W.at_clamped(z + 1, y, x) +
                                      W.at_clamped(z, y - 1, x) + W.at_clamped(z, y + 1, x) +
                                      W.at_clamped(z, y, x - 1) + W.at_clamped(z, y, x + 1) - DB(z, y, x)) /
                                     6.0f;
                    }
            });
        f64 sum = 0;
        for (s64 i = 0; i < ncell; ++i) sum += static_cast<f64>(W.data()[i]);
        const f32 mean = static_cast<f32>(sum / static_cast<f64>(ncell));
        parallel_for_z(dd, [&](s64 z) {
            for (s64 y = 0; y < dd.y; ++y)
                for (s64 x = 0; x < dd.x; ++x) W(z, y, x) -= mean;
        });
    }
    return wf;
}

// Assign each patch an integer winding from the Eulerian field. Each patch's real-valued winding is the
// mean θ over its cells (fragments of one wrap share a θ — coherent by construction, with NO discrete
// merge/link clustering). The field is gauge- and scale-free (an unknown θ-step per wrap from the
// estimate/convergence), so the per-wrap θ step is CALIBRATED from the graph's Link edges (Δwrap=±1
// pairs): θ-step = median |θ_a − θ_b| over Link edges. This fuses the field's global coherence with the
// graph's local scale — robust to both an inexact spacing estimate and an under-converged field (the
// θ and θ-step scale together). Updates g.wrap_lo/g.wrap_hi.
inline void assign_windings_from_field(segment::PatchGraph& g, const WindingField& wf) {
    if (g.patches.empty()) return;
    std::vector<f32> th(g.patches.size(), 0.0f);
    f32 mn = std::numeric_limits<f32>::max();
    for (usize i = 0; i < g.patches.size(); ++i) {
        const segment::Patch& p = g.patches[i];
        if (p.pos.empty()) continue;
        f64 s = 0;
        for (const Vec3f& q : p.pos) s += static_cast<f64>(wf.value(q));
        th[i] = static_cast<f32>(s / static_cast<f64>(p.pos.size()));
        mn = std::min(mn, th[i]);
    }
    // per-wrap θ step = median |Δθ| across Link edges (each spans exactly one wrap)
    std::vector<f32> steps;
    for (const segment::PatchEdge& e : g.edges)
        if (e.kind == segment::EdgeKind::Link)
            steps.push_back(std::abs(th[static_cast<usize>(e.a)] - th[static_cast<usize>(e.b)]));
    f32 dth = 1.0f;
    if (!steps.empty()) {
        std::nth_element(steps.begin(), steps.begin() + static_cast<s64>(steps.size() / 2), steps.end());
        dth = steps[steps.size() / 2];
    }
    if (dth < 1e-4f) dth = 1.0f;  // degenerate (no/zero-length links) -> leave θ unscaled
    g.wrap_lo = std::numeric_limits<s32>::max();
    g.wrap_hi = std::numeric_limits<s32>::min();
    for (usize i = 0; i < g.patches.size(); ++i) {
        const s32 w = static_cast<s32>(std::lround((th[i] - mn) / dth));
        g.patches[i].wrap = w;
        g.wrap_lo = std::min(g.wrap_lo, w);
        g.wrap_hi = std::max(g.wrap_hi, w);
    }
}

struct FieldFillParams {
    f32 step = 2.0f;         // tracer grid spacing (for the edge-stretch validation)
    int newton_iters = 6;    // Newton steps projecting a cell onto its wrap's level set
    int smooth_iters = 8;    // Laplacian position smoothing passes over the hole
    f32 max_edge = 2.6f;     // reject a filled cell whose edge to a neighbour exceeds this * step
    s64 max_hole = 200000;   // skip enclosed holes larger than this (a real void, not a dropout)
};

namespace detail {
// One Newton step set: move p onto {W = target} along ∇W (W changes ~1 per `spacing` voxels along g).
inline Vec3f project_to_wrap(const WindingField& wf, Vec3f p, f32 target, int iters) {
    for (int k = 0; k < iters; ++k) {
        const f32 e = target - wf.value(p);
        const Vec3f g = wf.grad(p);
        const f32 gg = dot(g, g);
        if (gg < 1e-12f) break;
        p = p + g * (e / gg);
    }
    return p;
}
}  // namespace detail

// Fill enclosed holes / weak-prediction gaps in a single-wrap patch by projecting each empty cell
// onto its wrap's level set in the field (i.e. onto the surface interpolated from the NEIGHBOURING
// wraps). Returns the number of cells filled. The patch's winding is `wrap`. This is the principled,
// neighbour-informed successor to the local-Laplacian fill_rivers/fill_holes: where the prediction
// dropped out, the geometry comes from where the adjacent wraps say the sheet must be.
inline s64 fill_surface_from_field(Surface& S, const WindingField& wf, s32 wrap, FieldFillParams fp = {}) {
    const int G = static_cast<int>(S.nu), H = static_cast<int>(S.nv);
    const int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};
    const usize NG = static_cast<usize>(G) * static_cast<usize>(H);
    // flood "outside" from the grid border through empty cells; remaining empties are enclosed holes.
    std::vector<u8> outside(NG, 0);
    std::vector<s64> stk;
    auto push = [&](int u, int v) {
        if (u < 0 || v < 0 || u >= G || v >= H) return;
        const usize id = static_cast<usize>(v * G + u);
        if (S.valid[id] || outside[id]) return;
        outside[id] = 1;
        stk.push_back(static_cast<s64>(id));
    };
    for (int u = 0; u < G; ++u) { push(u, 0); push(u, H - 1); }
    for (int v = 0; v < H; ++v) { push(0, v); push(G - 1, v); }
    while (!stk.empty()) {
        const s64 i = stk.back();
        stk.pop_back();
        const int u = static_cast<int>(i % G), v = static_cast<int>(i / G);
        for (int k = 0; k < 4; ++k) push(u + du4[k], v + dv4[k]);
    }

    // collect enclosed-hole components and fill each.
    std::vector<u8> seen(NG, 0);
    const f32 wt = static_cast<f32>(wrap);
    s64 filled = 0;
    for (int v0 = 1; v0 < H - 1; ++v0)
        for (int u0 = 1; u0 < G - 1; ++u0) {
            const usize id0 = static_cast<usize>(v0 * G + u0);
            if (S.valid[id0] || outside[id0] || seen[id0]) continue;
            std::vector<s64> comp{static_cast<s64>(id0)};
            seen[id0] = 1;
            for (usize h = 0; h < comp.size(); ++h) {
                const int cu = static_cast<int>(comp[h] % G), cv = static_cast<int>(comp[h] / G);
                for (int k = 0; k < 4; ++k) {
                    const int uu = cu + du4[k], vv = cv + dv4[k];
                    if (uu < 1 || vv < 1 || uu >= G - 1 || vv >= H - 1) continue;
                    const usize nid = static_cast<usize>(vv * G + uu);
                    if (!S.valid[nid] && !outside[nid] && !seen[nid]) { seen[nid] = 1; comp.push_back(static_cast<s64>(nid)); }
                }
            }
            if (static_cast<s64>(comp.size()) > fp.max_hole) continue;
            for (s64 ci : comp) S.coord[static_cast<usize>(ci)] = Vec3f{0, 0, 0};
            // Laplacian warm-start across the hole from the valid banks.
            for (int it = 0; it < static_cast<int>(comp.size()) + fp.smooth_iters; ++it)
                for (s64 ci : comp) {
                    const int cu = static_cast<int>(ci % G), cv = static_cast<int>(ci / G);
                    Vec3f sum{0, 0, 0};
                    int c = 0;
                    for (int k = 0; k < 4; ++k) {
                        const int uu = cu + du4[k], vv = cv + dv4[k];
                        if (uu < 0 || vv < 0 || uu >= G || vv >= H) continue;
                        const usize nid = static_cast<usize>(vv * G + uu);
                        if (S.valid[nid] || norm(S.coord[nid]) > 0) { sum = sum + S.coord[nid]; ++c; }
                    }
                    if (c) S.coord[static_cast<usize>(ci)] = sum / static_cast<f32>(c);
                }
            // project each onto the wrap's level set, then accept if its edges stay near step length.
            for (s64 ci : comp) {
                const usize id = static_cast<usize>(ci);
                if (norm(S.coord[id]) == 0) continue;
                S.coord[id] = detail::project_to_wrap(wf, S.coord[id], wt, fp.newton_iters);
                const int cu = static_cast<int>(ci % G), cv = static_cast<int>(ci / G);
                f32 maxe = 0;
                int nb = 0;
                for (int k = 0; k < 4; ++k) {
                    const int uu = cu + du4[k], vv = cv + dv4[k];
                    if (uu < 0 || vv < 0 || uu >= G || vv >= H) continue;
                    const usize nid = static_cast<usize>(vv * G + uu);
                    if (S.valid[nid]) { maxe = std::max(maxe, norm(S.coord[id] - S.coord[nid])); ++nb; }
                }
                if (nb >= 1 && maxe <= fp.max_edge * fp.step) { S.valid[id] = 1; ++filled; }
            }
        }
    return filled;
}

}  // namespace fenix::winding
