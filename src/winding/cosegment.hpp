// winding/cosegment.hpp — Stage D: the coupled patch ↔ winding-field refinement (the segment↔winding
// co-evolution). Each round: (1) analyze the patch population (spacing, same-sheet merges, integer
// windings); (2) build the coarse winding field, pinned only by HIGH-confidence cells; (3) refine
// every patch against that field — fill enclosed holes from the neighbouring wraps, and pull weak
// (low-confidence) cells onto their wrap's level set so they stay coherent with the wraps on either
// side. Repeat: corrected patches sharpen the field, the sharper field corrects more patches. Health
// is reported as the graph's winding conflicts + the field's radial monotonicity violation (a fold /
// wrap-overlap signal). This is how predictions that are weak on one wrap get "filled in" from its
// two neighbours, and how nearby non-merging patches keep the deformed spiral going coherently.
#pragma once

#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "segment/patch_graph.hpp"
#include "winding/patch_field.hpp"
#include "winding/winding_field.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace fenix::winding {

// Pull weak (low-confidence) VALID cells partway onto their wrap's level set, keeping them coherent
// with the neighbouring wraps. Clean cells are left alone. Returns the number of cells moved.
inline s64 field_consistency_snap(Surface& S, const WindingField& wf, s32 wrap, f32 conf_thresh,
                                  f32 weight, int newton_iters = 6) {
    if (!S.has_channels()) return 0;
    const f32 wt = static_cast<f32>(wrap);
    s64 moved = 0;
    for (usize id = 0; id < S.valid.size(); ++id) {
        if (!S.valid[id] || S.conf[id] >= conf_thresh) continue;
        const Vec3f p = S.coord[id];
        const Vec3f t = detail::project_to_wrap(wf, p, wt, newton_iters);
        S.coord[id] = p + (t - p) * weight;
        ++moved;
    }
    return moved;
}

struct CosegParams {
    Extent3 full{0, 0, 0};   // full-res volume extent (the field grid is this / field.ds)
    int rounds = 3;          // EM rounds (analyze → field → refine)
    FieldParams field{};     // winding-field build params
    FieldFillParams fill{};  // hole-fill params
    f32 conf_thresh = 1.0f;  // cells below this confidence are pulled toward the field
    f32 consist_weight = 0.5f;  // how far weak cells move toward the level set each round (0..1)
};

struct CosegReport {
    f32 spacing = 0;
    s32 clusters = 0;
    s32 wrap_lo = 0, wrap_hi = 0;
    s32 conflicts = 0;       // winding-assignment cycle conflicts (graph health)
    f32 monotonicity = 0;    // field radial-monotonicity violation fraction (fold/overlap health, ~0 good)
    s64 filled = 0;          // cells filled from the field in the last round
    s64 snapped = 0;         // weak cells pulled onto the field in the last round
};

// Run the coupled refinement in place on `sheets`. Patch i's winding is taken from the graph built
// each round (patches are built in sheet order, so graph patch i == sheets[i]).
inline CosegReport cosegment_refine(std::vector<Surface>& sheets, const annotate::Umbilicus& umb,
                                    segment::PatchGraphParams gp, CosegParams cp) {
    CosegReport rep;
    const int ds = std::max(1, cp.field.ds);
    for (int round = 0; round < cp.rounds; ++round) {
        const segment::PatchGraph g = segment::analyze_patches(sheets, umb, gp);
        rep.spacing = g.spacing;
        rep.clusters = g.cluster_count;
        rep.wrap_lo = g.wrap_lo;
        rep.wrap_hi = g.wrap_hi;
        rep.conflicts = g.winding_conflicts;
        if (gp.spacing <= 0) gp.spacing = g.spacing;  // lock spacing after the first robust estimate

        const WindingField wf = build_patch_winding_field(g.patches, cp.full, g.spacing, cp.field);
        s64 filled = 0, snapped = 0;
        for (usize i = 0; i < sheets.size(); ++i) {
            const s32 w = g.patches[i].wrap;
            if (w == segment::kUnassignedWrap) continue;
            filled += fill_surface_from_field(sheets[i], wf, w, cp.fill);
            snapped += field_consistency_snap(sheets[i], wf, w, cp.conf_thresh, cp.consist_weight);
        }
        rep.filled = filled;
        rep.snapped = snapped;

        if (round == cp.rounds - 1) {
            // field health: cast radial rays in the coarse field and measure monotonicity (a wrap that
            // folds back / overlaps a neighbour shows up as a non-increasing winding along the radius).
            // Restrict to the DATA BAND [r of innermost wrap, r of outermost wrap]: outside it the
            // field is a flat extrapolation, which would read as (spurious) non-monotonicity.
            annotate::Umbilicus uc;
            const f32 invds = 1.0f / static_cast<f32>(ds);
            for (usize k = 0; k < umb.z.size(); ++k) {
                uc.z.push_back(umb.z[k] * invds);
                uc.y.push_back(umb.y[k] * invds);
                uc.x.push_back(umb.x[k] * invds);
            }
            f32 rlo = 1e30f, rhi = 0.0f;
            for (const segment::Patch& p : g.patches)
                for (const Vec3f& q : p.pos) {
                    const Vec3f c = umb.center(q.z);
                    const f32 r = std::sqrt((q.y - c.y) * (q.y - c.y) + (q.x - c.x) * (q.x - c.x));
                    rlo = std::min(rlo, r);
                    rhi = std::max(rhi, r);
                }
            const Extent3 dd = wf.w.dims();
            const s64 zc = dd.z / 2;
            const s64 rmin_c = std::max<s64>(1, static_cast<s64>(rlo * invds) + 1);
            const s64 rmax_c = std::min<s64>(std::min(dd.y, dd.x) / 2, static_cast<s64>(rhi * invds));
            rep.monotonicity = rmax_c > rmin_c
                                   ? winding::monotonicity_violation(wf.w.view(), uc, zc, rmax_c, 180, rmin_c)
                                   : 0.0f;
        }
    }
    return rep;
}

}  // namespace fenix::winding
