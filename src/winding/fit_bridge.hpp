// winding/fit_bridge.hpp — the segment→winding bridge: turn a PatchGraph whose patches carry assigned
// integer windings (from the CT-valley discrete assign or the band-Eulerian solve) into the two fit
// constraint kinds the diffeomorphic fit consumes:
//   - FitConstraint (winding-TARGET): each high-confidence patch cell pinned to its patch's integer wrap.
//   - CoWindingGroup (radius CONSTANCY): each patch's cells share one (whatever) winding — geometry-only,
//     so the global spiral model can re-stack patches the pairwise graph mis-related, the natural seed
//     for the later EM re-assignment.
// This is how the fit (the heart) ingests the segment side's output. See winding/CLAUDE.md.
#pragma once

#include "segment/patch_graph.hpp"
#include "winding/diffeo_fit.hpp"

#include <vector>

namespace fenix::winding {

struct BridgeParams {
    f32 conf_min = 1.0f;   // a cell becomes a hard winding-TARGET only if its confidence >= this (>=1 = snapped)
    int stride = 4;        // subsample patch cells (bounds the constraint count on big real patches)
    bool co_winding = true;// also emit a per-patch CoWindingGroup (shared-winding geometry term)
    int min_group = 3;     // a CoWindingGroup needs at least this many cells
};

struct BridgeOut {
    std::vector<FitConstraint> targets;
    std::vector<CoWindingGroup> groups;
};

// Build fit constraints from a PatchGraph with assigned `wrap`s. Targets are gauge-relative integers
// (consistent within a connected component — fine for the fit, which only reads differences + one
// anchor). Cells are subsampled by `stride`; only conf>=conf_min cells become hard targets.
inline BridgeOut patches_to_constraints(const segment::PatchGraph& g, BridgeParams bp = {}) {
    BridgeOut out;
    const int st = std::max(1, bp.stride);
    for (const segment::Patch& p : g.patches) {
        if (p.wrap == segment::kUnassignedWrap) continue;
        CoWindingGroup grp;
        const bool has_conf = p.conf.size() == p.pos.size();
        for (usize i = 0; i < p.pos.size(); i += static_cast<usize>(st)) {
            const f32 conf = has_conf ? p.conf[i] : 1.0f;
            if (bp.co_winding) grp.points.push_back(p.pos[i]);
            if (conf >= bp.conf_min) out.targets.push_back({p.pos[i], static_cast<f32>(p.wrap)});
        }
        if (bp.co_winding && static_cast<int>(grp.points.size()) >= bp.min_group) out.groups.push_back(std::move(grp));
    }
    return out;
}

}  // namespace fenix::winding
