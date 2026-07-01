// eval/score.hpp — the Kaggle "Vesuvius Challenge - Surface Detection" composite score, on two
// BINARY masks (prediction vs ground truth):
//     Score = 0.30·TopoScore + 0.35·SurfaceDice@τ + 0.35·VOI_score      (higher = better, ∈[0,1])
// Ports taberna's `eval/score.c` / `scripts/official_score.py` onto fenix's existing pieces:
//   SurfaceDice@τ ← eval/nsd.hpp   |   VOI ← the union-foreground labelings here   |   TopoScore ← topo/betti.hpp
// τ=2.0 in voxel-spacing units; VOI_score = 1/(1+α·VOI_total), α=0.3; TopoScore = weighted Betti-F1
// over active homology dims (k=0 components, k=1 tunnels, k=2 cavities). See eval/CLAUDE.md.
#pragma once

#include "core/core.hpp"
#include "eval/nsd.hpp"
#include "geom/connected_components.hpp"
#include "topo/betti.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace fenix::eval {

// --- VOI over the UNION foreground -----------------------------------------------------------
// Label each mask by its 26-connected components; over the union of the two foregrounds, each voxel
// carries a pred-label and a gt-label (0 = "background cluster" where that mask is absent — NOT
// ignored, unlike the generic eval::voi). VOI = H(pred|gt) + H(gt|pred), lower is better.
struct VoiUnion {
    f64 split = 0;  // H(pred|gt) — over-segmentation (a wrap split into pieces)
    f64 merge = 0;  // H(gt|pred) — under-segmentation (a cross-wrap merge: catastrophic)
    [[nodiscard]] f64 total() const { return split + merge; }
};

inline VoiUnion voi_union(VolumeView<const u8> pred, VolumeView<const u8> gt) {
    auto lp = geom::connected_components(pred, geom::Conn::TwentySix);
    auto lg = geom::connected_components(gt, geom::Conn::TwentySix);
    VolumeView<const s32> P = lp.labels.view();
    VolumeView<const s32> G = lg.labels.view();
    // Parallel contingency tables: per-chunk local maps, merged serially (integer-exact).
    const s64 n = pred.size();
    const s64 nchunks = std::max<s64>(1, std::min<s64>(cpu_budget(), n));
    struct Local {
        std::unordered_map<s64, s64> np, ng, njoint;
        s64 total = 0;
    };
    std::vector<Local> locals(static_cast<usize>(nchunks));
    parallel_for(0, nchunks, [&](s64 c) {
        Local& L = locals[static_cast<usize>(c)];
        const s64 i0 = n * c / nchunks, i1 = n * (c + 1) / nchunks;
        for (s64 i = i0; i < i1; ++i) {
            const bool fp = pred.flat()[static_cast<usize>(i)] != 0;
            const bool fg = gt.flat()[static_cast<usize>(i)] != 0;
            if (!fp && !fg) continue;  // restrict to the union foreground
            const s64 s = fp ? static_cast<s64>(P.flat()[static_cast<usize>(i)]) : 0;  // 0 = background cluster
            const s64 g = fg ? static_cast<s64>(G.flat()[static_cast<usize>(i)]) : 0;
            ++L.np[s];
            ++L.ng[g];
            ++L.njoint[(s << 32) | static_cast<u32>(static_cast<u64>(g) & 0xffffffffull)];
            ++L.total;
        }
    });
    std::unordered_map<s64, s64> np, ng, njoint;
    s64 total = 0;
    for (Local& L : locals) {
        for (auto& [k, c] : L.np) np[k] += c;
        for (auto& [k, c] : L.ng) ng[k] += c;
        for (auto& [k, c] : L.njoint) njoint[k] += c;
        total += L.total;
    }
    if (total == 0) return {};
    const f64 N = static_cast<f64>(total);
    VoiUnion r;
    for (auto& [key, c] : njoint) {
        const s64 s = key >> 32;
        const s64 g = static_cast<s32>(key & 0xffffffff);
        const f64 pjoint = static_cast<f64>(c) / N;
        const f64 ps = static_cast<f64>(np[s]) / N;
        const f64 pg = static_cast<f64>(ng[g]) / N;
        r.merge += pjoint * std::log(pg / pjoint);  // H(gt|pred)
        r.split += pjoint * std::log(ps / pjoint);  // H(pred|gt)
    }
    return r;
}

// Bounded VOI score: 1/(1+α·VOI_total), α=0.3. 1 == identical partition, →0 as VOI grows.
inline f64 voi_score(f64 voi_total, f64 alpha = 0.3) { return 1.0 / (1.0 + alpha * voi_total); }

// --- TopoScore (Betti F1) --------------------------------------------------------------------
// Per-dim topological F1 from number-based Betti matching: matched = min(b_pred, b_gt),
// F1 = 2·matched/(b_pred+b_gt). Both zero ⇒ dim inactive (returns -1, skipped in the average).
// NOTE: this is the number-based PROXY (matches taberna's dim-1/2 candor); exact Betti-MATCHING
// via cubical PH (topo/) refines it later — see topo/CLAUDE.md.
inline f64 betti_f1(s64 bp, s64 bg) {
    if (bp == 0 && bg == 0) return -1.0;
    return 2.0 * static_cast<f64>(std::min(bp, bg)) / static_cast<f64>(bp + bg);
}

inline f64 topo_score(VolumeView<const u8> pred, VolumeView<const u8> gt,
                      std::array<f64, 3> w = {0.34, 0.33, 0.33}) {
    const topo::Betti bp = topo::betti_numbers(pred);
    const topo::Betti bg = topo::betti_numbers(gt);
    const std::array<f64, 3> f1{betti_f1(bp.b0, bg.b0), betti_f1(bp.b1, bg.b1), betti_f1(bp.b2, bg.b2)};
    f64 num = 0, den = 0;
    for (int k = 0; k < 3; ++k)
        if (f1[static_cast<usize>(k)] >= 0.0) { num += w[static_cast<usize>(k)] * f1[static_cast<usize>(k)]; den += w[static_cast<usize>(k)]; }
    return den > 0.0 ? num / den : 1.0;  // all dims inactive (both empty) ⇒ identical
}

// --- the composite ---------------------------------------------------------------------------
struct Score {
    f64 surface_dice = 0, voi_score = 0, topo_score = 0, total = 0;
};

// Kaggle composite on two binary masks. Edge cases follow the metric: both empty ⇒ all three = 1
// (total 1); one empty ⇒ SurfaceDice 0, VOI_score→small, TopoScore 0 (k=0 F1 = 0).
inline Score official_score(VolumeView<const u8> pred, VolumeView<const u8> gt, f32 tau = 2.0f) {
    Score s;
    s.surface_dice = nsd(pred, gt, tau);
    s.voi_score = eval::voi_score(voi_union(pred, gt).total());
    s.topo_score = topo_score(pred, gt);
    s.total = 0.30 * s.topo_score + 0.35 * s.surface_dice + 0.35 * s.voi_score;
    return s;
}

}  // namespace fenix::eval
