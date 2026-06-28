// segment/patch_graph.hpp — the multi-scale glue over a population of traced patches. Grow MANY
// seeds (segment::trace_volume), then reason about how the patches relate as wraps of the spiral:
//   * SAME sheet (touching, co-normal, ~0 normal gap)            -> MERGE  (L0->L1->L2)
//   * ADJACENT wrap (co-normal, normal gap ~ wrap spacing)       -> LINK   (Δwinding = ±1)
//   * inconsistent (anti-normal / |Δwinding|>=2 / cycle clash)   -> CONFLICT
// From the LINK/MERGE graph we assign each patch an integer winding number (a potential-DSU over ℤ:
// the discrete, exact form of thaumato's winding-angle relaxation), giving every patch its place in
// the nested wrap stack (L3). This is the segment-side scaffold the winding fit consumes. The metrics
// (signed normal gap, co-normality, co-deformation residual, wrap spacing) are the analysis the whole
// multi-scale method runs on. See segment/CLAUDE.md and winding/CLAUDE.md.
#pragma once

#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "core/eig.hpp"
#include "core/surface.hpp"
#include "geom/kdtree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fenix::segment {

inline constexpr s32 kUnassignedWrap = std::numeric_limits<s32>::min();

// A compact, graph-ready view of one traced patch: only its valid cells, with the per-cell normal
// oriented OUTWARD (away from the umbilicus) so signed gaps are consistent across patches.
struct Patch {
    s32 id = 0;
    std::vector<Vec3f> pos;   // valid-cell 3D coords (ZYX, voxels)
    std::vector<Vec3f> nrm;   // outward-oriented unit across-sheet normals
    std::vector<f32> conf;    // per-cell data confidence (>=1 == snapped onto a sheet)
    Vec3f centroid{0, 0, 0};
    Vec3f bb_lo{0, 0, 0}, bb_hi{0, 0, 0};
    f32 ang_mean = 0;         // mean winding angle about the umbilicus (turns); thaumato's f_init
    f32 ang_lo = 0, ang_hi = 0;  // angular extent (turns)
    f32 radial_align = 0;     // mean |n·r̂| — how radial the normals are (1 = ideal sheet)
    s32 wrap = kUnassignedWrap;  // assigned integer winding (Stage B)
    s32 cluster = 0;          // same-sheet merge component (Stage A)
    [[nodiscard]] s64 size() const { return static_cast<s64>(pos.size()); }
};

enum class EdgeKind : u8 { Merge, Link, Conflict };

struct PatchEdge {
    s32 a = 0, b = 0;
    f32 dist = 0;       // closest-approach distance between the two cell sets (voxels)
    f32 conormal = 0;   // mean n_a·n_b over correspondences (sign-aware; +1 = parallel same orientation)
    f32 gap = 0;        // mean signed normal gap (b - a)·n_a (voxels): +outward, -inward
    f32 codeform = 0;   // co-deformation residual: stddev of the a->b offset field (voxels)
    int dwrap = 0;      // round(gap / spacing): 0 = same sheet, ±1 = adjacent wrap
    f32 certainty = 0;  // edge confidence in [0,1]
    EdgeKind kind = EdgeKind::Conflict;
};

struct PatchGraph {
    std::vector<Patch> patches;
    std::vector<PatchEdge> edges;
    f32 spacing = 0;          // estimated wrap spacing (voxels)
    s32 cluster_count = 0;    // number of same-sheet merge clusters
    s32 wrap_lo = 0, wrap_hi = 0;   // assigned winding range
    s32 winding_conflicts = 0;      // edges inconsistent with the winding assignment
};

struct PatchGraphParams {
    f32 step = 2.0f;             // tracer grid spacing (voxels); same-wrap seams are ~this apart
    f32 spacing = 0.0f;          // wrap spacing (voxels); 0 = auto-estimate from the gaps
    int sample_stride = 4;       // subsample a patch's cells when probing closest approach
    f32 bootstrap_radius = 48.0f;// bbox-gap cutoff for candidate pairs (used before spacing is known)
    f32 search_frac = 2.5f;      // keep edges for pairs within search_frac * spacing
    f32 conormal_min = 0.4f;     // |n_a·n_b| below this => not parallel sheets => conflict
    f32 merge_conormal = 0.8f;   // merge needs strongly-aligned normals (else it's a crossing, not a sheet)
    f32 merge_gap_steps = 1.5f;  // merge needs |gap| < this*step (ABSOLUTE, not a fraction of spacing):
                                 // a same-sheet seam has gap~0 regardless of winding density, whereas a
                                 // spacing fraction balloons where local wrap spacing is small and fuses
                                 // adjacent wraps (the tiled over-merge: a 117-fragment giant ~13 wraps).
    f32 merge_dist_steps = 1.5f; // merge ALSO needs closest-approach dmin < this*step: same-sheet
                                 // fragments TOUCH (dmin ~ step); different wraps sit a full spacing apart.
    int max_graph_cells = 8000;  // cap cells per patch used by the graph (subsample) — bounds the
                                 // O(P^2 * cells) pairwise cost on big real patches (700k cells -> 8k)
    int max_corr = 96;           // correspondences kept per pair for the co-deformation residual
};

// --- build one Patch from a traced Surface (orient normals outward via the umbilicus radial) ---
inline Patch make_patch(const Surface& S, s32 id, const annotate::Umbilicus& umb, int max_cells = 8000) {
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    Patch P;
    P.id = id;
    const s64 G = S.nu;
    const bool ch = S.has_channels();
    s64 nvalid = 0;
    for (u8 b : S.valid) nvalid += b;
    const s64 stride = (max_cells > 0 && nvalid > max_cells) ? nvalid / max_cells + 1 : 1;  // subsample to cap
    Vec3f lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f}, csum{0, 0, 0};
    f64 ang_sum = 0, align_sum = 0;
    f32 amin = 1e30f, amax = -1e30f;
    s64 seen = 0;
    for (s64 v = 0; v < S.nv; ++v)
        for (s64 u = 0; u < G; ++u) {
            const usize idx = S.idx(u, v);
            if (!S.valid[idx]) continue;
            if ((seen++ % stride) != 0) continue;  // keep ~max_cells representative cells
            const Vec3f p = S.coord[idx];
            const Vec3f c = umb.center(p.z);
            Vec3f rad{0, p.y - c.y, p.x - c.x};
            const f32 rl = norm(rad);
            rad = rl > 1e-6f ? rad / rl : Vec3f{0, 1, 0};
            Vec3f n = ch ? S.normal[idx] : rad;
            if (norm(n) < 1e-6f) n = rad;
            n = normalized(n);
            // Keep the grower's RAW per-cell normal: it is consistently oriented WITHIN the patch
            // (parallel-transported frame), which is what makes a per-cell signed gap meaningful on a
            // curved sheet. Imposing a global axis here makes the gap cancel across a U-bend. Cross-
            // patch orientation is resolved later by propagation over the graph (orient_patches).
            const f32 turn = std::atan2(p.y - c.y, p.x - c.x) / two_pi;
            P.pos.push_back(p);
            P.nrm.push_back(n);
            P.conf.push_back(ch ? S.conf[idx] : 1.0f);
            csum = csum + p;
            lo = Vec3f{std::min(lo.z, p.z), std::min(lo.y, p.y), std::min(lo.x, p.x)};
            hi = Vec3f{std::max(hi.z, p.z), std::max(hi.y, p.y), std::max(hi.x, p.x)};
            ang_sum += static_cast<f64>(turn);
            amin = std::min(amin, turn);
            amax = std::max(amax, turn);
            align_sum += static_cast<f64>(std::abs(dot(n, rad)));
        }
    const f32 nn = static_cast<f32>(std::max<usize>(P.pos.size(), 1));
    P.centroid = csum / nn;
    P.bb_lo = lo;
    P.bb_hi = hi;
    P.ang_mean = static_cast<f32>(ang_sum) / nn;
    P.ang_lo = amin;
    P.ang_hi = amax;
    P.radial_align = static_cast<f32>(align_sum) / nn;
    return P;
}

namespace detail {
// minimum gap between two axis-aligned bounding boxes (0 if they overlap).
inline f32 bbox_gap(const Patch& A, const Patch& B) {
    const f32 dz = std::max({0.0f, A.bb_lo.z - B.bb_hi.z, B.bb_lo.z - A.bb_hi.z});
    const f32 dy = std::max({0.0f, A.bb_lo.y - B.bb_hi.y, B.bb_lo.y - A.bb_hi.y});
    const f32 dx = std::max({0.0f, A.bb_lo.x - B.bb_hi.x, B.bb_lo.x - A.bb_hi.x});
    return std::sqrt(dz * dz + dy * dy + dx * dx);
}

// union-find over ±1: propagate a consistent normal orientation along the patch manifold. Each node
// carries a sign relative to its parent; an edge "a and b have relative orientation rel (= sign of
// n_a·n_b)" fixes b's sign vs a. Spanning the graph from the most-co-normal edges first orients every
// patch consistently WITHOUT any global axis/umbilicus — robust to sheet curvature (U-bends).
struct SignDSU {
    std::vector<s32> par, sg;
    explicit SignDSU(s32 n) : par(static_cast<usize>(n)), sg(static_cast<usize>(n), 1) {
        for (s32 i = 0; i < n; ++i) par[static_cast<usize>(i)] = i;
    }
    s32 find(s32 x, s32& s) {
        if (par[static_cast<usize>(x)] == x) { s = 1; return x; }
        s32 ps = 1;
        const s32 r = find(par[static_cast<usize>(x)], ps);
        sg[static_cast<usize>(x)] *= ps;
        par[static_cast<usize>(x)] = r;
        s = sg[static_cast<usize>(x)];
        return r;
    }
    void unite(s32 a, s32 b, s32 rel) {
        s32 sa = 1, sb = 1;
        const s32 ra = find(a, sa), rb = find(b, sb);
        if (ra != rb) { par[static_cast<usize>(rb)] = ra; sg[static_cast<usize>(rb)] = rel * sa * sb; }
    }
};
}  // namespace detail

// Build the patch adjacency graph: closest-approach + co-normality + signed normal gap +
// co-deformation residual for every near pair, the wrap-spacing estimate, a consistent normal
// orientation propagated over the manifold, and the merge/link/conflict classification. Patches carry
// channels (the grower fills them); normals are kept locally-consistent (no global axis).
inline PatchGraph build_patch_graph(std::span<const Surface> sheets, const annotate::Umbilicus& umb,
                                    PatchGraphParams gp = {}) {
    PatchGraph g;
    // make_patch, the KdTree builds, and the O(P^2) pairwise metrics below are all per-element
    // independent (read-only on the inputs) -> parallelize each. Tiling the tracer multiplies the patch
    // count (whole-cube sheets -> hundreds of tile fragments), which made this the pipeline bottleneck;
    // it is also the out-of-core stitch, so it must scale. KdTree::nearest is const (no mutable state),
    // so many threads can query the same tree concurrently.
    g.patches.resize(sheets.size());
    parallel_for(0, static_cast<s64>(sheets.size()), [&](s64 i) {
        g.patches[static_cast<usize>(i)] = make_patch(sheets[static_cast<usize>(i)], static_cast<s32>(i), umb, gp.max_graph_cells);
    });
    const s32 P = static_cast<s32>(g.patches.size());
    std::vector<geom::KdTree> trees(static_cast<usize>(P));
    parallel_for(0, P, [&](s64 i) { trees[static_cast<usize>(i)] = geom::KdTree(g.patches[static_cast<usize>(i)].pos); });

    // gap/conormal are read with each patch's OWN normals (locally consistent); the gap sign is per-A
    // and only made globally consistent after orientation propagation below.
    struct Raw { s32 a, b; f32 dmin, dmean, gap, conormal, codeform; int n; };
    const f32 boot = gp.bootstrap_radius;
    const int stride = std::max(1, gp.sample_stride);
    // per-row buffers (row a owns pairs (a, b>a)) -> no cross-thread writes; concatenated in a-order
    // afterwards (identical to the old serial order, so all downstream stats are unchanged).
    std::vector<std::vector<Raw>> raws_a(static_cast<usize>(P));
    parallel_for(0, P, [&](s64 a_) {
        const s32 a = static_cast<s32>(a_);
        std::vector<Raw>& out = raws_a[static_cast<usize>(a)];
        const Patch& A = g.patches[static_cast<usize>(a)];
        for (s32 b = a + 1; b < P; ++b) {
            const Patch& B = g.patches[static_cast<usize>(b)];
            if (detail::bbox_gap(A, B) > boot) continue;
            struct Corr { f32 d, g, cn; Vec3f o, na; };
            std::vector<Corr> cs;
            f32 dmin = 1e30f;
            for (usize i = 0; i < A.pos.size(); i += static_cast<usize>(stride)) {
                const s64 j = trees[static_cast<usize>(b)].nearest(A.pos[i]);
                if (j < 0) continue;
                const Vec3f o = B.pos[static_cast<usize>(j)] - A.pos[i];
                const f32 d = norm(o);
                dmin = std::min(dmin, d);
                if (d <= boot) cs.push_back({d, dot(o, A.nrm[i]), dot(A.nrm[i], B.nrm[static_cast<usize>(j)]), o, A.nrm[i]});
            }
            const f32 band = dmin + 4.0f * gp.step;  // read the relationship near the closest approach
            f64 gsum = 0, cnsum = 0, dsum = 0;
            int n = 0;
            for (const Corr& c : cs)
                if (c.d <= band) { gsum += static_cast<f64>(c.g); cnsum += static_cast<f64>(c.cn); dsum += static_cast<f64>(c.d); ++n; }
            if (n < 3) continue;
            const f32 gap = static_cast<f32>(gsum / n);
            const f32 cn = static_cast<f32>(cnsum / n);
            const f32 dmean = static_cast<f32>(dsum / n);
            f64 vg = 0, tg = 0;  // co-deformation residual in the local frame (curvature-invariant)
            for (const Corr& c : cs)
                if (c.d <= band) {
                    const f32 dg = c.g - gap;
                    const Vec3f t = c.o - c.na * c.g;
                    vg += static_cast<f64>(dg * dg);
                    tg += static_cast<f64>(dot(t, t));
                }
            const f32 cdf = static_cast<f32>(std::sqrt((vg + tg) / static_cast<f64>(n)));
            out.push_back({a, b, dmin, dmean, gap, cn, cdf, n});
        }
    });
    std::vector<Raw> raws;
    for (const std::vector<Raw>& v : raws_a) raws.insert(raws.end(), v.begin(), v.end());

    // wrap spacing: gap to the NEAREST OUTWARD neighbour (per-patch min |gap| above the seam floor),
    // medianed. |gap| is orientation-free (local normals), so this is curvature-robust.
    f32 spacing = gp.spacing;
    if (spacing <= 0) {
        std::vector<f32> permin(static_cast<usize>(P), 1e30f);
        for (const Raw& r : raws)
            if (std::abs(r.conormal) > gp.conormal_min && std::abs(r.gap) > 1.5f * gp.step && std::abs(r.gap) < boot) {
                permin[static_cast<usize>(r.a)] = std::min(permin[static_cast<usize>(r.a)], std::abs(r.gap));
                permin[static_cast<usize>(r.b)] = std::min(permin[static_cast<usize>(r.b)], std::abs(r.gap));
            }
        std::vector<f32> mins;
        for (f32 m : permin)
            if (m < 1e29f) mins.push_back(m);
        if (!mins.empty()) {
            std::nth_element(mins.begin(), mins.begin() + static_cast<s64>(mins.size() / 2), mins.end());
            spacing = mins[mins.size() / 2];
        } else {
            spacing = 4.0f * gp.step;
        }
    }
    g.spacing = spacing;

    // propagate a consistent normal orientation over the manifold: span the most-co-normal edges
    // first, fixing each patch's sign relative to its neighbour (no global axis -> robust to U-bends).
    detail::SignDSU od(P);
    {
        std::vector<s32> ord(raws.size());
        for (usize i = 0; i < raws.size(); ++i) ord[i] = static_cast<s32>(i);
        std::sort(ord.begin(), ord.end(), [&](s32 x, s32 y) { return std::abs(raws[static_cast<usize>(x)].conormal) > std::abs(raws[static_cast<usize>(y)].conormal); });
        for (s32 ei : ord) {
            const Raw& r = raws[static_cast<usize>(ei)];
            if (std::abs(r.conormal) < 0.5f) break;  // sorted: nothing reliable left
            od.unite(r.a, r.b, r.conormal >= 0 ? 1 : -1);
        }
    }
    std::vector<s32> sgn(static_cast<usize>(P));
    for (s32 i = 0; i < P; ++i) { s32 s = 1; od.find(i, s); sgn[static_cast<usize>(i)] = s; }
    for (s32 i = 0; i < P; ++i)
        if (sgn[static_cast<usize>(i)] < 0)
            for (Vec3f& nn : g.patches[static_cast<usize>(i)].nrm) nn = nn * -1.0f;

    for (const Raw& r : raws) {
        if (r.dmin > gp.search_frac * spacing && std::abs(r.gap) > gp.search_frac * spacing) continue;
        const f32 sgap = r.gap * static_cast<f32>(sgn[static_cast<usize>(r.a)]);  // consistent outward gap
        PatchEdge e;
        e.a = r.a;
        e.b = r.b;
        e.dist = r.dmin;
        e.gap = sgap;
        e.conormal = r.conormal * static_cast<f32>(sgn[static_cast<usize>(r.a)] * sgn[static_cast<usize>(r.b)]);
        e.codeform = r.codeform;
        e.dwrap = static_cast<int>(std::lround(sgap / spacing));
        const f32 rgap = std::abs(sgap / spacing - static_cast<f32>(e.dwrap));
        const f32 cnabs = std::clamp(std::abs(r.conormal), 0.0f, 1.0f);
        const f32 cd = r.codeform / spacing;
        e.certainty = cnabs * std::exp(-3.0f * rgap) * std::exp(-1.5f * cd);
        // MERGE is orientation-free: strongly co-normal AND small |gap| (local normals don't cancel
        // across a bend, so two adjacent wraps keep |gap|~spacing and never merge).
        if (cnabs < gp.conormal_min) e.kind = EdgeKind::Conflict;
        else if (cnabs > gp.merge_conormal && std::abs(r.gap) < gp.merge_gap_steps * gp.step && r.dmin < gp.merge_dist_steps * gp.step) e.kind = EdgeKind::Merge;
        else if (std::abs(e.dwrap) == 1) e.kind = EdgeKind::Link;
        else e.kind = EdgeKind::Conflict;
        g.edges.push_back(e);
    }
    return g;
}

// Merge same-sheet patches (L0->L1->L2): union-find over the MERGE edges -> compact cluster ids on
// patch.cluster. Returns the cluster count.
inline s32 merge_same_sheet(PatchGraph& g) {
    const s32 P = static_cast<s32>(g.patches.size());
    std::vector<s32> par(static_cast<usize>(P));
    for (s32 i = 0; i < P; ++i) par[static_cast<usize>(i)] = i;
    auto find = [&](s32 x) {
        while (par[static_cast<usize>(x)] != x) { par[static_cast<usize>(x)] = par[static_cast<usize>(par[static_cast<usize>(x)])]; x = par[static_cast<usize>(x)]; }
        return x;
    };
    for (const PatchEdge& e : g.edges)
        if (e.kind == EdgeKind::Merge) par[static_cast<usize>(find(e.a))] = find(e.b);
    std::unordered_map<s32, s32> compact;
    for (s32 i = 0; i < P; ++i) {
        const s32 r = find(i);
        auto it = compact.find(r);
        if (it == compact.end()) it = compact.emplace(r, static_cast<s32>(compact.size())).first;
        g.patches[static_cast<usize>(i)].cluster = it->second;
    }
    g.cluster_count = static_cast<s32>(compact.size());
    return g.cluster_count;
}

struct WindingSolveParams {
    int iters = 600;          // weighted-Jacobi sweeps
    int reweight_every = 50;  // robustly down-weight violated edges every N sweeps
    f32 reweight_k = 2.0f;    // edge weight *= exp(-k * |residual|) (Geman-McClure-ish robustifier)
};

// Assign an integer winding number to each patch from the MERGE (Δ=0) + LINK (Δ=±1) edges. Real wraps
// have variable spacing, so the per-edge Δwrap is noisy and a HARD integer gauge would flag every
// imperfect cycle as a conflict. Instead solve the continuous least-squares winding (thaumato-style
// weighted Jacobi: f_i = Σ w(f_j ± t)/Σ w), robustly down-weighting edges that stay violated, anchor
// one node per connected component, snap to the nearest integer, and normalize each component to start
// at 0. `winding_conflicts` is then the number of edges still violated AFTER the snap (the genuine
// inconsistencies, not every noisy gap). On clean data this reproduces the exact integer gauge.
inline void assign_windings(PatchGraph& g, WindingSolveParams sp = {}) {
    const s32 P = static_cast<s32>(g.patches.size());
    struct E { s32 a, b, t; f32 w0; };
    std::vector<E> es;
    for (const PatchEdge& e : g.edges)
        if (e.kind != EdgeKind::Conflict)
            es.push_back({e.a, e.b, (e.kind == EdgeKind::Merge) ? 0 : e.dwrap, std::max(e.certainty, 1e-3f)});

    // connected components (for anchoring + per-component normalization).
    std::vector<s32> par(static_cast<usize>(P));
    for (s32 i = 0; i < P; ++i) par[static_cast<usize>(i)] = i;
    auto find = [&](s32 x) { while (par[static_cast<usize>(x)] != x) { par[static_cast<usize>(x)] = par[static_cast<usize>(par[static_cast<usize>(x)])]; x = par[static_cast<usize>(x)]; } return x; };
    for (const E& e : es) par[static_cast<usize>(find(e.a))] = find(e.b);
    std::vector<u8> anchored(static_cast<usize>(P), 0);
    std::unordered_map<s32, s32> rep;  // one representative node per component -> anchored at 0
    for (s32 i = 0; i < P; ++i) {
        const s32 r = find(i);
        if (rep.emplace(r, i).second) anchored[static_cast<usize>(i)] = 1;
    }

    // adjacency: (neighbour, t-sign-from-this-node, edge index).
    std::vector<std::vector<std::tuple<s32, s32, usize>>> adj(static_cast<usize>(P));
    for (usize i = 0; i < es.size(); ++i) {
        adj[static_cast<usize>(es[i].a)].push_back({es[i].b, -es[i].t, i});
        adj[static_cast<usize>(es[i].b)].push_back({es[i].a, +es[i].t, i});
    }
    std::vector<f64> f(static_cast<usize>(P), 0.0);
    std::vector<f32> ew(es.size());
    for (usize i = 0; i < es.size(); ++i) ew[i] = es[i].w0;
    for (int it = 0; it < sp.iters; ++it) {
        std::vector<f64> nf(static_cast<usize>(P));
        for (s32 i = 0; i < P; ++i) {
            if (anchored[static_cast<usize>(i)]) { nf[static_cast<usize>(i)] = 0.0; continue; }
            f64 num = 0, den = 0;
            for (const auto& [j, ts, ei] : adj[static_cast<usize>(i)]) {
                num += static_cast<f64>(ew[ei]) * (f[static_cast<usize>(j)] + static_cast<f64>(ts));
                den += static_cast<f64>(ew[ei]);
            }
            nf[static_cast<usize>(i)] = den > 0 ? num / den : f[static_cast<usize>(i)];
        }
        f.swap(nf);
        if (sp.reweight_every > 0 && (it % sp.reweight_every) == sp.reweight_every - 1)
            for (usize e = 0; e < es.size(); ++e) {
                const f64 r = f[static_cast<usize>(es[e].b)] - f[static_cast<usize>(es[e].a)] - static_cast<f64>(es[e].t);
                ew[e] = es[e].w0 * static_cast<f32>(std::exp(-static_cast<f64>(sp.reweight_k) * std::abs(r)));
            }
    }

    std::vector<s32> raw(static_cast<usize>(P));
    for (s32 i = 0; i < P; ++i) raw[static_cast<usize>(i)] = static_cast<s32>(std::lround(f[static_cast<usize>(i)]));
    std::unordered_map<s32, s32> cmin;
    for (s32 i = 0; i < P; ++i) {
        const s32 r = find(i);
        auto it = cmin.find(r);
        if (it == cmin.end() || raw[static_cast<usize>(i)] < it->second) cmin[r] = raw[static_cast<usize>(i)];
    }
    s32 lo = std::numeric_limits<s32>::max(), hi = std::numeric_limits<s32>::min();
    for (s32 i = 0; i < P; ++i) {
        const s32 w = raw[static_cast<usize>(i)] - cmin[find(i)];
        g.patches[static_cast<usize>(i)].wrap = w;
        lo = std::min(lo, w);
        hi = std::max(hi, w);
    }
    s32 conflicts = 0;
    for (const E& e : es)
        if ((g.patches[static_cast<usize>(e.b)].wrap - g.patches[static_cast<usize>(e.a)].wrap) != e.t) ++conflicts;
    g.wrap_lo = P ? lo : 0;
    g.wrap_hi = P ? hi : 0;
    g.winding_conflicts = conflicts;
}

// End-to-end Stage A+B: build the graph, merge same-sheet patches, assign integer windings.
inline PatchGraph analyze_patches(std::span<const Surface> sheets, const annotate::Umbilicus& umb,
                                  PatchGraphParams gp = {}) {
    PatchGraph g = build_patch_graph(sheets, umb, gp);
    merge_same_sheet(g);
    assign_windings(g);
    return g;
}

}  // namespace fenix::segment
