// segment/grow.hpp — generational surface region-grower (the VC-style tracer, first-party).
// Grows a (u,v) grid of 3D points outward from a seed across the predicted-surface field:
// each new grid point is extrapolated from its already-placed neighbours and then SNAPPED onto
// the sheet by a short search along the local across-sheet normal (so it locks to the current
// wrap and cannot jump to an adjacent one — the key to not cutting across wraps), with periodic
// Laplacian relaxation + re-snap for regularity. Normals come from the structure tensor of a
// downsampled copy of the field (computed natively here). Produces a Surface (tifxyz/.fxsurf).
#pragma once

#include "core/core.hpp"
#include "core/eig.hpp"
#include "core/sampling.hpp"
#include "core/surface.hpp"
#include "segment/ct_valley.hpp"
#include "segment/structure_tensor.hpp"

#include <algorithm>
#include <functional>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fenix::segment {

// Flat open-addressing (linear-probe) s64->s64 map for the 3D occupancy / injectivity guard.
// The tracer hammers this (27 probes per candidate); std::unordered_map's node-per-entry +
// malloc churn dominated the grow loop, so we use a cache-friendly flat table. Keys are spatial
// bin codes (always positive in practice); kEmpty is the slot sentinel.
struct OccMap {
    static constexpr s64 kEmpty = std::numeric_limits<s64>::min();
    std::vector<s64> keys_, vals_;
    s64 mask_ = 0, count_ = 0;
    OccMap() { alloc(1024); }
    static s64 mix(s64 k) {
        u64 x = static_cast<u64>(k);
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
        return static_cast<s64>(x);
    }
    void alloc(s64 cap) {
        s64 c = 16; while (c < cap) c <<= 1;
        keys_.assign(static_cast<usize>(c), kEmpty); vals_.assign(static_cast<usize>(c), 0);
        mask_ = c - 1; count_ = 0;
    }
    void rehash(s64 cap) {
        std::vector<s64> ok = std::move(keys_), ov = std::move(vals_);
        alloc(cap);
        for (usize i = 0; i < ok.size(); ++i) if (ok[i] != kEmpty) set(ok[i], ov[i]);
    }
    void reserve(s64 n) { if ((n * 4) > (mask_ + 1) * 3) rehash(n * 2); }
    [[nodiscard]] s64* get(s64 k) {
        s64 h = mix(k) & mask_;
        while (keys_[static_cast<usize>(h)] != kEmpty) {
            if (keys_[static_cast<usize>(h)] == k) return &vals_[static_cast<usize>(h)];
            h = (h + 1) & mask_;
        }
        return nullptr;
    }
    void set(s64 k, s64 v) {
        if ((count_ + 1) * 4 >= (mask_ + 1) * 3) rehash((mask_ + 1) * 2);
        s64 h = mix(k) & mask_;
        while (keys_[static_cast<usize>(h)] != kEmpty) {
            if (keys_[static_cast<usize>(h)] == k) { vals_[static_cast<usize>(h)] = v; return; }
            h = (h + 1) & mask_;
        }
        keys_[static_cast<usize>(h)] = k; vals_[static_cast<usize>(h)] = v; ++count_;
    }
    [[nodiscard]] s64 size() const { return count_; }
};

// Low-resolution across-sheet normal field (sampled in full-res voxel coordinates).
struct NormalField {
    int ds = 8;
    Vec3f org{0, 0, 0};  // world coord of the field's (0,0,0) — windows of a larger volume
    Volume<f32> nz, ny, nx;
    [[nodiscard]] Vec3f at(Vec3f cw) const {
        const Vec3f c = cw - org;
        const Vec3f p{c.z / static_cast<f32>(ds), c.y / static_cast<f32>(ds), c.x / static_cast<f32>(ds)};
        Vec3f n{sample_trilinear(nz.view(), p), sample_trilinear(ny.view(), p), sample_trilinear(nx.view(), p)};
        const f32 l = norm(n);
        return l > 1e-6f ? n / l : Vec3f{1, 0, 0};
    }
};

// Compute the across-sheet normal field natively: mean-downsample `vol` by `ds`, take the
// structure tensor, keep the largest-eigenvalue eigenvector (across-sheet direction).
template <class T>
inline NormalField compute_normal_field(VolumeView<const T> vol, int ds, StParams st = {1.0f, 2.0f}) {
    const Extent3 d = vol.dims();
    const Extent3 dd{d.z / ds, d.y / ds, d.x / ds};
    Volume<f32> small(dd);
    auto sv = small.view();
    const f32 inv = 1.0f / static_cast<f32>(ds * ds * ds);
    parallel_for_z(dd, [&](s64 z) {
        for (s64 y = 0; y < dd.y; ++y)
            for (s64 x = 0; x < dd.x; ++x) {
                f32 acc = 0;
                for (int dz = 0; dz < ds; ++dz)
                    for (int dy = 0; dy < ds; ++dy)
                        for (int dx = 0; dx < ds; ++dx)
                            acc += static_cast<f32>(vol(z * ds + dz, y * ds + dy, x * ds + dx));
                sv(z, y, x) = acc * inv;
            }
    });
    SheetField sf = structure_tensor(small.view(), st);
    NormalField nf{ds, Vec3f{0, 0, 0}, Volume<f32>(dd), Volume<f32>(dd), Volume<f32>(dd)};
    for (s64 i = 0; i < dd.count(); ++i) {
        nf.nz.flat()[i] = sf.normal[static_cast<usize>(i)].z;
        nf.ny.flat()[i] = sf.normal[static_cast<usize>(i)].y;
        nf.nx.flat()[i] = sf.normal[static_cast<usize>(i)].x;
    }
    return nf;
}

// Combined sheet-likelihood data term: the ML surface PREDICTION plus the raw-CT density RIDGE.
// Predictions are imperfect (they drop out at cracks -> the "rivers"); the papyrus sheet is still a
// bright ridge in the CT there, so combining the two lets geometry lock onto real surface where the
// prediction failed. Each signal is normalized by its own accept threshold so `value()>=1` is the
// single, scale-free "this is a sheet" test; the snap maximizes `value` along the normal to find the
// ridge in whichever signal is present. ct is optional (empty view / ct_thresh<=0 -> prediction only).
template <class T>
struct DataField {
    VolumeView<const T> pred{};
    VolumeView<const T> ct{};
    f32 surf_thresh = 0.15f;  // prediction accept level (field units)
    f32 ct_thresh = 0.0f;     // CT ridge accept level (CT units); <=0 disables the CT term
    f32 ct_weight = 1.0f;     // down/up-weight the CT term in the combined max
    f32 ct_ds = 1.0f;         // CT grid is downsampled by this factor vs `pred` (1 = same resolution)
    f32 ct_skip = 1.5f;       // PER-LINE CT short-circuit threshold used by snap_to_sheet: if the
                              // prediction's ridge along the snap line is at least this strong, the snap
                              // never samples CT (the CT fallback only RAISES weak spots, so it cannot
                              // move a confident prediction lock). This drops the entire second 128 MiB
                              // volume gather on on-sheet snaps — the tracer's #1 data-term bandwidth
                              // lever — and keeps the coarse/permissive CT from sprawling growth where
                              // the prediction is already good. <=0 = always use the combined term.
    Vec3f org{0, 0, 0};       // world coord of pred's (0,0,0): ALL public accessors take WORLD
                              // coords and convert internally (streaming windows of a big volume)
    [[nodiscard]] bool has_ct() const { return ct_thresh > 0.0f && ct.dims().count() > 0; }
    // prediction-only sheetness (the trusted signal): pred / accept-level. One volume.
    [[nodiscard]] f32 pred_value(Vec3f pw) const { return sample_trilinear(pred, pw - org) / surf_thresh; }
    // Map a prediction-space coord into the (possibly coarser, ct_ds>1) CT grid. Any consumer that
    // samples `ct` directly with a prediction-space coord (not just value()) must go through this —
    // ct_valley's crosses_valley() guards in particular, which used to bypass it (grow.hpp:378,960).
    [[nodiscard]] Vec3f ct_coord(Vec3f pw) const {
        const Vec3f p = pw - org;
        return ct_ds == 1.0f ? p
                              : Vec3f{(p.z + 0.5f) / ct_ds - 0.5f, (p.y + 0.5f) / ct_ds - 0.5f,
                                      (p.x + 0.5f) / ct_ds - 0.5f};
    }
    // normalized sheetness: >=1 on a sheet by EITHER signal (prediction OR CT ridge). The CT term may
    // live on a coarser grid (ct_ds>1) to save RAM/compute — map the prediction-space coord into it.
    [[nodiscard]] f32 value(Vec3f p) const {
        const f32 pv = pred_value(p);
        if (!has_ct()) return pv;
        const f32 cv = ct_weight * sample_trilinear(ct, ct_coord(p)) / ct_thresh;
        return std::max(pv, cv);
    }
};

// Mean-pool a field by integer factor `s` -> a coarse f32 field (in the input's units). The coarse
// level is where the expensive growth runs; native is for refinement only (coarse-to-fine tracing).
template <class T>
inline Volume<f32> downscale_field(VolumeView<const T> v, int s) {
    const Extent3 d = v.dims();
    const Extent3 dd{d.z / s, d.y / s, d.x / s};
    Volume<f32> o(dd);
    auto ov = o.view();
    const f32 inv = 1.0f / static_cast<f32>(s * s * s);
    parallel_for_z(dd, [&](s64 z) {
        for (s64 y = 0; y < dd.y; ++y)
            for (s64 x = 0; x < dd.x; ++x) {
                f32 a = 0;
                for (int dz = 0; dz < s; ++dz)
                    for (int dy = 0; dy < s; ++dy)
                        for (int dx = 0; dx < s; ++dx)
                            a += static_cast<f32>(v(z * s + dz, y * s + dy, x * s + dx));
                ov(z, y, x) = a * inv;
            }
    });
    return o;
}

struct GrowParams {
    f32 step = 2.0f;          // grid spacing (voxels)
    f32 surf_thresh = 0.15f;  // min field value (in the field's own units) to accept a point
    f32 ct_thresh = 0.0f;     // raw-CT density above which it's papyrus (CT units); <=0 = CT term off
    f32 ct_weight = 1.0f;     // weight of the CT ridge vs the prediction in the combined data term
    f32 ct_ds = 1.0f;         // CT-term grid downsample vs the prediction (1 = same res; e.g. 2 = half)
    f32 ct_skip = 1.5f;       // skip the CT data-term sample where the prediction alone is >= this
                              // (a multiple of surf_thresh): CT is only a crack fallback, so this halves
                              // the data-term memory traffic where the prediction is strong. <=0 = off.
    f32 snap_radius = 4.0f;   // +/- search along the normal; keep < half the inter-wrap spacing
    int max_gen = 4000;       // generation cap
    int grid = 1400;          // (u,v) grid size
    int fold_thresh = 10;     // reject a point landing in a 3D bin owned by a uv-cell >this far away
    f32 bin_size = 2.0f;      // 3D occupancy bin (voxels) for the injectivity guard
    f32 lambda = 3.0f;        // ARAP data weight (pull onto sheet vs. stay isometric)
    int fit_every = 0;        // interleave a light ARAP fit every N generations (0 = grow free first)
    int final_outer = 12;     // MAX outer ARAP iterations in the final global polish (a convergence cap)
    int final_inner = 25;     // Gauss-Seidel sweeps per outer (more => more global propagation)
    f32 arap_tol = 0.15f;     // adaptive ARAP stop: end the outer loop early once the MEAN per-vertex
                              // move in an iteration drops below arap_tol*step voxels (diminishing
                              // returns), rather than always running `final_outer`. 0 = fixed count.
                              // ~0.15 measured strictly better than fixed-12 on paris4 (fewer folds &
                              // self-intersections, smoother, more coverage) — heavy ARAP over-folds.
    f32 ct_barrier = 0.0f;    // CT-valley growth BARRIER (prominence fraction; 0 = off). When >0 and a CT
                              // volume is present, reject a frontier cell whose step from a parent crosses
                              // a CT density saddle (an inter-wrap air gap, see ct_valley.hpp) — this stops
                              // growth drifting onto the ADJACENT wrap where the ML prediction fuses two
                              // touching wraps into one bright ridge (the snap could otherwise lock onto
                              // the neighbour's ridge in tightly-wound regions). ~0.12 matches the winding.
    int max_bridge = 2;       // weak-field bridging during growth: max consecutive weak-field cells
                              // the geometry may carry across before giving up (0 = off; reject on
                              // weak field). Default 2 (SMALL, per below) — the ARAP data-snap now has the
                              // CT-valley injectivity guard so a bridge can't drift to the adjacent wrap;
                              // measured +7% coverage (1.31M→1.40M cells) at fold/self-intersect 0 on
                              // paris4. Keep SMALL (2-3) so a bridge crosses a thin crack but not into the
                              // adjacent wrap (which would self-intersect). The smooth alternative to
                              // post-hoc river-filling — pairs with soft_gate+fit_every.
    bool soft_gate = false;   // when bridging, place weak-field cells by CONFIDENCE-blended geometry
                              // (pure extrapolation where there's no signal, partial snap where the
                              // ridge is present-but-weak) instead of a hard snap/reject. Decouples
                              // geometry from the data term (VC3D-style) so cracks are spanned, not cut.
    int river_radius = 2;     // POST-fill thin "river"/crack channels: morphological closing radius
                              // of the valid mask. Fills tributaries up to ~2*radius wide but leaves
                              // genuine wide voids ("the bay") open. 0 = off.
    std::vector<u8> uv_mask;  // optional grid*grid (row-major, id=v*grid+u) target SHAPE in the (u,v)
                              // flattened domain: 1 = growth allowed here, 0 = forbidden. Empty =
                              // unconstrained (grow until the sheet runs out). Growth fills mask ∩
                              // {where the sheet exists} and the final result is clipped to the mask.
    bool mask_gate = true;    // true: the mask GATES the frontier (growth can't cross gaps -> one
                              // connected patch). false: grow unconstrained, only CLIP to the mask at
                              // the end -> fills MULTIPLE disconnected mask components from the one
                              // continuous underlying sheet (e.g. mainland + islands of a country).
    // GLOBAL winding-coherence gate (type-erased so segment stays independent of the
    // winding module): winding_fn(p) returns the fitted model's STEPPED wrap index
    // (winding_at: constant along a wrap, ±1 jumps only at the theta branch cut).
    // The gate transports a per-cell RESIDUAL: res(child) = w + round(res(parents) - w) —
    // branch snapping absorbs the legit ±1 cut jumps (a sheet may spiral around many
    // turns), while a RADIAL ramp/hop accumulates a genuine ±1 that rounding cannot hide
    // (per-step drift < 0.5). Reject when |res - res(seed)| > winding_tol (fixed global
    // reference; no EMA — an EMA walks with the growth and leaked 1.97 windings) or the
    // pre-snap deviation from the parents exceeds winding_jump.
    std::function<f32(Vec3f)> winding_fn;
    f32 winding_tol = 0.0f;   // 0 = off; total residual budget vs the seed (~0.5 incl. model bias)
    f32 winding_jump = 0.4f;  // per-step |res - parent res| after branch snap (model noise bound)
};

namespace detail {
// Snap c onto the sheet ridge by maximizing the combined sheetness along +/- n; parabolic sub-voxel
// refine. Returns {position, NORMALIZED sheetness there (>=1 == on a sheet), signed offset traveled}.
template <class T>
inline std::tuple<Vec3f, f32, f32> snap_to_sheet(const DataField<T>& fld, Vec3f c, Vec3f n, f32 R) {
    const f32 dt = 0.5f;
    // argmax of `sv` along +/- n: a coarse step scan then a parabolic sub-voxel refine at the peak.
    auto scan = [&](auto&& sv) {
        f32 best = -1e30f, bestt = 0;
        for (f32 t = -R; t <= R + 1e-3f; t += dt) { const f32 v = sv(c + n * t); if (v > best) { best = v; bestt = t; } }
        return std::pair<f32, f32>{best, bestt};
    };
    auto refine = [&](auto&& sv, f32 best, f32 bestt) {
        const f32 vm = sv(c + n * (bestt - dt)), vp = sv(c + n * (bestt + dt));
        const f32 den = vm - 2.0f * best + vp;
        f32 sub = std::abs(den) > 1e-6f ? 0.5f * (vm - vp) / den : 0.0f;
        sub = std::clamp(sub, -1.0f, 1.0f);
        const f32 tt = bestt + sub * dt;
        const Vec3f pos = c + n * tt;
        return std::tuple<Vec3f, f32, f32>{pos, sv(pos), tt};
    };
    auto pred = [&](Vec3f q) { return fld.pred_value(q); };  // one volume
    auto comb = [&](Vec3f q) { return fld.value(q); };       // pred + CT (two volumes)
    if (!fld.has_ct()) { auto [b, bt] = scan(pred); return refine(pred, b, bt); }
    // Per-line CT short-circuit: scan the prediction alone; if its ridge clears ct_skip, the CT
    // fallback can only raise weaker spots, never beat that peak -> lock on the prediction and skip
    // the entire CT volume gather. Only a weak prediction (a possible crack/river) pays for the
    // second pass that brings CT in. This is where the data-term bandwidth actually drops.
    if (fld.ct_skip > 0.0f) {
        auto [pb, pbt] = scan(pred);
        if (pb >= fld.ct_skip) return refine(pred, pb, pbt);
    }
    auto [b, bt] = scan(comb);
    return refine(comb, b, bt);
}

// --- small 3x3 helpers for the ARAP local step (component index 0/1/2 == z/y/x) ---
inline f32 cmp(const Vec3f& v, int i) { return i == 0 ? v.z : (i == 1 ? v.y : v.x); }
inline f32 det3(const Mat3f& a) {
    return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) -
           a.m[0][1] * (a.m[1][0] * a.m[2][2] - a.m[1][2] * a.m[2][0]) +
           a.m[0][2] * (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]);
}
// Nearest rotation to C (polar factor) via C * (CᵀC)^(-1/2), with a reflection fix.
inline Mat3f polar_rot(const Mat3f& C) {
    Mat3f CtC{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) CtC.m[i][j] += C.m[k][i] * C.m[k][j];
    const auto e = sym_eig3<f32>(CtC.m[0][0], CtC.m[1][1], CtC.m[2][2], CtC.m[0][1], CtC.m[0][2], CtC.m[1][2]);
    f32 s[3] = {1, 1, 1};
    if (det3(C) < 0) s[2] = -1;  // flip smallest-singular-value direction
    Mat3f is{};
    for (int k = 0; k < 3; ++k) {
        const f32 c = s[k] / std::sqrt(std::max(e.values[k], 1e-8f));
        const Vec3f vk = e.vectors[k];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) is.m[i][j] += c * cmp(vk, i) * cmp(vk, j);
    }
    return C * is;
}

// Local/global ARAP surface fit: treat the (u,v) grid as a flat rest mesh (edge=step) deformed
// onto the sheet. LOCAL step: per-vertex best rotation of the rest 1-ring onto the current
// positions. GLOBAL step: in-place Gauss-Seidel solve placing each vertex to match its rotated
// rest edges to neighbours PLUS a data term pulling it onto the sheet (snap). The global coupling
// distributes distortion over the whole patch — this is what removes the radial fan that purely
// local growth/relaxation produces.
template <class T>
inline int arap_fit(Surface& S, const DataField<T>& fld, const NormalField& nf, const GrowParams& p,
                    int outer, int inner, f32 lambda, bool interior_only = false) {
    const int G = static_cast<int>(S.nu);
    const Extent3 D = fld.pred.dims();
    const f32 mgn = p.snap_radius + 2.0f;
    auto inb = [&](Vec3f c) {
        return c.z >= mgn && c.y >= mgn && c.x >= mgn && c.z < static_cast<f32>(D.z) - mgn &&
               c.y < static_cast<f32>(D.y) - mgn && c.x < static_cast<f32>(D.x) - mgn;
    };
    auto rest = [&](int u, int v) { return Vec3f{0.0f, static_cast<f32>(v) * p.step, static_cast<f32>(u) * p.step}; };
    const usize NG = static_cast<usize>(G) * static_cast<usize>(G);
    std::vector<Mat3f> R(NG, Mat3f::identity());
    std::vector<Vec3f> Tg(NG);
    std::vector<u8> hasT(NG, 0);
    const int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};

    // The grid is ~10% occupied; build a compact list of interior valid cells once (validity is
    // fixed across the fit — only coords move) and iterate THAT, never the full G*G. Split by
    // (u+v) parity for red-black Gauss-Seidel: the 4-neighbour stencil only couples opposite
    // colours, so each colour sub-sweep is race-free and parallelizes exactly.
    std::array<std::vector<s64>, 2> color;
    std::vector<s64> cells;
    for (int v = 1; v < G - 1; ++v)
        for (int u = 1; u < G - 1; ++u) {
            if (!S.is_valid(u, v)) continue;
            const s64 id = static_cast<s64>(v) * G + u;
            cells.push_back(id);
            color[static_cast<usize>((u + v) & 1)].push_back(id);
        }
    const s64 NC = static_cast<s64>(cells.size());

    auto sweep_color = [&](const std::vector<s64>& cl) {
        parallel_for(0, static_cast<s64>(cl.size()), [&](s64 ci) {
            const s64 id = cl[static_cast<usize>(ci)];
            const int u = static_cast<int>(id % G), v = static_cast<int>(id / G);
            const usize i = static_cast<usize>(id);
            const Vec3f Ri = rest(u, v);
            Vec3f acc{0, 0, 0};
            f32 w = 0;
            int nnb = 0;
            for (int k = 0; k < 4; ++k) {
                const int uu = u + du4[k], vv = v + dv4[k];
                if (!S.is_valid(uu, vv)) continue;
                ++nnb;
                const usize j = S.idx(uu, vv);
                const Vec3f d = Ri - rest(uu, vv);
                const Vec3f term = (R[i] * d + R[j] * d) * 0.5f;
                acc = acc + (S.coord[j] + term);
                w += 1.0f;
            }
            if (w == 0) return;
            if (interior_only && nnb < 4) return;  // leave the frontier free to grow
            if (hasT[i]) { acc = acc + Tg[i] * lambda; w += lambda; }
            S.coord[i] = acc / w;
        });
    };

    // Adaptive convergence: stop the outer loop once the surface stops moving (mean per-vertex move <
    // tol). The mesh settles geometrically, so a fixed `outer` either wastes iterations on easy patches
    // or over-folds them; iterating to diminishing returns is both faster and avoids the slight
    // over-fold that the extra fixed iterations cause. `outer` is then just the safety cap.
    const f32 tol = p.arap_tol * p.step;  // mean-move tolerance (voxels); <=0 -> run all `outer`
    std::vector<Vec3f> prev;
    if (tol > 0.0f) prev.resize(static_cast<usize>(NC));
    int done = 0;
    for (int o = 0; o < outer; ++o) {
        ++done;
        if (tol > 0.0f)
            for (s64 ci = 0; ci < NC; ++ci)
                prev[static_cast<usize>(ci)] = S.coord[static_cast<usize>(cells[static_cast<usize>(ci)])];
        // data targets: project each vertex onto the sheet (per-vertex independent -> parallel)
        parallel_for(0, NC, [&](s64 ci) {
            const s64 id = cells[static_cast<usize>(ci)];
            const usize i = static_cast<usize>(id);
            const Vec3f P = S.coord[i];
            auto [q, val, tt] = snap_to_sheet(fld, P, nf.at(P), p.snap_radius);
            // Injectivity guard: reject a data-snap that crosses a CT inter-wrap saddle (would pull this
            // vertex onto the NEIGHBOUR wrap). The growth loop guards this (crosses_valley / ct_barrier),
            // but the heavy final ARAP polish did NOT — so a strong data pull could drift a vertex across a
            // wrap, undoing the growth-time injectivity. Same CT-barrier the grower uses.
            const bool cross = p.ct_barrier > 0.0f && fld.has_ct() &&
                               segment::crosses_valley(fld.ct, fld.ct_coord(P), fld.ct_coord(q), p.ct_barrier, 0.5f);
            hasT[i] = (val >= 0.5f && std::abs(tt) < p.snap_radius && inb(q) && !cross) ? 1 : 0;
            Tg[i] = hasT[i] ? q : P;
        });
        // local rotations (polar of the rest->current 1-ring covariance) — per-vertex independent
        parallel_for(0, NC, [&](s64 ci) {
            const s64 id = cells[static_cast<usize>(ci)];
            const int u = static_cast<int>(id % G), v = static_cast<int>(id / G);
            Mat3f C{};
            int cnt = 0;
            const Vec3f Pi = S.coord[static_cast<usize>(id)], Ri = rest(u, v);
            for (int k = 0; k < 4; ++k) {
                const int uu = u + du4[k], vv = v + dv4[k];
                if (!S.is_valid(uu, vv)) continue;
                const Vec3f eP = Pi - S.at(uu, vv), eR = Ri - rest(uu, vv);
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) C.m[a][b] += cmp(eP, a) * cmp(eR, b);
                ++cnt;
            }
            R[static_cast<usize>(id)] = cnt >= 2 ? polar_rot(C) : Mat3f::identity();
        });
        // global red-black Gauss-Seidel sweeps
        for (int s = 0; s < inner; ++s) { sweep_color(color[0]); sweep_color(color[1]); }
        // diminishing-returns check: MEAN per-vertex displacement this iteration (run >=2 so the first
        // settle never short-circuits). Mean (not max) because a few boundary vertices oscillate
        // between snap targets forever — their max move never decays, but the bulk of the sheet settles
        // fast, and the mean captures that. O(NC) — ~1% of one inner sweep, so the early exit pays off.
        if (tol > 0.0f && o >= 1) {
            f64 sum = 0;
            for (s64 ci = 0; ci < NC; ++ci) {
                const usize i = static_cast<usize>(cells[static_cast<usize>(ci)]);
                sum += static_cast<f64>(norm(S.coord[i] - prev[static_cast<usize>(ci)]));
            }
            if (NC > 0 && sum / static_cast<f64>(NC) < static_cast<f64>(tol)) break;
        }
    }
    return done;
}

// Drop points off the sheet or attached by TEAR edges (long edges = the streaks). Tear-aware.
template <class T>
inline void cleanup_outliers(Surface& S, const DataField<T>& fld, const NormalField& nf, const GrowParams& p) {
    const int G = static_cast<int>(S.nu);
    const int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};
    for (int pass = 0; pass < 4; ++pass) {
        std::vector<usize> kill;
        for (int v = 0; v < G; ++v)
            for (int u = 0; u < G; ++u) {
                if (!S.is_valid(u, v)) continue;
                const Vec3f P = S.at(u, v);
                const f32 val = fld.value(P);
                int nbr = 0, bad = 0, shortn = 0;
                for (int k = 0; k < 4; ++k) {
                    const int uu = u + du4[k], vv = v + dv4[k];
                    if (uu < 0 || vv < 0 || uu >= G || vv >= G || !S.is_valid(uu, vv)) continue;
                    ++nbr;
                    const f32 d = norm(P - S.at(uu, vv));
                    if (d > 1.8f * p.step) ++bad;       // tear edge
                    if (d < 0.5f * p.step) ++shortn;    // collapsed edge -> sliver source
                }
                // Low field only kills a BOUNDARY cell (nbr<4): an interior weak-field cell is a
                // bridge across a crack — geometry holds it, so keep it (this is what stops the
                // pipeline re-punching the rivers that growth just bridged). Tear/collapse/isolated
                // kills are geometric and still apply everywhere.
                const bool interior = nbr == 4;
                // soft_gate intentionally carries weak-field bridge cells at the boundary too -> only
                // cull boundary cells with essentially NO signal; geometry/topology checks still apply.
                const f32 cull = p.soft_gate ? 0.3f : 0.8f;
                if ((val < cull && !interior) || bad >= 2 || shortn >= 1 || nbr <= 1) kill.push_back(S.idx(u, v));
            }
        if (kill.empty()) break;
        for (usize i : kill) S.valid[i] = 0;
    }
}

// Remove sliver triangles (bad aspect, edges OK) by dropping the apex vertex (the one nearest its
// opposite edge). These are mostly ragged-boundary triangles; iterated to convergence. Degenerate
// geometry is killed HERE (in the tracer) so downstream (flatten/codec/render) sees clean meshes.
inline void remove_slivers(Surface& S, const GrowParams& p) {
    const int G = static_cast<int>(S.nu);
    auto distline = [](Vec3f x, Vec3f a, Vec3f b) {
        const Vec3f ab = b - a, ax = x - a;
        const f32 L = norm(ab);
        return L > 1e-6f ? norm(cross(ax, ab)) / L : norm(ax);
    };
    auto aspect = [&](Vec3f A, Vec3f B, Vec3f C) {
        const f32 me = std::max({norm(B - A), norm(C - A), norm(C - B)});
        return me > 1e-6f ? (0.5f * norm(cross(B - A, C - A))) / (me * me) : 0.0f;
    };
    for (int pass = 0; pass < 4; ++pass) {
        std::vector<usize> kill;
        auto consider = [&](int au, int av, int bu, int bv, int cu, int cv) {
            const Vec3f A = S.at(au, av), B = S.at(bu, bv), C = S.at(cu, cv);
            if (aspect(A, B, C) >= 0.08f) return;
            const f32 da = distline(A, B, C), db = distline(B, A, C), dc = distline(C, A, B);
            if (da <= db && da <= dc) kill.push_back(S.idx(au, av));
            else if (db <= dc) kill.push_back(S.idx(bu, bv));
            else kill.push_back(S.idx(cu, cv));
        };
        for (int v = 0; v + 1 < S.nv; ++v)
            for (int u = 0; u + 1 < G; ++u) {
                const bool a = S.is_valid(u, v), b = S.is_valid(u + 1, v), c = S.is_valid(u, v + 1), d = S.is_valid(u + 1, v + 1);
                if (a && b && c) consider(u, v, u + 1, v, u, v + 1);
                if (b && d && c) consider(u + 1, v, u + 1, v + 1, u, v + 1);
            }
        if (kill.empty()) break;
        for (usize i : kill) S.valid[i] = 0;
    }
}

// Keep only connected components (4-conn) of >= min_size valid cells; drop the rest. Removes the
// dust islands that fragment a patch (villa/thaumato keep-largest-component, generalized).
inline void keep_large_components(Surface& S, s64 min_size) {
    const int G = static_cast<int>(S.nu), Hh = static_cast<int>(S.nv);
    std::vector<s64> par(static_cast<usize>(G) * static_cast<usize>(Hh));
    for (usize i = 0; i < par.size(); ++i) par[i] = static_cast<s64>(i);
    std::function<s64(s64)> find = [&](s64 x) { while (par[static_cast<usize>(x)] != x) { par[static_cast<usize>(x)] = par[static_cast<usize>(par[static_cast<usize>(x)])]; x = par[static_cast<usize>(x)]; } return x; };
    auto uni = [&](s64 a, s64 b) { par[static_cast<usize>(find(a))] = find(b); };
    for (int v = 0; v < Hh; ++v)
        for (int u = 0; u < G; ++u) {
            if (!S.is_valid(u, v)) continue;
            if (u + 1 < G && S.is_valid(u + 1, v)) uni(static_cast<s64>(v) * G + u, static_cast<s64>(v) * G + u + 1);
            if (v + 1 < Hh && S.is_valid(u, v + 1)) uni(static_cast<s64>(v) * G + u, static_cast<s64>(v + 1) * G + u);
        }
    std::unordered_map<s64, s64> sz;
    for (int v = 0; v < Hh; ++v) for (int u = 0; u < G; ++u) if (S.is_valid(u, v)) sz[find(static_cast<s64>(v) * G + u)]++;
    for (int v = 0; v < Hh; ++v)
        for (int u = 0; u < G; ++u)
            if (S.is_valid(u, v) && sz[find(static_cast<s64>(v) * G + u)] < min_size) S.valid[S.idx(u, v)] = 0;
}

// Fill SMALL enclosed holes (<= max_hole cells) by bilinear interpolation of position + snap. Only
// small holes are closed — large enclosed regions are real damage/concavities and bridging them
// would create huge tears/distortion (so they're left as boundary). This removes the thousands of
// 1-2 cell holes that make a patch look like swiss cheese, without touching genuine voids.
template <class T>
inline void fill_holes(Surface& S, const DataField<T>& fld, const NormalField& nf, const GrowParams& p, s64 max_hole = 25) {
    const int G = static_cast<int>(S.nu), Hh = static_cast<int>(S.nv);
    const int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};
    std::vector<u8> outside(static_cast<usize>(G) * static_cast<usize>(Hh), 0);
    std::vector<s64> stk;
    auto push = [&](int u, int v) { if (u >= 0 && v >= 0 && u < G && v < Hh && !S.is_valid(u, v) && !outside[static_cast<usize>(v) * G + u]) { outside[static_cast<usize>(v) * G + u] = 1; stk.push_back(static_cast<s64>(v) * G + u); } };
    for (int u = 0; u < G; ++u) { push(u, 0); push(u, Hh - 1); }
    for (int v = 0; v < Hh; ++v) { push(0, v); push(G - 1, v); }
    while (!stk.empty()) { const s64 i = stk.back(); stk.pop_back(); const int u = static_cast<int>(i % G), v = static_cast<int>(i / G); for (int k = 0; k < 4; ++k) push(u + du4[k], v + dv4[k]); }

    std::vector<u8> seen(static_cast<usize>(G) * static_cast<usize>(Hh), 0);
    for (int v = 1; v < Hh - 1; ++v)
        for (int u = 1; u < G - 1; ++u) {
            const usize id = static_cast<usize>(v) * G + u;
            if (S.is_valid(u, v) || outside[id] || seen[id]) continue;
            // BFS this enclosed hole component
            std::vector<s64> comp{static_cast<s64>(id)};
            seen[id] = 1;
            for (usize h = 0; h < comp.size(); ++h) {
                const int cu = static_cast<int>(comp[h] % G), cv = static_cast<int>(comp[h] / G);
                for (int k = 0; k < 4; ++k) {
                    const int uu = cu + du4[k], vv = cv + dv4[k];
                    if (uu < 1 || vv < 1 || uu >= G - 1 || vv >= Hh - 1) continue;
                    const usize nid = static_cast<usize>(vv) * G + uu;
                    if (!S.is_valid(uu, vv) && !outside[nid] && !seen[nid]) { seen[nid] = 1; comp.push_back(static_cast<s64>(nid)); }
                }
            }
            if (static_cast<s64>(comp.size()) > max_hole) continue;  // genuine void -> leave it
            for (s64 ci : comp) S.coord[static_cast<usize>(ci)] = Vec3f{0, 0, 0};
            for (int it = 0; it < static_cast<int>(comp.size()) + 4; ++it)  // Laplacian fill (small -> cheap)
                for (s64 ci : comp) {
                    const int cu = static_cast<int>(ci % G), cv = static_cast<int>(ci / G);
                    Vec3f sum{0, 0, 0}; int c = 0;
                    for (int k = 0; k < 4; ++k) { const int uu = cu + du4[k], vv = cv + dv4[k]; if (S.is_valid(uu, vv) || (uu >= 0 && vv >= 0 && uu < G && vv < Hh && norm(S.coord[static_cast<usize>(vv) * G + uu]) > 0)) { sum = sum + S.coord[static_cast<usize>(vv) * G + uu]; ++c; } }
                    if (c) S.coord[static_cast<usize>(ci)] = sum / static_cast<f32>(c);
                }
            for (s64 ci : comp) {
                if (norm(S.coord[static_cast<usize>(ci)]) == 0) continue;
                const Vec3f cpos = S.coord[static_cast<usize>(ci)];
                auto [qq, val, tt] = snap_to_sheet(fld, cpos, nf.at(cpos), p.snap_radius);
                S.coord[static_cast<usize>(ci)] = (val >= 1.0f && std::abs(tt) < p.snap_radius) ? qq : cpos;
                S.valid[static_cast<usize>(ci)] = 1;
            }
        }
}

// Fill thin "river"/crack channels that cut INTO the sheet from its boundary (not enclosed, so
// fill_holes leaves them). A morphological CLOSING of the valid mask by `radius` fills concavities
// up to ~2*radius wide — i.e. the tributaries — while genuine wide voids ("the bay") stay open.
// The filled cells get a Laplacian interpolation between their two healthy banks (a smooth,
// non-folding minimal bridge — the banks are the boundary conditions) then a light snap. This is
// the controllable, global-by-construction analog of greedy weak-field bridging (which folds).
template <class T>
inline void fill_rivers(Surface& S, const DataField<T>& fld, const NormalField& nf, const GrowParams& p, int radius) {
    if (radius <= 0) return;
    const int G = static_cast<int>(S.nu), Hh = static_cast<int>(S.nv);
    const usize NG = static_cast<usize>(G) * static_cast<usize>(Hh);
    // separable box morphology of the binary valid mask (dilate=OR, erode=AND over a 1D window)
    auto morph = [&](const std::vector<u8>& in, std::vector<u8>& out, bool dilate) {
        std::vector<u8> tmp(NG);
        for (int v = 0; v < Hh; ++v)
            for (int u = 0; u < G; ++u) {
                int acc = dilate ? 0 : 1;
                for (int du = -radius; du <= radius; ++du) {
                    const int uu = u + du;
                    const u8 val = (uu < 0 || uu >= G) ? 0 : in[static_cast<usize>(v) * G + uu];
                    acc = dilate ? (acc | val) : (acc & val);
                }
                tmp[static_cast<usize>(v) * G + u] = static_cast<u8>(acc);
            }
        for (int v = 0; v < Hh; ++v)
            for (int u = 0; u < G; ++u) {
                int acc = dilate ? 0 : 1;
                for (int dv = -radius; dv <= radius; ++dv) {
                    const int vv = v + dv;
                    const u8 val = (vv < 0 || vv >= Hh) ? 0 : tmp[static_cast<usize>(vv) * G + u];
                    acc = dilate ? (acc | val) : (acc & val);
                }
                out[static_cast<usize>(v) * G + u] = static_cast<u8>(acc);
            }
    };
    std::vector<u8> m(NG), dil(NG), closed(NG);
    for (usize i = 0; i < NG; ++i) m[i] = S.valid[i];
    morph(m, dil, true);
    morph(dil, closed, false);
    std::vector<s64> riv;
    for (int v = 1; v < Hh - 1; ++v)
        for (int u = 1; u < G - 1; ++u) {
            const usize id = static_cast<usize>(v) * G + u;
            if (!S.valid[id] && closed[id]) riv.push_back(static_cast<s64>(id));
        }
    if (riv.empty()) return;
    const int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};
    for (s64 ci : riv) S.coord[static_cast<usize>(ci)] = Vec3f{0, 0, 0};
    for (int it = 0; it < radius * 6 + 8; ++it)  // Laplacian to convergence across the channel width
        for (s64 ci : riv) {
            const int cu = static_cast<int>(ci % G), cv = static_cast<int>(ci / G);
            Vec3f sum{0, 0, 0}; int c = 0;
            for (int k = 0; k < 4; ++k) {
                const int uu = cu + du4[k], vv = cv + dv4[k];
                if (uu < 0 || vv < 0 || uu >= G || vv >= Hh) continue;
                const usize nid = static_cast<usize>(vv) * G + uu;
                if (S.valid[nid] || norm(S.coord[nid]) > 0) { sum = sum + S.coord[nid]; ++c; }
            }
            if (c) S.coord[static_cast<usize>(ci)] = sum / static_cast<f32>(c);
        }
    for (s64 ci : riv) {
        if (norm(S.coord[static_cast<usize>(ci)]) == 0) continue;
        // Keep the smooth Laplacian bridge between banks; only a GENTLE snap (half radius) so a
        // bridge that happens to pass near real sheet locks on, but a crack bridge is NOT yanked to
        // an offset wall (that was the tear source). The final ARAP smooths these as geometry-only.
        const Vec3f cpos = S.coord[static_cast<usize>(ci)];
        auto [qq, val, tt] = snap_to_sheet(fld, cpos, nf.at(cpos), p.snap_radius * 0.5f);
        S.coord[static_cast<usize>(ci)] = (val >= 1.0f && std::abs(tt) < p.snap_radius * 0.5f) ? qq : cpos;
    }
    // Validate a bridge cell ONLY if its edges to existing (snapped) neighbours stay near step-length.
    // A low-stretch bridge = the sheet was continuous there (a prediction-artifact crack) -> fill it.
    // A high-stretch bridge = the grid channel spans a real 3D gap (sheet physically separated) ->
    // leave it open (bridging would be fake surface + the smearing). This is the artifact-vs-real-gap
    // discriminator; the genuine fix for artifact cracks is the CT-ridge data term (next stage).
    for (s64 ci : riv) {
        const usize id = static_cast<usize>(ci);
        if (norm(S.coord[id]) == 0) continue;
        const int cu = static_cast<int>(ci % G), cv = static_cast<int>(ci / G);
        f32 maxe = 0; int nb = 0;
        for (int k = 0; k < 4; ++k) {
            const int uu = cu + du4[k], vv = cv + dv4[k];
            if (uu < 0 || vv < 0 || uu >= G || vv >= Hh) continue;
            const usize nid = static_cast<usize>(vv) * G + uu;
            if (S.valid[nid] || norm(S.coord[nid]) > 0) { maxe = std::max(maxe, norm(S.coord[id] - S.coord[nid])); ++nb; }
        }
        if (nb >= 1 && maxe <= 1.6f * p.step) S.valid[id] = 1;  // low-stretch -> real crack, fill
    }
}

// Persist the per-cell across-sheet normal + data confidence as Surface channels (the inputs the
// patch-graph / winding fit need). The normal comes from the local (u,v) tangent frame (central
// difference of the placed coords — more accurate than the coarse NormalField), falling back to the
// coarse field at the boundary; confidence is the combined data term at the final position. Done once
// at the end of a grow (validity/coords are settled), so it reflects the cleaned surface.
template <class T>
inline void fill_surface_channels(Surface& S, const DataField<T>& fld, const NormalField& nf) {
    const int G = static_cast<int>(S.nu);
    S.alloc_channels();
    parallel_for_z(Extent3{S.nv, 1, 1}, [&](s64 v) {
        for (int u = 0; u < G; ++u) {
            const usize id = S.idx(u, v);
            if (!S.valid[id]) continue;
            const Vec3f P = S.coord[id];
            Vec3f n{0, 0, 0};
            const bool cu = u > 0 && u + 1 < G && S.is_valid(u - 1, v) && S.is_valid(u + 1, v);
            const bool cv = v > 0 && v + 1 < S.nv && S.is_valid(u, static_cast<int>(v) - 1) && S.is_valid(u, static_cast<int>(v) + 1);
            if (cu && cv) n = cross(S.at(u + 1, v) - S.at(u - 1, v), S.at(u, v + 1) - S.at(u, v - 1));
            if (norm(n) < 1e-6f) n = nf.at(P);
            S.normal[id] = normalized(n);
            S.conf[id] = fld.value(P);
        }
    });
}
}  // namespace detail

// Geometric quality of a traced Surface — the acceptance gates: consistent spacing, no folds,
// and injectivity (no self-intersection / wrap-crossing). All "want ~0" except coverage (~1).
struct SurfQuality {
    s64 valid = 0;
    f64 spacing_cv = 0, spacing_oor = 0;  // edge-length coeff-of-variation; frac outside [.5,1.5]xmedian
    f64 min_edge = 0, frac_short = 0;      // shortest edge; frac of edges < 0.5xmedian (collapses)
    f64 degen_tri = 0;                     // frac of grid triangles with aspect (area/maxedge^2) < 0.08
    f64 fold_rate = 0;                     // frac of adjacent vertex-normals that flip (local folds)
    f64 normal_smooth = 0;                 // mean dihedral angle between adjacent vertex normals (deg)
    f64 boundary_frac = 0;                 // frac of valid cells on the boundary (<4 valid neighbours)
    f64 bad_angle = 0;                     // frac of triangles with a min angle < 20 deg
    f64 overlap = 0, distant_fold = 0;     // frac of points sharing a 3D bin; ... with a uv-distant cell
    f64 coverage = 0;                      // unique bins / points (1 = injective)
};
inline SurfQuality surface_quality(const Surface& S, f32 bin_size, int fold_thresh) {
    const s64 G = S.nu;
    SurfQuality q;
    q.valid = S.valid_count();
    // spacing
    std::vector<f32> E;
    E.reserve(static_cast<usize>(q.valid * 2));
    for (s64 v = 0; v < S.nv; ++v)
        for (s64 u = 0; u < G; ++u) {
            if (!S.is_valid(u, v)) continue;
            if (u + 1 < G && S.is_valid(u + 1, v)) E.push_back(norm(S.at(u + 1, v) - S.at(u, v)));
            if (v + 1 < S.nv && S.is_valid(u, v + 1)) E.push_back(norm(S.at(u, v + 1) - S.at(u, v)));
        }
    f32 med = 0;
    if (!E.empty()) {
        f64 m = 0; for (f32 e : E) m += e; m /= static_cast<f64>(E.size());
        f64 s2 = 0; for (f32 e : E) s2 += (e - m) * (e - m); s2 /= static_cast<f64>(E.size());
        std::vector<f32> tmp = E; std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
        med = tmp[tmp.size() / 2];
        q.spacing_cv = std::sqrt(s2) / (m + 1e-9);
        s64 oor = 0, sh = 0; q.min_edge = 1e30;
        for (f32 e : E) { if (e < 0.5f * med || e > 1.5f * med) ++oor; if (e < 0.5f * med) ++sh; q.min_edge = std::min(q.min_edge, static_cast<f64>(e)); }
        q.spacing_oor = static_cast<f64>(oor) / static_cast<f64>(E.size());
        q.frac_short = static_cast<f64>(sh) / static_cast<f64>(E.size());
    }
    // degenerate-triangle fraction (sliver aspect): the geometry the flattener can't tolerate
    {
        s64 nt = 0, bad = 0, ba = 0;
        auto aspect = [&](Vec3f A, Vec3f B, Vec3f C) {
            const f64 ar = 0.5 * norm(cross(B - A, C - A));
            const f64 me = std::max({norm(B - A), norm(C - A), norm(C - B)});
            return me > 1e-6 ? ar / (me * me) : 0.0;
        };
        const f64 cos20 = 0.9397;
        auto small_angle = [&](Vec3f A, Vec3f B, Vec3f C) {  // any triangle angle < 20 deg?
            auto cosat = [](Vec3f P, Vec3f Q, Vec3f R) { const Vec3f a = Q - P, b = R - P; const f64 d = norm(a) * norm(b); return d > 1e-9 ? dot(a, b) / d : 1.0; };
            return std::max({cosat(A, B, C), cosat(B, A, C), cosat(C, A, B)}) > cos20;
        };
        for (s64 v = 0; v + 1 < S.nv; ++v)
            for (s64 u = 0; u + 1 < G; ++u) {
                const bool a = S.is_valid(u, v), b = S.is_valid(u + 1, v), c = S.is_valid(u, v + 1), d = S.is_valid(u + 1, v + 1);
                if (a && b && c) { ++nt; const Vec3f A = S.at(u, v), B = S.at(u + 1, v), C = S.at(u, v + 1); if (aspect(A, B, C) < 0.08) ++bad; if (small_angle(A, B, C)) ++ba; }
                if (b && d && c) { ++nt; const Vec3f A = S.at(u + 1, v), B = S.at(u + 1, v + 1), C = S.at(u, v + 1); if (aspect(A, B, C) < 0.08) ++bad; if (small_angle(A, B, C)) ++ba; }
            }
        q.degen_tri = nt ? static_cast<f64>(bad) / static_cast<f64>(nt) : 0;
        q.bad_angle = nt ? static_cast<f64>(ba) / static_cast<f64>(nt) : 0;
    }
    // local fold rate via central-difference vertex normals
    auto nrm_at = [&](s64 u, s64 v) -> Vec3f {
        const Vec3f tu = S.at(u + 1, v) - S.at(u - 1, v), tv = S.at(u, v + 1) - S.at(u, v - 1);
        return normalized(cross(tu, tv));
    };
    auto core = [&](s64 u, s64 v) {
        return u > 0 && v > 0 && u + 1 < G && v + 1 < S.nv && S.is_valid(u, v) && S.is_valid(u - 1, v) &&
               S.is_valid(u + 1, v) && S.is_valid(u, v - 1) && S.is_valid(u, v + 1);
    };
    s64 flips = 0, ftot = 0;
    f64 dih = 0;
    for (s64 v = 1; v + 1 < S.nv; ++v)
        for (s64 u = 1; u + 1 < G; ++u) {
            if (!core(u, v)) continue;
            const Vec3f n = nrm_at(u, v);
            auto pair = [&](s64 uu, s64 vv) { if (!core(uu, vv)) return; ++ftot; const f64 d = std::clamp(static_cast<f64>(dot(n, nrm_at(uu, vv))), -1.0, 1.0); if (d < 0) ++flips; dih += std::acos(d); };
            pair(u + 1, v); pair(u, v + 1);
        }
    q.fold_rate = ftot ? static_cast<f64>(flips) / static_cast<f64>(ftot) : 0;
    q.normal_smooth = ftot ? dih / static_cast<f64>(ftot) * 180.0 / 3.14159265 : 0;
    // boundary fraction: valid cells with < 4 valid 4-neighbours
    {
        s64 nb = 0, val = 0;
        for (s64 v = 0; v < S.nv; ++v)
            for (s64 u = 0; u < G; ++u) {
                if (!S.is_valid(u, v)) continue;
                ++val;
                int c = (u > 0 && S.is_valid(u - 1, v)) + (u + 1 < G && S.is_valid(u + 1, v)) + (v > 0 && S.is_valid(u, v - 1)) + (v + 1 < S.nv && S.is_valid(u, v + 1));
                if (c < 4) ++nb;
            }
        q.boundary_frac = val ? static_cast<f64>(nb) / static_cast<f64>(val) : 0;
    }
    // injectivity: 3D bin sharing
    std::unordered_map<s64, s64> occ;
    occ.reserve(static_cast<usize>(q.valid));
    const f32 ib = 1.0f / bin_size;
    const s64 BS = 1 << 20;
    s64 ov = 0, dfold = 0, total = 0;
    for (s64 v = 0; v < S.nv; ++v)
        for (s64 u = 0; u < G; ++u) {
            if (!S.is_valid(u, v)) continue;
            ++total;
            const Vec3f c = S.at(u, v);
            const s64 key = (static_cast<s64>(c.z * ib)) * BS * BS + (static_cast<s64>(c.y * ib)) * BS + static_cast<s64>(c.x * ib);
            auto it = occ.find(key);
            if (it == occ.end()) { occ[key] = static_cast<s64>(v) * G + u; continue; }
            ++ov;
            const int ou = static_cast<int>(it->second % G), ovv = static_cast<int>(it->second / G);
            if (std::abs(static_cast<int>(u) - ou) + std::abs(static_cast<int>(v) - ovv) > fold_thresh) ++dfold;
        }
    if (total) { q.overlap = static_cast<f64>(ov) / static_cast<f64>(total); q.distant_fold = static_cast<f64>(dfold) / static_cast<f64>(total); }
    q.coverage = total ? static_cast<f64>(occ.size()) / static_cast<f64>(total) : 0;
    return q;
}

// Grow a sheet from `seed` (ZYX voxel coords) across the prediction field `f` (high on the sheet),
// optionally fused with the raw CT `ct` (the sheet's density ridge) so growth survives prediction
// drop-outs at cracks. If `shared_occ` is given, the 3D occupancy is shared across sheets (with this
// `sheet_id`): a cell landing in a bin owned by ANOTHER sheet is rejected, so multiple sheets tile
// the volume without overlapping (full-volume tracing). Without it, a local map enforces injectivity.
// Final winding-coherence FILTER: BFS from the seed cell transporting the branch-snapped
// residual across the finished grid, invalidating cells beyond tol. The growth-loop gate
// cannot see cells created by the POST passes (fill_holes/fill_rivers/ARAP filled an
// enclosed gate-rejected ramp region and leaked a wrap — measured 1611/4299 cells), so
// the same rule runs once over the final surface, catching every creation path.
inline void winding_filter(Surface& S, const std::function<f32(Vec3f)>& fn, f32 tol, f32 jump, int cu, int cv) {
    if (!fn || tol <= 0 || !S.is_valid(cu, cv)) return;
    const s64 G = S.nu;
    std::vector<f32> res(static_cast<usize>(G) * static_cast<usize>(S.nv), 1e30f);
    std::vector<s64> q;
    const f32 ref = fn(S.at(cu, cv));
    if (!(std::abs(ref) < 1e30f)) return;
    res[S.idx(cu, cv)] = ref;
    q.push_back(static_cast<s64>(cv) * G + cu);
    const int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};
    while (!q.empty()) {
        const s64 id = q.back();
        q.pop_back();
        const int u = static_cast<int>(id % G), v = static_cast<int>(id / G);
        const f32 pr = res[static_cast<usize>(id)];
        for (int k = 0; k < 4; ++k) {
            const int uu = u + du4[k], vv = v + dv4[k];
            if (uu < 0 || vv < 0 || uu >= G || vv >= S.nv) continue;
            const usize j = S.idx(uu, vv);
            if (!S.valid[j] || res[j] < 1e29f) continue;
            const f32 w = fn(S.coord[j]);
            if (!(std::abs(w) < 1e30f)) { S.valid[j] = 0; continue; }
            const f32 r = w + std::round(pr - w);
            if (std::abs(r - pr) > jump || std::abs(r - ref) > tol) continue;  // unreachable within rule
            res[j] = r;
            q.push_back(static_cast<s64>(vv) * G + uu);
        }
    }
    // anything the coherent BFS could not reach is off-wrap or disconnected -> drop
    for (usize j = 0; j < res.size(); ++j)
        if (S.valid[j] && res[j] > 1e29f) S.valid[j] = 0;
}

template <class T>
inline Surface grow_surface(VolumeView<const T> f, VolumeView<const T> ct, const NormalField& nf, Vec3f seed,
                            GrowParams p, OccMap* shared_occ = nullptr, s64 sheet_id = 0) {
    const int G = p.grid, C = G / 2;
    const Extent3 D = f.dims();
    const DataField<T> fld{f, ct, p.surf_thresh, p.ct_thresh, p.ct_weight, p.ct_ds, p.ct_skip};
    const f32 mgn = p.snap_radius + 2.0f;
    auto inb = [&](Vec3f c) {
        return c.z >= mgn && c.y >= mgn && c.x >= mgn && c.z < static_cast<f32>(D.z) - mgn &&
               c.y < static_cast<f32>(D.y) - mgn && c.x < static_cast<f32>(D.x) - mgn;
    };
    Surface S(G, G);
    const usize NG = static_cast<usize>(G) * static_cast<usize>(G);
    std::vector<u8> dead(NG, 0);
    std::vector<Vec3f> Tu(NG), Tv(NG);  // per-vertex tangent frame (parallel-transported)
    auto N = [&](Vec3f c) { return nf.at(c); };
    auto snap = [&](Vec3f c, Vec3f n) { return detail::snap_to_sheet(fld, c, n, p.snap_radius); };
    // Re-express frame (tu,tv) in the tangent plane of normal n (Gram-Schmidt) — parallel transport.
    auto transport = [&](Vec3f tu, Vec3f tv, Vec3f n) -> std::pair<Vec3f, Vec3f> {
        Vec3f u = tu - n * dot(tu, n);
        if (norm(u) < 1e-4f) { const Vec3f e = (std::abs(n.x) < 0.9f) ? Vec3f{0, 0, 1} : Vec3f{1, 0, 0}; u = e - n * dot(e, n); }
        u = normalized(u);
        Vec3f v = tv - n * dot(tv, n);
        v = v - u * dot(v, u);
        if (norm(v) < 1e-4f) v = cross(n, u);
        return {u, normalized(v)};
    };
    auto V = [&](int u, int v) { return S.is_valid(u, v); };

    // --- seed patch (3x3), frame parallel-transported from a projected global frame ---
    const Vec3f n0 = N(seed);
    auto [sp, sv0, st0] = snap(seed, n0);
    (void)sv0; (void)st0;
    auto [su, sv] = transport(Vec3f{0, 0, 1}, Vec3f{0, 1, 0}, n0);
    S.set(C, C, sp);
    Tu[S.idx(C, C)] = su; Tv[S.idx(C, C)] = sv;
    for (int dv = -1; dv <= 1; ++dv)
        for (int du = -1; du <= 1; ++du) {
            if (du == 0 && dv == 0) continue;
            const Vec3f c = sp + su * (static_cast<f32>(du) * p.step) + sv * (static_cast<f32>(dv) * p.step);
            if (!inb(c)) continue;
            auto [q, vv, tt] = snap(c, N(c));
            (void)tt;
            if (vv >= 1.0f && inb(q)) {
                S.set(C + du, C + dv, q);
                auto [fu, fv] = transport(su, sv, N(q));
                Tu[S.idx(C + du, C + dv)] = fu; Tv[S.idx(C + du, C + dv)] = fv;
            }
        }

    // 3D occupancy for the injectivity guard: bin -> owning cell index. A new point whose bin
    // (or 26-neighbourhood) is already owned by a uv-DISTANT cell is a self-intersection -> reject.
    OccMap local_occ;
    OccMap& occ = shared_occ ? *shared_occ : local_occ;
    if (!shared_occ) occ.reserve(static_cast<s64>(NG) / 4);
    const f32 ibin = 1.0f / p.bin_size;
    const s64 BS = 1 << 20;
    const s64 SHIFT = static_cast<s64>(1) << 32;  // pack (sheet_id, cell) into one value
    auto binkey = [&](Vec3f c, int oz, int oy, int ox) {
        return (static_cast<s64>(c.z * ibin) + oz) * BS * BS + (static_cast<s64>(c.y * ibin) + oy) * BS +
               (static_cast<s64>(c.x * ibin) + ox);
    };
    auto fold_conflict = [&](Vec3f q, int u, int v) {
        for (int oz = -1; oz <= 1; ++oz)
            for (int oy = -1; oy <= 1; ++oy)
                for (int ox = -1; ox <= 1; ++ox) {
                    const s64* it = occ.get(binkey(q, oz, oy, ox));
                    if (!it) continue;
                    const s64 osheet = *it / SHIFT, cell = *it % SHIFT;
                    if (osheet != sheet_id) return true;  // owned by another sheet -> don't overlap
                    const int ou = static_cast<int>(cell % G), ov = static_cast<int>(cell / G);
                    if (std::abs(u - ou) + std::abs(v - ov) > p.fold_thresh) return true;
                }
        return false;
    };
    auto claim = [&](Vec3f q, int u, int v) { occ.set(binkey(q, 0, 0, 0), sheet_id * SHIFT + (static_cast<s64>(v) * G + u)); };
    for (int dv = -1; dv <= 1; ++dv)
        for (int du = -1; du <= 1; ++du)
            if (V(C + du, C + dv)) claim(S.at(C + du, C + dv), C + du, C + dv);

    // winding-coherence: per-cell transported RESIDUAL (branch-snapped wrap index) +
    // the seed's residual as the fixed global reference
    f32 wind_ref = 0;
    std::vector<f32> wcell;  // residual memory per uv cell
    if (p.winding_tol > 0 && p.winding_fn) {
        wcell.assign(NG, 1e30f);
        if (V(C, C)) wind_ref = p.winding_fn(S.at(C, C));
        for (int dv = -1; dv <= 1; ++dv)
            for (int du = -1; du <= 1; ++du)
                if (V(C + du, C + dv)) {
                    const f32 w = p.winding_fn(S.at(C + du, C + dv));
                    wcell[S.idx(C + du, C + dv)] = w + std::round(wind_ref - w);
                }
    }

    const int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};
    // Frontier-queue growth: only touch cells adjacent to the live boundary instead of rescanning
    // the whole (mostly-empty) G*G grid every generation. Each placed cell queues its empty
    // neighbours for the next wave; every queued cell is resolved (placed or marked dead) exactly
    // once — so the BFS waves reproduce the old "generations" while turning O(G^2 * gens) into
    // O(cells touched). The per-wave sort keeps row-major order (matches the old visit order).
    std::vector<u8> queued(NG, 0);
    std::vector<u8> bdepth(NG, 0);  // weak-field "bridge" distance from the nearest snapped anchor (0 = snapped)
    std::vector<s64> frontier, nextf;
    const bool gate_mask = !p.uv_mask.empty() && p.mask_gate;
    auto enqueue = [&](int u, int v) {
        if (u < 1 || v < 1 || u >= G - 1 || v >= G - 1) return;
        const s64 id = static_cast<s64>(v) * G + u;
        if (gate_mask && !p.uv_mask[static_cast<usize>(id)]) return;  // outside the target shape -> stop
        if (S.valid[static_cast<usize>(id)] || dead[static_cast<usize>(id)] || queued[static_cast<usize>(id)]) return;
        queued[static_cast<usize>(id)] = 1;
        nextf.push_back(id);
    };
    for (int dv = -1; dv <= 1; ++dv)
        for (int du = -1; du <= 1; ++du)
            if (V(C + du, C + dv))
                for (int k = 0; k < 4; ++k) enqueue(C + du + du4[k], C + dv + dv4[k]);
    frontier.swap(nextf);

    for (int gen = 0; gen < p.max_gen && !frontier.empty(); ++gen) {
        nextf.clear();
        std::sort(frontier.begin(), frontier.end());
        int placed = 0;
        for (const s64 id : frontier) {
            const int u = static_cast<int>(id % G), v = static_cast<int>(id / G);
            // Predict from each valid neighbour by stepping along ITS transported frame toward
            // (u,v); average. Consistent frames -> no fan; single-neighbour OK -> no stall.
            Vec3f pred{0, 0, 0}, accu{0, 0, 0}, accv{0, 0, 0};
            int np = 0;
            for (int k = 0; k < 4; ++k) {
                const int nu = u + du4[k], nv = v + dv4[k];
                if (!V(nu, nv)) continue;
                const usize ni = S.idx(nu, nv);
                const f32 dU = static_cast<f32>(u - nu), dV = static_cast<f32>(v - nv);
                pred = pred + (S.at(nu, nv) + (Tu[ni] * dU + Tv[ni] * dV) * p.step);
                accu = accu + Tu[ni]; accv = accv + Tv[ni];
                ++np;
            }
            if (np == 0) continue;
            const Vec3f c = pred / static_cast<f32>(np);
            if (!inb(c)) { dead[static_cast<usize>(id)] = 1; continue; }
            const Vec3f nrm = N(c);
            auto [q, val, tt] = snap(c, nrm);
            // Decouple geometry from data: a point that locks onto the sheet (strong field, in-range
            // snap) is "snapped"; otherwise the GEOMETRY carries the surface across the weak spot at
            // the extrapolated position `c` (a "bridge"), bounded by max_bridge so thin cracks/rivers
            // get crossed but genuine wide voids run out of bridge budget and stay open boundaries.
            const bool inq = inb(q);
            const bool snapped = inq && val >= 1.0f && std::abs(tt) <= p.snap_radius - 0.51f;
            u8 bd = 0;
            Vec3f place = q;
            if (!snapped) {
                if (p.max_bridge <= 0) { dead[static_cast<usize>(id)] = 1; continue; }
                int mind = 255;
                for (int k = 0; k < 4; ++k) {
                    const int uu = u + du4[k], vv = v + dv4[k];
                    if (V(uu, vv)) mind = std::min(mind, static_cast<int>(bdepth[S.idx(uu, vv)]));
                }
                bd = static_cast<u8>(std::min(254, mind + 1));
                if (bd > p.max_bridge) { dead[static_cast<usize>(id)] = 1; continue; }  // out of bridge budget -> honest void
                // soft_gate: blend extrapolated geometry `c` with the in-range ridge `q` by confidence
                // (0 below val 0.5 -> pure geometry; 1 at val>=1.0 -> full snap). Off -> pure geometry.
                const f32 w = (p.soft_gate && inq) ? std::clamp((val - 0.5f) * 2.0f, 0.0f, 1.0f) : 0.0f;
                place = c + (q - c) * w;
            }
            if (!inb(place)) { dead[static_cast<usize>(id)] = 1; continue; }
            bool ok = true;
            for (int k = 0; k < 4; ++k) {
                const int uu = u + du4[k], vv = v + dv4[k];
                if (V(uu, vv)) { const f32 d = norm(place - S.at(uu, vv)); if (d > 2.5f * p.step || d < 0.5f * p.step) ok = false; }  // no collapse/tear
            }
            if (!ok) { dead[static_cast<usize>(id)] = 1; continue; }
            // CT-valley BARRIER: reject a step that crosses an inter-wrap air gap (the prediction fused
            // two touching wraps, so the snap drifted onto the neighbour). Checked against a parent on the
            // current wrap — a within-wrap step stays high (no saddle); a cross-wrap jump crosses one.
            if (p.ct_barrier > 0.0f && fld.has_ct()) {
                bool jumped = false;
                for (int k = 0; k < 4 && !jumped; ++k) {
                    const int uu = u + du4[k], vv = v + dv4[k];
                    if (V(uu, vv))
                        jumped = segment::crosses_valley(fld.ct, fld.ct_coord(S.at(uu, vv)), fld.ct_coord(place),
                                                          p.ct_barrier, 0.5f);
                }
                if (jumped) { dead[static_cast<usize>(id)] = 1; continue; }
            }
            if (fold_conflict(place, u, v)) { dead[static_cast<usize>(id)] = 1; continue; }  // injectivity guard
            if (p.winding_tol > 0 && p.winding_fn) {
                const f32 w = p.winding_fn(place);
                // magnitude-bound finiteness (std::isfinite folds away under -ffast-math)
                if (!(std::abs(w) < 1e30f)) { dead[static_cast<usize>(id)] = 1; continue; }
                f32 wp = 0;
                int nwp = 0;
                for (int k = 0; k < 4; ++k) {
                    const int uu = u + du4[k], vv = v + dv4[k];
                    if (V(uu, vv) && wcell[S.idx(uu, vv)] < 1e29f) {
                        wp += wcell[S.idx(uu, vv)];
                        ++nwp;
                    }
                }
                const f32 pref = nwp > 0 ? wp / static_cast<f32>(nwp) : wind_ref;
                const f32 res = w + std::round(pref - w);  // absorb legit ±1 branch-cut jumps
                if (std::abs(res - pref) > p.winding_jump ||
                    std::abs(res - wind_ref) > p.winding_tol) {
                    dead[static_cast<usize>(id)] = 1;
                    continue;
                }
                wcell[static_cast<usize>(id)] = res;
            }
            S.set(u, v, place);
            claim(place, u, v);
            bdepth[static_cast<usize>(id)] = bd;
            auto [fu, fv] = transport(normalized(accu), normalized(accv), N(place));
            Tu[S.idx(u, v)] = fu; Tv[S.idx(u, v)] = fv;
            ++placed;
            for (int k = 0; k < 4; ++k) enqueue(u + du4[k], v + dv4[k]);
        }

        if (p.fit_every > 0 && (gen % p.fit_every) == 0) detail::arap_fit<T>(S, fld, nf, p, 2, 2, p.lambda, /*interior_only=*/true);
        frontier.swap(nextf);
        if (placed == 0) break;
    }
    detail::cleanup_outliers<T>(S, fld, nf, p);                                       // remove tears/collapses/off-sheet
    detail::remove_slivers(S, p);                                                     // remove bad-aspect slivers
    detail::keep_large_components(S, 200);                                            // drop dust islands (de-fragment)
    detail::fill_rivers<T>(S, fld, nf, p, p.river_radius);                            // bridge thin tributary cracks (smooth, banked)
    detail::fill_holes<T>(S, fld, nf, p, 200);                                        // close enclosed holes
    detail::arap_fit<T>(S, fld, nf, p, p.final_outer, p.final_inner, p.lambda, false); // polish (also smooths fills)
    detail::cleanup_outliers<T>(S, fld, nf, p);                                       // final tear/collapse sweep
    detail::remove_slivers(S, p);                                                     // final sliver sweep
    detail::keep_large_components(S, 200);                                            // final de-fragment
    detail::fill_rivers<T>(S, fld, nf, p, p.river_radius);                            // re-bridge any rivers re-opened by cleanup
    detail::fill_holes<T>(S, fld, nf, p, 120);                                        // close holes punched by the final cleanup
    detail::arap_fit<T>(S, fld, nf, p, 4, 12, p.lambda, false);                       // light polish of the filled cells
    if (p.winding_tol > 0 && p.winding_fn)                                            // drop post-pass wrap leaks
        winding_filter(S, p.winding_fn, p.winding_tol, p.winding_jump, C, C);
    if (!p.uv_mask.empty())                                                           // clip fill spillover to the target shape
        for (s64 i = 0; i < static_cast<s64>(NG); ++i)
            if (!p.uv_mask[static_cast<usize>(i)]) S.valid[static_cast<usize>(i)] = 0;
    detail::fill_surface_channels<T>(S, fld, nf);                                     // persist per-cell normal + confidence
    return S;
}

// Refine a COARSE-traced surface to native resolution: upsample the (u,v) grid by `scale`, scale the
// coordinates from coarse-voxel space to native, snap each point onto the native field along the
// native normal, then ARAP-polish + clean. The coarse trace did the hard topology cheaply; this only
// adds native detail (local snaps) — the coarse-to-fine path the user asked for (fast + native).
template <class T>
inline Surface refine_to_native(const Surface& C, int scale, VolumeView<const T> fine,
                                const NormalField& nf, GrowParams p, VolumeView<const T> fine_ct = {}) {
    const s64 Gc = C.nu;
    const s64 Gf = Gc * scale;
    Surface S(Gf, Gf);
    const Extent3 D = fine.dims();
    const DataField<T> fld{fine, fine_ct, p.surf_thresh, p.ct_thresh, p.ct_weight, p.ct_ds, p.ct_skip};
    const f32 mgn = p.snap_radius + 2.0f;
    auto inb = [&](Vec3f c) { return c.z >= mgn && c.y >= mgn && c.x >= mgn && c.z < D.z - mgn && c.y < D.y - mgn && c.x < D.x - mgn; };
    parallel_for_z(Extent3{Gf, 1, 1}, [&](s64 vf) {
        for (s64 uf = 0; uf < Gf; ++uf) {
            const f32 cu = (static_cast<f32>(uf) + 0.5f) / static_cast<f32>(scale) - 0.5f;
            const f32 cv = (static_cast<f32>(vf) + 0.5f) / static_cast<f32>(scale) - 0.5f;
            const s64 u0 = static_cast<s64>(std::floor(cu)), v0 = static_cast<s64>(std::floor(cv));
            if (u0 < 0 || v0 < 0 || u0 + 1 >= Gc || v0 + 1 >= Gc) continue;
            if (!(C.is_valid(u0, v0) && C.is_valid(u0 + 1, v0) && C.is_valid(u0, v0 + 1) && C.is_valid(u0 + 1, v0 + 1))) continue;
            const f32 fu = cu - static_cast<f32>(u0), fv = cv - static_cast<f32>(v0);
            const Vec3f p00 = C.at(u0, v0), p10 = C.at(u0 + 1, v0), p01 = C.at(u0, v0 + 1), p11 = C.at(u0 + 1, v0 + 1);
            const Vec3f pc = p00 * ((1 - fu) * (1 - fv)) + p10 * (fu * (1 - fv)) + p01 * ((1 - fu) * fv) + p11 * (fu * fv);
            const Vec3f pn = pc * static_cast<f32>(scale);  // coarse-voxel -> native-voxel
            if (!inb(pn)) continue;
            auto [q, val, tt] = detail::snap_to_sheet(fld, pn, nf.at(pn), p.snap_radius);
            S.set(uf, vf, (val >= 1.0f && std::abs(tt) < p.snap_radius && inb(q)) ? q : pn);
        }
    });
    detail::arap_fit<T>(S, fld, nf, p, p.final_outer, p.final_inner, p.lambda, false);
    detail::cleanup_outliers<T>(S, fld, nf, p);
    detail::remove_slivers(S, p);
    detail::keep_large_components(S, 200);
    detail::fill_rivers<T>(S, fld, nf, p, p.river_radius);
    detail::fill_holes<T>(S, fld, nf, p, 20);
    detail::fill_surface_channels<T>(S, fld, nf);
    return S;
}

struct VolumeResult {
    std::vector<Surface> sheets;
    s64 occupied_bins = 0;    // distinct 3D bins covered by all sheets
    s64 seed_candidates = 0;  // high-surf seed candidates considered
};

// Trace the WHOLE volume: auto-seed at the strongest uncovered sheet voxels and grow non-overlapping
// sheets (shared 3D occupancy) until the seed budget / sheet cap is hit. Each accepted sheet tiles a
// distinct part of the scroll. `seed_stride` subsamples seed candidates; `seed_thresh` is the minimum
// field value to seed on (in the field's units); `min_valid` rejects tiny sheets.
template <class T>
inline VolumeResult trace_volume(VolumeView<const T> f, const NormalField& nf, GrowParams p,
                                 int max_sheets, s64 min_valid, int seed_stride, f32 seed_thresh,
                                 VolumeView<const T> ct = {}) {
    const Extent3 D = f.dims();
    struct Cand { f32 val; Vec3f c; };
    std::vector<Cand> cands;
    for (s64 z = seed_stride; z < D.z - seed_stride; z += seed_stride)
        for (s64 y = seed_stride; y < D.y - seed_stride; y += seed_stride)
            for (s64 x = seed_stride; x < D.x - seed_stride; x += seed_stride) {
                const f32 val = static_cast<f32>(f(z, y, x));
                if (val >= seed_thresh) cands.push_back({val, Vec3f{static_cast<f32>(z), static_cast<f32>(y), static_cast<f32>(x)}});
            }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.val > b.val; });

    VolumeResult R;
    R.seed_candidates = static_cast<s64>(cands.size());
    OccMap occ;
    const f32 ibin = 1.0f / p.bin_size;
    const s64 BS = 1 << 20;
    auto binof = [&](Vec3f c) { return (static_cast<s64>(c.z * ibin)) * BS * BS + (static_cast<s64>(c.y * ibin)) * BS + static_cast<s64>(c.x * ibin); };
    s64 next_id = 1;
    for (const auto& cd : cands) {
        if (static_cast<int>(R.sheets.size()) >= max_sheets) break;
        if (occ.get(binof(cd.c))) continue;  // region already covered
        Surface S = grow_surface<T>(f, ct, nf, cd.c, p, &occ, next_id++);
        if (S.valid_count() >= min_valid) R.sheets.push_back(std::move(S));
    }
    R.occupied_bins = static_cast<s64>(occ.size());
    return R;
}

namespace detail {
// Process ONE tile: from contiguous process-region pred/ct blocks (`tf`,`tc`), trace sheet-fragments,
// clip each to the core (expanded by `overlap`), and translate to GLOBAL coords (`porg` = process-region
// origin; `clo`/`chi` = the core box in tile-local coords). The per-tile body shared by the in-core
// (trace_volume_tiled) and streamed (trace_volume_streamed) tilers — they differ ONLY in how the tile
// blocks are obtained (resident crop vs zarr region fetch).
template <class T>
inline std::vector<Surface> trace_one_tile(VolumeView<const T> tf, VolumeView<const T> tc, Index3 porg,
                                           Index3 clo, Index3 chi, GrowParams p, s64 min_valid,
                                           int seed_stride, f32 seed_thresh, int nf_ds, int overlap) {
    const Extent3 pe = tf.dims();
    const bool has_ct = tc.dims().count() > 0;
    const NormalField tnf = compute_normal_field<T>(tf, nf_ds);
    const s64 maxpe = std::max({pe.z, pe.y, pe.x});
    GrowParams pt = p;  // grid sized so a center-seeded fragment reaches the farthest tile corner +margin
    pt.grid = static_cast<int>(std::clamp<s64>(static_cast<s64>(1.8f * static_cast<f32>(maxpe) / p.step) + 8, 64, 1024));
    const f32 ibin = 1.0f / p.bin_size;
    const s64 BS = 1 << 20;
    auto binof = [&](Vec3f c) { return (static_cast<s64>(c.z * ibin)) * BS * BS + (static_cast<s64>(c.y * ibin)) * BS + static_cast<s64>(c.x * ibin); };
    struct Cand { f32 val; Vec3f c; };
    std::vector<Cand> cands;  // seeds in the CORE only (halo seeds belong to the neighbour tile)
    for (s64 z = clo.z; z < chi.z; z += seed_stride)
        for (s64 y = clo.y; y < chi.y; y += seed_stride)
            for (s64 x = clo.x; x < chi.x; x += seed_stride) {
                const f32 val = static_cast<f32>(tf(z, y, x));
                if (val >= seed_thresh) cands.push_back({val, Vec3f{static_cast<f32>(z), static_cast<f32>(y), static_cast<f32>(x)}});
            }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.val > b.val; });
    const VolumeView<const T> ctv = has_ct ? tc : VolumeView<const T>{};
    OccMap occ;
    std::vector<Surface> out;
    s64 id = 1;
    const Vec3f off{static_cast<f32>(porg.z), static_cast<f32>(porg.y), static_cast<f32>(porg.x)};
    for (const auto& cd : cands) {
        if (occ.get(binof(cd.c))) continue;
        Surface S = grow_surface<T>(tf, ctv, tnf, cd.c, pt, &occ, id++);
        const s64 kz0 = std::max<s64>(0, clo.z - overlap), kz1 = std::min<s64>(pe.z, chi.z + overlap);
        const s64 ky0 = std::max<s64>(0, clo.y - overlap), ky1 = std::min<s64>(pe.y, chi.y + overlap);
        const s64 kx0 = std::max<s64>(0, clo.x - overlap), kx1 = std::min<s64>(pe.x, chi.x + overlap);
        s64 vc = 0;
        for (usize i = 0; i < S.valid.size(); ++i) {
            if (!S.valid[i]) continue;
            const Vec3f c = S.coord[i];
            if (c.z < static_cast<f32>(kz0) || c.z >= static_cast<f32>(kz1) || c.y < static_cast<f32>(ky0) ||
                c.y >= static_cast<f32>(ky1) || c.x < static_cast<f32>(kx0) || c.x >= static_cast<f32>(kx1))
                S.valid[i] = 0;
            else
                ++vc;
        }
        if (vc < min_valid) continue;
        for (usize i = 0; i < S.valid.size(); ++i)
            if (S.valid[i]) S.coord[i] = S.coord[i] + off;
        out.push_back(std::move(S));
    }
    return out;
}
}  // namespace detail

// Tiled whole-volume trace — the cache-locality (and out-of-core on-ramp) form of trace_volume.
// The non-tiled tracer grows each sheet across the WHOLE cube, so a single grow's scattered snap reads
// roam a 256 MiB (pred+CT) working set that dwarfs L3 -> DRAM-latency bound. Here we partition the
// volume into `tile_core`^3 tiles, each padded by `halo`, COPY the (tile+2*halo) sub-region of pred+CT
// into CONTIGUOUS blocks (the packing is what wins: a packed ~(tile+2*halo)^3 x2 u8 of a few MB is
// L2/L3-resident, so the snap reads hit cache), trace sheet-FRAGMENTS inside the tile, clip them to the
// core, and translate coords back to global. Fragments of one physical sheet meet at core seams and are
// re-stitched by the patch graph (merge_same_sheet) into one coherent unwrap — the same machinery that
// will stitch streamed tiles out-of-core. Per-tile occupancy guards injectivity within a tile; cross-
// tile coverage is disjoint by construction (cores tile the volume). `min_valid` here is a FRAGMENT
// threshold (smaller than a whole-cube sheet). `max_sheets` is a total fragment cap.
template <class T>
inline VolumeResult trace_volume_tiled(VolumeView<const T> f, VolumeView<const T> ct, GrowParams p,
                                       int max_sheets, s64 min_valid, int seed_stride, f32 seed_thresh,
                                       int tile_core, int halo, int overlap = 0, int nf_ds = 8) {
    const Extent3 D = f.dims();
    const bool has_ct = ct.dims().count() > 0;
    // The tiles are DISJOINT cores (cores tile the volume; each fragment is clipped to its core and a
    // per-tile OccMap guards injectivity within the tile) -> the tiles are independent and the trace runs
    // one per thread. Growth is the wall-clock cost and was single-threaded; here the OUTER tile loop is
    // the parallel level and each tile body wraps its per-tile kernels (normal field, ARAP, packing) in a
    // SerialRegion so they don't nest. Result is order-independent (the patch graph re-stitches), so
    // parallel == serial modulo fragment order + which fragments a `max_sheets` truncation keeps.
    struct Tile { s64 tz, ty, tx; };
    std::vector<Tile> tiles;
    for (s64 tz = 0; tz < D.z; tz += tile_core)
        for (s64 ty = 0; ty < D.y; ty += tile_core)
            for (s64 tx = 0; tx < D.x; tx += tile_core) tiles.push_back({tz, ty, tx});
    std::vector<std::vector<Surface>> per_tile(tiles.size());  // one slot per tile -> no write contention
    // DYNAMIC schedule: tile cost is wildly uneven (dense papyrus tiles dwarf edge/air tiles), so static
    // ranges starve most cores waiting on the few heavy tiles.
    parallel_for_dynamic(0, static_cast<s64>(tiles.size()), [&](s64 ti) {
        SerialRegion serial;  // per-tile kernels serial (this is the one parallel level)
        const Tile t = tiles[static_cast<usize>(ti)];
        const Index3 porg{std::max<s64>(0, t.tz - halo), std::max<s64>(0, t.ty - halo), std::max<s64>(0, t.tx - halo)};
        const Extent3 pe{std::min<s64>(D.z, t.tz + tile_core + halo) - porg.z,
                         std::min<s64>(D.y, t.ty + tile_core + halo) - porg.y,
                         std::min<s64>(D.x, t.tx + tile_core + halo) - porg.x};
        if (pe.z < 16 || pe.y < 16 || pe.x < 16) return;  // too thin to seed/normal-field
        // contiguous copies of the process region (this packing is the cache win)
        Volume<T> tf(pe), tc;
        { auto src = f.crop(porg, pe); auto dv = tf.view(); for (s64 z = 0; z < pe.z; ++z) for (s64 y = 0; y < pe.y; ++y) for (s64 x = 0; x < pe.x; ++x) dv(z, y, x) = src(z, y, x); }
        if (has_ct) { tc = Volume<T>(pe); auto src = ct.crop(porg, pe); auto dv = tc.view(); for (s64 z = 0; z < pe.z; ++z) for (s64 y = 0; y < pe.y; ++y) for (s64 x = 0; x < pe.x; ++x) dv(z, y, x) = src(z, y, x); }
        const Index3 clo{t.tz - porg.z, t.ty - porg.y, t.tx - porg.x};
        const Index3 chi{std::min<s64>(pe.z, clo.z + tile_core), std::min<s64>(pe.y, clo.y + tile_core), std::min<s64>(pe.x, clo.x + tile_core)};
        per_tile[static_cast<usize>(ti)] = detail::trace_one_tile<T>(
            tf.view(), has_ct ? tc.view() : VolumeView<const T>{}, porg, clo, chi, p, min_valid, seed_stride, seed_thresh, nf_ds, overlap);
    });
    VolumeResult R;
    for (auto& frags : per_tile)
        for (Surface& s : frags) {
            if (static_cast<int>(R.sheets.size()) >= max_sheets) break;
            R.sheets.push_back(std::move(s));
        }
    FENIX_INFO("segment", "tiled trace: {} fragments (core={}, halo={}, {} tiles, {})", R.sheets.size(),
               tile_core, halo, tiles.size(), g_parallel_serial ? "serial" : "parallel");
    return R;
}

}  // namespace fenix::segment
