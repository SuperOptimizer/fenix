// winding/diffeo_fit.hpp — the differentiable diffeomorphic fit (the capstone, first in-core slice).
// Optimizes the SVF flow LATTICE (analytic reverse-mode gradient via flow_point_backward — the bulk
// deformation, thousands of params), the global affine (analytic readout; the 2x2 matrix-exp via a 4-FD
// contraction), and dr_per_winding, so constraint points attain their target winding and co-winding
// groups share a winding. This is spiral-v2's objective in miniature: the two loss archetypes
// (winding-target/DT snap + co-winding radius constancy) + flow regularizers, AdamW, coarse->fine.
// The stub fit.hpp::fit_spiral (dr+affine, finite-diff) stays as the Stage-0 warm-start.
// See winding/CLAUDE.md, docs/research/spiral-v2.md.
#pragma once

#include "core/core.hpp"
#include "core/optimize.hpp"
#include "winding/fit.hpp"  // FitConstraint, FitResult
#include "winding/spiral_model.hpp"
#include "winding/transforms.hpp"  // Mat2, expm2

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <thread>
#include <vector>

namespace fenix::winding {

// A set of points known to lie on ONE (unknown) winding — penalize variance of their winding value.
struct CoWindingGroup {
    std::vector<Vec3f> points;
};

// A relative-winding constraint: W(b) − W(a) = delta (the "+1/+2/+3 radial line" annotation —
// spiral-v2's rel_winding term). No absolute winding needed; pairs chain along a crossing line.
struct RelWindingConstraint {
    Vec3f a, b;
    f32 delta = 1.0f;
    f32 weight = 1.0f;
};

struct DiffeoFitConfig {
    int iters_affine = 250;     // Stage 0: dr + global affine, flow frozen at identity
    int iters_flow = 500;       // Stage 1: unfreeze the flow lattice
    f32 lr_affine = 0.03f;
    f32 lr_flow = 0.05f;
    f32 lambda_cowind = 1.0f;   // weight of the co-winding variance term vs the winding-target term
    f32 lambda_rel = 1.0f;      // weight of the relative-winding term vs the winding-target term
    f32 lambda_l2 = 1e-3f;      // flow velocity L2 (keep the deformation small)
    f32 lambda_smooth = 1e-2f;  // flow velocity 6-neighbour graph-Laplacian smoothness
    f32 r_min = 2.0f;           // skip constraints within this radius of the umbilicus (theta ill-cond.)
    f32 expm_fd_eps = 1e-3f;    // central-FD step for d(expm2)/d(affine logit) (only 4 scalars)
    Extent3 flow_dims{16, 16, 16};  // coarse velocity lattice
    Vec3f domain_lo{0, 0, 0};
    Vec3f domain_hi{0, 0, 0};   // hi<=lo => derive the box from the constraint bbox (+margin)
    int flow_steps = 8;
    bool fit_flow = true;       // false => affine-only (skip Stage 1)
    bool flow_pyramid = true;   // Stage 1 runs COARSE->FINE: lattice halved down to >=4/axis,
                                // each level warm-started by trilinear upsampling of the coarser
                                // solution (classic diffeomorphic multi-resolution: big basins
                                // first, detail last). false = single-shot at flow_dims.
    bool continuous = false;    // targets are CONTINUOUS windings (corpus bridge): read/backprop
                                // winding_cont (no theta term) instead of the stepped winding_at
    int affine_bands = 0;       // per-z-band residual affines (0 = global affine only). Fitted in
                                // Stage 1 alongside the flow — the z capacity the global affine lacks.
    f32 lambda_affine_smooth = 1e-2f;  // 1D Laplacian over adjacent band params
    int gap_logits = 0;         // per-winding gap-expander logits to FIT (0 = gap frozen at its
                                // current table). Unfreezes non-uniform inter-wrap spacing — real
                                // scrolls compress toward the core. Fitted in Stage 1.
    f32 lambda_gap_smooth = 1e-2f;  // 1D Laplacian over adjacent logits
    f32 lambda_gap_zero = 1e-2f;    // pull the logit MEAN to 0 (a uniform shift is dr's job)
    int threads = 0;            // per-constraint backward parallelism (0 => hardware_concurrency)
};

namespace detail {
// Accumulated gradient for one fit step. Affine/dr are scalars; the flow lattice grads are f64 arrays
// sized to the lattice (gz unused while vz is frozen, but computed for correctness/symmetry).
struct FitGrad {
    f64 g_dr = 0, g_ty = 0, g_tx = 0, g_off = 0;
    f64 G_Mi[2][2] = {{0, 0}, {0, 0}};  // adjoint of the inverse-affine matrix entries
    std::vector<f64> gz, gy, gx;
    std::vector<f64> Gb;  // per-band adjoints, 6 per band: [G00,G01,G10,G11, g_ty, g_tx]
    std::vector<f64> Gg;  // per-winding gap-logit adjoints (size = gap_logits when fitted)
};

// Reverse-mode of winding_at(p) given seed gW = dL/dW: accumulate into `acc`. Mirrors spiral_model's
// forward chain EXACTLY: q0=p-umbilicus -> qf=flow^{-1}(q0) -> q=affine^{-1}(qf) -> (r,theta) -> W.
inline void winding_backward(const SpiralModel& m, Vec3f p, f64 gW, FitGrad& acc, bool fit_flow, f32 r_min,
                             bool continuous = false) {
    constexpr f64 two_pi = 2.0 * std::numbers::pi_v<f64>;
    const Vec3f c = m.umbilicus.empty() ? Vec3f{p.z, 0, 0} : m.umbilicus.center(p.z);
    const Vec3f q0{p.z, p.y - c.y, p.x - c.x};
    const bool use_flow = fit_flow && m.has_flow;
    const Vec3f qf = use_flow ? flow_point(m.flow, q0, m.flow_steps, -1.0f) : q0;
    const Mat2 Mi = expm2(-m.affine.a, -m.affine.b, -m.affine.c, -m.affine.d);
    const f32 v0 = qf.y - m.affine.ty, v1 = qf.x - m.affine.tx;
    const f32 uy = Mi.m00 * v0 + Mi.m01 * v1, ux = Mi.m10 * v0 + Mi.m11 * v1;
    // band residual affine (innermost): q = Mi_b (u - t_b)
    const s64 kb = m.affine_bands.band_of(p.z);
    f32 qy = uy, qx = ux, w0 = 0, w1 = 0;
    Mat2 Mib{1, 0, 0, 1};
    if (kb >= 0) {
        const AffineYX& ab = m.affine_bands.bands[static_cast<usize>(kb)];
        Mib = expm2(-ab.a, -ab.b, -ab.c, -ab.d);
        w0 = uy - ab.ty;
        w1 = ux - ab.tx;
        qy = Mib.m00 * w0 + Mib.m01 * w1;
        qx = Mib.m10 * w0 + Mib.m11 * w1;
    }
    const f64 r = std::sqrt(static_cast<f64>(qy) * qy + static_cast<f64>(qx) * qx);
    if (r < r_min) return;  // near the umbilicus theta is ill-conditioned
    const f64 r_ideal = static_cast<f64>(m.gap.inverse(static_cast<f32>(r)));
    const f64 dr = m.dr_per_winding;
    // W = gap.inverse(r)/dr - theta/2pi + winding_offset. The HONEST gap chain: inside
    // segment k, inverse(r) = (k + (r - acc_k)/seg_k)*dr with seg_i = dr*e^{l_i}, so
    //   d inv/dr    = dr/seg_k = e^{-l_k}
    //   d inv/dl_k  = -frac*dr           (frac = (r - acc_k)/seg_k)
    //   d inv/dl_i  = -seg_i*dr/seg_k    (i < k: the whole boundary stack shifts)
    // With no logits, seg = dr and everything degrades to the old identity path.
    f64 dinv_dr = 1.0;
    if (!m.gap.logits.empty()) {
        f64 acc_r = 0;
        usize k = m.gap.logits.size();
        f64 seg_k = dr, frac = 0;
        for (usize i = 0; i < m.gap.logits.size(); ++i) {
            const f64 seg = dr * std::exp(static_cast<f64>(m.gap.logits[i]));
            if (r < acc_r + seg) {
                k = i;
                seg_k = seg;
                frac = (r - acc_r) / seg;
                break;
            }
            acc_r += seg;
        }
        if (k < m.gap.logits.size()) {
            dinv_dr = dr / seg_k;
            if (!acc.Gg.empty()) {
                const f64 gwd = gW / dr;  // dL/d inv
                acc.Gg[k] += gwd * (-frac * dr);
                for (usize i = 0; i < k; ++i)
                    acc.Gg[i] += gwd * (-(dr * std::exp(static_cast<f64>(m.gap.logits[i]))) * dr / seg_k);
            }
        }  // past the table: inverse is identity-sloped, no logit dependence
    }
    // W = r_ideal/dr - theta/2pi + winding_offset
    acc.g_off += gW;  // dW/d(winding_offset) = 1
    acc.g_dr += gW * (-r_ideal / (dr * dr));
    const f64 a_r = gW * (dinv_dr / dr);
    const f64 a_th = continuous ? 0.0 : gW * (-1.0 / two_pi);  // winding_cont has no theta term
    const f64 a_qy = a_r * (qy / r) + a_th * (qx / (r * r));
    const f64 a_qx = a_r * (qx / r) + a_th * (-qy / (r * r));
    f64 a_uy = a_qy, a_ux = a_qx;
    if (kb >= 0) {
        a_uy = static_cast<f64>(Mib.m00) * a_qy + static_cast<f64>(Mib.m10) * a_qx;
        a_ux = static_cast<f64>(Mib.m01) * a_qy + static_cast<f64>(Mib.m11) * a_qx;
        if (!acc.Gb.empty()) {
            f64* gb = acc.Gb.data() + 6 * kb;
            gb[0] += a_qy * w0;
            gb[1] += a_qy * w1;
            gb[2] += a_qx * w0;
            gb[3] += a_qx * w1;
            gb[4] += -a_uy;
            gb[5] += -a_ux;
        }
    }
    acc.G_Mi[0][0] += a_uy * v0;
    acc.G_Mi[0][1] += a_uy * v1;
    acc.G_Mi[1][0] += a_ux * v0;
    acc.G_Mi[1][1] += a_ux * v1;
    const f64 a_v0 = static_cast<f64>(Mi.m00) * a_uy + static_cast<f64>(Mi.m10) * a_ux;
    const f64 a_v1 = static_cast<f64>(Mi.m01) * a_uy + static_cast<f64>(Mi.m11) * a_ux;
    acc.g_ty += -a_v0;
    acc.g_tx += -a_v1;
    if (use_flow) {
        const Vec3f a_qf{0.0f, static_cast<f32>(a_v0), static_cast<f32>(a_v1)};
        FlowGradAccum ga{acc.gz.data(), acc.gy.data(), acc.gx.data()};
        flow_point_backward(m.flow, q0, m.flow_steps, -1.0f, a_qf, ga);
    }
}

// total loss (for reporting + the synthetic test): mean winding-target sq + lambda*co-winding
// variance + lambda*mean weighted relative-winding sq.
inline f64 fit_loss(const SpiralModel& m, std::span<const FitConstraint> wcs,
                    std::span<const CoWindingGroup> groups, f32 lambda_cw, f32 r_min,
                    std::span<const RelWindingConstraint> rels = {}, f32 lambda_rel = 1.0f,
                    bool continuous = false) {
    auto W = [&](Vec3f p) { return continuous ? m.winding_cont(p) : m.winding_at(p); };
    f64 s = 0;
    const f64 M = std::max<usize>(1, wcs.size());
    for (const FitConstraint& c : wcs) {
        const f64 e = static_cast<f64>(W(c.scroll_pt)) - c.target_winding;
        s += c.weight * e * e / M;
    }
    for (const CoWindingGroup& g : groups) {
        if (g.points.size() < 2) continue;
        f64 mean = 0;
        for (Vec3f p : g.points) mean += W(p);
        mean /= static_cast<f64>(g.points.size());
        f64 var = 0;
        for (Vec3f p : g.points) {
            const f64 e = W(p) - mean;
            var += e * e;
        }
        s += static_cast<f64>(lambda_cw) * var / static_cast<f64>(g.points.size());
    }
    const f64 R = static_cast<f64>(std::max<usize>(1, rels.size()));
    for (const RelWindingConstraint& rc : rels) {
        const f64 e = static_cast<f64>(W(rc.b)) - W(rc.a) - rc.delta;
        s += static_cast<f64>(lambda_rel) * rc.weight * e * e / R;
    }
    (void)r_min;
    return s;
}
}  // namespace detail

// Fit `model` in place to the winding-target constraints + co-winding groups + relative-winding
// pairs. Stage 0 fits dr+affine with the flow frozen at identity (a global warm-start); Stage 1
// unfreezes the coarse flow lattice.
inline FitResult fit_spiral_diffeo(SpiralModel& model, std::span<const FitConstraint> wcs,
                                   std::span<const CoWindingGroup> groups,
                                   std::span<const RelWindingConstraint> rels, DiffeoFitConfig cfg = {}) {
    FitResult res;
    res.initial_loss = static_cast<f32>(
        detail::fit_loss(model, wcs, groups, cfg.lambda_cowind, cfg.r_min, rels, cfg.lambda_rel, cfg.continuous));

    // domain box for the flow lattice: explicit, or the constraint bbox + a margin. The flow is sampled
    // in UMBILICUS-CENTERED space (q0 = p - center, the input to flow_point in to_canonical), so the box
    // is over the CENTERED constraint points, not raw scroll coords.
    Vec3f lo = cfg.domain_lo, hi = cfg.domain_hi;
    if (!(hi.z > lo.z && hi.y > lo.y && hi.x > lo.x)) {
        lo = Vec3f{1e30f, 1e30f, 1e30f};
        hi = Vec3f{-1e30f, -1e30f, -1e30f};
        auto ext = [&](Vec3f p) {
            const Vec3f cc = model.umbilicus.empty() ? Vec3f{p.z, 0, 0} : model.umbilicus.center(p.z);
            const Vec3f q{p.z, p.y - cc.y, p.x - cc.x};
            lo = Vec3f{std::min(lo.z, q.z), std::min(lo.y, q.y), std::min(lo.x, q.x)};
            hi = Vec3f{std::max(hi.z, q.z), std::max(hi.y, q.y), std::max(hi.x, q.x)};
        };
        for (const FitConstraint& c : wcs) ext(c.scroll_pt);
        for (const CoWindingGroup& g : groups)
            for (Vec3f p : g.points) ext(p);
        for (const RelWindingConstraint& rc : rels) {
            ext(rc.a);
            ext(rc.b);
        }
        const Vec3f m{8, 8, 8};
        lo = lo - m;
        hi = hi + m;
    }
    // WARM-START-SAFE setup: keep an existing flow lattice / band stack when their shapes
    // already match (the EM regauge refit path); only (re)initialize on shape change.
    const Extent3 fd = cfg.flow_dims;
    Extent3 fd0 = fd;  // first-allocated level: coarsest of the pyramid
    if (cfg.flow_pyramid)
        while (fd0.z / 2 >= 4 && fd0.y / 2 >= 4 && fd0.x / 2 >= 4) fd0 = {fd0.z / 2, fd0.y / 2, fd0.x / 2};
    if (!(model.has_flow && model.flow.vy.dims() == fd)) {
        model.flow.vz = Volume<f32>::zeros(fd0);
        model.flow.vy = Volume<f32>::zeros(fd0);
        model.flow.vx = Volume<f32>::zeros(fd0);
        model.flow.lat_lo = lo;
        model.flow.lat_scale = Vec3f{static_cast<f32>(fd0.z - 1) / std::max(1e-3f, hi.z - lo.z),
                                     static_cast<f32>(fd0.y - 1) / std::max(1e-3f, hi.y - lo.y),
                                     static_cast<f32>(fd0.x - 1) / std::max(1e-3f, hi.x - lo.x)};
    }
    model.flow_steps = cfg.flow_steps;

    // per-z-band residual affines: identity bands over the domain z-range; Stage 0 fits the
    // global affine through them (identity = transparent), Stage 1 unfreezes them.
    const int K = std::max(0, cfg.gap_logits);
    if (K > 0 && static_cast<int>(model.gap.logits.size()) != K) {
        model.gap.dr = model.dr_per_winding;
        model.gap.logits.assign(static_cast<usize>(K), 0.0f);
    }
    const int B = std::max(0, cfg.affine_bands);
    if (B > 0 && static_cast<int>(model.affine_bands.bands.size()) != B) {
        model.affine_bands.z0 = lo.z;
        model.affine_bands.dz = std::max(1.0f, (hi.z - lo.z) / static_cast<f32>(B));
        model.affine_bands.bands.assign(static_cast<usize>(B), AffineYX{});
    }

    // d(expm2(-L))/d(logit) by central FD at an arbitrary L (a,b,c,d) — cheap closed-form expm2.
    auto dMi_at = [&](f32 a, f32 b, f32 c, f32 d, int which) -> Mat2 {
        f32* tgt = which == 0 ? &a : (which == 1 ? &b : (which == 2 ? &c : &d));
        const f32 o = *tgt;
        *tgt = o + cfg.expm_fd_eps;
        const Mat2 mp = expm2(-a, -b, -c, -d);
        *tgt = o - cfg.expm_fd_eps;
        const Mat2 mm = expm2(-a, -b, -c, -d);
        *tgt = o;
        const f32 inv = 1.0f / (2.0f * cfg.expm_fd_eps);
        return {(mp.m00 - mm.m00) * inv, (mp.m01 - mm.m01) * inv, (mp.m10 - mm.m10) * inv, (mp.m11 - mm.m11) * inv};
    };
    auto dMi = [&](int which) -> Mat2 {
        return dMi_at(model.affine.a, model.affine.b, model.affine.c, model.affine.d, which);
    };

    FENIX_INFO("winding", "diffeo fit: {} targets, {} groups, {} rel pairs, flow {}x{}x{}, dr={:.2f}, initial loss {:.4f}",
               wcs.size(), groups.size(), rels.size(), fd.z, fd.y, fd.x, static_cast<double>(model.dr_per_winding),
               static_cast<double>(res.initial_loss));

    auto run_stage = [&](int iters, f32 lr, bool flow) {
        FENIX_INFO("winding", "fit stage '{}': {} iters, lr {:.3f}", flow ? "flow" : "affine", iters, static_cast<double>(lr));
        FENIX_SCOPE_TIMER("winding", flow ? "fit stage flow" : "fit stage affine");
        model.has_flow = flow;
        // [dr, affine(6), winding_offset | bands(6B) | gaps(K) | vy(N), vx(N)] — Stage 1 only
        const Extent3 ld = model.flow.vy.dims();  // CURRENT lattice (pyramid level)
        const usize N = static_cast<usize>(ld.count());
        const usize NB = flow ? static_cast<usize>(6 * B) : 0;
        const usize NG = flow ? static_cast<usize>(K) : 0;
        const usize NP = flow ? 8 + NB + NG + 2 * N : 8;
        const usize GO = 8 + NB;       // gap-logit offset
        const usize FO = GO + NG;      // flow offset
        std::vector<f32> P(NP), G(NP);
        P[0] = model.dr_per_winding;
        P[1] = model.affine.a;
        P[2] = model.affine.b;
        P[3] = model.affine.c;
        P[4] = model.affine.d;
        P[5] = model.affine.ty;
        P[6] = model.affine.tx;
        P[7] = model.winding_offset;
        if (flow) {
            for (usize k = 0; k < static_cast<usize>(B); ++k) {
                const AffineYX& ab = model.affine_bands.bands[k];
                P[8 + 6 * k] = ab.a;
                P[8 + 6 * k + 1] = ab.b;
                P[8 + 6 * k + 2] = ab.c;
                P[8 + 6 * k + 3] = ab.d;
                P[8 + 6 * k + 4] = ab.ty;
                P[8 + 6 * k + 5] = ab.tx;
            }
            for (usize i = 0; i < NG; ++i) P[GO + i] = model.gap.logits[i];
            for (usize i = 0; i < N; ++i) P[FO + i] = model.flow.vy.view().data()[i];
            for (usize i = 0; i < N; ++i) P[FO + N + i] = model.flow.vx.view().data()[i];
        }
        AdamW opt(NP, {.lr = lr});

        auto Wof = [&](Vec3f pt) { return cfg.continuous ? model.winding_cont(pt) : model.winding_at(pt); };
        // Flatten both loss terms into one work list of (point, loss-coeff, ref) so the per-constraint
        // backward — the bottleneck — parallelizes over T chunks with per-chunk grad buffers (reduced
        // after; winding_at is const/thread-safe, winding_backward writes only its own buffer).
        struct Item { Vec3f p; f64 coeff; f64 ref; int gidx; };  // gidx<0: target (ref=t); else group/rel (ref=means[g])
        std::vector<Item> items;
        const f64 Mn = static_cast<f64>(std::max<usize>(1, wcs.size()));
        for (const FitConstraint& c : wcs)
            items.push_back({c.scroll_pt, 2.0 * c.weight / Mn, static_cast<f64>(c.target_winding), -1});
        for (usize gi = 0; gi < groups.size(); ++gi) {
            if (groups[gi].points.size() < 2) continue;
            const f64 co = 2.0 * static_cast<f64>(cfg.lambda_cowind) / static_cast<f64>(groups[gi].points.size());
            for (Vec3f p : groups[gi].points) items.push_back({p, co, 0.0, static_cast<int>(gi)});
        }
        // Relative pairs ride the same worklist: two items per pair, each seeded against a per-iteration
        // reference in the means[] tail (ref_b = W(a)+delta, ref_a = W(b)-delta), so seed = ±coeff·e.
        const int rel_base = static_cast<int>(groups.size());
        const f64 Rn = static_cast<f64>(std::max<usize>(1, rels.size()));
        for (usize ri = 0; ri < rels.size(); ++ri) {
            const f64 co = 2.0 * static_cast<f64>(cfg.lambda_rel) * rels[ri].weight / Rn;
            items.push_back({rels[ri].b, co, 0.0, rel_base + 2 * static_cast<int>(ri)});
            items.push_back({rels[ri].a, co, 0.0, rel_base + 2 * static_cast<int>(ri) + 1});
        }
        const s64 K = static_cast<s64>(items.size());
        int T = cfg.threads > 0 ? cfg.threads : static_cast<int>(std::thread::hardware_concurrency());
        T = std::clamp(T, 1, 64);
        std::vector<f64> means(groups.size() + 2 * rels.size(), 0.0);
        std::vector<detail::FitGrad> accs(static_cast<usize>(T));

        for (int it = 0; it < iters; ++it) {
            model.dr_per_winding = P[0];
            model.affine.a = P[1];
            model.affine.b = P[2];
            model.affine.c = P[3];
            model.affine.d = P[4];
            model.affine.ty = P[5];
            model.affine.tx = P[6];
            model.winding_offset = P[7];
            if (flow) {
                for (usize k = 0; k < static_cast<usize>(B); ++k) {
                    AffineYX& ab = model.affine_bands.bands[k];
                    ab.a = P[8 + 6 * k];
                    ab.b = P[8 + 6 * k + 1];
                    ab.c = P[8 + 6 * k + 2];
                    ab.d = P[8 + 6 * k + 3];
                    ab.ty = P[8 + 6 * k + 4];
                    ab.tx = P[8 + 6 * k + 5];
                }
                for (usize i = 0; i < NG; ++i) model.gap.logits[i] = P[GO + i];
                std::copy(P.begin() + static_cast<s64>(FO), P.begin() + static_cast<s64>(FO + N),
                          model.flow.vy.view().data());
                std::copy(P.begin() + static_cast<s64>(FO + N), P.end(), model.flow.vx.view().data());
            }
            if ((it % 100) == 0 && static_cast<int>(LogLevel::debug) >= static_cast<int>(log_level()))
                FENIX_DEBUG("winding", "fit it {}: loss {:.4f}", it,
                            detail::fit_loss(model, wcs, groups, cfg.lambda_cowind, cfg.r_min, rels, cfg.lambda_rel, cfg.continuous));
            // co-winding group means + rel-pair references under the CURRENT model (forward only;
            // const => parallel-safe).
            parallel_for(0, static_cast<s64>(groups.size()), [&](s64 gi) {
                if (groups[static_cast<usize>(gi)].points.size() < 2) { means[static_cast<usize>(gi)] = 0; return; }
                f64 m = 0;
                for (Vec3f p : groups[static_cast<usize>(gi)].points) m += Wof(p);
                means[static_cast<usize>(gi)] = m / static_cast<f64>(groups[static_cast<usize>(gi)].points.size());
            });
            parallel_for(0, static_cast<s64>(rels.size()), [&](s64 ri) {
                const RelWindingConstraint& rc = rels[static_cast<usize>(ri)];
                means[static_cast<usize>(rel_base) + 2 * static_cast<usize>(ri)] =
                    static_cast<f64>(Wof(rc.a)) + rc.delta;
                means[static_cast<usize>(rel_base) + 2 * static_cast<usize>(ri) + 1] =
                    static_cast<f64>(Wof(rc.b)) - rc.delta;
            });
            // per-chunk backward into per-chunk buffers (no shared-state race).
            parallel_for(0, T, [&](s64 c) {
                detail::FitGrad& a = accs[static_cast<usize>(c)];
                a = detail::FitGrad{};
                if (flow) { a.gz.assign(N, 0.0); a.gy.assign(N, 0.0); a.gx.assign(N, 0.0); a.Gb.assign(NB, 0.0); a.Gg.assign(NG, 0.0); }
                const s64 lo = c * K / T, hi = (c + 1) * K / T;
                for (s64 k = lo; k < hi; ++k) {
                    const Item& it2 = items[static_cast<usize>(k)];
                    const f64 ref = it2.gidx < 0 ? it2.ref : means[static_cast<usize>(it2.gidx)];
                    const f64 seed = it2.coeff * (Wof(it2.p) - ref);
                    detail::winding_backward(model, it2.p, seed, a, flow, cfg.r_min, cfg.continuous);
                }
            });
            // reduce per-chunk buffers -> acc.
            detail::FitGrad acc;
            if (flow) { acc.gz.assign(N, 0.0); acc.gy.assign(N, 0.0); acc.gx.assign(N, 0.0); acc.Gb.assign(NB, 0.0); acc.Gg.assign(NG, 0.0); }
            for (int c = 0; c < T; ++c) {
                const detail::FitGrad& a = accs[static_cast<usize>(c)];
                acc.g_dr += a.g_dr;
                acc.g_ty += a.g_ty;
                acc.g_tx += a.g_tx;
                acc.g_off += a.g_off;
                acc.G_Mi[0][0] += a.G_Mi[0][0];
                acc.G_Mi[0][1] += a.G_Mi[0][1];
                acc.G_Mi[1][0] += a.G_Mi[1][0];
                acc.G_Mi[1][1] += a.G_Mi[1][1];
                if (flow) {
                    for (usize i = 0; i < N; ++i) { acc.gz[i] += a.gz[i]; acc.gy[i] += a.gy[i]; acc.gx[i] += a.gx[i]; }
                    for (usize i = 0; i < NB; ++i) acc.Gb[i] += a.Gb[i];
                    for (usize i = 0; i < NG; ++i) acc.Gg[i] += a.Gg[i];
                }
            }
            // scalars + affine-logit contraction.
            G[0] = static_cast<f32>(acc.g_dr);
            G[5] = static_cast<f32>(acc.g_ty);
            G[6] = static_cast<f32>(acc.g_tx);
            G[7] = static_cast<f32>(acc.g_off);
            for (int w = 0; w < 4; ++w) {
                const Mat2 dm = dMi(w);
                G[static_cast<usize>(1 + w)] = static_cast<f32>(
                    acc.G_Mi[0][0] * dm.m00 + acc.G_Mi[0][1] * dm.m01 + acc.G_Mi[1][0] * dm.m10 + acc.G_Mi[1][1] * dm.m11);
            }
            if (flow) {  // band grads: 4 logit contractions per band + direct translations + 1D smoothness
                for (usize k = 0; k < static_cast<usize>(B); ++k) {
                    const f64* gb = acc.Gb.data() + 6 * k;
                    const AffineYX& ab = model.affine_bands.bands[k];
                    for (int w = 0; w < 4; ++w) {
                        const Mat2 dm = dMi_at(ab.a, ab.b, ab.c, ab.d, w);
                        G[8 + 6 * k + static_cast<usize>(w)] = static_cast<f32>(
                            gb[0] * dm.m00 + gb[1] * dm.m01 + gb[2] * dm.m10 + gb[3] * dm.m11);
                    }
                    G[8 + 6 * k + 4] = static_cast<f32>(gb[4]);
                    G[8 + 6 * k + 5] = static_cast<f32>(gb[5]);
                    // 1D Laplacian over adjacent bands, per parameter (keeps the stack smooth in z)
                    for (int j = 0; j < 6; ++j) {
                        f32 lapv = 0;
                        int deg = 0;
                        if (k > 0) { lapv += P[8 + 6 * k + static_cast<usize>(j)] - P[8 + 6 * (k - 1) + static_cast<usize>(j)]; ++deg; }
                        if (k + 1 < static_cast<usize>(B)) { lapv += P[8 + 6 * k + static_cast<usize>(j)] - P[8 + 6 * (k + 1) + static_cast<usize>(j)]; ++deg; }
                        if (deg) G[8 + 6 * k + static_cast<usize>(j)] += cfg.lambda_affine_smooth * lapv;
                    }
                }
                // gap-logit grads + 1D smoothness + mean-zero pull
                if (NG > 0) {
                    f64 lmean = 0;
                    for (usize i = 0; i < NG; ++i) lmean += P[GO + i];
                    lmean /= static_cast<f64>(NG);
                    for (usize i = 0; i < NG; ++i) {
                        f32 lap = 0;
                        if (i > 0) lap += P[GO + i] - P[GO + i - 1];
                        if (i + 1 < NG) lap += P[GO + i] - P[GO + i + 1];
                        G[GO + i] = static_cast<f32>(acc.Gg[i] + cfg.lambda_gap_smooth * lap +
                                                     cfg.lambda_gap_zero * lmean);
                    }
                }
                // flow lattice grad = data grad + L2 + 6-neighbour Laplacian smoothness
                const f32* vy = model.flow.vy.view().data();
                const f32* vx = model.flow.vx.view().data();
                const s64 Lz = ld.z, Ly = ld.y, Lx = ld.x;
                auto lap = [&](const f32* v, s64 iz, s64 iy, s64 ix) {
                    const s64 i = (iz * Ly + iy) * Lx + ix;
                    f64 s = 0;
                    int deg = 0;
                    auto nb = [&](s64 z, s64 y, s64 x) { s += v[(z * Ly + y) * Lx + x]; ++deg; };
                    if (iz > 0) nb(iz - 1, iy, ix);
                    if (iz + 1 < Lz) nb(iz + 1, iy, ix);
                    if (iy > 0) nb(iz, iy - 1, ix);
                    if (iy + 1 < Ly) nb(iz, iy + 1, ix);
                    if (ix > 0) nb(iz, iy, ix - 1);
                    if (ix + 1 < Lx) nb(iz, iy, ix + 1);
                    return static_cast<f64>(deg) * v[i] - s;  // (Σ (v_i - v_n))
                };
                for (s64 iz = 0; iz < Lz; ++iz)
                    for (s64 iy = 0; iy < Ly; ++iy)
                        for (s64 ix = 0; ix < Lx; ++ix) {
                            const usize i = static_cast<usize>((iz * Ly + iy) * Lx + ix);
                            G[FO + i] = static_cast<f32>(acc.gy[i] + cfg.lambda_l2 * vy[i] + cfg.lambda_smooth * lap(vy, iz, iy, ix));
                            G[FO + N + i] = static_cast<f32>(acc.gx[i] + cfg.lambda_l2 * vx[i] + cfg.lambda_smooth * lap(vx, iz, iy, ix));
                        }
            }
            opt.step(P, G);
            // band params are RESIDUALS — clamp so an under-constrained band can't run away
            // (expm2 overflows past |logit|~89 -> inf -> NaN windings; seen on real data)
            for (usize k = 0; k < static_cast<usize>(B) * (flow ? 1u : 0u); ++k)
                for (int j = 0; j < 6; ++j) {
                    const f32 lim = j < 4 ? 1.5f : 512.0f;
                    P[8 + 6 * k + static_cast<usize>(j)] = std::clamp(P[8 + 6 * k + static_cast<usize>(j)], -lim, lim);
                }
            for (usize i = 0; i < NG; ++i) P[GO + i] = std::clamp(P[GO + i], -0.7f, 0.7f);
        }
        model.dr_per_winding = P[0];
        model.affine.a = P[1];
        model.affine.b = P[2];
        model.affine.c = P[3];
        model.affine.d = P[4];
        model.affine.ty = P[5];
        model.affine.tx = P[6];
        model.winding_offset = P[7];
        if (flow) {
            for (usize k = 0; k < static_cast<usize>(B); ++k) {
                AffineYX& ab = model.affine_bands.bands[k];
                ab.a = P[8 + 6 * k];
                ab.b = P[8 + 6 * k + 1];
                ab.c = P[8 + 6 * k + 2];
                ab.d = P[8 + 6 * k + 3];
                ab.ty = P[8 + 6 * k + 4];
                ab.tx = P[8 + 6 * k + 5];
            }
            for (usize i = 0; i < NG; ++i) model.gap.logits[i] = P[GO + i];
            std::copy(P.begin() + static_cast<s64>(FO), P.begin() + static_cast<s64>(FO + N),
                      model.flow.vy.view().data());
            std::copy(P.begin() + static_cast<s64>(FO + N), P.end(), model.flow.vx.view().data());
        }
    };

    run_stage(cfg.iters_affine, cfg.lr_affine, false);  // Stage 0: dr + affine, flow at identity
    if (cfg.fit_flow) {
        // pyramid schedule: halve flow_dims down to >=4/axis, coarsest first
        std::vector<Extent3> levels{fd};
        if (cfg.flow_pyramid)
            while (levels.back().z / 2 >= 4 && levels.back().y / 2 >= 4 && levels.back().x / 2 >= 4)
                levels.push_back({levels.back().z / 2, levels.back().y / 2, levels.back().x / 2});
        std::reverse(levels.begin(), levels.end());
        const int iters_per = std::max(50, cfg.iters_flow / static_cast<int>(levels.size()));
        for (usize li = 0; li < levels.size(); ++li) {
            const Extent3 nd = levels[li];
            if (model.flow.vy.dims() != nd) {
                // trilinear-upsample the current solution onto the finer lattice (same box)
                auto up = [&](const Volume<f32>& src) {
                    const Extent3 sd = src.dims();
                    Volume<f32> dst = Volume<f32>::zeros(nd);
                    for (s64 z = 0; z < nd.z; ++z)
                        for (s64 y = 0; y < nd.y; ++y)
                            for (s64 x = 0; x < nd.x; ++x)
                                dst(z, y, x) = sample_trilinear(
                                    src.view(),
                                    Vec3f{static_cast<f32>(z) * static_cast<f32>(sd.z - 1) / static_cast<f32>(std::max<s64>(1, nd.z - 1)),
                                          static_cast<f32>(y) * static_cast<f32>(sd.y - 1) / static_cast<f32>(std::max<s64>(1, nd.y - 1)),
                                          static_cast<f32>(x) * static_cast<f32>(sd.x - 1) / static_cast<f32>(std::max<s64>(1, nd.x - 1))});
                    return dst;
                };
                model.flow.vz = up(model.flow.vz);
                model.flow.vy = up(model.flow.vy);
                model.flow.vx = up(model.flow.vx);
                model.flow.lat_scale = Vec3f{static_cast<f32>(nd.z - 1) / std::max(1e-3f, hi.z - lo.z),
                                             static_cast<f32>(nd.y - 1) / std::max(1e-3f, hi.y - lo.y),
                                             static_cast<f32>(nd.x - 1) / std::max(1e-3f, hi.x - lo.x)};
            }
            FENIX_INFO("winding", "flow pyramid level {}/{}: {}x{}x{}", li + 1, levels.size(), nd.z, nd.y, nd.x);
            run_stage(iters_per, cfg.lr_flow, true);
        }
    }
    model.has_flow = cfg.fit_flow;

    res.final_loss = static_cast<f32>(
        detail::fit_loss(model, wcs, groups, cfg.lambda_cowind, cfg.r_min, rels, cfg.lambda_rel, cfg.continuous));
    res.iters = cfg.iters_affine + (cfg.fit_flow ? cfg.iters_flow : 0);
    FENIX_INFO("winding", "diffeo fit done: loss {:.4f} -> {:.4f} ({} iters)", static_cast<double>(res.initial_loss),
               static_cast<double>(res.final_loss), res.iters);
    return res;
}

inline FitResult fit_spiral_diffeo(SpiralModel& model, std::span<const FitConstraint> wcs,
                                   std::span<const CoWindingGroup> groups, DiffeoFitConfig cfg = {}) {
    return fit_spiral_diffeo(model, wcs, groups, {}, cfg);
}

}  // namespace fenix::winding
