// winding/fit.hpp — the unified diffeomorphic FIT (capstone): adjust the SpiralModel's
// parameters so constraint points attain their target winding numbers, minimizing a
// radius/winding-consistency loss with AdamW. This is the spiral-v2 fit in miniature:
// the full version adds the flow lattice + gap logits + the richer loss vocabulary
// (patch/track radius+DT, winding annotations, lasagna spacing, sym-Dirichlet). Gradients
// here are finite-difference over a small global parameter set (dr + per-slice affine).
// See winding/CLAUDE.md, docs/research/spiral-v2.md.
#pragma once

#include "core/core.hpp"
#include "winding/spiral_model.hpp"

#include <span>
#include <vector>

namespace fenix::winding {

struct FitConstraint {
    Vec3f scroll_pt;
    f32 target_winding;
};

struct FitConfig {
    int iters = 400;
    f32 lr = 0.02f;
    f32 fd_eps = 1e-3f;  // finite-difference step
};

struct FitResult {
    f32 initial_loss = 0;
    f32 final_loss = 0;
    int iters = 0;
};

namespace detail {
// Parameter vector layout: [dr, a, b, c, d, ty, tx] of the SpiralModel (global affine).
inline void pack(const SpiralModel& m, std::span<f32> p) {
    p[0] = m.dr_per_winding;
    p[1] = m.affine.a;
    p[2] = m.affine.b;
    p[3] = m.affine.c;
    p[4] = m.affine.d;
    p[5] = m.affine.ty;
    p[6] = m.affine.tx;
}
inline void unpack(std::span<const f32> p, SpiralModel& m) {
    m.dr_per_winding = p[0];
    m.affine.a = p[1];
    m.affine.b = p[2];
    m.affine.c = p[3];
    m.affine.d = p[4];
    m.affine.ty = p[5];
    m.affine.tx = p[6];
}
inline f32 loss(const SpiralModel& m, std::span<const FitConstraint> cs) {
    f32 s = 0;
    for (const auto& c : cs) {
        const f32 e = m.winding_at(c.scroll_pt) - c.target_winding;
        s += e * e;
    }
    return cs.empty() ? 0.0f : s / static_cast<f32>(cs.size());
}
}  // namespace detail

// Fit `model` in place. The model's umbilicus/flow/gap are held fixed; the global affine +
// dr_per_winding are optimized to satisfy the constraints.
inline FitResult fit_spiral(SpiralModel& model, std::span<const FitConstraint> constraints,
                            FitConfig cfg = {}) {
    constexpr usize NP = 7;
    std::vector<f32> p(NP);
    detail::pack(model, p);

    FitResult r;
    r.initial_loss = detail::loss(model, constraints);

    AdamW opt(NP, {.lr = cfg.lr});
    std::vector<f32> g(NP);
    SpiralModel scratch = model;
    for (int it = 0; it < cfg.iters; ++it) {
        // Central finite-difference gradient over the small parameter set.
        for (usize i = 0; i < NP; ++i) {
            const f32 orig = p[i];
            p[i] = orig + cfg.fd_eps;
            detail::unpack(p, scratch);
            const f32 lp = detail::loss(scratch, constraints);
            p[i] = orig - cfg.fd_eps;
            detail::unpack(p, scratch);
            const f32 lm = detail::loss(scratch, constraints);
            p[i] = orig;
            g[i] = (lp - lm) / (2.0f * cfg.fd_eps);
        }
        opt.step(p, g);
    }
    detail::unpack(p, model);
    r.final_loss = detail::loss(model, constraints);
    r.iters = cfg.iters;
    return r;
}

}  // namespace fenix::winding
