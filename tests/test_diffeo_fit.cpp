// test_diffeo_fit.cpp — the differentiable diffeomorphic spiral fit (winding/diffeo_fit.hpp).
//  (1) GRADIENT GATE: the analytic reverse-mode flow-lattice gradient (flow_point_backward + the
//      winding readout) must match central finite-difference of the loss to < 5% — this gates the
//      whole fit (a wrong RK4 adjoint or lattice scatter silently breaks optimization).
//  (2) RECOVERY: build a deformed spiral with windings known BY CONSTRUCTION (ideal canonical spiral
//      pushed through a ground-truth affine + flow), fit from identity, and require it to recover the
//      winding field + stay invertible.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "winding/diffeo_fit.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

using namespace fenix;
using namespace fenix::winding;

namespace {
constexpr f32 kPi = std::numbers::pi_v<f32>;

// A model with empty umbilicus, identity affine, and a smooth non-uniform flow lattice on a box.
SpiralModel make_flow_model(Extent3 fd, Vec3f lo, Vec3f hi, f32 amp, int steps) {
    SpiralModel m;
    m.dr_per_winding = 8.0f;
    m.flow.vz = Volume<f32>::zeros(fd);
    m.flow.vy = Volume<f32>::zeros(fd);
    m.flow.vx = Volume<f32>::zeros(fd);
    m.flow.lat_lo = lo;
    m.flow.lat_scale = Vec3f{static_cast<f32>(fd.z - 1) / (hi.z - lo.z), static_cast<f32>(fd.y - 1) / (hi.y - lo.y),
                             static_cast<f32>(fd.x - 1) / (hi.x - lo.x)};
    m.flow_steps = steps;
    m.has_flow = true;
    auto vy = m.flow.vy.view(), vx = m.flow.vx.view();
    for (s64 iz = 0; iz < fd.z; ++iz)
        for (s64 iy = 0; iy < fd.y; ++iy)
            for (s64 ix = 0; ix < fd.x; ++ix) {
                const f32 wy = lo.y + static_cast<f32>(iy) / m.flow.lat_scale.y;
                const f32 wx = lo.x + static_cast<f32>(ix) / m.flow.lat_scale.x;
                vy(iz, iy, ix) = amp * std::sin(wx * 0.05f);
                vx(iz, iy, ix) = amp * std::cos(wy * 0.04f);
            }
    return m;
}
}  // namespace

TEST(diffeo_fit_flow_gradient_matches_finite_difference) {
    // small lattice + a few off-grid constraints; nonzero flow so the RK4 adjoint (steps>1) is exercised.
    SpiralModel m = make_flow_model(Extent3{4, 5, 5}, Vec3f{0, -40, -40}, Vec3f{20, 40, 40}, 2.0f, 4);
    std::vector<FitConstraint> cs;
    for (int k = 0; k < 6; ++k) {
        const f32 th = -2.0f + 0.6f * static_cast<f32>(k);
        const f32 r = 12.0f + 4.0f * static_cast<f32>(k);
        cs.push_back({Vec3f{6.0f + static_cast<f32>(k), r * std::sin(th), r * std::cos(th)}, static_cast<f32>(k % 3)});
    }
    const std::span<const FitConstraint> wcs(cs);
    const std::span<const CoWindingGroup> none{};

    // analytic flow-lattice gradient of L = mean (W-t)^2 (data term only; no regularizers).
    const usize N = static_cast<usize>(m.flow.vy.dims().count());
    ::fenix::winding::detail::FitGrad acc;
    acc.gz.assign(N, 0.0);
    acc.gy.assign(N, 0.0);
    acc.gx.assign(N, 0.0);
    const f64 M = static_cast<f64>(cs.size());
    for (const FitConstraint& c : cs) {
        const f64 W = m.winding_at(c.scroll_pt);
        ::fenix::winding::detail::winding_backward(m, c.scroll_pt, (2.0 / M) * (W - c.target_winding), acc, true, 2.0f);
    }

    // central FD of the loss w.r.t. each vy/vx lattice value.
    const f32 eps = 1e-2f;
    f64 max_rel = 0;
    int checked = 0;
    auto check_axis = [&](Volume<f32>& vol, const std::vector<f64>& g) {
        f32* d = vol.view().data();
        for (usize i = 0; i < N; ++i) {
            const f32 o = d[i];
            d[i] = o + eps;
            const f64 lp = ::fenix::winding::detail::fit_loss(m, wcs, none, 0.0f, 2.0f);
            d[i] = o - eps;
            const f64 lm = ::fenix::winding::detail::fit_loss(m, wcs, none, 0.0f, 2.0f);
            d[i] = o;
            const f64 fd = (lp - lm) / (2.0 * eps);
            if (std::abs(fd) < 1e-6 && std::abs(g[i]) < 1e-6) continue;  // both ~0: skip
            const f64 rel = std::abs(fd - g[i]) / (std::abs(fd) + std::abs(g[i]) + 1e-9);
            max_rel = std::max(max_rel, rel);
            ++checked;
        }
    };
    check_axis(m.flow.vy, acc.gy);
    check_axis(m.flow.vx, acc.gx);
    std::printf("  [flow-grad: checked %d cells, max rel-err %.4f]\n", checked, max_rel);
    REQUIRE(checked > 10);
    CHECK(max_rel < 5e-2);

    // dr analytic-vs-FD too.
    const f32 deps = 1e-2f, o = m.dr_per_winding;
    m.dr_per_winding = o + deps;
    const f64 lp = ::fenix::winding::detail::fit_loss(m, wcs, none, 0.0f, 2.0f);
    m.dr_per_winding = o - deps;
    const f64 lm = ::fenix::winding::detail::fit_loss(m, wcs, none, 0.0f, 2.0f);
    m.dr_per_winding = o;
    const f64 dr_fd = (lp - lm) / (2.0 * deps);
    CHECK(std::abs(dr_fd - acc.g_dr) / (std::abs(dr_fd) + std::abs(acc.g_dr) + 1e-9) < 5e-2);
}

TEST(diffeo_fit_recovers_synthetic_spiral) {
    // ground-truth deformation: nonzero affine + a smooth flow; spiral pitch dr=8.
    SpiralModel truth = make_flow_model(Extent3{6, 16, 16}, Vec3f{0, -75, -75}, Vec3f{40, 75, 75}, 2.5f, 8);
    truth.affine = AffineYX{0.08f, 0.12f, -0.07f, 0.04f, 2.0f, -1.0f};
    const f32 dr = truth.dr_per_winding, two_pi = 2.0f * kPi;

    // points with winding known BY CONSTRUCTION: canonical q on integer winding w, pushed to scroll.
    auto canon_to_scroll = [&](int w, f32 th, f32 z) {
        const f32 r = dr * (static_cast<f32>(w) + th / two_pi);  // r_ideal s.t. shifted_radius = w*dr
        return truth.to_scroll(Vec3f{z, r * std::sin(th), r * std::cos(th)});
    };
    std::vector<FitConstraint> cs, held;
    std::vector<CoWindingGroup> groups;
    for (int w = 1; w <= 6; ++w) {
        CoWindingGroup grp;
        for (int zi = 0; zi < 5; ++zi) {
            const f32 z = 4.0f + 7.0f * static_cast<f32>(zi);
            for (int ai = 0; ai < 24; ++ai) {
                const f32 th = -kPi + 0.25f + (two_pi - 0.5f) * static_cast<f32>(ai) / 23.0f;
                const Vec3f p = canon_to_scroll(w, th, z);
                cs.push_back({p, static_cast<f32>(w)});
                grp.points.push_back(p);
            }
            for (int ai = 0; ai < 10; ++ai) {  // held-out angles
                const f32 th = -kPi + 0.6f + (two_pi - 1.2f) * (static_cast<f32>(ai) + 0.5f) / 10.0f;
                held.push_back({canon_to_scroll(w, th, z), static_cast<f32>(w)});
            }
        }
        groups.push_back(std::move(grp));
    }

    SpiralModel fit;  // empty umbilicus, identity affine, dr=8 default; flow allocated by the fit
    fit.dr_per_winding = 8.0f;
    DiffeoFitConfig cfg;
    cfg.flow_dims = Extent3{6, 16, 16};
    cfg.iters_affine = 300;
    cfg.iters_flow = 700;
    cfg.lr_affine = 0.05f;
    cfg.lr_flow = 0.08f;
    cfg.lambda_cowind = 0.5f;
    cfg.lambda_l2 = 1e-4f;
    cfg.lambda_smooth = 1e-3f;
    cfg.flow_steps = 8;
    const FitResult r = fit_spiral_diffeo(fit, cs, groups, cfg);

    f64 se = 0;
    for (const FitConstraint& c : cs) {
        const f64 e = fit.winding_at(c.scroll_pt) - c.target_winding;
        se += e * e;
    }
    const f64 rmse = std::sqrt(se / static_cast<f64>(cs.size()));
    f64 he = 0;
    for (const FitConstraint& c : held) he += std::abs(fit.winding_at(c.scroll_pt) - c.target_winding);
    he /= static_cast<f64>(held.size());
    f64 max_rt = 0;
    for (usize i = 0; i < cs.size(); i += 17) {
        const Vec3f rt = fit.to_scroll(fit.to_canonical(cs[i].scroll_pt));
        max_rt = std::max(max_rt, static_cast<f64>(norm(rt - cs[i].scroll_pt)));
    }
    std::printf("  [recover: loss %.4f -> %.4f | winding RMSE %.3f | held-out MAE %.3f | invert %.3f]\n",
                r.initial_loss, r.final_loss, rmse, he, max_rt);
    CHECK(r.final_loss < 0.05f * r.initial_loss);
    CHECK(r.final_loss < 0.02f);
    CHECK(rmse < 0.1);
    CHECK(he < 0.25);
    CHECK(max_rt < 0.5);
}

TEST(band_affine_gradient_matches_finite_difference) {
    // Nonzero bands + a few constraints: the per-band adjoint chain (Mi_b transpose + logit
    // contraction) must match central FD of the data-term loss.
    SpiralModel m;
    m.dr_per_winding = 8.0f;
    m.affine = {.a = 0.02f, .b = -0.01f, .c = 0.015f, .d = -0.01f, .ty = 0.5f, .tx = -0.3f};
    m.affine_bands.z0 = 0;
    m.affine_bands.dz = 10;
    m.affine_bands.bands = {{.a = 0.03f, .b = 0.01f, .c = -0.02f, .d = 0.02f, .ty = 1.0f, .tx = -0.7f},
                            {.a = -0.02f, .b = 0.02f, .c = 0.01f, .d = -0.03f, .ty = -0.5f, .tx = 0.9f}};
    std::vector<FitConstraint> cs;
    for (int k = 0; k < 8; ++k) {
        const f32 th = -2.5f + 0.7f * static_cast<f32>(k);
        const f32 r = 10.0f + 3.0f * static_cast<f32>(k);
        cs.push_back({Vec3f{2.5f * static_cast<f32>(k), r * std::sin(th), r * std::cos(th)},
                      0.3f * static_cast<f32>(k)});
    }
    const std::span<const FitConstraint> wcs(cs);
    const std::span<const CoWindingGroup> none{};
    const bool cont = true;  // exercise the continuous readout path too

    ::fenix::winding::detail::FitGrad acc;
    acc.Gb.assign(12, 0.0);
    const f64 M = static_cast<f64>(cs.size());
    for (const FitConstraint& c : cs) {
        const f64 seed = 2.0 / M * (static_cast<f64>(m.winding_cont(c.scroll_pt)) - c.target_winding);
        ::fenix::winding::detail::winding_backward(m, c.scroll_pt, seed, acc, false, 2.0f, cont);
    }
    // contract logit adjoints exactly like the fit loop
    const f32 eps = 1e-3f;
    for (usize k = 0; k < 2; ++k) {
        AffineYX& ab = m.affine_bands.bands[k];
        f32* prm[6] = {&ab.a, &ab.b, &ab.c, &ab.d, &ab.ty, &ab.tx};
        for (int j = 0; j < 6; ++j) {
            f64 analytic;
            if (j < 4) {
                f32 a = ab.a, b = ab.b, c2 = ab.c, d = ab.d;
                f32* tgt = j == 0 ? &a : (j == 1 ? &b : (j == 2 ? &c2 : &d));
                const f32 o = *tgt;
                *tgt = o + eps;
                const Mat2 mp = expm2(-a, -b, -c2, -d);
                *tgt = o - eps;
                const Mat2 mm = expm2(-a, -b, -c2, -d);
                *tgt = o;
                const Mat2 dm{(mp.m00 - mm.m00) / (2 * eps), (mp.m01 - mm.m01) / (2 * eps),
                              (mp.m10 - mm.m10) / (2 * eps), (mp.m11 - mm.m11) / (2 * eps)};
                const f64* gb = acc.Gb.data() + 6 * k;
                analytic = gb[0] * dm.m00 + gb[1] * dm.m01 + gb[2] * dm.m10 + gb[3] * dm.m11;
            } else {
                analytic = acc.Gb[6 * k + static_cast<usize>(j)];
            }
            const f32 o = *prm[j];
            *prm[j] = o + eps;
            const f64 lp = ::fenix::winding::detail::fit_loss(m, wcs, none, 0, 2.0f, {}, 1.0f, cont);
            *prm[j] = o - eps;
            const f64 lm = ::fenix::winding::detail::fit_loss(m, wcs, none, 0, 2.0f, {}, 1.0f, cont);
            *prm[j] = o;
            const f64 fd = (lp - lm) / (2.0 * eps);
            REQUIRE(std::isfinite(analytic));
            if (std::abs(fd) > 1e-6)
                CHECK(std::abs(analytic - fd) / std::max(1e-6, std::abs(fd)) < 0.05);
        }
    }
}

TEST(band_affine_capacity_beats_global_only) {
    // Ground truth: a spiral whose CENTER shifts with z (piecewise) — expressible by per-z-band
    // translations, inexpressible by one global affine. Bands must fit far better.
    constexpr f32 two_pi = 2.0f * kPi;
    std::vector<FitConstraint> cs;
    for (int zi = 0; zi < 4; ++zi)
        for (int k = 0; k < 60; ++k) {
            const f32 w = 1.0f + 0.05f * static_cast<f32>(k);
            const f32 th = two_pi * w;
            const f32 r = 8.0f * w;
            const f32 sy = 6.0f * static_cast<f32>(zi % 2 ? 1 : -1);   // center jumps per z-band
            const f32 sx = 4.0f * static_cast<f32>(zi >= 2 ? 1 : -1);
            cs.push_back({Vec3f{static_cast<f32>(10 * zi + 5), sy + r * std::sin(th), sx + r * std::cos(th)}, w});
        }
    auto run = [&](int bands) {
        SpiralModel m;
        m.dr_per_winding = 8.0f;
        DiffeoFitConfig fc;
        fc.iters_affine = 150;
        fc.iters_flow = 300;
        fc.flow_dims = {4, 6, 6};
        fc.continuous = true;
        fc.affine_bands = bands;
        fc.lambda_affine_smooth = 0;  // the jump IS the signal here
        fit_spiral_diffeo(m, cs, {}, fc);
        return ::fenix::winding::detail::fit_loss(m, cs, {}, 0, 2.0f, {}, 1.0f, true);
    };
    const f64 loss_global = run(0);
    const f64 loss_banded = run(4);
    std::printf("  [capacity: global %.4f banded %.4f]\n", loss_global, loss_banded);
    REQUIRE(std::isfinite(loss_banded));
    CHECK(loss_banded < loss_global * 0.5);  // bands express the per-z shift; global cannot
}
