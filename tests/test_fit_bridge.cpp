// test_fit_bridge.cpp — the segment->winding bridge (winding/fit_bridge.hpp) end to end: build patches
// on a known deformed spiral (each patch = one wrap, GT winding), turn them into fit constraints, fit
// the diffeomorphic model from identity, and require it to recover the winding field. Validates that the
// fit ingests segment's PatchGraph output (the path the real pipeline takes).
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "winding/fit_bridge.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using namespace fenix;
using namespace fenix::winding;

namespace {
constexpr f32 kPi = std::numbers::pi_v<f32>;

// affine + smooth-flow deformed spiral (ground truth), pitch dr=8.
SpiralModel make_truth() {
    SpiralModel m;
    m.dr_per_winding = 8.0f;
    const Extent3 fd{6, 16, 16};
    const Vec3f lo{0, -75, -75}, hi{40, 75, 75};
    m.flow.vz = Volume<f32>::zeros(fd);
    m.flow.vy = Volume<f32>::zeros(fd);
    m.flow.vx = Volume<f32>::zeros(fd);
    m.flow.lat_lo = lo;
    m.flow.lat_scale = Vec3f{static_cast<f32>(fd.z - 1) / (hi.z - lo.z), static_cast<f32>(fd.y - 1) / (hi.y - lo.y),
                             static_cast<f32>(fd.x - 1) / (hi.x - lo.x)};
    m.flow_steps = 8;
    m.has_flow = true;
    auto vy = m.flow.vy.view(), vx = m.flow.vx.view();
    for (s64 iz = 0; iz < fd.z; ++iz)
        for (s64 iy = 0; iy < fd.y; ++iy)
            for (s64 ix = 0; ix < fd.x; ++ix) {
                const f32 wy = lo.y + static_cast<f32>(iy) / m.flow.lat_scale.y;
                const f32 wx = lo.x + static_cast<f32>(ix) / m.flow.lat_scale.x;
                vy(iz, iy, ix) = 2.5f * std::sin(wx * 0.05f);
                vx(iz, iy, ix) = 2.5f * std::cos(wy * 0.04f);
            }
    m.affine = AffineYX{0.08f, 0.12f, -0.07f, 0.04f, 2.0f, -1.0f};
    return m;
}
}  // namespace

TEST(fit_bridge_recovers_windings_from_patches) {
    const SpiralModel truth = make_truth();
    const f32 dr = truth.dr_per_winding, two_pi = 2.0f * kPi;
    auto c2s = [&](int w, f32 th, f32 z) {
        const f32 r = dr * (static_cast<f32>(w) + th / two_pi);
        return truth.to_scroll(Vec3f{z, r * std::sin(th), r * std::cos(th)});
    };

    // one Patch per wrap (cells over angle x z), GT winding -> a PatchGraph the bridge consumes.
    segment::PatchGraph g;
    std::vector<FitConstraint> gt;  // for evaluation only
    for (int w = 1; w <= 6; ++w) {
        segment::Patch p;
        p.id = w;
        p.wrap = w;
        for (int zi = 0; zi < 5; ++zi) {
            const f32 z = 4.0f + 7.0f * static_cast<f32>(zi);
            for (int ai = 0; ai < 20; ++ai) {
                const f32 th = -kPi + 0.3f + (two_pi - 0.6f) * static_cast<f32>(ai) / 19.0f;
                const Vec3f q = c2s(w, th, z);
                p.pos.push_back(q);
                p.conf.push_back(1.0f);
                gt.push_back({q, static_cast<f32>(w)});
            }
        }
        g.patches.push_back(std::move(p));
    }

    BridgeParams bp;
    bp.conf_min = 1.0f;
    bp.stride = 1;
    const BridgeOut b = patches_to_constraints(g, bp);
    REQUIRE(b.targets.size() > 100);
    REQUIRE(b.groups.size() == 6);

    SpiralModel fit;
    fit.dr_per_winding = 8.0f;
    DiffeoFitConfig cfg;
    cfg.flow_dims = Extent3{6, 16, 16};
    cfg.iters_affine = 300;
    cfg.iters_flow = 700;
    cfg.lr_affine = 0.05f;
    cfg.lr_flow = 0.08f;
    cfg.lambda_cowind = 0.5f;
    cfg.flow_steps = 8;
    const FitResult r = fit_spiral_diffeo(fit, b.targets, b.groups, cfg);

    f64 se = 0;
    for (const FitConstraint& c : gt) {
        const f64 e = fit.winding_at(c.scroll_pt) - c.target_winding;
        se += e * e;
    }
    const f64 rmse = std::sqrt(se / static_cast<f64>(gt.size()));
    std::printf("  [bridge: %zu targets, %zu groups | loss %.4f -> %.4f | winding RMSE %.3f]\n",
                b.targets.size(), b.groups.size(), r.initial_loss, r.final_loss, rmse);
    CHECK(r.final_loss < 0.05f * r.initial_loss);
    CHECK(rmse < 0.1);
}
