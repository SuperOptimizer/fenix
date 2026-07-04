// winding/winding.hpp — the `winding` stage: the P2 REAL-DATA PILOT (winding/CLAUDE.md
// roadmap): corpus segment meshes -> cosegment (coupled patch-graph + Eulerian winding
// assignment) -> fit_bridge -> the diffeomorphic spiral fit. The first end-to-end spiral
// fit of a real scroll region; the tracer replaces "corpus meshes" as the patch source
// later without changing this driver.
//   fenix winding surf=<fxsurf>... umb=y,x|<umb.toml> [bridge=corpus|patch] [stride=4] [rounds=2]
//                 [eulerian=1] [iters_affine=250] [iters_flow=500] [flow=12] [bstride=6]
//                 [conf=0] [spacing=0] [holdout=0] [abands=0] [flowz=0] [out=<fxmodel>]
// holdout=K: the LAST K loaded meshes are excluded from the fit and scored afterwards —
// per held-out mesh, winding_at along its own unwrapped turn must be constant up to one
// gauge, so RMSE after subtracting the per-mesh median offset is a TRUE generalization
// number (the overfitting firewall for the fit, mirroring eval-set).
// bridge=corpus (default): each mesh is a MULTI-WRAP spiral segment — per-cell continuous
// winding from the unwrapped angular turn + a per-component Archimedean base gauge
// (corpus_bridge.hpp). bridge=patch: the tracer-fragment path (cosegment + one integer
// winding per patch, fit_bridge.hpp).
// umb: straight umbilicus axis at constant (y,x) — Paris4's axis is near-vertical; a
// curved umbilicus TOML replaces it later (annotate/). stride subsamples each mesh's uv
// grid (memory + graph cost); bstride subsamples bridge constraints on top of that.
#pragma once

#include "core/core.hpp"

#include "annotate/umbilicus.hpp"
#include "annotate/umbilicus_fit.hpp"
#include "io/surface.hpp"
#include "winding/corpus_bridge.hpp"
#include "winding/cosegment.hpp"
#include "winding/diffeo_fit.hpp"
#include "winding/fit.hpp"
#include "winding/fit_bridge.hpp"
#include "winding/flow.hpp"
#include "winding/model_io.hpp"
#include "winding/relax.hpp"
#include "winding/spiral_fit.hpp"
#include "winding/spiral_model.hpp"
#include "winding/transforms.hpp"
#include "winding/winding_field.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::winding {

namespace detail {
// uv-subsample a surface (every k-th cell) — bounds patch-graph and fit cost on
// multi-million-cell corpus meshes while keeping their full spatial footprint
inline Surface subsample_sheet(const Surface& s, s64 k) {
    if (k <= 1) return s;
    Surface d((s.nu - 1) / k + 1, (s.nv - 1) / k + 1);
    d.scale_u = s.scale_u * static_cast<f32>(k);
    d.scale_v = s.scale_v * static_cast<f32>(k);
    for (s64 v = 0; v < d.nv; ++v)
        for (s64 u = 0; u < d.nu; ++u)
            if (s.is_valid(u * k, v * k)) d.set(u, v, s.at(u * k, v * k));
    return d;
}
}  // namespace detail

inline Expected<int> run(std::span<const std::string_view> args, Context&) {
    s64 stride = 4, rounds = 2, eulerian = 1, iters_affine = 250, iters_flow = 500, flow = 12, flowz = 0,
        bstride = 6, holdout = 0, abands = 0;
    f64 umb_y = -1, umb_x = -1, conf = 0, spacing = 0;
    std::string out_path, bridge = "corpus", umb_toml;
    std::vector<std::string> surf_paths;
    for (const auto a : args) {
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("stride=", stride) || num("rounds=", rounds) || num("eulerian=", eulerian) ||
            num("iters_affine=", iters_affine) || num("iters_flow=", iters_flow) || num("flow=", flow) ||
            num("bstride=", bstride) || num("conf=", conf) || num("spacing=", spacing) ||
            num("holdout=", holdout) || num("abands=", abands) || num("flowz=", flowz))
            continue;
        if (a.starts_with("bridge=")) {
            bridge = std::string(a.substr(7));
            continue;
        }
        if (a.starts_with("umb=")) {
            const auto t = a.substr(4);
            const auto comma = t.find(',');
            if (comma != std::string_view::npos) {
                std::from_chars(t.data(), t.data() + comma, umb_y);
                std::from_chars(t.data() + comma + 1, t.data() + t.size(), umb_x);
            } else {
                umb_toml = std::string(t);  // curved axis from `fenix umbilicus`
            }
            continue;
        }
        if (a.starts_with("surf=")) {
            surf_paths.emplace_back(a.substr(5));
            continue;
        }
        if (a.starts_with("out=")) {
            out_path = std::string(a.substr(4));
            continue;
        }
        return err(Errc::invalid_argument, "winding: unknown arg '" + std::string(a) + "'");
    }
    if (surf_paths.empty() || (umb_y < 0 && umb_toml.empty()))
        return err(Errc::invalid_argument,
                   "usage: winding surf=<fxsurf>... umb=y,x [stride=] [rounds=] [eulerian=] "
                   "[iters_affine=] [iters_flow=] [flow=] [bstride=] [conf=] [out=<fxmodel>]");

    // 1) load + subsample the patch sources (corpus meshes for the pilot; tracer output later)
    std::vector<Surface> sheets;
    f32 z_lo = 1e30f, z_hi = -1e30f;
    for (const auto& sp : surf_paths) {
        auto s = io::read_fxsurf(sp);
        if (!s) return std::unexpected(s.error());
        Surface d = detail::subsample_sheet(*s, stride);
        if (d.valid_count() < 16) continue;
        for (s64 v = 0; v < d.nv; ++v)
            for (s64 u = 0; u < d.nu; ++u)
                if (d.is_valid(u, v)) {
                    z_lo = std::min(z_lo, d.at(u, v).z);
                    z_hi = std::max(z_hi, d.at(u, v).z);
                }
        sheets.push_back(std::move(d));
    }
    std::vector<Surface> held_out;
    if (holdout > 0 && holdout < static_cast<s64>(sheets.size()) - 1) {
        held_out.assign(std::make_move_iterator(sheets.end() - holdout), std::make_move_iterator(sheets.end()));
        sheets.resize(sheets.size() - static_cast<usize>(holdout));
    }
    if (sheets.size() < 2) return err(Errc::invalid_argument, "winding: need >=2 usable sheets");
    log(LogLevel::info, "winding: {} sheets (stride {}), {} held out, z [{:.0f},{:.0f}]", sheets.size(),
        stride, held_out.size(), z_lo, z_hi);

    // 2) the axis: a curved umbilicus TOML (umb=<path>, from `fenix umbilicus`) or a
    // straight fallback at constant (y,x)
    annotate::Umbilicus umb;
    if (!umb_toml.empty()) {
        auto lu = annotate::load_umbilicus(umb_toml);
        if (!lu) return std::unexpected(lu.error());
        umb = std::move(*lu);
        log(LogLevel::info, "winding: curved umbilicus from {} ({} control points)", umb_toml, umb.z.size());
    } else {
        umb.z = {z_lo, z_hi};
        umb.y = {static_cast<f32>(umb_y), static_cast<f32>(umb_y)};
        umb.x = {static_cast<f32>(umb_x), static_cast<f32>(umb_x)};
    }

    // 3+4) constraints, by input kind. corpus: per-cell continuous winding from each
    // mesh's own unwrapped turn (multi-wrap segments). patch: cosegment + integer wraps.
    f32 fit_spacing = 0;
    std::vector<FitConstraint> targets;
    std::vector<CoWindingGroup> groups;
    if (bridge == "corpus") {
        CorpusBridgeParams cbp;
        cbp.stride = static_cast<int>(bstride);
        cbp.spacing = static_cast<f32>(spacing);
        CorpusBridgeOut cb = corpus_to_constraints(sheets, umb, cbp);
        log(LogLevel::info,
            "winding: corpus bridge — {} components, spacing {:.1f} vox/wrap, bases [{:.1f},{:.1f}], {} targets",
            cb.components, cb.spacing, cb.base_lo, cb.base_hi, cb.targets.size());
        fit_spacing = cb.spacing;
        targets = std::move(cb.targets);
    } else if (bridge == "patch") {
        segment::PatchGraphParams gp;
        CosegParams cp;
        cp.rounds = static_cast<int>(rounds);
        cp.eulerian = eulerian != 0;
        const CosegReport rep = cosegment_refine(sheets, umb, gp, cp);
        log(LogLevel::info,
            "winding: cosegment — spacing {:.1f}, clusters {}, wraps [{},{}], conflicts {}, monotonicity {:.4f}",
            rep.spacing, rep.clusters, rep.wrap_lo, rep.wrap_hi, rep.conflicts, rep.monotonicity);
        segment::PatchGraph g = segment::build_patch_graph(sheets, umb, gp);
        segment::merge_same_sheet(g);
        if (cp.eulerian) {
            const WindingField ewf = build_eulerian_winding_field(g.patches, cp.full, g.spacing, cp.efield);
            assign_windings_from_field(g, ewf);
        } else {
            segment::assign_windings(g);
        }
        BridgeParams bp;
        bp.stride = static_cast<int>(bstride);
        bp.conf_min = static_cast<f32>(conf);
        BridgeOut b = patches_to_constraints(g, bp);
        fit_spacing = rep.spacing;
        targets = std::move(b.targets);
        groups = std::move(b.groups);
    } else {
        return err(Errc::invalid_argument, "winding: bridge must be corpus|patch");
    }
    if (targets.size() < 64) return err(Errc::invalid_argument, "winding: too few fit constraints");

    // 5) the diffeomorphic fit: dr seeded from the measured wrap spacing; flow lattice over
    // the umbilicus-centered bbox of the constraints (the frame the flow acts in)
    SpiralModel model;
    model.dr_per_winding = fit_spacing > 0 ? fit_spacing : 20.0f;
    model.umbilicus = umb;
    Vec3f lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (const auto& t : targets) {
        const Vec3f c = umb.center(t.scroll_pt.z);
        lo.z = std::min(lo.z, t.scroll_pt.z);
        hi.z = std::max(hi.z, t.scroll_pt.z);
        lo.y = std::min(lo.y, t.scroll_pt.y - c.y);
        hi.y = std::max(hi.y, t.scroll_pt.y - c.y);
        lo.x = std::min(lo.x, t.scroll_pt.x - c.x);
        hi.x = std::max(hi.x, t.scroll_pt.x - c.x);
    }
    const Extent3 fd{flowz > 0 ? flowz : std::max<s64>(4, flow / 2), flow, flow};
    model.flow.vz = Volume<f32>::zeros(fd);
    model.flow.vy = Volume<f32>::zeros(fd);
    model.flow.vx = Volume<f32>::zeros(fd);
    model.flow.lat_lo = lo;
    model.flow.lat_scale = Vec3f{static_cast<f32>(fd.z - 1) / std::max(1.0f, hi.z - lo.z),
                                 static_cast<f32>(fd.y - 1) / std::max(1.0f, hi.y - lo.y),
                                 static_cast<f32>(fd.x - 1) / std::max(1.0f, hi.x - lo.x)};
    model.flow_steps = 8;
    model.has_flow = true;

    DiffeoFitConfig fc;
    fc.iters_affine = static_cast<int>(iters_affine);
    fc.iters_flow = static_cast<int>(iters_flow);
    fc.flow_dims = fd;
    fc.domain_lo = lo;
    fc.domain_hi = hi;
    fc.continuous = bridge == "corpus";  // corpus targets are continuous windings
    fc.affine_bands = static_cast<int>(abands);
    const FitResult res = fit_spiral_diffeo(model, targets, groups, fc);

    // the honest per-cell check: winding residual on the targets through the fitted model
    f64 se = 0;
    s64 n = 0;
    for (const auto& t : targets) {
        const f64 w = fc.continuous ? model.winding_cont(t.scroll_pt) : model.winding_at(t.scroll_pt);
        const f64 e = w - static_cast<f64>(t.target_winding);
        se += e * e;
        ++n;
    }
    log(LogLevel::info,
        "winding: FIT — loss {:.4f} -> {:.4f} ({} iters) | dr {:.2f} vox/winding | target RMSE {:.3f} "
        "windings over {} cells",
        res.initial_loss,
        res.final_loss,
        res.iters,
        model.dr_per_winding,
        n ? std::sqrt(se / static_cast<f64>(n)) : -1.0,
        n);
    if (!held_out.empty()) {
        const HoldoutScore hs = score_holdout(model, held_out, umb, static_cast<int>(bstride));
        log(LogLevel::info,
            "winding: HOLDOUT — RMSE {:.3f} windings over {} cells ({} components, {} meshes never seen "
            "by the fit)",
            hs.rmse, hs.cells, hs.components, held_out.size());
    }
    if (!out_path.empty()) {
        if (auto w = write_fxmodel(out_path, model); !w) return std::unexpected(w.error());
        log(LogLevel::info, "winding: model -> {}", out_path);
    }
    return 0;
}

}  // namespace fenix::winding

FENIX_REGISTER_STAGE(winding,
                     "diffeomorphic spiral fit pilot (meshes -> cosegment -> bridge -> fit)",
                     ::fenix::winding::run)
