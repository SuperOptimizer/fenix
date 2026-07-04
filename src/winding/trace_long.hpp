// winding/trace_long.hpp — SPIRAL-GUIDED LONG TRACING: grow ONE sheet as far as the data
// allows, gated by the fitted spiral model so it can never switch wraps. The published
// GP segments' defining quality is length + coherence along a single sheet; the local
// grower's guards (CT-valley, injectivity) are short-range, but the fitted .fxmodel knows
// the CONTINUOUS winding of every point — a true sheet holds it constant, a wrap-hop
// jumps it by ~1. This stage closes that loop: model-gated single-seed growth on a large
// in-core region, one big continuous chart out.
//   fenix trace-long pred=<fxvol> ct=<fxvol> model=<fxmodel> origin=z,y,x seed=z,y,x
//                    out=<fxsurf> [grid=3000] [step=2] [wtol=0.5] [wjump=0.4] [thresh=0.10]
//                    [barrier=0.12] [bridge=4] [arap_tol=0.15] [maxgen=100000]
// origin: the block's absolute corner (model lives in scroll coords). seed: ABSOLUTE.
// Reports: cells, spatial extent, winding span (should stay < wtol), surf-qc-style
// evidence is the caller's next step (surf-qc the output).
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/surface.hpp"
#include "segment/grow.hpp"
#include "winding/model_io.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <span>
#include <string>
#include <string_view>

namespace fenix::winding {

inline Expected<int> run_trace_long(std::span<const std::string_view> args, Context&) {
    std::string pred_path, ct_path, model_path, out_path;
    f64 oz = -1, oy = -1, ox = -1, sz = -1, sy = -1, sx = -1;
    segment::GrowParams gp;
    gp.surf_thresh = 0.10f;
    gp.ct_barrier = 0.12f;
    gp.max_bridge = 4;
    gp.winding_tol = 0.5f;
    s64 grid = 3000, maxgen = 100000;
    for (const auto a : args) {
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            using V = std::remove_reference_t<decltype(v)>;
            if constexpr (std::is_floating_point_v<V>) {
                f64 d = 0;
                std::from_chars(t.data(), t.data() + t.size(), d);
                v = static_cast<V>(d);
            } else {
                std::from_chars(t.data(), t.data() + t.size(), v);
            }
            return true;
        };
        auto triple = [&](std::string_view key, f64& z, f64& y, f64& x) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            const auto c1 = t.find(','), c2 = t.rfind(',');
            if (c1 == std::string_view::npos || c2 == c1) return false;
            std::from_chars(t.data(), t.data() + c1, z);
            std::from_chars(t.data() + c1 + 1, t.data() + c2, y);
            std::from_chars(t.data() + c2 + 1, t.data() + t.size(), x);
            return true;
        };
        if (num("grid=", grid) || num("step=", gp.step) || num("wtol=", gp.winding_tol) ||
            num("wjump=", gp.winding_jump) ||
            num("thresh=", gp.surf_thresh) || num("barrier=", gp.ct_barrier) ||
            num("bridge=", gp.max_bridge) || num("arap_tol=", gp.arap_tol) || num("maxgen=", maxgen))
            continue;
        if (triple("origin=", oz, oy, ox) || triple("seed=", sz, sy, sx)) continue;
        if (a.starts_with("pred=")) { pred_path = std::string(a.substr(5)); continue; }
        if (a.starts_with("ct=")) { ct_path = std::string(a.substr(3)); continue; }
        if (a.starts_with("model=")) { model_path = std::string(a.substr(6)); continue; }
        if (a.starts_with("out=")) { out_path = std::string(a.substr(4)); continue; }
        return err(Errc::invalid_argument, "trace-long: unknown arg '" + std::string(a) + "'");
    }
    if (pred_path.empty() || out_path.empty() || oz < 0 || sz < 0)
        return err(Errc::invalid_argument,
                   "usage: trace-long pred=<fxvol> ct=<fxvol> model=<fxmodel> origin=z,y,x seed=z,y,x "
                   "out=<fxsurf> [grid=3000] [step=2] [wtol=0.35] [thresh=] [barrier=] [bridge=] "
                   "[arap_tol=] [maxgen=]");

    auto pa = codec::VolumeArchive::open(pred_path);
    if (!pa) return std::unexpected(pa.error());
    auto pred = pa->read_volume_as<u8>();
    if (!pred) return std::unexpected(pred.error());
    Volume<u8> ct;
    if (!ct_path.empty()) {
        auto ca = codec::VolumeArchive::open(ct_path);
        if (!ca) return std::unexpected(ca.error());
        auto cv = ca->read_volume_as<u8>();
        if (!cv) return std::unexpected(cv.error());
        ct = std::move(*cv);
    }
    if (gp.ct_barrier > 0 && ct.dims().count() == 0) gp.ct_barrier = 0;

    const Vec3f org{static_cast<f32>(oz), static_cast<f32>(oy), static_cast<f32>(ox)};
    SpiralModel model;
    if (!model_path.empty()) {
        auto m = read_fxmodel(model_path);
        if (!m) return std::unexpected(m.error());
        model = std::move(*m);
        gp.winding_fn = [&model, org](Vec3f p) { return model.winding_at(p + org); };  // STEPPED (gate branch-snaps)
        log(LogLevel::info, "trace-long: model winding gate ON (wtol {:.2f})", gp.winding_tol);
    } else {
        gp.winding_tol = 0;  // no model -> no gate
    }

    gp.grid = static_cast<int>(grid);
    gp.max_gen = static_cast<int>(maxgen);
    const Vec3f seed{static_cast<f32>(sz - oz), static_cast<f32>(sy - oy), static_cast<f32>(sx - ox)};
    const Extent3 D = pred->dims();
    if (seed.z < 0 || seed.y < 0 || seed.x < 0 || seed.z >= static_cast<f32>(D.z) ||
        seed.y >= static_cast<f32>(D.y) || seed.x >= static_cast<f32>(D.x))
        return err(Errc::invalid_argument, "trace-long: seed outside the block");

    log(LogLevel::info, "trace-long: block {}x{}x{}, seed local ({:.0f},{:.0f},{:.0f}), grid {}", D.z,
        D.y, D.x, seed.z, seed.y, seed.x, grid);
    const segment::NormalField nf = segment::compute_normal_field<u8>(pred->view(), 8);
    Surface S = segment::grow_surface<u8>(pred->view(),
                                          ct.dims().count() > 0 ? ct.view() : VolumeView<const u8>{}, nf,
                                          seed, gp);
    const s64 nvalid = S.valid_count();
    if (nvalid < 100) return err(Errc::invalid_argument, "trace-long: growth died near the seed");

    // report: extent + winding span through the model (the coherence proof)
    Vec3f lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    f32 w_lo = 1e30f, w_hi = -1e30f;
    for (usize i = 0; i < S.coord.size(); ++i) {
        if (!S.valid[i]) continue;
        const Vec3f p = S.coord[i];
        lo = Vec3f{std::min(lo.z, p.z), std::min(lo.y, p.y), std::min(lo.x, p.x)};
        hi = Vec3f{std::max(hi.z, p.z), std::max(hi.y, p.y), std::max(hi.x, p.x)};
        if (gp.winding_fn) {
            const f32 w = gp.winding_fn(p);
            if (std::abs(w) < 1e30f) {
                w_lo = std::min(w_lo, w);
                w_hi = std::max(w_hi, w);
            }
        }
        S.coord[i] = p + org;  // write in ABSOLUTE coords (surf-qc / view / eval ready)
    }
    log(LogLevel::info,
        "trace-long: {} cells, extent {:.0f}x{:.0f}x{:.0f} vox, winding span {:.3f} (gate {:.2f})",
        nvalid, hi.z - lo.z, hi.y - lo.y, hi.x - lo.x,
        gp.winding_fn ? (w_hi - w_lo) : -1.0f, gp.winding_tol);
    if (auto w = io::write_fxsurf(out_path, S); !w) return std::unexpected(w.error());
    log(LogLevel::info, "trace-long: -> {}", out_path);
    return 0;
}

}  // namespace fenix::winding

namespace {
[[maybe_unused]] const int fenix_stage_trace_long = ::fenix::register_stage(::fenix::Stage{
    "trace-long", "spiral-guided long tracing: one sheet, model-gated, no wrap switching",
    ::fenix::winding::run_trace_long});
}  // namespace
