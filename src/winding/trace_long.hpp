// winding/trace_long.hpp — SPIRAL-GUIDED LONG TRACING: grow ONE sheet as far as the data
// allows, gated by the fitted spiral model so it can never switch wraps. The published
// GP segments' defining quality is length + coherence along a single sheet; the local
// grower's guards (CT-valley, injectivity) are short-range, but the fitted .fxmodel knows
// the CONTINUOUS winding of every point — a true sheet holds it constant, a wrap-hop
// jumps it by ~1. This stage closes that loop: model-gated single-seed growth on a large
// in-core region, one big continuous chart out.
//   in-core:  trace-long pred=<fxvol> ct=<fxvol> model=<fxmodel> origin=z,y,x seed=z,y,x
//                    out=<fxsurf> [grid=3000] [step=2] [wtol=0.5] [wjump=0.4] [thresh=0.10]
//                    [barrier=0.12] [bridge=4] [arap_tol=0.15] [maxgen=100000]
//   STREAMED: trace-long stream=1 ct=<cache.fxvol@zarr-url> model= seed=z,y,x full=Z,Y,X
//                    out= [window=384] [windows=32] [pred=<cache@url>] ...
//     The streaming frontier: windows fetched on demand around the live growth
//     (segment/stream_grow.hpp) — no block ceiling on segment length. Without pred= the
//     data term is structure-tensor sheetness computed per window from the CT (classical;
//     wire per-window ML inference later). All coords ABSOLUTE; origin unused.
// origin: the block's absolute corner (model lives in scroll coords). seed: ABSOLUTE.
// Reports: cells, spatial extent, winding span (should stay < wtol), surf-qc-style
// evidence is the caller's next step (surf-qc the output).
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/surface.hpp"
#include "segment/grow.hpp"
#include "segment/stream_grow.hpp"
#include "ml/ml_api.hpp"
#include "segment/structure_tensor.hpp"
#include "winding/model_io.hpp"

#include <algorithm>
#include <charconv>
#include <memory>
#include <optional>
#include <cmath>
#include <span>
#include <string>
#include <string_view>

namespace fenix::winding {

inline Expected<int> run_trace_long(std::span<const std::string_view> args, Context&) {
    std::string pred_path, ct_path, model_path, out_path, weights_path;
    f64 oz = -1, oy = -1, ox = -1, sz = -1, sy = -1, sx = -1;
    segment::GrowParams gp;
    gp.surf_thresh = 0.10f;
    gp.ct_barrier = 0.12f;
    gp.max_bridge = 4;
    gp.winding_tol = 0.5f;
    s64 grid = 3000, maxgen = 100000, stream = 0, window = 384, windows = 32;
    f64 fz = 0, fy = 0, fx = 0;
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
        if (num("stream=", stream) || num("window=", window) || num("windows=", windows)) continue;
        if (triple("origin=", oz, oy, ox) || triple("seed=", sz, sy, sx) || triple("full=", fz, fy, fx))
            continue;
        if (a.starts_with("pred=")) { pred_path = std::string(a.substr(5)); continue; }
        if (a.starts_with("ct=")) { ct_path = std::string(a.substr(3)); continue; }
        if (a.starts_with("model=")) { model_path = std::string(a.substr(6)); continue; }
        if (a.starts_with("weights=")) { weights_path = std::string(a.substr(8)); continue; }
        if (a.starts_with("out=")) { out_path = std::string(a.substr(4)); continue; }
        return err(Errc::invalid_argument, "trace-long: unknown arg '" + std::string(a) + "'");
    }
    if ((stream == 0 && (pred_path.empty() || oz < 0)) || out_path.empty() || sz < 0 ||
        (stream != 0 && (ct_path.empty() || fz <= 0)))
        return err(Errc::invalid_argument,
                   "usage: trace-long pred=<fxvol> ct=<fxvol> model=<fxmodel> origin=z,y,x seed=z,y,x "
                   "out=<fxsurf> [grid=3000] [step=2] [wtol=0.35] [thresh=] [barrier=] [bridge=] "
                   "[arap_tol=] [maxgen=]");

    if (stream != 0) {
        // --- STREAMING FRONTIER: windows fetched on demand around the live growth ---
        auto ca = codec::VolumeArchive::open(ct_path);
        if (!ca) return std::unexpected(ca.error());
        ca->reserve_cache(u64{8} << 30);
        std::optional<codec::VolumeArchive> parch;
        if (!pred_path.empty()) {
            auto pa2 = codec::VolumeArchive::open(pred_path);
            if (!pa2) return std::unexpected(pa2.error());
            parch.emplace(std::move(*pa2));
        }
        SpiralModel model;
        if (!model_path.empty()) {
            auto m = read_fxmodel(model_path);
            if (!m) return std::unexpected(m.error());
            model = std::move(*m);
            gp.winding_fn = [mm = std::make_shared<SpiralModel>(std::move(model))](Vec3f p2) {
                return mm->winding_at(p2);
            };
            log(LogLevel::info, "trace-long[stream]: model winding gate ON (wtol {:.2f})", gp.winding_tol);
        } else {
            gp.winding_tol = 0;
        }
        gp.grid = static_cast<int>(grid);
        gp.max_gen = static_cast<int>(maxgen);
        const Extent3 wdim{window, window, window};
        segment::WindowFetch fetch = [&](Vec3f wlo, Extent3 wd, Volume<u8>& pred_out,
                                         Volume<u8>& ct_out) -> bool {
            ct_out = Volume<u8>::zeros(wd);
            if (auto g = ca->gather_box_u8(0, static_cast<s64>(wlo.z), static_cast<s64>(wlo.y),
                                           static_cast<s64>(wlo.x), wd.z, wd.y, wd.x, ct_out.view().data());
                !g) {
                log(LogLevel::error, "trace-long[stream]: CT fetch failed at ({:.0f},{:.0f},{:.0f})", wlo.z,
                    wlo.y, wlo.x);
                return false;
            }
            if (parch) {
                pred_out = Volume<u8>::zeros(wd);
                if (auto g = parch->gather_box_u8(0, static_cast<s64>(wlo.z), static_cast<s64>(wlo.y),
                                                  static_cast<s64>(wlo.x), wd.z, wd.y, wd.x,
                                                  pred_out.view().data());
                    !g)
                    return false;
            } else if (!weights_path.empty()) {
                // per-window ML inference (torch-free hook; FENIX_ML builds only)
                auto pw = ml::predict_surface_window(ct_out.view(), weights_path);
                if (!pw) {
                    log(LogLevel::error, "trace-long[stream]: window inference failed: {}", pw.error().message);
                    return false;
                }
                pred_out = std::move(*pw);
            } else {
                // classical data term: structure-tensor sheetness of the window CT (u8 0..255)
                pred_out = segment::structure_tensor_sheetness<u8, u8>(ct_out.view(), {1.0f, 2.0f}, 256);
            }
            return true;
        };
        // u8 field units: prob 0..255 (ML) or sheetness 0..255 -> thresh in raw u8 units
        if (!parch && gp.surf_thresh <= 1.0f) gp.surf_thresh = weights_path.empty() ? 25.0f : 26.0f;
        segment::StreamGrower grower(gp, Vec3f{0, 0, 0}, Vec3f{static_cast<f32>(fz), static_cast<f32>(fy),
                                                               static_cast<f32>(fx)},
                                     wdim);
        segment::StreamGrowStats st;
        auto S = grower.run(Vec3f{static_cast<f32>(sz), static_cast<f32>(sy), static_cast<f32>(sx)}, fetch,
                            static_cast<int>(windows), &st);
        if (!S) return std::unexpected(S.error());
        Vec3f lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
        for (usize i = 0; i < S->coord.size(); ++i)
            if (S->valid[i]) {
                const Vec3f q = S->coord[i];
                lo = Vec3f{std::min(lo.z, q.z), std::min(lo.y, q.y), std::min(lo.x, q.x)};
                hi = Vec3f{std::max(hi.z, q.z), std::max(hi.y, q.y), std::max(hi.x, q.x)};
            }
        log(LogLevel::info,
            "trace-long[stream]: {} cells over {} windows ({} paused unresolved), extent {:.0f}x{:.0f}x{:.0f} vox",
            st.cells, st.windows, st.paused_final, hi.z - lo.z, hi.y - lo.y, hi.x - lo.x);
        if (auto w = io::write_fxsurf(out_path, *S); !w) return std::unexpected(w.error());
        log(LogLevel::info, "trace-long[stream]: -> {}", out_path);
        return 0;
    }

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

    // report: extent + TRANSPORTED-RESIDUAL span (raw winding_at legitimately spans ~1
    // across the theta cut; the residual is the actual coherence metric the gate enforces)
    Vec3f lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (usize i = 0; i < S.coord.size(); ++i) {
        if (!S.valid[i]) continue;
        const Vec3f p = S.coord[i];
        lo = Vec3f{std::min(lo.z, p.z), std::min(lo.y, p.y), std::min(lo.x, p.x)};
        hi = Vec3f{std::max(hi.z, p.z), std::max(hi.y, p.y), std::max(hi.x, p.x)};
    }
    f32 r_lo = 1e30f, r_hi = -1e30f;
    if (gp.winding_fn) {
        // BFS residual transport from the largest-valid-region anchor (grid centre if valid)
        const s64 G = S.nu;
        std::vector<f32> res(S.coord.size(), 1e30f);
        s64 a0 = -1;
        for (usize i = 0; i < S.valid.size() && a0 < 0; ++i)
            if (S.valid[i]) a0 = static_cast<s64>(i);
        const usize ci = S.idx(G / 2, S.nv / 2);
        if (S.valid[ci]) a0 = static_cast<s64>(ci);
        if (a0 >= 0) {
            std::vector<s64> q{a0};
            res[static_cast<usize>(a0)] = gp.winding_fn(S.coord[static_cast<usize>(a0)]);
            while (!q.empty()) {
                const s64 id = q.back();
                q.pop_back();
                const s64 u = id % G, v = id / G;
                const f32 pr = res[static_cast<usize>(id)];
                const s64 nb[4][2] = {{u - 1, v}, {u + 1, v}, {u, v - 1}, {u, v + 1}};
                for (const auto& [uu, vv] : nb) {
                    if (uu < 0 || vv < 0 || uu >= G || vv >= S.nv) continue;
                    const usize j = S.idx(uu, vv);
                    if (!S.valid[j] || res[j] < 1e29f) continue;
                    const f32 w = gp.winding_fn(S.coord[j]);
                    if (!(std::abs(w) < 1e30f)) continue;
                    res[j] = w + std::round(pr - w);
                    q.push_back(static_cast<s64>(j));
                }
            }
            for (usize i = 0; i < res.size(); ++i)
                if (S.valid[i] && res[i] < 1e29f) {
                    r_lo = std::min(r_lo, res[i]);
                    r_hi = std::max(r_hi, res[i]);
                }
        }
    }
    for (usize i = 0; i < S.coord.size(); ++i)
        if (S.valid[i]) S.coord[i] = S.coord[i] + org;  // ABSOLUTE coords out
    log(LogLevel::info,
        "trace-long: {} cells, extent {:.0f}x{:.0f}x{:.0f} vox, residual span {:.3f} (gate tol {:.2f})",
        nvalid, hi.z - lo.z, hi.y - lo.y, hi.x - lo.x,
        gp.winding_fn ? (r_hi - r_lo) : -1.0f, gp.winding_tol);
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
