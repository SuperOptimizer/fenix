// segment/trace_eval.hpp — the TRACER BENCHMARK: predict-surface output + CT -> our NLLS
// tracer -> scored against PUBLISHED corpus meshes clipped to the block. The tracer's
// eval-set: every tracer change becomes a measured recall/precision/fragmentation delta
// against the same GT instead of a vibe. GT meshes are whole-scroll (absolute coords);
// origin= translates them into the block's local frame.
//   fenix trace-eval pred=<fxvol> gt=<fxsurf>... origin=z,y,x [ct=<fxvol>]
//                    [thresh=0.10] [step=2] [tile=128] [halo=24] [max_sheets=512]
//                    [min_valid=400] [seed_stride=24] [barrier=0.12] [bridge=2]
//                    [arap_tol=0.15] [out=<prefix>]
// Metrics: GT recall@2/@4 vox (fraction of GT cells with a traced point nearby — did we
// find the published sheet), trace precision@2/@4 (fraction of traced cells near ANY GT
// cell — but GT only covers what was published, so hallucination shows as low precision
// ONLY where GT is dense; read it with the coverage caveat), mean/median GT distance,
// sheets + cells (fragmentation). out= writes the traced sheets for view-chunk.
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/surface.hpp"
#include "segment/grow.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fenix::segment {

namespace detail {

// minimal uniform hash grid over a point set for radius queries (block-local, small)
struct PointGrid {
    f32 cell = 4.0f;
    std::unordered_map<u64, std::vector<u32>> bins;
    std::vector<Vec3f> pts;

    static u64 key(s64 z, s64 y, s64 x) {
        return (static_cast<u64>(z) & 0x1FFFFF) << 42 | (static_cast<u64>(y) & 0x1FFFFF) << 21 |
               (static_cast<u64>(x) & 0x1FFFFF);
    }
    void build(std::vector<Vec3f> p, f32 c) {
        cell = c;
        pts = std::move(p);
        bins.reserve(pts.size());
        for (u32 i = 0; i < pts.size(); ++i) {
            const Vec3f& q = pts[i];
            bins[key(static_cast<s64>(q.z / cell), static_cast<s64>(q.y / cell), static_cast<s64>(q.x / cell))]
                .push_back(i);
        }
    }
    // squared distance to the nearest point within one cell ring (max query radius = cell)
    [[nodiscard]] f32 nearest_d2(Vec3f q) const {
        const s64 bz = static_cast<s64>(q.z / cell), by = static_cast<s64>(q.y / cell),
                  bx = static_cast<s64>(q.x / cell);
        f32 best = 1e30f;
        for (s64 dz = -1; dz <= 1; ++dz)
            for (s64 dy = -1; dy <= 1; ++dy)
                for (s64 dx = -1; dx <= 1; ++dx) {
                    const auto it = bins.find(key(bz + dz, by + dy, bx + dx));
                    if (it == bins.end()) continue;
                    for (u32 i : it->second) {
                        const Vec3f d = pts[i] - q;
                        best = std::min(best, d.z * d.z + d.y * d.y + d.x * d.x);
                    }
                }
        return best;
    }
};

// recall/precision core, extracted so the oracle test can hit it directly: fraction of
// `from` points (stride-4 subsample) within 2/4 vox of the nearest `to` point. mean/med
// distances are clamped at the grid probe radius (4.0) — diagnostics only, the @2/@4
// fractions compare the UNclamped distance.
struct ScoreStats {
    f64 r2 = 0, r4 = 0, mean_d = 0, med_d = 0;
};
inline ScoreStats score_points(const std::vector<Vec3f>& from, const PointGrid& to) {
    ScoreStats st;
    std::vector<f32> ds;
    ds.reserve(from.size() / 4 + 1);
    s64 n2 = 0, n4 = 0;
    for (usize i = 0; i < from.size(); i += 4) {
        const f32 d = std::sqrt(to.nearest_d2(from[i]));
        ds.push_back(std::min(d, 4.0f));  // grid probe radius caps at one cell
        if (d <= 2.0f) ++n2;
        if (d <= 4.0f) ++n4;
    }
    const f64 n = static_cast<f64>(ds.size());
    st.r2 = static_cast<f64>(n2) / n;
    st.r4 = static_cast<f64>(n4) / n;
    f64 s = 0;
    for (f32 d : ds) s += d;
    st.mean_d = s / n;
    const auto mid = ds.begin() + static_cast<std::ptrdiff_t>(ds.size() / 2);
    std::nth_element(ds.begin(), mid, ds.end());
    st.med_d = *mid;
    return st;
}

}  // namespace detail

inline Expected<int> run_trace_eval(std::span<const std::string_view> args, Context&) {
    std::string pred_path, ct_path, out_prefix;
    std::vector<std::string> gt_paths;
    f64 oz = -1, oy = -1, ox = -1;
    GrowParams gp;
    gp.surf_thresh = 0.10f;
    gp.ct_barrier = 0.12f;
    s64 tile = 128, halo = 24, max_sheets = 512, min_valid = 400, seed_stride = 24;
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
        if (num("thresh=", gp.surf_thresh) || num("step=", gp.step) || num("tile=", tile) ||
            num("halo=", halo) || num("max_sheets=", max_sheets) || num("min_valid=", min_valid) ||
            num("seed_stride=", seed_stride) || num("barrier=", gp.ct_barrier) ||
            num("bridge=", gp.max_bridge) || num("arap_tol=", gp.arap_tol))
            continue;
        if (a.starts_with("pred=")) { pred_path = std::string(a.substr(5)); continue; }
        if (a.starts_with("ct=")) { ct_path = std::string(a.substr(3)); continue; }
        if (a.starts_with("gt=")) { gt_paths.emplace_back(a.substr(3)); continue; }
        if (a.starts_with("out=")) { out_prefix = std::string(a.substr(4)); continue; }
        if (a.starts_with("origin=")) {
            const auto t = a.substr(7);
            const auto c1 = t.find(','), c2 = t.rfind(',');
            if (c1 == std::string_view::npos || c2 == c1) continue;
            std::from_chars(t.data(), t.data() + c1, oz);
            std::from_chars(t.data() + c1 + 1, t.data() + c2, oy);
            std::from_chars(t.data() + c2 + 1, t.data() + t.size(), ox);
            continue;
        }
        return err(Errc::invalid_argument, "trace-eval: unknown arg '" + std::string(a) + "'");
    }
    if (pred_path.empty() || gt_paths.empty() || oz < 0)
        return err(Errc::invalid_argument,
                   "usage: trace-eval pred=<fxvol> gt=<fxsurf>... origin=z,y,x [ct=<fxvol>] "
                   "[thresh=] [step=] [tile=] [halo=] [max_sheets=] [min_valid=] [seed_stride=] "
                   "[barrier=] [bridge=] [arap_tol=] [out=<prefix>]");

    auto pa = codec::VolumeArchive::open(pred_path);
    if (!pa) return std::unexpected(pa.error());
    auto pred = pa->read_volume_as<u8>();
    if (!pred) return std::unexpected(pred.error());
    const Extent3 D = pred->dims();
    Volume<u8> ct;
    if (!ct_path.empty()) {
        auto ca = codec::VolumeArchive::open(ct_path);
        if (!ca) return std::unexpected(ca.error());
        auto cv = ca->read_volume_as<u8>();
        if (!cv) return std::unexpected(cv.error());
        ct = std::move(*cv);
    }
    if (gp.ct_barrier > 0 && ct.dims().count() == 0) gp.ct_barrier = 0;  // barrier needs CT

    // GT: clip whole-scroll meshes to the block, translate to local coords
    const Vec3f org{static_cast<f32>(oz), static_cast<f32>(oy), static_cast<f32>(ox)};
    std::vector<Vec3f> gt_pts;
    for (const auto& gpath : gt_paths) {
        auto g = io::read_fxsurf(gpath);
        if (!g) return std::unexpected(g.error());
        for (usize i = 0; i < g->coord.size(); ++i) {
            if (!g->valid[i]) continue;
            const Vec3f p = g->coord[i] - org;
            if (p.z >= 0 && p.y >= 0 && p.x >= 0 && p.z < static_cast<f32>(D.z) &&
                p.y < static_cast<f32>(D.y) && p.x < static_cast<f32>(D.x))
                gt_pts.push_back(p);
        }
    }
    if (gt_pts.size() < 100)
        return err(Errc::invalid_argument, "trace-eval: <100 GT cells inside the block — wrong origin?");
    log(LogLevel::info, "trace-eval: {} GT cells in block {}x{}x{}, {} meshes", gt_pts.size(), D.z, D.y,
        D.x, gt_paths.size());

    // trace
    const VolumeResult res =
        ct.dims().count() > 0
            ? trace_volume_tiled<u8>(pred->view(), ct.view(), gp, static_cast<int>(max_sheets), min_valid,
                                     static_cast<int>(seed_stride), 0, static_cast<int>(tile),
                                     static_cast<int>(halo))
            : trace_volume_tiled<u8>(pred->view(), VolumeView<const u8>{}, gp, static_cast<int>(max_sheets),
                                     min_valid, static_cast<int>(seed_stride), 0, static_cast<int>(tile),
                                     static_cast<int>(halo));
    std::vector<Vec3f> tr_pts;
    s64 total_cells = 0;
    for (const Surface& s : res.sheets) {
        for (usize i = 0; i < s.coord.size(); ++i)
            if (s.valid[i]) {
                tr_pts.push_back(s.coord[i]);
                ++total_cells;
            }
    }
    log(LogLevel::info, "trace-eval: traced {} sheets, {} cells", res.sheets.size(), total_cells);
    if (tr_pts.empty()) return err(Errc::invalid_argument, "trace-eval: tracer produced nothing");

    // score
    detail::PointGrid tg, gg;
    tg.build(tr_pts, 4.0f);
    gg.build(gt_pts, 4.0f);
    const detail::ScoreStats rec = detail::score_points(gt_pts, tg);
    const detail::ScoreStats pre = detail::score_points(tr_pts, gg);
    const f64 rec2 = rec.r2, rec4 = rec.r4, rec_mean = rec.mean_d, rec_med = rec.med_d;
    const f64 pre2 = pre.r2, pre4 = pre.r4, pre_mean = pre.mean_d, pre_med = pre.med_d;
    log(LogLevel::info,
        "trace-eval: RECALL @2 {:.3f} @4 {:.3f} (mean d {:.2f}, med {:.2f}) | PRECISION @2 {:.3f} @4 "
        "{:.3f} (mean d {:.2f}, med {:.2f}) | {} sheets / {} cells (thresh {:.2f} step {:.1f} barrier "
        "{:.2f} bridge {} arap_tol {:.2f})",
        rec2, rec4, rec_mean, rec_med, pre2, pre4, pre_mean, pre_med, res.sheets.size(), total_cells,
        gp.surf_thresh, gp.step, gp.ct_barrier, gp.max_bridge, gp.arap_tol);

    if (!out_prefix.empty()) {
        // written in ABSOLUTE scroll coords (block origin added back) so surf-qc /
        // view-chunk / surf-consist can consume them directly against the full volume.
        // Largest-first, so _t0 is always the biggest traced sheet.
        std::vector<usize> order(res.sheets.size());
        for (usize i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](usize a2, usize b2) {
            return res.sheets[a2].valid_count() > res.sheets[b2].valid_count();
        });
        s64 w = 0;
        for (usize i : order) {
            Surface s = res.sheets[i];
            for (usize j = 0; j < s.coord.size(); ++j)
                if (s.valid[j]) s.coord[j] = s.coord[j] + org;
            if (auto r = io::write_fxsurf(out_prefix + "_t" + std::to_string(w) + ".fxsurf", s); !r)
                return std::unexpected(r.error());
            ++w;
        }
        log(LogLevel::info, "trace-eval: {} traced sheets (absolute coords, largest first) -> {}_t*.fxsurf",
            w, out_prefix);
    }
    return 0;
}

}  // namespace fenix::segment

namespace {
[[maybe_unused]] const int fenix_stage_trace_eval = ::fenix::register_stage(::fenix::Stage{
    "trace-eval", "tracer benchmark: predict->trace->score vs published meshes",
    ::fenix::segment::run_trace_eval});
}  // namespace
