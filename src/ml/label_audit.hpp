// ml/label_audit.hpp — `fenix label-audit`: model-vs-label disagreement mining (torch-free).
// Once a trained model is decent, places where it CONFIDENTLY contradicts a mesh label are
// label errors more often than model errors — the third label-quality oracle (training
// dynamics) made queryable. Given a prediction block (u8 sheet-prob .fxvol, e.g. from
// predict-surface) and meshes in the same scroll frame:
//   per valid uv cell inside the block: sample pred at the surface point, charitably
//   maxed over ±1 voxel along the stencil normal (registration slop must not read as
//   disagreement); per uv TILE: n, mean prob, miss% (prob < miss threshold).
// Output: one TSV per mesh (u0 v0 n mean_p miss%) sorted worst-first — the human review
// queue — plus a per-mesh summary line. With pred = the production teacher this is
// teacher-as-auditor; with pred = the current student it is self-disagreement mining.
// The same TSV drives trust-mask tightening and surf-sheet crop rendering.
//   fenix label-audit <pred.fxvol> <oz> <oy> <ox> <fxsurf...>
//                     [tile=256] [miss=64] [out=audit]
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "io/surface.hpp"
#include "ml/surf_qc.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::ml {

inline Expected<int> run_label_audit(std::span<const std::string_view> args, Context&) {
    if (args.size() < 5)
        return err(Errc::invalid_argument,
                   "usage: label-audit <pred.fxvol> <oz> <oy> <ox> <fxsurf...> [tile=256] [miss=64] [out=audit]");
    auto pi = [](std::string_view t) {
        s64 v = 0;
        std::from_chars(t.data(), t.data() + t.size(), v);
        return v;
    };
    const Index3 org{pi(args[1]), pi(args[2]), pi(args[3])};
    s64 tile = 256, miss_thr = 64;
    std::string out_prefix = "audit";
    std::vector<std::string> meshes;
    for (usize i = 4; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("tile=", tile) || num("miss=", miss_thr)) continue;
        if (a.starts_with("out=")) {
            out_prefix = std::string(a.substr(4));
            continue;
        }
        meshes.emplace_back(a);
    }
    if (meshes.empty()) return err(Errc::invalid_argument, "label-audit: no meshes");

    auto pred = codec::VolumeArchive::open(std::string(args[0]));
    if (!pred) return std::unexpected(pred.error());
    const Extent3 dims = pred->dims();

    for (const auto& mp : meshes) {
        auto s = io::read_fxsurf(mp);
        if (!s) return std::unexpected(s.error());
        const s64 tu = (s->nu + tile - 1) / tile, tv = (s->nv + tile - 1) / tile;
        struct Cell {
            f64 psum = 0;
            s64 n = 0, nmiss = 0;
        };
        std::vector<Cell> cells(static_cast<usize>(tu * tv));
        s64 n_total = 0;
        for (s64 v = 0; v < s->nv; ++v)
            for (s64 u = 0; u < s->nu; ++u) {
                if (!s->is_valid(u, v)) continue;
                const Vec3f p = s->at(u, v);
                const s64 z = static_cast<s64>(std::lround(p.z)) - org.z;
                const s64 y = static_cast<s64>(std::lround(p.y)) - org.y;
                const s64 x = static_cast<s64>(std::lround(p.x)) - org.x;
                if (z < 1 || y < 1 || x < 1 || z + 1 >= dims.z || y + 1 >= dims.y || x + 1 >= dims.x) continue;
                // charitable read: max prob over the surface point ±1 along the stencil
                // normal — sub-voxel registration slop must not read as disagreement
                const auto nm = detail::stencil_normal(*s, u, v);
                u8 best = 0;
                for (int t = -1; t <= 1; ++t) {
                    Vec3f q = p;
                    if (nm) q = p + *nm * static_cast<f32>(t);
                    const s64 qz = static_cast<s64>(std::lround(q.z)) - org.z;
                    const s64 qy = static_cast<s64>(std::lround(q.y)) - org.y;
                    const s64 qx = static_cast<s64>(std::lround(q.x)) - org.x;
                    if (qz < 0 || qy < 0 || qx < 0 || qz >= dims.z || qy >= dims.y || qx >= dims.x) continue;
                    u8 b;
                    if (!pred->gather_box_u8(0, qz, qy, qx, 1, 1, 1, &b)) continue;
                    best = std::max(best, b);
                }
                Cell& c = cells[static_cast<usize>((v / tile) * tu + (u / tile))];
                c.psum += best;
                ++c.n;
                c.nmiss += best < static_cast<u8>(miss_thr);
                ++n_total;
            }
        if (n_total == 0) {
            std::printf("label-audit %s  OUTSIDE-BLOCK (0 points in the prediction volume)\n", mp.c_str());
            continue;
        }
        // sort tiles worst-first (by miss fraction, then low mean prob); write the queue
        struct Row {
            s64 u0, v0, n, nmiss;
            f64 mean_p;
        };
        std::vector<Row> rows;
        f64 g_p = 0;
        s64 g_n = 0, g_miss = 0;
        for (s64 tj = 0; tj < tv; ++tj)
            for (s64 ti = 0; ti < tu; ++ti) {
                const Cell& c = cells[static_cast<usize>(tj * tu + ti)];
                if (c.n < 8) continue;
                rows.push_back({ti * tile, tj * tile, c.n, c.nmiss, c.psum / static_cast<f64>(c.n)});
                g_p += c.psum;
                g_n += c.n;
                g_miss += c.nmiss;
            }
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
            const f64 ma = static_cast<f64>(a.nmiss) / static_cast<f64>(a.n);
            const f64 mb = static_cast<f64>(b.nmiss) / static_cast<f64>(b.n);
            return ma != mb ? ma > mb : a.mean_p < b.mean_p;
        });
        std::string base = mp;
        if (const auto sl = base.rfind('/'); sl != std::string::npos) base = base.substr(sl + 1);
        if (const auto dot = base.rfind(".fxsurf"); dot != std::string::npos) base.resize(dot);
        const std::string tsv = out_prefix + "_" + base + ".tsv";
        if (std::FILE* f = std::fopen(tsv.c_str(), "w")) {
            std::fprintf(f, "#u0\tv0\tn\tmean_p\tmiss_pct\n");
            for (const Row& r : rows)
                std::fprintf(f,
                             "%lld\t%lld\t%lld\t%.1f\t%.1f\n",
                             static_cast<long long>(r.u0),
                             static_cast<long long>(r.v0),
                             static_cast<long long>(r.n),
                             r.mean_p,
                             100.0 * static_cast<f64>(r.nmiss) / static_cast<f64>(r.n));
            std::fclose(f);
        }
        std::printf("label-audit %s  n=%lld  mean_p %.1f  miss %.1f%%  tiles %zu  -> %s\n",
                    mp.c_str(),
                    static_cast<long long>(n_total),
                    g_n ? g_p / static_cast<f64>(g_n) : 0.0,
                    g_n ? 100.0 * static_cast<f64>(g_miss) / static_cast<f64>(g_n) : 0.0,
                    rows.size(),
                    tsv.c_str());
    }
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_label_audit = ::fenix::register_stage(::fenix::Stage{
    "label-audit", "model-vs-label disagreement mining (review queue per mesh)", ::fenix::ml::run_label_audit});
}  // namespace
