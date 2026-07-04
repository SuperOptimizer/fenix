// ml/stroke_score.hpp — `fenix stroke-score`: human-drawn strokes as the label referee
// (torch-free). The strongest label oracle is a human tracing the sheets they SEE on a
// slice (fenix view's stroke tools, drawn BLIND — no mesh overlay to anchor them), then
// scoring the corpus meshes against that drawn truth:
//   MATCHED  med distance <= tol       -> the mesh is confirmed where the human looked
//   OFFSET   tol < med <= 3*tol        -> systematic misregistration, repair candidate
//   MISSED   no mesh within 3*tol      -> a sheet the human sees that no mesh traces
//            (or the region where a QC-failed mesh SHOULD have been)
// Distances are point-to-SURFACE (bilinear refinement, same machinery as surf-consist).
//   fenix stroke-score <anno.toml> <fxsurf...> [tol=3] [crop=z,y,x] [out=strokes.tsv]
// crop: strokes were drawn on a CROPPED region (fenix view of an eval block) — shift
// them by the block origin into the meshes' absolute frame.
#pragma once

#include "annotate/annotation.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "io/surface.hpp"
#include "ml/surf_consist.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::ml {

inline Expected<int> run_stroke_score(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2)
        return err(Errc::invalid_argument,
                   "usage: stroke-score <anno.toml> <fxsurf...> [tol=3] [crop=z,y,x] [out=strokes.tsv]");
    f64 tol = 3;
    Vec3f crop{0, 0, 0};
    std::string out_path;
    std::vector<std::string> mesh_paths;
    for (usize i = 1; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("tol=", tol)) continue;
        if (a.starts_with("crop=")) {
            f64 cz = 0, cy = 0, cx = 0;
            const auto t = a.substr(5);
            const auto c1 = t.find(','), c2 = t.rfind(',');
            if (c1 == std::string_view::npos || c2 == c1)
                return err(Errc::invalid_argument, "stroke-score: crop wants z,y,x");
            std::from_chars(t.data(), t.data() + c1, cz);
            std::from_chars(t.data() + c1 + 1, t.data() + c2, cy);
            std::from_chars(t.data() + c2 + 1, t.data() + t.size(), cx);
            crop = Vec3f{static_cast<f32>(cz), static_cast<f32>(cy), static_cast<f32>(cx)};
            continue;
        }
        if (a.starts_with("out=")) {
            out_path = std::string(a.substr(4));
            continue;
        }
        mesh_paths.emplace_back(a);
    }
    if (mesh_paths.empty()) return err(Errc::invalid_argument, "stroke-score: no meshes");

    auto anno = annotate::load_annotations(std::string(args[0]));
    if (!anno) return std::unexpected(anno.error());
    if (anno->strokes.empty()) return err(Errc::invalid_argument, "stroke-score: no strokes in " + std::string(args[0]));

    // load meshes as refinable clouds (coarse seed + bilinear surface, surf-consist infra)
    std::vector<detail::MeshCloud> clouds;
    for (const auto& mp : mesh_paths) {
        auto mc = detail::load_cloud(mp, 1u << 20, /*keep_surf=*/true);
        if (!mc) return std::unexpected(mc.error());
        mc->build_grid(std::max(6.0f, mc->spacing));
        clouds.push_back(std::move(*mc));
    }

    std::FILE* tf = out_path.empty() ? nullptr : std::fopen(out_path.c_str(), "w");
    if (tf) std::fprintf(tf, "#stroke\tmesh\tn\tmed\tp90\tverdict\n");
    s64 n_match = 0, n_off = 0, n_miss = 0;
    for (usize si = 0; si < anno->strokes.size(); ++si) {
        const auto& st = anno->strokes[si];
        if (st.points.size() < 3) continue;
        // best mesh for this stroke = the one with the lowest median distance
        f64 best_med = 1e30, best_p90 = 0;
        s64 best_mesh = -1;
        for (usize mi = 0; mi < clouds.size(); ++mi) {
            auto& B = clouds[mi];
            const f32 R = B.spacing * 0.75f + static_cast<f32>(3 * tol);
            std::vector<f32> dists;
            for (const Vec3f& sp : st.points) {
                const Vec3f p = sp + crop;
                const s64 j = B.nearest(p, R);
                if (j < 0) continue;
                Vec3f cp{};
                const f32 d =
                    B.refined_dist(p, B.uvs[static_cast<usize>(j)][0], B.uvs[static_cast<usize>(j)][1], 1.5f, &cp);
                if (d <= static_cast<f32>(3 * tol)) dists.push_back(d);
            }
            if (dists.size() < st.points.size() / 2) continue;  // mesh covers under half the stroke
            std::sort(dists.begin(), dists.end());
            const f64 med = dists[dists.size() / 2];
            if (med < best_med) {
                best_med = med;
                best_p90 = dists[dists.size() * 9 / 10];
                best_mesh = static_cast<s64>(mi);
            }
        }
        const char* verdict = best_mesh < 0 ? "MISSED" : best_med <= tol ? "MATCHED" : "OFFSET";
        (best_mesh < 0 ? n_miss : best_med <= tol ? n_match : n_off)++;
        const std::string mesh_name = best_mesh < 0 ? "-" : clouds[static_cast<usize>(best_mesh)].path;
        std::printf("stroke-score %s (%zu pts)  mesh %s  med %.1f  p90 %.1f  %s\n",
                    st.name.empty() ? ("stroke" + std::to_string(si)).c_str() : st.name.c_str(),
                    st.points.size(),
                    mesh_name.c_str(),
                    best_mesh < 0 ? -1.0 : best_med,
                    best_mesh < 0 ? -1.0 : best_p90,
                    verdict);
        if (tf)
            std::fprintf(tf,
                         "%s\t%s\t%zu\t%.2f\t%.2f\t%s\n",
                         st.name.empty() ? ("stroke" + std::to_string(si)).c_str() : st.name.c_str(),
                         mesh_name.c_str(),
                         st.points.size(),
                         best_mesh < 0 ? -1.0 : best_med,
                         best_mesh < 0 ? -1.0 : best_p90,
                         verdict);
    }
    if (tf) std::fclose(tf);
    std::printf("stroke-score: %lld matched, %lld offset, %lld missed (tol %.1f)\n",
                static_cast<long long>(n_match),
                static_cast<long long>(n_off),
                static_cast<long long>(n_miss),
                tol);
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_stroke_score = ::fenix::register_stage(::fenix::Stage{
    "stroke-score", "score corpus meshes against human-drawn slice strokes", ::fenix::ml::run_stroke_score});
}  // namespace
