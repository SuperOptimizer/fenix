// test_mesh_qa.cpp — run the intrinsic mesh-quality analysis on every traced patch and print a QA
// table, flagging tearing / holes / folds / self-intersection / degeneracy / distortion.
// Usage: test_mesh_qa <voldir>  (dir of sheet_NN/{x,y,z}.f32+valid.u8+meta.txt)
#include "core/core.hpp"
#include "core/surface.hpp"
#include "eval/mesh_quality.hpp"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace fenix;
namespace fs = std::filesystem;

static Surface load_sheet(const std::string& d) {
    s64 G = 0; { FILE* f = std::fopen((d + "/meta.txt").c_str(), "r"); if (f) { long g = 0; (void)std::fscanf(f, "%ld", &g); G = g; std::fclose(f); } }
    Surface S(G, G);
    if (G <= 0) return S;
    const usize NG = static_cast<usize>(G * G);
    auto rd = [&](const std::string& nm, usize b) { std::vector<u8> v(b); FILE* f = std::fopen((d + "/" + nm).c_str(), "rb"); if (f) { (void)std::fread(v.data(), 1, b, f); std::fclose(f); } return v; };
    auto xb = rd("x.f32", NG * 4), yb = rd("y.f32", NG * 4), zb = rd("z.f32", NG * 4), mb = rd("valid.u8", NG);
    const f32 *X = reinterpret_cast<const f32*>(xb.data()), *Y = reinterpret_cast<const f32*>(yb.data()), *Z = reinterpret_cast<const f32*>(zb.data());
    for (usize i = 0; i < NG; ++i) if (mb[i]) { S.coord[i] = Vec3f{Z[i], Y[i], X[i]}; S.valid[i] = 1; }
    return S;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "data/fenix_vol";
    std::vector<std::string> sheets;
    for (auto& e : fs::directory_iterator(dir)) if (e.is_directory() && e.path().filename().string().rfind("sheet_", 0) == 0) sheets.push_back(e.path().string());
    std::sort(sheets.begin(), sheets.end());
    if (sheets.empty()) { std::printf("no sheets in %s\n", dir.c_str()); return 0; }

    std::printf("%-9s %8s %5s %5s | %6s %6s %6s | %6s %7s %6s | %6s %6s | %7s %7s | %6s %6s | flags\n",
                "patch", "valid", "cmp", "hole", "edgeCV", "p99/m", "long%", "degen%", "minang", "bad%", "fold%", "selfX%", "sDir.mn", "sDir99", "curv", "bnd%");
    eval::MeshQuality agg{}; s64 totv = 0, bad_patches = 0;
    for (auto& d : sheets) {
        Surface S = load_sheet(d);
        auto q = eval::analyze_mesh(S);
        std::string flags;
        if (q.holes > 0) flags += "HOLES ";
        if (q.frac_long > 0.01) flags += "TEAR ";
        if (q.fold_detj > 0.005) flags += "FOLD ";
        if (q.self_intersect > 0.01) flags += "SELFX ";
        if (q.degen_tri > 0.005) flags += "DEGEN ";
        if (q.sdirichlet_mean > 0.6) flags += "DISTORT ";
        if (q.min_angle_deg < 10) flags += "SLIVER ";
        if (!flags.empty()) ++bad_patches;
        std::printf("%-9s %8lld %5lld %5lld | %6.3f %6.2f %6.2f | %6.2f %7.1f %6.2f | %6.2f %6.2f | %7.3f %7.2f | %6.2f %6.1f | %s\n",
                    fs::path(d).filename().string().c_str(), (long long)q.valid, (long long)q.components, (long long)q.holes,
                    q.edge_cv, q.edge_p99_ratio, 100 * q.frac_long, 100 * q.degen_tri, q.min_angle_deg, 100 * q.bad_angle,
                    100 * q.fold_detj, 100 * q.self_intersect, q.sdirichlet_mean, q.sdirichlet_p99, q.curvature_mean, 100 * q.boundary_frac,
                    flags.empty() ? "ok" : flags.c_str());
        const f64 w = static_cast<f64>(q.valid); totv += q.valid;
        agg.holes += q.holes; agg.frac_long += w * q.frac_long; agg.degen_tri += w * q.degen_tri; agg.fold_detj += w * q.fold_detj;
        agg.self_intersect += w * q.self_intersect; agg.sdirichlet_mean += w * q.sdirichlet_mean; agg.curvature_mean += w * q.curvature_mean;
        agg.edge_cv += w * q.edge_cv; agg.boundary_frac += w * q.boundary_frac;
    }
    const f64 iv = totv ? 1.0 / static_cast<f64>(totv) : 0;
    std::printf("\nAGGREGATE (valid-weighted): edgeCV %.3f  long(tear)%% %.2f  degen%% %.3f  fold%% %.3f  selfX%% %.3f  sDirichlet %.3f  curv %.3f  bnd%% %.1f  | total holes %lld  bad patches %lld/%zu\n",
                agg.edge_cv * iv, 100 * agg.frac_long * iv, 100 * agg.degen_tri * iv, 100 * agg.fold_detj * iv, 100 * agg.self_intersect * iv,
                agg.sdirichlet_mean * iv, agg.curvature_mean * iv, 100 * agg.boundary_frac * iv, (long long)agg.holes, (long long)bad_patches, sheets.size());
    return 0;
}
