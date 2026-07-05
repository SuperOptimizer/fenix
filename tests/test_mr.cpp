// test_mr.cpp — multi-resolution (coarse-to-fine) tracing demo: trace rough on a downscaled field,
// refine to native; compare time + quality against a direct native trace.
// Usage: test_mr <surf.fxvol> sz sy sx [coarse_factor grid]
#include "core/core.hpp"
#include "core/surface.hpp"
#include "bench_vol.hpp"
#include "segment/grow.hpp"
#include "eval/mesh_quality.hpp"

#include <chrono>
#include <cstdio>

using namespace fenix;
using clk = std::chrono::steady_clock;
static double sec(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static void qa(const char* tag, const Surface& S, VolumeView<const f32> field) {
    auto q = eval::analyze_mesh(S);
    double df = 0; s64 nv = 0;
    for (s64 i = 0; i < S.nu * S.nv; ++i) if (S.valid[static_cast<usize>(i)]) { df += sample_trilinear(field, S.coord[static_cast<usize>(i)]); ++nv; }
    std::printf("%-14s valid %lld  cmp %lld  holes %lld  edgeCV %.3f  tear%% %.2f  degen%% %.3f  minang %.1f  sDir %.3f  selfX%% %.3f  data_fid %.2f\n",
                tag, (long long)nv, (long long)q.components, (long long)q.holes, q.edge_cv, 100 * q.frac_long, 100 * q.degen_tri,
                q.min_angle_deg, q.sdirichlet_mean, 100 * q.self_intersect, nv ? df / static_cast<double>(nv) : 0);
}

int main(int argc, char** argv) {
    if (argc < 5) { std::printf("need <surf.fxvol> sz sy sx [coarse_factor grid]\n"); return 1; }
    Vec3f seed{static_cast<f32>(std::atof(argv[2])), static_cast<f32>(std::atof(argv[3])), static_cast<f32>(std::atof(argv[4]))};
    const int cf = argc > 5 ? std::atoi(argv[5]) : 4;
    const int grid = argc > 6 ? std::atoi(argv[6]) : 800;
    auto vr = bench::load_f32(argv[1]);
    if (!vr) { std::printf("read fail\n"); return 1; }
    Volume<f32> vol = std::move(*vr);
    f32 mx = 0; for (s64 i = 0; i < vol.dims().count(); ++i) mx = std::max(mx, vol.flat()[i]);
    if (mx > 2) for (s64 i = 0; i < vol.dims().count(); ++i) vol.flat()[i] /= 255.0f;

    segment::GrowParams p; p.step = 2; p.snap_radius = 3; p.fold_thresh = 6; p.surf_thresh = 0.15f; p.grid = grid; p.max_gen = 6000;

    // --- native-direct ---
    auto t0 = clk::now();
    auto nf = segment::compute_normal_field<f32>(vol.view(), 8);
    Surface dir = segment::grow_surface<f32>(vol.view(), VolumeView<const f32>{}, nf, seed, p);
    auto t1 = clk::now();
    std::printf("[native-direct]  %.1fs\n", sec(t0, t1));
    qa("native-direct", dir, vol.view());

    // --- coarse-to-fine ---
    auto t2 = clk::now();
    Volume<f32> coarse = segment::downscale_field<f32>(vol.view(), cf);
    auto nfc = segment::compute_normal_field<f32>(coarse.view(), 8 / cf > 0 ? 8 / cf : 1);
    segment::GrowParams pc = p; pc.grid = grid / cf + 16;
    Surface cs = segment::grow_surface<f32>(coarse.view(), VolumeView<const f32>{}, nfc, seed / static_cast<f32>(cf), pc);
    auto t3 = clk::now();
    Surface ref = segment::refine_to_native<f32>(cs, cf, vol.view(), nf, p);
    auto t4 = clk::now();
    std::printf("[coarse %dx]     trace %.1fs (coarse grid %lld, %lld pts)  refine %.1fs  total %.1fs\n",
                cf, sec(t2, t3), (long long)cs.nu, (long long)cs.valid_count(), sec(t3, t4), sec(t2, t4));
    qa("coarse-refined", ref, vol.view());
    return 0;
}
