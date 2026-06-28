// test_trace_perf — crop a cube to <csz>^3 and run the trace pipeline with per-stage timing, for
// perf profiling + optimization. No rendering / dumping; pure compute.
// Usage: test_trace_perf <ct.nrrd> <surf.nrrd> [csz maxsheets seedstride thresh ctweight]
#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "io/nrrd.hpp"
#include "preprocess/aircut.hpp"
#include "segment/grow.hpp"
#include "segment/patch_graph.hpp"

#include <chrono>
#include <cstdio>
#include <string>

using namespace fenix;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

static Volume<u8> downsample_u8(VolumeView<const u8> v, int s) {
    const Extent3 d = v.dims();
    const Extent3 dd{d.z / s, d.y / s, d.x / s};
    Volume<u8> o(dd);
    auto ov = o.view();
    const int n = s * s * s;
    parallel_for_z(dd, [&](s64 z) {
        for (s64 y = 0; y < dd.y; ++y)
            for (s64 x = 0; x < dd.x; ++x) {
                int acc = 0;
                for (int dz = 0; dz < s; ++dz)
                    for (int dy = 0; dy < s; ++dy)
                        for (int dx = 0; dx < s; ++dx) acc += v(z * s + dz, y * s + dy, x * s + dx);
                ov(z, y, x) = static_cast<u8>(acc / n);
            }
    });
    return o;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: test_trace_perf <ct.nrrd> <surf.nrrd> [csz maxsheets seedstride thresh ctweight]\n");
        return 0;
    }
    const std::string ct_path = argv[1], surf_path = argv[2];
    const s64 csz = argc > 3 ? std::atoll(argv[3]) : 512;
    const int max_sheets = argc > 4 ? std::atoi(argv[4]) : 24;
    const int seed_stride = argc > 5 ? std::atoi(argv[5]) : 32;
    const f32 thr = argc > 6 ? static_cast<f32>(std::atof(argv[6])) : 0.12f;
    const f32 ctw = argc > 7 ? static_cast<f32>(std::atof(argv[7])) : 0.8f;
    const int ctds = argc > 8 ? std::atoi(argv[8]) : 1;  // CT-term downsample (1 = full res)
    const int fouter = argc > 9 ? std::atoi(argv[9]) : 12;   // final ARAP outer iters
    const int finner = argc > 10 ? std::atoi(argv[10]) : 25;  // final ARAP inner sweeps

    auto pm = io::nrrd_max(surf_path), cm = io::nrrd_max(ct_path);
    if (!pm || !cm) { std::printf("read failed\n"); return 1; }
    auto ctr = io::read_nrrd_u8(ct_path, (*cm > 2.0f) ? 1.0f : 255.0f);
    auto pr = io::read_nrrd_u8(surf_path, (*pm > 2.0f) ? 1.0f : 255.0f);
    if (!ctr || !pr) { std::printf("read failed\n"); return 1; }
    const Volume<u8> ctf = std::move(*ctr), predf = std::move(*pr);
    const Extent3 D = predf.dims();

    // contiguous central <csz>^3 crop of both volumes
    const s64 oz = (D.z - csz) / 2, oy = (D.y - csz) / 2, ox = (D.x - csz) / 2;
    Volume<u8> ct(Extent3{csz, csz, csz}), pred(Extent3{csz, csz, csz});
    {
        auto cv = ct.view();
        auto pv = pred.view();
        auto cf = ctf.view();
        auto pf = predf.view();
        parallel_for_z(Extent3{csz, csz, csz}, [&](s64 z) {
            for (s64 y = 0; y < csz; ++y)
                for (s64 x = 0; x < csz; ++x) { cv(z, y, x) = cf(oz + z, oy + y, ox + x); pv(z, y, x) = pf(oz + z, oy + y, ox + x); }
        });
    }
    const f32 ctt = preprocess::air_cut<u8>(ct.view(), 0.0f, 256.0f);
    std::printf("crop %lld^3 (from %lldx%lldx%lld @ %lld,%lld,%lld)  air-cut=%.0f\n",
                (long long)csz, (long long)D.z, (long long)D.y, (long long)D.x, (long long)oz, (long long)oy, (long long)ox, ctt);

    segment::GrowParams gp;
    gp.step = 2.0f;
    gp.surf_thresh = thr * 255.0f;
    gp.snap_radius = gp.step * 1.5f;
    gp.fold_thresh = 6;
    gp.grid = static_cast<int>(csz);
    gp.river_radius = 2;
    gp.ct_thresh = ctt;
    gp.ct_weight = ctw;
    gp.ct_ds = static_cast<f32>(ctds);
    gp.final_outer = fouter;
    gp.final_inner = finner;
    const f32 seed_thresh = std::max(gp.surf_thresh * 1.5f, 56.0f);
    // CT term on a downsampled grid (ct_ds>1): less memory traffic in the snap data-term hot loop.
    Volume<u8> ctsmall;
    if (ctds > 1) ctsmall = downsample_u8(ct.view(), ctds);
    const VolumeView<const u8> ct_term = ctds > 1 ? ctsmall.view() : ct.view();

    auto t0 = clk::now();
    const segment::NormalField nf = segment::compute_normal_field<u8>(pred.view(), 8);
    auto t1 = clk::now();
    segment::VolumeResult R = segment::trace_volume<u8>(pred.view(), nf, gp, max_sheets, 3000, seed_stride, seed_thresh, ct_term);
    auto t2 = clk::now();
    annotate::Umbilicus umb;
    umb.z = {0.0f, static_cast<f32>(csz)};
    umb.y = {static_cast<f32>(csz) * 0.5f, static_cast<f32>(csz) * 0.5f};
    umb.x = {static_cast<f32>(csz) * 0.5f, static_cast<f32>(csz) * 0.5f};
    segment::PatchGraphParams pgp;
    pgp.step = gp.step;
    segment::PatchGraph g = segment::analyze_patches(R.sheets, umb, pgp);
    auto t3 = clk::now();

    s64 valid = 0;
    f64 fold_w = 0, dfold_w = 0;  // valid-weighted quality (want ~0)
    for (const Surface& s : R.sheets) {
        const segment::SurfQuality q = segment::surface_quality(s, gp.bin_size, gp.fold_thresh);
        valid += q.valid;
        fold_w += q.fold_rate * static_cast<f64>(q.valid);
        dfold_w += q.distant_fold * static_cast<f64>(q.valid);
    }
    const f64 inv = valid ? 1.0 / static_cast<f64>(valid) : 0.0;
    std::printf("normal_field %.0f ms\ntrace        %.0f ms  (sheets=%zu valid=%lld  fold%%=%.3f selfX%%=%.3f)\ngraph        %.0f ms  (spacing=%.1f clusters=%d wraps[%d..%d] conflicts=%d)\nTOTAL        %.0f ms  [outer=%d inner=%d ct_ds=%d]\n",
                ms(t0, t1), ms(t1, t2), R.sheets.size(), (long long)valid, 100.0 * fold_w * inv, 100.0 * dfold_w * inv,
                ms(t2, t3), static_cast<double>(g.spacing), g.cluster_count, g.wrap_lo, g.wrap_hi, g.winding_conflicts,
                ms(t0, t3), fouter, finner, ctds);
    return 0;
}
