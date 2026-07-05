// test_trace_perf — crop a cube to <csz>^3 and run the trace pipeline with per-stage timing, for
// perf profiling + optimization. No rendering / dumping; pure compute.
// Usage: test_trace_perf <ct.fxvol> <surf.fxvol> [csz maxsheets seedstride thresh ctweight]
#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "bench_vol.hpp"
#include "preprocess/aircut.hpp"
#include "segment/grow.hpp"
#include "segment/patch_graph.hpp"

#include <functional>
#include <cstdlib>
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
        std::printf("usage: test_trace_perf <ct.fxvol> <surf.fxvol> [csz maxsheets seedstride thresh ctweight ctds fouter finner ctskip araptol]\n");
        return 0;
    }
    const std::string ct_path = argv[1], surf_path = argv[2];
    const s64 csz = argc > 3 ? std::atoll(argv[3]) : 512;
    const int max_sheets = argc > 4 ? std::atoi(argv[4]) : 24;
    const int seed_stride = argc > 5 ? std::atoi(argv[5]) : 32;
    const f32 thr = argc > 6 ? static_cast<f32>(std::atof(argv[6])) : 0.12f;
    const f32 ctw = argc > 7 ? static_cast<f32>(std::atof(argv[7])) : 0.8f;
    const int ctds = argc > 8 ? std::atoi(argv[8]) : 1;  // CT-term downsample (1 = full res)
    const int fouter = argc > 9 ? std::atoi(argv[9]) : 12;   // max final ARAP outer iters (cap)
    const int finner = argc > 10 ? std::atoi(argv[10]) : 25;  // final ARAP inner sweeps
    const f32 ctskip = argc > 11 ? static_cast<f32>(std::atof(argv[11])) : 1.5f;   // skip CT where pred>=this (0=off)
    const f32 araptol = argc > 12 ? static_cast<f32>(std::atof(argv[12])) : 0.03f;  // adaptive ARAP stop (0=fixed)
    const int qbits = argc > 13 ? std::atoi(argv[13]) : 8;   // quantize PRED to this many bits (8 = no change)
    const int ctbits = argc > 14 ? std::atoi(argv[14]) : 8;  // quantize CT to this many bits (8 = no change)
    const int tile_core = argc > 15 ? std::atoi(argv[15]) : 0;  // >0 = tiled tracer with this core tile size
    const int halo = argc > 16 ? std::atoi(argv[16]) : 24;      // tile halo (growth-context overlap)
    const int overlap = argc > 17 ? std::atoi(argv[17]) : 0;    // seam overlap for fragment stitching

    auto pm = bench::peak(surf_path), cm = bench::peak(ct_path);
    if (!pm || !cm) { std::printf("read failed\n"); return 1; }
    auto ctr = bench::load_u8(ct_path, (*cm > 2.0f) ? 1.0f : 255.0f);
    auto pr = bench::load_u8(surf_path, (*pm > 2.0f) ? 1.0f : 255.0f);
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
    // Dynamic-range probe: round pred/CT to N levels (still stored u8, so this isolates the QUALITY
    // effect of reduced bit-depth from any packing/bandwidth change). 8 = unchanged. Independent so the
    // pred can be probed without disturbing the CT-histogram-derived air-cut (and vice-versa).
    auto qz = [&](Volume<u8>& vol, int bits) {
        if (bits < 1 || bits >= 8) return;
        const f32 sc = static_cast<f32>((1 << bits) - 1) / 255.0f;
        auto v = vol.view();
        parallel_for_z(vol.dims(), [&](s64 z) {
            for (s64 y = 0; y < vol.dims().y; ++y)
                for (s64 x = 0; x < vol.dims().x; ++x)
                    v(z, y, x) = static_cast<u8>(std::lround(std::lround(static_cast<f32>(v(z, y, x)) * sc) / sc));
        });
    };
    qz(pred, qbits); qz(ct, ctbits);
    const f32 ctt = preprocess::air_cut<u8>(ct.view(), 0.0f, 256.0f);
    FENIX_INFO("perf", "crop {}^3 (from {}x{}x{} @ {},{},{})  air-cut={:.0f} predbits={} ctbits={}",
               csz, D.z, D.y, D.x, oz, oy, ox, static_cast<double>(ctt), qbits, ctbits);

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
    gp.ct_skip = ctskip;
    gp.arap_tol = araptol;
    const f32 seed_thresh = std::max(gp.surf_thresh * 1.5f, 56.0f);
    // CT term on a downsampled grid (ct_ds>1): less memory traffic in the snap data-term hot loop.
    Volume<u8> ctsmall;
    if (ctds > 1) ctsmall = downsample_u8(ct.view(), ctds);
    const VolumeView<const u8> ct_term = ctds > 1 ? ctsmall.view() : ct.view();

    auto t0 = clk::now();
    const segment::NormalField nf = segment::compute_normal_field<u8>(pred.view(), 8);
    auto t1 = clk::now();
    segment::VolumeResult R;
    if (tile_core > 0)  // tiled tracer: fragment threshold (smaller than a whole-cube sheet) + a high cap
        R = segment::trace_volume_tiled<u8>(pred.view(), ct.view(), gp, 100000, 300, seed_stride, seed_thresh, tile_core, halo, overlap);
    else
        R = segment::trace_volume<u8>(pred.view(), nf, gp, max_sheets, 3000, seed_stride, seed_thresh, ct_term);
    auto t2 = clk::now();
    // (the tiled tracer logs its own fragment count via FENIX_INFO("segment", ...))
    annotate::Umbilicus umb;
    umb.z = {0.0f, static_cast<f32>(csz)};
    umb.y = {static_cast<f32>(csz) * 0.5f, static_cast<f32>(csz) * 0.5f};
    umb.x = {static_cast<f32>(csz) * 0.5f, static_cast<f32>(csz) * 0.5f};
    segment::PatchGraphParams pgp;
    pgp.step = gp.step;
    auto ga = clk::now();
    segment::PatchGraph g = segment::build_patch_graph<u8>(R.sheets, umb, ct.view(), pgp);  // CT-valley Δwrap
    auto gb = clk::now();
    segment::merge_same_sheet(g, 2);  // consensus gate: cut fused-weld bridges (no transitive over-merge)
    auto gc = clk::now();
    segment::assign_windings(g);
    auto t3 = clk::now();
    {
        int nm = 0, nl = 0, nc = 0;
        for (const segment::PatchEdge& e : g.edges)
            nm += e.kind == segment::EdgeKind::Merge, nl += e.kind == segment::EdgeKind::Link, nc += e.kind == segment::EdgeKind::Conflict;
        std::vector<int> csz(static_cast<usize>(std::max(1, g.cluster_count)), 0);
        for (const segment::Patch& pp : g.patches) csz[static_cast<usize>(pp.cluster)]++;
        int singletons = 0, biggest = 0;
        for (int s : csz) { singletons += (s == 1); biggest = std::max(biggest, s); }
        FENIX_INFO("perf", "graph phases: build={:.0f} merge={:.0f} winding={:.0f} ms | edges M={} L={} C={} | clusters: {} singletons={} biggest={}",
                   ms(ga, gb), ms(gb, gc), ms(gc, t3), nm, nl, nc, g.cluster_count, singletons, biggest);
        // gap-distribution diagnostic (the "predictions touching" test): for co-normal (parallel-sheet)
        // pairs, |gap|/spacing should peak near 1 (one wrap apart). A big mass below ~0.5 means adjacent
        // wraps TOUCH (gap collapses) -> dwrap rounds to 0 -> not a Link -> the winding counter stalls.
        int gh[8] = {0}, dwh[6] = {0};  // gh: |gap|/spacing in .25 bins (last >=2); dwh: |dwrap| 0..5+
        for (const segment::PatchEdge& e : g.edges) {
            if (std::abs(e.conormal) < pgp.conormal_min) continue;
            gh[std::min(7, static_cast<int>(std::abs(e.gap) / std::max(1e-3f, g.spacing) / 0.25f))]++;
            dwh[std::min(5, std::abs(e.dwrap))]++;
        }
        FENIX_INFO("perf", "conormal-edge |gap|/spacing hist (.25 bins, last>=2): {} {} {} {} {} {} {} {}",
                   gh[0], gh[1], gh[2], gh[3], gh[4], gh[5], gh[6], gh[7]);
        FENIX_INFO("perf", "conormal-edge |dwrap| hist 0..5+: {} {} {} {} {} {}", dwh[0], dwh[1], dwh[2], dwh[3], dwh[4], dwh[5]);
        // connectivity: does the Δwrap=±1 (Link) graph span all patches, or does dropping the dwrap>=2
        // edges shatter it into islands the winding solve then re-gauges to 0 (collapsing the range)?
        auto components = [&](int max_dwrap) {
            std::vector<int> par(g.patches.size());
            for (usize i = 0; i < par.size(); ++i) par[i] = static_cast<int>(i);
            std::function<int(int)> find = [&](int x) { while (par[static_cast<usize>(x)] != x) x = par[static_cast<usize>(x)] = par[static_cast<usize>(par[static_cast<usize>(x)])]; return x; };
            for (const segment::PatchEdge& e : g.edges)
                if (std::abs(e.conormal) >= pgp.conormal_min && std::abs(e.dwrap) >= 1 && std::abs(e.dwrap) <= max_dwrap)
                    par[static_cast<usize>(find(e.a))] = find(e.b);
            int c = 0;
            for (usize i = 0; i < par.size(); ++i) c += (find(static_cast<int>(i)) == static_cast<int>(i));
            return c;
        };
        FENIX_INFO("perf", "winding-graph components: Link-only(dwrap=1)={}  Link+dwrap2={}  (patches={})",
                   components(1), components(2), g.patches.size());
    }

    s64 valid = 0;
    f64 fold_w = 0, dfold_w = 0, nsm_w = 0;  // valid-weighted quality (fold/selfX want ~0; nsm = mean dihedral deg)
    for (const Surface& s : R.sheets) {
        const segment::SurfQuality q = segment::surface_quality(s, gp.bin_size, gp.fold_thresh);
        valid += q.valid;
        fold_w += q.fold_rate * static_cast<f64>(q.valid);
        dfold_w += q.distant_fold * static_cast<f64>(q.valid);
        nsm_w += q.normal_smooth * static_cast<f64>(q.valid);
    }
    const f64 inv = valid ? 1.0 / static_cast<f64>(valid) : 0.0;
    FENIX_INFO("perf", "normal_field {:.0f} ms", ms(t0, t1));
    FENIX_INFO("perf", "trace {:.0f} ms  (sheets={} valid={}  fold%={:.3f} selfX%={:.3f} dihed={:.2f})",
               ms(t1, t2), R.sheets.size(), valid, 100.0 * fold_w * inv, 100.0 * dfold_w * inv, nsm_w * inv);
    FENIX_INFO("perf", "graph {:.0f} ms  (spacing={:.1f} clusters={} wraps[{}..{}] conflicts={})",
               ms(t2, t3), static_cast<double>(g.spacing), g.cluster_count, g.wrap_lo, g.wrap_hi, g.winding_conflicts);
    FENIX_INFO("perf", "TOTAL {:.0f} ms  [outer<={} inner={} ct_ds={} ct_skip={:.2f} arap_tol={:.3f} qbits={}]",
               ms(t0, t3), fouter, finner, ctds, static_cast<double>(ctskip), static_cast<double>(araptol), qbits);
    return 0;
}
