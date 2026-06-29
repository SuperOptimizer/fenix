// test_trace_fit.cpp — Phase 2 end to end on REAL data: trace -> CT-valley patch graph + winding ->
// estimate the umbilicus (polar centre) from the patch normals -> bridge the assigned windings into the
// diffeomorphic fit -> report whether the fitted global model reproduces the windings COHERENTLY (the
// continuous, fold-free winding field that unrolling samples through T^-1). The fit consumes the
// assigned windings; it globalizes + smooths them (it does not invent wraps — that is the later EM).
#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "io/nrrd.hpp"
#include "preprocess/aircut.hpp"
#include "segment/grow.hpp"
#include "segment/patch_graph.hpp"
#include "winding/fit_bridge.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace fenix;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: test_trace_fit <ct.nrrd> <surf.nrrd> [csz tile_core halo]\n");
        return 0;
    }
    const std::string ct_path = argv[1], surf_path = argv[2];
    const s64 csz = argc > 3 ? std::atoll(argv[3]) : 512;
    const int tile_core = argc > 4 ? std::atoi(argv[4]) : 128;
    const int halo = argc > 5 ? std::atoi(argv[5]) : 24;

    auto pm = io::nrrd_max(surf_path), cm = io::nrrd_max(ct_path);
    if (!pm || !cm) { std::printf("read failed\n"); return 1; }
    auto ctr = io::read_nrrd_u8(ct_path, (*cm > 2.0f) ? 1.0f : 255.0f);
    auto pr = io::read_nrrd_u8(surf_path, (*pm > 2.0f) ? 1.0f : 255.0f);
    if (!ctr || !pr) { std::printf("read failed\n"); return 1; }
    const Volume<u8> ctf = std::move(*ctr), predf = std::move(*pr);
    const Extent3 D = predf.dims();
    const s64 oz = (D.z - csz) / 2, oy = (D.y - csz) / 2, ox = (D.x - csz) / 2;
    Volume<u8> ct(Extent3{csz, csz, csz}), pred(Extent3{csz, csz, csz});
    {
        auto cv = ct.view(), pv = pred.view();
        auto cf = ctf.view(), pf = predf.view();
        parallel_for_z(Extent3{csz, csz, csz}, [&](s64 z) {
            for (s64 y = 0; y < csz; ++y)
                for (s64 x = 0; x < csz; ++x) { cv(z, y, x) = cf(oz + z, oy + y, ox + x); pv(z, y, x) = pf(oz + z, oy + y, ox + x); }
        });
    }
    const f32 ctt = preprocess::air_cut<u8>(ct.view(), 0.0f, 256.0f);

    segment::GrowParams gp;
    gp.step = 2.0f;
    gp.surf_thresh = 0.12f * 255.0f;
    gp.snap_radius = gp.step * 1.5f;
    gp.fold_thresh = 6;
    gp.grid = static_cast<int>(csz);
    gp.river_radius = 2;
    gp.ct_thresh = ctt;
    gp.ct_weight = 0.8f;
    gp.ct_barrier = 0.12f;
    const f32 seed_thresh = std::max(gp.surf_thresh * 1.5f, 56.0f);

    auto t0 = clk::now();
    const segment::VolumeResult R = segment::trace_volume_tiled<u8>(pred.view(), ct.view(), gp, 100000, 300, 32, seed_thresh, tile_core, halo);
    auto t1 = clk::now();

    // initial dummy umbilicus (chunk centre) — only seeds make_patch's normal reference; SignDSU orients.
    annotate::Umbilicus umb;
    umb.z = {0.0f, static_cast<f32>(csz)};
    umb.y = {static_cast<f32>(csz) * 0.5f, static_cast<f32>(csz) * 0.5f};
    umb.x = {static_cast<f32>(csz) * 0.5f, static_cast<f32>(csz) * 0.5f};
    segment::PatchGraphParams pgp;
    pgp.step = gp.step;
    segment::PatchGraph g = segment::build_patch_graph<u8>(R.sheets, umb, ct.view(), pgp);
    segment::merge_same_sheet(g, 2);
    segment::assign_windings(g);
    auto t2 = clk::now();

    // estimate the polar centre from the (now consistently-oriented) patch normals: least-squares
    // intersection of the per-cell radial normal lines, [Σ(I - n nᵀ)] c = Σ(I - n nᵀ) p  in yx.
    f64 A00 = 0, A01 = 0, A11 = 0, b0 = 0, b1 = 0;
    for (const segment::Patch& p : g.patches)
        for (usize i = 0; i < p.pos.size(); i += 8) {
            f32 ny = p.nrm[i].y, nx = p.nrm[i].x;
            const f32 nn = std::sqrt(ny * ny + nx * nx);
            if (nn < 1e-3f) continue;
            ny /= nn;
            nx /= nn;
            const f64 m00 = 1.0 - ny * ny, m01 = -static_cast<f64>(ny) * nx, m11 = 1.0 - nx * nx;
            A00 += m00; A01 += m01; A11 += m11;
            b0 += m00 * p.pos[i].y + m01 * p.pos[i].x;
            b1 += m01 * p.pos[i].y + m11 * p.pos[i].x;
        }
    const f64 det = A00 * A11 - A01 * A01;
    const f32 cy = std::abs(det) > 1e-6 ? static_cast<f32>((A11 * b0 - A01 * b1) / det) : csz * 0.5f;
    const f32 cx = std::abs(det) > 1e-6 ? static_cast<f32>((A00 * b1 - A01 * b0) / det) : csz * 0.5f;
    umb.y = {cy, cy};
    umb.x = {cx, cx};
    FENIX_INFO("fit", "trace {:.1f}s | graph {:.1f}s sheets={} clusters={} wraps[{}..{}] | umbilicus est (y={:.0f} x={:.0f})",
               secs(t0, t1), secs(t1, t2), R.sheets.size(), g.cluster_count, g.wrap_lo, g.wrap_hi,
               static_cast<double>(cy), static_cast<double>(cx));

    // bridge the assigned windings -> fit constraints, and fit the diffeomorphic model.
    winding::BridgeParams bp;
    bp.stride = 6;
    const winding::BridgeOut br = winding::patches_to_constraints(g, bp);
    winding::SpiralModel m;
    m.umbilicus = umb;
    m.dr_per_winding = std::max(2.0f, g.spacing);
    winding::DiffeoFitConfig cfg;
    cfg.flow_dims = Extent3{8, 24, 24};
    cfg.iters_affine = 250;
    cfg.iters_flow = 500;
    cfg.lr_affine = 0.04f;
    cfg.lr_flow = 0.06f;
    cfg.lambda_cowind = 0.3f;
    cfg.flow_steps = 6;
    auto t3 = clk::now();
    const winding::FitResult fr = winding::fit_spiral_diffeo(m, br.targets, br.groups, cfg);
    auto t4 = clk::now();

    // report: how well the fitted CONTINUOUS field reproduces the assigned (discrete) windings, and the
    // fitted winding range (per-patch, at the centroid). RMSE small => the global model is coherent with
    // the assignment; range ~ the input range (the fit globalizes, it does not add wraps — that is EM).
    f64 se = 0;
    s64 nc = 0;
    s32 wlo = 1 << 30, whi = -(1 << 30);
    for (const segment::Patch& p : g.patches) {
        if (p.wrap == segment::kUnassignedWrap || p.pos.empty()) continue;
        const f32 w = m.winding_at(p.centroid);
        se += (static_cast<f64>(w) - p.wrap) * (static_cast<f64>(w) - p.wrap);
        ++nc;
        const s32 rw = static_cast<s32>(std::lround(w));
        wlo = std::min(wlo, rw);
        whi = std::max(whi, rw);
    }
    FENIX_INFO("fit", "fit {:.1f}s ({} targets, {} groups) | loss {:.4f} -> {:.4f} | field-vs-assigned RMSE {:.3f} | fitted wraps[{}..{}]",
               secs(t3, t4), br.targets.size(), br.groups.size(), static_cast<double>(fr.initial_loss),
               static_cast<double>(fr.final_loss), nc ? std::sqrt(se / static_cast<f64>(nc)) : 0.0, nc ? wlo : 0, nc ? whi : 0);
    return 0;
}
