// test_trace_merge.cpp — denser multi-seed trace (CT-ridge data term) + same-sheet MERGE. Renders
// two axial-slice panels: segments coloured by raw index (fragmented) vs by merged cluster (fragments
// of one physical sheet share a colour). Reports spacing / clusters / windings / conflicts.
// Usage: test_trace_merge <ct.nrrd> <surf.nrrd> [grid maxsheets seedstride thresh ctweight z0 out]
#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "io/jpeg.hpp"
#include "io/nrrd.hpp"
#include "preprocess/aircut.hpp"
#include "segment/grow.hpp"
#include "segment/patch_graph.hpp"
#include "winding/cosegment.hpp"
#include "winding/patch_field.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

using namespace fenix;
using clk = std::chrono::steady_clock;

static void hsv(f32 h, f32 s, f32 v, u8& R, u8& G, u8& B) {
    h = h - std::floor(h);
    const int i = static_cast<int>(h * 6.0f) % 6;
    const f32 f = h * 6.0f - std::floor(h * 6.0f);
    const f32 p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    f32 r = v, g = v, b = v;
    if (i == 0) { r = v; g = t; b = p; } else if (i == 1) { r = q; g = v; b = p; }
    else if (i == 2) { r = p; g = v; b = t; } else if (i == 3) { r = p; g = q; b = v; }
    else if (i == 4) { r = t; g = p; b = v; } else { r = v; g = p; b = q; }
    R = static_cast<u8>(r * 255); G = static_cast<u8>(g * 255); B = static_cast<u8>(b * 255);
}

// CT gray + red prediction overlay + segments painted with a per-sheet RGB colour `col[k]`.
static void draw_panel(io::Image& img, int ox, int oy, int W, int H, int z, VolumeView<const u8> ct,
                       VolumeView<const u8> pred, const std::vector<Surface>& sheets,
                       const std::vector<std::array<u8, 3>>& col, f32 band) {
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const f32 g = static_cast<f32>(ct(z, y, x));
            const f32 p = static_cast<f32>(pred(z, y, x)) / 255.0f;
            const f32 a = std::clamp(p * 1.5f, 0.0f, 0.6f);
            const usize o = (static_cast<usize>(oy + y) * static_cast<usize>(img.w) + static_cast<usize>(ox + x)) * 3;
            img.px[o] = static_cast<u8>(g * (1 - a) + 255.0f * a);
            img.px[o + 1] = static_cast<u8>(g * (1 - a));
            img.px[o + 2] = static_cast<u8>(g * (1 - a));
        }
    for (usize k = 0; k < sheets.size(); ++k) {
        const Surface& Sf = sheets[k];
        for (usize i = 0; i < Sf.valid.size(); ++i) {
            if (!Sf.valid[i]) continue;
            const Vec3f c = Sf.coord[i];
            if (std::abs(c.z - static_cast<f32>(z)) > band) continue;
            const int x = static_cast<int>(std::lround(c.x)), y = static_cast<int>(std::lround(c.y));
            for (int dy = -1; dy <= 0; ++dy)
                for (int dx = -1; dx <= 0; ++dx) {
                    const int X = ox + x + dx, Y = oy + y + dy;
                    if (X < ox || Y < oy || X >= ox + W || Y >= oy + H) continue;
                    const usize o = (static_cast<usize>(Y) * static_cast<usize>(img.w) + static_cast<usize>(X)) * 3;
                    img.px[o] = col[k][0]; img.px[o + 1] = col[k][1]; img.px[o + 2] = col[k][2];
                }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: test_trace_merge <ct.nrrd> <surf.nrrd> [grid maxsheets seedstride thresh ctweight z0 out]\n");
        return 0;
    }
    const std::string ct_path = argv[1], surf_path = argv[2];
    const int grid = argc > 3 ? std::atoi(argv[3]) : 1024;
    const int max_sheets = argc > 4 ? std::atoi(argv[4]) : 60;
    const int seed_stride = argc > 5 ? std::atoi(argv[5]) : 32;
    const f32 thr = argc > 6 ? static_cast<f32>(std::atof(argv[6])) : 0.12f;
    const f32 ctw = argc > 7 ? static_cast<f32>(std::atof(argv[7])) : 0.8f;
    const int z0 = argc > 8 ? std::atoi(argv[8]) : 512;
    const std::string out = argc > 9 ? argv[9] : "data/fenix_trace_merge.jpg";
    const int tile_core = argc > 10 ? std::atoi(argv[10]) : 0;  // >0 = tiled tracer
    const int halo = argc > 11 ? std::atoi(argv[11]) : 24;
    const int euler = argc > 12 ? std::atoi(argv[12]) : 0;      // 1 = Eulerian (normal-driven) winding
    const int eds = argc > 13 ? std::atoi(argv[13]) : 8;        // Eulerian field downsample
    const int eiters = argc > 14 ? std::atoi(argv[14]) : 1500;  // Eulerian GS iterations
    const int coseg = argc > 15 ? std::atoi(argv[15]) : 0;      // 1 = Stage-D cosegment fill (neighbour-informed)

    auto pm = io::nrrd_max(surf_path), cm = io::nrrd_max(ct_path);
    if (!pm || !cm) { std::printf("read failed\n"); return 1; }
    auto ctr = io::read_nrrd_u8(ct_path, (*cm > 2.0f) ? 1.0f : 255.0f);
    auto pr = io::read_nrrd_u8(surf_path, (*pm > 2.0f) ? 1.0f : 255.0f);
    if (!ctr || !pr) { std::printf("read failed\n"); return 1; }
    Volume<u8> ct = std::move(*ctr), pred = std::move(*pr);
    const Extent3 D = pred.dims();
    const f32 ctt = preprocess::air_cut<u8>(ct.view(), 0.0f, 256.0f);

    segment::GrowParams gp;
    gp.step = 2.0f;
    gp.surf_thresh = thr * 255.0f;
    gp.snap_radius = gp.step * 1.5f;
    gp.fold_thresh = 6;
    gp.grid = grid;
    gp.river_radius = 2;
    gp.ct_thresh = ctt;
    gp.ct_weight = ctw;
    gp.ct_barrier = 0.12f;  // CT-valley growth barrier: don't drift across a touching wrap (ct_valley.hpp)
    const f32 seed_thresh = std::max(gp.surf_thresh * 1.5f, 56.0f);

    auto t0 = clk::now();
    const segment::NormalField nf = segment::compute_normal_field<u8>(pred.view(), 8);
    segment::VolumeResult R = tile_core > 0
        ? segment::trace_volume_tiled<u8>(pred.view(), ct.view(), gp, 100000, 300, seed_stride, seed_thresh, tile_core, halo)
        : segment::trace_volume<u8>(pred.view(), nf, gp, max_sheets, 3000, seed_stride, seed_thresh, ct.view());
    auto t1 = clk::now();

    // patch graph: a 1024^3 region of a scroll is locally parallel laminations (umbilicus effectively
    // outside) -> orient to the PCA dominant normal axis.
    annotate::Umbilicus umb;
    umb.z = {0.0f, static_cast<f32>(D.z)};
    umb.y = {static_cast<f32>(D.y) * 0.5f, static_cast<f32>(D.y) * 0.5f};
    umb.x = {static_cast<f32>(D.x) * 0.5f, static_cast<f32>(D.x) * 0.5f};
    segment::PatchGraphParams pgp;
    pgp.step = gp.step;

    // Optional Stage-D cosegment: coherent (band-Eulerian) windings -> fill each fragment's weak/empty
    // cells from the NEIGHBOURING wraps via the winding field. The original "fill the blank" goal.
    if (coseg) {
        s64 vbefore = 0;
        for (const Surface& s : R.sheets) vbefore += s.valid_count();
        winding::CosegParams cp;
        cp.full = D;
        cp.rounds = 3;
        cp.eulerian = true;
        cp.efield.ds = eds;
        cp.efield.iters = eiters;
        cp.efield.band = std::max(2, static_cast<int>(std::lround(static_cast<double>(gp.step * 3.5f) / std::max(1, eds))) + 2);
        cp.field.ds = eds;
        cp.field.iters = 120;
        cp.fill.step = gp.step;
        const winding::CosegReport rep = winding::cosegment_refine(R.sheets, umb, pgp, cp);
        s64 vafter = 0;
        for (const Surface& s : R.sheets) vafter += s.valid_count();
        FENIX_INFO("merge", "coseg: valid {} -> {} (+{} filled from neighbours)  spacing={:.1f} clusters={} wraps[{}..{}] monotonicity={:.3f}",
                    vbefore, vafter, vafter - vbefore, static_cast<double>(rep.spacing),
                    rep.clusters, rep.wrap_lo, rep.wrap_hi, static_cast<double>(rep.monotonicity));
    }

    segment::PatchGraph g = segment::build_patch_graph<u8>(R.sheets, umb, ct.view(), pgp);  // CT-valley Δwrap
    segment::merge_same_sheet(g, 2);  // consensus gate: cut fused-weld bridges (no transitive over-merge)
    if (euler) {  // robust stitch: integer windings from the global normal-driven Eulerian field
        winding::FieldParams fp;
        fp.ds = eds;
        fp.iters = eiters;
        fp.band = std::max(2, static_cast<int>(std::lround(static_cast<double>(g.spacing) / std::max(1, eds))) + 2);
        winding::WindingField wf = winding::build_eulerian_winding_field(g.patches, D, g.spacing, fp);
        winding::assign_windings_from_field(g, wf);
    } else {
        segment::assign_windings(g);
    }
    auto t2 = clk::now();

    FENIX_INFO("merge", "trace {:.1f}s  sheets={}  | graph {:.1f}s spacing={:.1f} clusters={} wraps[{}..{}] conflicts={}",
               std::chrono::duration<double>(t1 - t0).count(), R.sheets.size(),
               std::chrono::duration<double>(t2 - t1).count(), static_cast<double>(g.spacing),
               g.cluster_count, g.wrap_lo, g.wrap_hi, g.winding_conflicts);
    int merged_groups = 0;
    {
        std::vector<int> sz(static_cast<usize>(std::max(1, g.cluster_count)), 0);
        for (const segment::Patch& p : g.patches) sz[static_cast<usize>(p.cluster)]++;
        for (int s : sz) if (s >= 2) ++merged_groups;
    }
    FENIX_INFO("merge", "merge: {} segments -> {} clusters ({} multi-segment groups joined)",
               R.sheets.size(), g.cluster_count, merged_groups);

    const usize N = R.sheets.size();
    const f32 wr = static_cast<f32>(std::max(1, g.wrap_hi - g.wrap_lo));
    std::vector<std::array<u8, 3>> c_idx(N), c_clu(N), c_win(N);
    for (usize k = 0; k < N; ++k) {
        hsv(std::fmod(static_cast<f32>(k) * 0.61803f, 1.0f), 0.9f, 1.0f, c_idx[k][0], c_idx[k][1], c_idx[k][2]);
        hsv(std::fmod(static_cast<f32>(g.patches[k].cluster) * 0.61803f, 1.0f), 0.9f, 1.0f, c_clu[k][0], c_clu[k][1], c_clu[k][2]);
        hsv(0.7f * (1.0f - static_cast<f32>(g.patches[k].wrap - g.wrap_lo) / wr), 0.95f, 1.0f, c_win[k][0], c_win[k][1], c_win[k][2]);
    }

    const int W = static_cast<int>(D.x), H = static_cast<int>(D.y), pad = 8;
    io::Image img;
    img.w = 3 * W + 4 * pad;
    img.h = H + 2 * pad;
    img.comps = 3;
    img.px.assign(static_cast<usize>(img.w) * static_cast<usize>(img.h) * 3, 18);
    draw_panel(img, pad, pad, W, H, z0, ct.view(), pred.view(), R.sheets, c_idx, 2.0f);
    draw_panel(img, 2 * pad + W, pad, W, H, z0, ct.view(), pred.view(), R.sheets, c_clu, 2.0f);
    draw_panel(img, 3 * pad + 2 * W, pad, W, H, z0, ct.view(), pred.view(), R.sheets, c_win, 2.0f);
    if (io::write_jpeg(out, img, 92)) FENIX_INFO("merge", "wrote {} (per-segment | per merged-sheet | winding {}..{}, z={})", out, g.wrap_lo, g.wrap_hi, z0);
    else FENIX_ERROR("merge", "jpeg write failed");
    return 0;
}
