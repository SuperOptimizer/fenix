// test_multiseg_render.cpp — multi-segment trace on a 1024^3 cube + a composited overlay render:
// raw CT (grayscale) <- semi-transparent red surface-prediction overlay <- traced segments as bright
// fully-opaque HSV colours (one hue per segment). Three axial (constant-z) slices side by side.
// Usage: test_multiseg_render <ct.nrrd> <surf.nrrd> [grid maxsheets seedstride thresh z0 outdir]
#include "core/core.hpp"
#include "io/jpeg.hpp"
#include "io/nrrd.hpp"
#include "segment/grow.hpp"

#include <chrono>
#include <cmath>
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: test_multiseg_render <ct.nrrd> <surf.nrrd> [grid maxsheets seedstride thresh z0 outdir]\n");
        return 0;
    }
    const std::string ct_path = argv[1], surf_path = argv[2];
    const int grid = argc > 3 ? std::atoi(argv[3]) : 1024;
    const int max_sheets = argc > 4 ? std::atoi(argv[4]) : 32;
    const int seed_stride = argc > 5 ? std::atoi(argv[5]) : 48;
    const f32 thr = argc > 6 ? static_cast<f32>(std::atof(argv[6])) : 0.12f;
    const int z0 = argc > 7 ? std::atoi(argv[7]) : 512;
    const std::string out = argc > 8 ? argv[8] : "data/fenix_multiseg.jpg";

    // CT for display (0..255 already); prediction scaled to u8 for the trace.
    auto pm = io::nrrd_max(surf_path);
    auto cm = io::nrrd_max(ct_path);
    if (!pm || !cm) { std::printf("read failed\n"); return 1; }
    const f32 pscale = (*pm > 2.0f) ? 1.0f : 255.0f;
    auto ctr = io::read_nrrd_u8(ct_path, (*cm > 2.0f) ? 1.0f : 255.0f);
    auto pr = io::read_nrrd_u8(surf_path, pscale);
    if (!ctr || !pr) { std::printf("read failed\n"); return 1; }
    Volume<u8> ct = std::move(*ctr), pred = std::move(*pr);
    const Extent3 D = pred.dims();
    auto cv = ct.view();
    std::printf("CT %lldx%lldx%lld  pred max=%.2f (pscale=%.0f)\n", (long long)D.z, (long long)D.y, (long long)D.x, *pm, pscale);

    segment::GrowParams gp;
    gp.step = 2.0f;
    gp.surf_thresh = thr * 255.0f;
    gp.snap_radius = gp.step * 1.5f;
    gp.fold_thresh = 6;
    gp.grid = grid;
    gp.river_radius = 2;

    auto t0 = clk::now();
    const segment::NormalField nf = segment::compute_normal_field<u8>(pred.view(), 8);
    auto t1 = clk::now();
    const f32 seed_thresh = std::max(gp.surf_thresh * 1.5f, 56.0f);
    segment::VolumeResult R = segment::trace_volume<u8>(pred.view(), nf, gp, max_sheets, /*min_valid=*/3000,
                                                        seed_stride, seed_thresh, VolumeView<const u8>{});
    auto t2 = clk::now();
    std::printf("normal %.1fs  trace %.1fs  sheets=%zu (of %lld candidates)\n",
                std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(),
                R.sheets.size(), (long long)R.seed_candidates);
    if (R.sheets.empty()) { std::printf("no sheets traced\n"); return 0; }

    // --- composite render: 3 axial slices (z0-160, z0, z0+160), each CT|pred|segments ---
    const int W = static_cast<int>(D.x), H = static_cast<int>(D.y), pad = 8;
    const int zslices[3] = {z0 - 160, z0, z0 + 160};
    const f32 band = 2.0f;          // a segment cell is drawn on a slice if |cell.z - z| <= band
    io::Image img;
    img.w = 3 * W + 4 * pad;
    img.h = H + 2 * pad;
    img.comps = 3;
    img.px.assign(static_cast<usize>(img.w) * static_cast<usize>(img.h) * 3, 18);
    const usize N = R.sheets.size();

    for (int s = 0; s < 3; ++s) {
        const int z = zslices[s];
        if (z < 0 || z >= D.z) continue;
        const int ox = pad + s * (W + pad), oy = pad;
        // base: CT grayscale + semi-transparent red prediction overlay
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const f32 g = static_cast<f32>(cv(z, y, x));
                const f32 p = static_cast<f32>(pred.view()(z, y, x)) / 255.0f;
                const f32 a = std::clamp(p * 1.5f, 0.0f, 0.6f);  // red overlay alpha
                const usize o = (static_cast<usize>(oy + y) * static_cast<usize>(img.w) + static_cast<usize>(ox + x)) * 3;
                img.px[o] = static_cast<u8>(g * (1 - a) + 255.0f * a);
                img.px[o + 1] = static_cast<u8>(g * (1 - a));
                img.px[o + 2] = static_cast<u8>(g * (1 - a));
            }
        // segments on top: bright opaque HSV per segment (golden-ratio hue spacing)
        for (usize k = 0; k < N; ++k) {
            u8 cr, cg, cb;
            hsv(std::fmod(static_cast<f32>(k) * 0.61803f, 1.0f), 0.9f, 1.0f, cr, cg, cb);
            const Surface& Sf = R.sheets[k];
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
                        img.px[o] = cr; img.px[o + 1] = cg; img.px[o + 2] = cb;
                    }
            }
        }
    }
    if (io::write_jpeg(out, img, 92)) std::printf("wrote %s (%dx%d): %zu segments over CT+pred at z=%d,%d,%d\n",
                                                  out.c_str(), img.w, img.h, N, zslices[0], zslices[1], zslices[2]);
    else std::printf("jpeg write failed\n");
    return 0;
}
