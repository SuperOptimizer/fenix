// test_trace_improve.cpp — does the CT ridge data term help the trace span prediction dropouts?
// Trace the same cube twice (prediction-only vs prediction+CT-ridge), report coverage, and render a
// side-by-side at one axial slice: CT grayscale + red prediction overlay + segments (HSV by index).
// Usage: test_trace_improve <ct.fxvol> <surf.fxvol> [grid maxsheets seedstride thresh ctweight z0 out]
#include "core/core.hpp"
#include "io/jpeg.hpp"
#include "bench_vol.hpp"
#include "preprocess/aircut.hpp"
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

static s64 total_valid(const segment::VolumeResult& R) {
    s64 n = 0;
    for (const Surface& s : R.sheets) n += s.valid_count();
    return n;
}

// draw one panel: CT gray + red prediction overlay + segments (HSV by index) within band of slice z.
static void draw_panel(io::Image& img, int ox, int oy, int W, int H, int z, VolumeView<const u8> ct,
                       VolumeView<const u8> pred, const segment::VolumeResult& R, f32 band) {
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
    for (usize k = 0; k < R.sheets.size(); ++k) {
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: test_trace_improve <ct.fxvol> <surf.fxvol> [grid maxsheets seedstride thresh ctweight z0 out]\n");
        return 0;
    }
    const std::string ct_path = argv[1], surf_path = argv[2];
    const int grid = argc > 3 ? std::atoi(argv[3]) : 1024;
    const int max_sheets = argc > 4 ? std::atoi(argv[4]) : 32;
    const int seed_stride = argc > 5 ? std::atoi(argv[5]) : 48;
    const f32 thr = argc > 6 ? static_cast<f32>(std::atof(argv[6])) : 0.12f;
    const f32 ctw = argc > 7 ? static_cast<f32>(std::atof(argv[7])) : 0.8f;
    const int z0 = argc > 8 ? std::atoi(argv[8]) : 512;
    const std::string out = argc > 9 ? argv[9] : "data/fenix_trace_improve.jpg";

    auto pm = bench::peak(surf_path), cm = bench::peak(ct_path);
    if (!pm || !cm) { std::printf("read failed\n"); return 1; }
    auto ctr = bench::load_u8(ct_path, (*cm > 2.0f) ? 1.0f : 255.0f);
    auto pr = bench::load_u8(surf_path, (*pm > 2.0f) ? 1.0f : 255.0f);
    if (!ctr || !pr) { std::printf("read failed\n"); return 1; }
    Volume<u8> ct = std::move(*ctr), pred = std::move(*pr);
    const Extent3 D = pred.dims();
    const f32 ctt = preprocess::air_cut<u8>(ct.view(), 0.0f, 256.0f);  // Otsu valley -> papyrus/air split
    std::printf("CT %lldx%lldx%lld  air-cut=%.0f  ct_weight=%.2f\n", (long long)D.z, (long long)D.y, (long long)D.x, ctt, ctw);

    segment::GrowParams gp;
    gp.step = 2.0f;
    gp.surf_thresh = thr * 255.0f;
    gp.snap_radius = gp.step * 1.5f;  // tight: span a crack via CT without hopping to the next wrap
    gp.fold_thresh = 6;
    gp.grid = grid;
    gp.river_radius = 2;
    const f32 seed_thresh = std::max(gp.surf_thresh * 1.5f, 56.0f);
    const segment::NormalField nf = segment::compute_normal_field<u8>(pred.view(), 8);

    // A: prediction only
    segment::GrowParams gpA = gp;
    auto a0 = clk::now();
    segment::VolumeResult RA = segment::trace_volume<u8>(pred.view(), nf, gpA, max_sheets, 3000, seed_stride, seed_thresh, VolumeView<const u8>{});
    auto a1 = clk::now();
    // B: prediction + CT ridge (so growth survives prediction dropouts at cracks)
    segment::GrowParams gpB = gp;
    gpB.ct_thresh = ctt;
    gpB.ct_weight = ctw;
    segment::VolumeResult RB = segment::trace_volume<u8>(pred.view(), nf, gpB, max_sheets, 3000, seed_stride, seed_thresh, ct.view());
    auto a2 = clk::now();

    const s64 vA = total_valid(RA), vB = total_valid(RB);
    std::printf("A pred-only : %.1fs  sheets=%zu  valid=%lld\n", std::chrono::duration<double>(a1 - a0).count(), RA.sheets.size(), (long long)vA);
    std::printf("B pred+CT   : %.1fs  sheets=%zu  valid=%lld  (%+.1f%% coverage)\n",
                std::chrono::duration<double>(a2 - a1).count(), RB.sheets.size(), (long long)vB,
                100.0 * static_cast<double>(vB - vA) / static_cast<double>(std::max<s64>(vA, 1)));

    const int W = static_cast<int>(D.x), H = static_cast<int>(D.y), pad = 8;
    io::Image img;
    img.w = 2 * W + 3 * pad;
    img.h = H + 2 * pad;
    img.comps = 3;
    img.px.assign(static_cast<usize>(img.w) * static_cast<usize>(img.h) * 3, 18);
    draw_panel(img, pad, pad, W, H, z0, ct.view(), pred.view(), RA, 2.0f);
    draw_panel(img, 2 * pad + W, pad, W, H, z0, ct.view(), pred.view(), RB, 2.0f);
    if (io::write_jpeg(out, img, 92)) std::printf("wrote %s (left: pred-only | right: pred+CT, z=%d)\n", out.c_str(), z0);
    else std::printf("jpeg write failed\n");
    return 0;
}
