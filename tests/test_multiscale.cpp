// test_multiscale.cpp — end-to-end driver for the multi-scale tracer on a real prediction volume:
// grow MANY seeds (trace_volume) -> patch adjacency graph (spacing, same-sheet MERGE, integer
// winding LINK) -> coupled patch↔winding-field refinement (fill holes from neighbouring wraps, pull
// weak cells onto the field). Prints the graph/health metrics and dumps per-patch grids (with wrap +
// cluster ids) for rendering. Standalone main; with no args it just prints usage (CI-safe).
// Usage: test_multiscale <surf.nrrd> [grid maxsheets seedstride thresh rounds outdir]
#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "io/jpeg.hpp"
#include "io/nrrd.hpp"
#include "segment/grow.hpp"
#include "segment/patch_graph.hpp"
#include "winding/cosegment.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace fenix;
using clk = std::chrono::steady_clock;

static s64 total_valid(const std::vector<Surface>& sheets) {
    s64 n = 0;
    for (const Surface& s : sheets) n += s.valid_count();
    return n;
}

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

// 3 top-down (x,y) panels: cells coloured by winding number, by merge-cluster, by depth.
static void render_panels(const std::string& out, const std::vector<Surface>& sheets,
                          const segment::PatchGraph& gf, Extent3 D) {
    const int W = static_cast<int>(D.x), H = static_cast<int>(D.y), pad = 6;
    io::Image img;
    img.w = 3 * W + 4 * pad;
    img.h = H + 2 * pad;
    img.comps = 3;
    img.px.assign(static_cast<usize>(img.w) * static_cast<usize>(img.h) * 3, 14);
    auto put = [&](int panel, int x, int y, u8 r, u8 g, u8 b) {
        const int X = pad + panel * (W + pad) + x, Y = pad + y;
        if (X < 0 || Y < 0 || X >= img.w || Y >= img.h) return;
        const usize o = (static_cast<usize>(Y) * static_cast<usize>(img.w) + static_cast<usize>(X)) * 3;
        img.px[o] = r; img.px[o + 1] = g; img.px[o + 2] = b;
    };
    f32 zlo = 1e9f, zhi = -1e9f;
    for (const Surface& S : sheets)
        for (usize i = 0; i < S.valid.size(); ++i)
            if (S.valid[i]) { zlo = std::min(zlo, S.coord[i].z); zhi = std::max(zhi, S.coord[i].z); }
    const f32 wr = static_cast<f32>(std::max(1, gf.wrap_hi - gf.wrap_lo));
    const f32 zr = std::max(1.0f, zhi - zlo);
    for (usize k = 0; k < sheets.size(); ++k) {
        const Surface& S = sheets[k];
        u8 wr_r, wr_g, wr_b, cl_r, cl_g, cl_b;
        hsv(0.66f * (1.0f - static_cast<f32>(gf.patches[k].wrap - gf.wrap_lo) / wr), 0.85f, 0.95f, wr_r, wr_g, wr_b);
        hsv(std::fmod(static_cast<f32>(gf.patches[k].cluster) * 0.137f, 1.0f), 0.7f, 0.95f, cl_r, cl_g, cl_b);
        for (usize i = 0; i < S.valid.size(); ++i) {
            if (!S.valid[i]) continue;
            const Vec3f p = S.coord[i];
            const int x = static_cast<int>(std::lround(p.x)), y = static_cast<int>(std::lround(p.y));
            u8 zr_, zg_, zb_;
            hsv(0.66f * (1.0f - (p.z - zlo) / zr), 0.8f, 0.9f, zr_, zg_, zb_);
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    put(0, x + dx, y + dy, wr_r, wr_g, wr_b);
                    put(1, x + dx, y + dy, cl_r, cl_g, cl_b);
                    put(2, x + dx, y + dy, zr_, zg_, zb_);
                }
        }
    }
    if (io::write_jpeg(out, img, 92)) FENIX_INFO("multiscale", "wrote {} ({}x{}): winding | clusters | depth", out, img.w, img.h);
    else FENIX_ERROR("multiscale", "jpeg write failed: {}", out);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: test_multiscale <surf.nrrd> [grid maxsheets seedstride thresh rounds outdir]\n");
        return 0;
    }
    const std::string path = argv[1];
    const int grid = argc > 2 ? std::atoi(argv[2]) : 700;
    const int max_sheets = argc > 3 ? std::atoi(argv[3]) : 24;
    const int seed_stride = argc > 4 ? std::atoi(argv[4]) : 40;
    const f32 thr = argc > 5 ? static_cast<f32>(std::atof(argv[5])) : 0.10f;
    const int rounds = argc > 6 ? std::atoi(argv[6]) : 3;
    const std::string outdir = argc > 7 ? argv[7] : "data/fenix_multiscale";

    auto pmx = io::nrrd_max(path);
    if (!pmx) { FENIX_ERROR("multiscale", "read failed: {}", pmx.error().message); return 1; }
    const f32 pscale = (*pmx > 2.0f) ? 1.0f : 255.0f;
    auto volr = io::read_nrrd_u8(path, pscale);
    if (!volr) { FENIX_ERROR("multiscale", "read failed: {}", volr.error().message); return 1; }
    Volume<u8> vol = std::move(*volr);
    const Extent3 D = vol.dims();
    FENIX_INFO("multiscale", "loaded {} -> u8 {}x{}x{} (pscale={:.0f})", path, D.z, D.y, D.x, static_cast<double>(pscale));

    segment::GrowParams gp;
    gp.step = 2.0f;
    gp.surf_thresh = thr * 255.0f;
    gp.snap_radius = gp.step * 1.5f;
    gp.fold_thresh = 6;
    gp.grid = grid;
    gp.river_radius = 2;

    auto t0 = clk::now();
    const segment::NormalField nf = segment::compute_normal_field<u8>(vol.view(), 8);
    auto t1 = clk::now();
    const f32 seed_thresh = std::max(gp.surf_thresh * 1.6f, 48.0f);
    segment::VolumeResult R = segment::trace_volume<u8>(vol.view(), nf, gp, max_sheets, /*min_valid=*/2000,
                                                        seed_stride, seed_thresh, VolumeView<const u8>{});
    auto t2 = clk::now();
    FENIX_INFO("multiscale", "normal {:.1f}s  trace {:.1f}s  sheets={} (of {} seed candidates, {} bins)",
               std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(),
               R.sheets.size(), R.seed_candidates, R.occupied_bins);
    if (R.sheets.empty()) { FENIX_WARN("multiscale", "no sheets traced (raise thresh / lower seed_stride)"); return 0; }

    // umbilicus: a straight axis at the crop centre (a crop is usually off the true scroll centre; the
    // graph's gaps/merges are 3D so an approximate centre only sets the outward sign + diagnostics).
    annotate::Umbilicus umb;
    umb.z = {0.0f, static_cast<f32>(D.z)};
    umb.y = {static_cast<f32>(D.y) * 0.5f, static_cast<f32>(D.y) * 0.5f};
    umb.x = {static_cast<f32>(D.x) * 0.5f, static_cast<f32>(D.x) * 0.5f};

    segment::PatchGraphParams pgp;
    pgp.step = gp.step;
    segment::PatchGraph g = segment::analyze_patches(R.sheets, umb, pgp);
    int nm = 0, nl = 0, nc = 0;
    for (const segment::PatchEdge& e : g.edges) {
        nm += e.kind == segment::EdgeKind::Merge;
        nl += e.kind == segment::EdgeKind::Link;
        nc += e.kind == segment::EdgeKind::Conflict;
    }
    FENIX_INFO("multiscale", "GRAPH spacing={:.1f} patches={} clusters={} wraps[{}..{}] edges: merge={} link={} conflict={} winding_conflicts={}",
               static_cast<double>(g.spacing), g.patches.size(), g.cluster_count, g.wrap_lo, g.wrap_hi, nm, nl, nc, g.winding_conflicts);

    const s64 vbefore = total_valid(R.sheets);
    winding::CosegParams cp;
    cp.full = D;
    cp.rounds = rounds;
    cp.field.ds = 4;
    cp.field.iters = 80;
    cp.fill.step = gp.step;
    cp.conf_thresh = 1.0f;
    cp.consist_weight = 0.5f;
    auto t3 = clk::now();
    const winding::CosegReport rep = winding::cosegment_refine(R.sheets, umb, pgp, cp);
    auto t4 = clk::now();
    const s64 vafter = total_valid(R.sheets);
    FENIX_INFO("multiscale", "REFINE {:.1f}s rounds={} valid {} -> {} (+{}, {:.1f}%) filled={} snapped={} monotonicity={:.3f} conflicts={}",
               std::chrono::duration<double>(t4 - t3).count(), rounds, vbefore, vafter, vafter - vbefore,
               100.0 * static_cast<double>(vafter - vbefore) / static_cast<double>(std::max<s64>(vbefore, 1)),
               rep.filled, rep.snapped, static_cast<double>(rep.monotonicity), rep.conflicts);

    // dump per-patch grids + a manifest (id wrap cluster nu nv valid) for rendering.
    const segment::PatchGraph gf = segment::analyze_patches(R.sheets, umb, pgp);  // final wraps/clusters
    std::string mk = "mkdir -p " + outdir;
    (void)std::system(mk.c_str());
    FILE* man = std::fopen((outdir + "/manifest.txt").c_str(), "w");
    std::fprintf(man, "# spacing %.2f clusters %d wraps %d %d\n", static_cast<double>(gf.spacing), gf.cluster_count, gf.wrap_lo, gf.wrap_hi);
    for (usize i = 0; i < R.sheets.size(); ++i) {
        const Surface& S = R.sheets[i];
        const s64 G2 = S.nu * S.nv;
        std::vector<f32> X(static_cast<usize>(G2)), Y(static_cast<usize>(G2)), Z(static_cast<usize>(G2));
        std::vector<u8> M(static_cast<usize>(G2));
        for (s64 k = 0; k < G2; ++k) {
            Z[static_cast<usize>(k)] = S.coord[static_cast<usize>(k)].z;
            Y[static_cast<usize>(k)] = S.coord[static_cast<usize>(k)].y;
            X[static_cast<usize>(k)] = S.coord[static_cast<usize>(k)].x;
            M[static_cast<usize>(k)] = S.valid[static_cast<usize>(k)];
        }
        char base[256];
        std::snprintf(base, sizeof base, "%s/p%02zu", outdir.c_str(), i);
        auto dump = [&](const char* suf, const void* d, usize bytes) {
            FILE* fp = std::fopen((std::string(base) + suf).c_str(), "wb");
            std::fwrite(d, 1, bytes, fp);
            std::fclose(fp);
        };
        dump("_x.f32", X.data(), X.size() * 4);
        dump("_y.f32", Y.data(), Y.size() * 4);
        dump("_z.f32", Z.data(), Z.size() * 4);
        dump("_valid.u8", M.data(), M.size());
        std::fprintf(man, "%zu %d %d %lld %lld %lld\n", i, gf.patches[i].wrap, gf.patches[i].cluster,
                     (long long)S.nu, (long long)S.nv, (long long)S.valid_count());
    }
    std::fclose(man);
    FENIX_INFO("multiscale", "wrote {}/p##_[xyz].f32 + _valid.u8 + manifest.txt", outdir);
    render_panels(outdir + "/render.jpg", R.sheets, gf, D);
    return 0;
}
