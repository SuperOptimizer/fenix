// test_grow.cpp — drive the native surface grower on a real surface-prediction volume and dump
// the traced grid (raw x/y/z/valid) for rendering/comparison. Standalone main (not a unit test).
// Usage: test_grow <surf.nrrd|big_surf.zarr> <sz> <sy> <sx> [step thresh maxgen grid outdir]
#include "core/core.hpp"
#include "io/nrrd.hpp"
#include "io/scan_meta.hpp"
#include "preprocess/aircut.hpp"
#include "preprocess/deconv.hpp"
#include "segment/grow.hpp"
#include "segment/structure_tensor.hpp"
#include "eval/mesh_quality.hpp"

#include <chrono>
#include <cstdio>
#include <string>

using namespace fenix;
using clk = std::chrono::steady_clock;

// Reassemble a uint8 OME-zarr (raw 128^3 nested chunks) into a Volume<u8>. shape/chunk hardcoded.
static Volume<u8> load_zarr_u8(const std::string& root, s64 N, s64 C) {
    Volume<u8> vol(Extent3{N, N, N});
    auto v = vol.view();
    const s64 nc = N / C;
    std::vector<u8> buf(static_cast<usize>(C * C * C));
    for (s64 cz = 0; cz < nc; ++cz)
        for (s64 cy = 0; cy < nc; ++cy)
            for (s64 cx = 0; cx < nc; ++cx) {
                char path[512];
                std::snprintf(path, sizeof path, "%s/0/%lld/%lld/%lld", root.c_str(), (long long)cz, (long long)cy, (long long)cx);
                FILE* fp = std::fopen(path, "rb");
                if (!fp) continue;  // missing chunk -> zeros
                size_t n = std::fread(buf.data(), 1, buf.size(), fp);
                std::fclose(fp);
                if (n != buf.size()) continue;
                for (s64 z = 0; z < C; ++z)
                    for (s64 y = 0; y < C; ++y)
                        for (s64 x = 0; x < C; ++x)
                            v(cz * C + z, cy * C + y, cx * C + x) = buf[static_cast<usize>((z * C + y) * C + x)];
            }
    return vol;
}

template <class T>
static void run(VolumeView<const T> vol, VolumeView<const T> ct, Vec3f seed, segment::GrowParams gp, f32 ds, const std::string& outdir) {
    auto t0 = clk::now();
    auto nf = segment::compute_normal_field<T>(vol, static_cast<int>(ds));
    auto t1 = clk::now();
    Surface S = segment::grow_surface<T>(vol, ct, nf, seed, gp);
    auto t2 = clk::now();

    const s64 nvalid = S.valid_count();
    double area = 0;
    Vec3f lo{1e9f, 1e9f, 1e9f}, hi{-1e9f, -1e9f, -1e9f};
    for (s64 v = 0; v < S.nv; ++v)
        for (s64 u = 0; u < S.nu; ++u) {
            if (!S.is_valid(u, v)) continue;
            const Vec3f c = S.at(u, v);
            lo = {std::min(lo.z, c.z), std::min(lo.y, c.y), std::min(lo.x, c.x)};
            hi = {std::max(hi.z, c.z), std::max(hi.y, c.y), std::max(hi.x, c.x)};
            if (u + 1 < S.nu && S.is_valid(u + 1, v) && v + 1 < S.nv && S.is_valid(u, v + 1))
                area += norm(S.at(u + 1, v) - c) * norm(S.at(u, v + 1) - c);
        }
    std::printf("normal %.1fs  grow %.1fs  valid=%lld  area~%.0f vx^2  bbox z[%.0f,%.0f] y[%.0f,%.0f] x[%.0f,%.0f]\n",
                std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(),
                (long long)nvalid, area, lo.z, hi.z, lo.y, hi.y, lo.x, hi.x);
    auto Q = eval::analyze_mesh(S);
    double dfid = 0;
    for (s64 i = 0; i < S.nu * S.nv; ++i) if (S.valid[static_cast<usize>(i)]) dfid += sample_trilinear(vol, S.coord[static_cast<usize>(i)]);
    dfid = nvalid ? dfid / static_cast<double>(nvalid) : 0;
    std::printf("QA  cmp %lld  holes %lld  edgeCV %.3f  p99/m %.2f  tear%% %.2f  degen%% %.3f  minang %.1f  fold%% %.3f  selfX%% %.3f  sDir %.3f  curv %.3f  bnd%% %.1f  data_fid %.1f\n",
                (long long)Q.components, (long long)Q.holes, Q.edge_cv, Q.edge_p99_ratio, 100 * Q.frac_long, 100 * Q.degen_tri, Q.min_angle_deg,
                100 * Q.fold_detj, 100 * Q.self_intersect, Q.sdirichlet_mean, Q.curvature_mean, 100 * Q.boundary_frac, dfid);

    std::string mk = "mkdir -p " + outdir; (void)std::system(mk.c_str());
    const s64 G = S.nu;
    std::vector<f32> X(static_cast<usize>(G * G)), Y(X.size()), Z(X.size());
    std::vector<u8> M(X.size());
    for (s64 i = 0; i < G * G; ++i) { Z[static_cast<usize>(i)] = S.coord[static_cast<usize>(i)].z; Y[static_cast<usize>(i)] = S.coord[static_cast<usize>(i)].y; X[static_cast<usize>(i)] = S.coord[static_cast<usize>(i)].x; M[static_cast<usize>(i)] = S.valid[static_cast<usize>(i)]; }
    auto dump = [&](const std::string& nm, const void* d, usize bytes) { FILE* fp = std::fopen((outdir + "/" + nm).c_str(), "wb"); std::fwrite(d, 1, bytes, fp); std::fclose(fp); };
    dump("x.f32", X.data(), X.size() * 4); dump("y.f32", Y.data(), Y.size() * 4);
    dump("z.f32", Z.data(), Z.size() * 4); dump("valid.u8", M.data(), M.size());
    FILE* mf = std::fopen((outdir + "/meta.txt").c_str(), "w"); std::fprintf(mf, "%lld\n", (long long)G); std::fclose(mf);
    std::printf("wrote %s/{x,y,z}.f32 valid.u8 (G=%lld)\n", outdir.c_str(), (long long)G);
}

int main(int argc, char** argv) {
    if (argc < 5) { std::printf("need <surf.nrrd|.zarr> sz sy sx [step thresh maxgen grid outdir]\n"); return 1; }
    const std::string path = argv[1];
    Vec3f seed{static_cast<f32>(std::atof(argv[2])), static_cast<f32>(std::atof(argv[3])), static_cast<f32>(std::atof(argv[4]))};
    segment::GrowParams gp;
    if (argc > 5) gp.step = static_cast<f32>(std::atof(argv[5]));
    f32 thr = argc > 6 ? static_cast<f32>(std::atof(argv[6])) : 0.15f;
    if (argc > 7) gp.max_gen = std::atoi(argv[7]);
    if (argc > 8) gp.grid = std::atoi(argv[8]);
    gp.snap_radius = gp.step * 1.5f;  // tighter -> can't hop to an adjacent wrap
    gp.fold_thresh = 6;
    const std::string outdir = argc > 9 ? argv[9] : "data/fenix_trace";
    const f32 ds = argc > 10 ? static_cast<f32>(std::atof(argv[10])) : 8.0f;
    if (argc > 11) gp.max_bridge = std::atoi(argv[11]);
    if (argc > 12) gp.river_radius = std::atoi(argv[12]);
    const std::string ct_path = argc > 13 ? argv[13] : "";    // raw CT for the combined data term
    if (argc > 14) gp.ct_thresh = static_cast<f32>(std::atof(argv[14]));
    const std::string meta_path = argc > 15 ? argv[15] : "";  // metadata.json (volume root or file) -> deconv params
    const int ct_sheet = argc > 16 ? std::atoi(argv[16]) : 0; // 1 = use structure-tensor sheetness as the CT term

    const bool is_zarr = path.size() > 5 && path.substr(path.size() - 5) == ".zarr";
    if (is_zarr) {
        gp.surf_thresh = thr * 255.0f;  // field is u8 0..255
        std::printf("loading %s as uint8 ...\n", path.c_str());
        Volume<u8> vol = load_zarr_u8(path, 2048, 128);
        Volume<u8> ct;
        if (!ct_path.empty()) {
            std::printf("loading CT %s ...\n", ct_path.c_str()); ct = load_zarr_u8(ct_path, 2048, 128);
            const f32 cut = preprocess::air_cut<u8>(ct.view(), 0.0f, 256.0f);   // zero low-density background
            if (gp.ct_thresh <= 0) gp.ct_thresh = cut;
            std::printf("air-cut threshold (Otsu valley) = %.0f\n", cut);
        }
        std::printf("loaded 2048^3 u8 seed(%.0f,%.0f,%.0f) step=%.1f thresh=%.0f grid=%d ct_thresh=%.0f\n", seed.z, seed.y, seed.x, gp.step, gp.surf_thresh, gp.grid, gp.ct_thresh);
        run<u8>(vol.view(), ct_path.empty() ? VolumeView<const u8>{} : ct.view(), seed, gp, ds, outdir);
    } else {
        // f32 NRRD inputs, but keep the RESIDENT volumes u8 (4x less RAM than f32 — matches the
        // production zarr path). Loads STREAM straight to u8 (the full f32 is never resident).
        auto pmx = io::nrrd_max(path);
        if (!pmx) { std::printf("read failed: %s\n", pmx.error().message.c_str()); return 1; }
        const f32 mx = *pmx;
        const f32 pscale = (mx > 2.0f) ? 1.0f : 255.0f;  // 0..1 preds -> 0..255; already-0..255 preds as-is
        auto volr = io::read_nrrd_u8(path, pscale);
        if (!volr) { std::printf("read failed: %s\n", volr.error().message.c_str()); return 1; }
        Volume<u8> vol = std::move(*volr);
        gp.surf_thresh = thr * 255.0f;

        Volume<u8> ct;
        if (!ct_path.empty()) {
            auto ctr = io::read_nrrd_u8(ct_path, 1.0f);  // CT is already 0..255
            if (!ctr) { std::printf("CT read failed: %s\n", ctr.error().message.c_str()); return 1; }
            ct = std::move(*ctr);
            const f32 cut = preprocess::air_cut<u8>(ct.view(), 0.0f, 256.0f);  // zero low-density background
            std::printf("air-cut threshold (Otsu valley) = %.1f\n", cut);
            // metadata-matched deconv sigma (unsharp at the recon's own sigma restores the high
            // frequencies Paganin phase retrieval low-passed). 0 => no deconv.
            f32 sig = 0.0f;
            if (!meta_path.empty()) {
                auto sm = io::fetch_scan_meta(meta_path);
                if (sm) {
                    std::printf("scan meta: voxel=%.2fum energy=%.0fkeV paganin_db=%.0f unsharp(sig=%.2f coeff=%.2f)\n",
                                sm->voxel_um, sm->energy_keV, sm->paganin_delta_beta, sm->unsharp_sigma, sm->unsharp_coeff);
                    sig = sm->unsharp_sigma > 0 ? sm->unsharp_sigma : 1.2f;
                } else std::printf("scan meta fetch failed: %s\n", sm.error().message.c_str());
            }
            if (ct_sheet) {
                // Coarse (ds=2) sheetness term: ~8x less structure-tensor work AND the resident term
                // stays ds^3 smaller (the grower samples it via gp.ct_ds). ct_sheetness_coarse frees
                // the full-res CT after downsampling, so peak RAM stays low. Deconv folded, air-cut gated.
                const int sds = 2;
                ct = segment::ct_sheetness_coarse<u8>(ct, cut, sig, sds);
                gp.ct_ds = static_cast<f32>(sds);
                if (gp.ct_thresh <= 0) gp.ct_thresh = 0.35f * 255.0f;  // sheetness 0..1 -> u8
                std::printf("CT term = structure-tensor sheetness u8 (coarse ds=%d, deconv sig=%.2f), ct_thresh=%.0f\n", sds, sig, gp.ct_thresh);
            } else if (gp.ct_thresh <= 0) {
                gp.ct_thresh = cut;  // intensity term, CT units 0..255
            }
        }
        std::printf("vol %lldx%lldx%lld pred->u8 (max %.3f) seed(%.0f,%.0f,%.0f) step=%.1f thresh=%.0f grid=%d ct_thresh=%.0f\n",
                    (long long)vol.dims().z, (long long)vol.dims().y, (long long)vol.dims().x, mx, seed.z, seed.y, seed.x, gp.step, gp.surf_thresh, gp.grid, gp.ct_thresh);
        run<u8>(vol.view(), ct_path.empty() ? VolumeView<const u8>{} : ct.view(), seed, gp, ds, outdir);
    }
    return 0;
}
