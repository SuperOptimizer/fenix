// test_trace.cpp — full-volume tracing: auto-seed across the 2048^3 surface-prediction cube and grow
// many non-overlapping sheets. Reports aggregate geometry-quality metrics + dumps all sheet vertices
// (z,y,x + sheet id) for a cross-section / coverage render. Standalone main.
// Usage: test_trace <big_surf.zarr> [max_sheets seed_thresh min_valid grid outdir]
#include "core/core.hpp"
#include "core/surface.hpp"
#include "segment/grow.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace fenix;

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
                if (!fp) continue;
                if (std::fread(buf.data(), 1, buf.size(), fp) == buf.size())
                    for (s64 z = 0; z < C; ++z) for (s64 y = 0; y < C; ++y) for (s64 x = 0; x < C; ++x)
                        v(cz * C + z, cy * C + y, cx * C + x) = buf[static_cast<usize>((z * C + y) * C + x)];
                std::fclose(fp);
            }
    return vol;
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "data/big_surf.zarr";
    const int max_sheets = argc > 2 ? std::atoi(argv[2]) : 40;
    const f32 seed_thr = argc > 3 ? static_cast<f32>(std::atof(argv[3])) : 0.30f;
    const s64 min_valid = argc > 4 ? std::atoll(argv[4]) : 20000;
    const int grid = argc > 5 ? std::atoi(argv[5]) : 1500;
    const std::string outdir = argc > 6 ? argv[6] : "data/fenix_vol";

    std::printf("loading %s ...\n", path.c_str());
    Volume<u8> vol = load_zarr_u8(path, 2048, 128);
    segment::GrowParams gp;
    gp.step = 2; gp.snap_radius = 3; gp.fold_thresh = 6; gp.surf_thresh = seed_thr * 255.0f; gp.grid = grid; gp.max_gen = 6000;

    auto t0 = std::chrono::steady_clock::now();
    auto nf = segment::compute_normal_field<u8>(vol.view(), 8);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("normal field %.1fs; tracing whole cube (max %d sheets, seed_thr %.0f) ...\n",
                std::chrono::duration<double>(t1 - t0).count(), max_sheets, seed_thr * 255.0f);
    auto R = segment::trace_volume<u8>(vol.view(), nf, gp, max_sheets, min_valid, 16, seed_thr * 255.0f);
    auto t2 = std::chrono::steady_clock::now();

    // aggregate stats + quality (valid-weighted means)
    s64 tot_valid = 0; double tot_area = 0, wcv = 0, wdf = 0, wcov = 0, wns = 0, wbf = 0, wba = 0, wfold = 0;
    std::string mk = "mkdir -p " + outdir; (void)std::system(mk.c_str());
    FILE* pf = std::fopen((outdir + "/pts.f32").c_str(), "wb");
    FILE* idf = std::fopen((outdir + "/ids.u16").c_str(), "wb");
    s64 npts = 0;
    for (usize si = 0; si < R.sheets.size(); ++si) {
        const Surface& S = R.sheets[si];
        auto Q = segment::surface_quality(S, gp.bin_size, gp.fold_thresh);
        const f64 w = static_cast<f64>(Q.valid);
        tot_valid += Q.valid; wcv += w * Q.spacing_cv; wcov += w * Q.coverage; wns += w * Q.normal_smooth;
        wbf += w * Q.boundary_frac; wba += w * Q.bad_angle; wfold += w * Q.distant_fold;
        // dump this sheet's grid (x/y/z/valid) for per-patch flattened-face rendering
        {
            char sd[512]; std::snprintf(sd, sizeof sd, "%s/sheet_%02zu", outdir.c_str(), si);
            std::string smk = "mkdir -p "; smk += sd; (void)std::system(smk.c_str());
            const s64 NG = S.nu * S.nv;
            std::vector<f32> X(static_cast<usize>(NG)), Yc(X.size()), Zc(X.size());
            std::vector<u8> Mv(X.size());
            for (s64 i = 0; i < NG; ++i) { Zc[static_cast<usize>(i)] = S.coord[static_cast<usize>(i)].z; Yc[static_cast<usize>(i)] = S.coord[static_cast<usize>(i)].y; X[static_cast<usize>(i)] = S.coord[static_cast<usize>(i)].x; Mv[static_cast<usize>(i)] = S.valid[static_cast<usize>(i)]; }
            auto wrs = [&](const char* nm, const void* d, usize b) { std::string p = sd; p += "/"; p += nm; FILE* fp = std::fopen(p.c_str(), "wb"); std::fwrite(d, 1, b, fp); std::fclose(fp); };
            wrs("x.f32", X.data(), X.size() * 4); wrs("y.f32", Yc.data(), Yc.size() * 4); wrs("z.f32", Zc.data(), Zc.size() * 4); wrs("valid.u8", Mv.data(), Mv.size());
            std::string mp = sd; mp += "/meta.txt"; FILE* mfp = std::fopen(mp.c_str(), "w"); std::fprintf(mfp, "%lld\n", (long long)S.nu); std::fclose(mfp);
        }
        for (s64 v = 0; v < S.nv; ++v)
            for (s64 u = 0; u < S.nu; ++u) {
                if (!S.is_valid(u, v)) continue;
                const Vec3f c = S.at(u, v);
                float zyx[3] = {c.z, c.y, c.x}; u16 id = static_cast<u16>(si + 1);
                std::fwrite(zyx, 4, 3, pf); std::fwrite(&id, 2, 1, idf); ++npts;
                if (u + 1 < S.nu && S.is_valid(u + 1, v) && v + 1 < S.nv && S.is_valid(u, v + 1))
                    tot_area += norm(S.at(u + 1, v) - c) * norm(S.at(u, v + 1) - c);
                wdf += static_cast<f64>(sample_trilinear(vol.view(), c));
            }
    }
    std::fclose(pf); std::fclose(idf);
    const f64 iv = tot_valid ? 1.0 / static_cast<f64>(tot_valid) : 0;
    std::printf("\n=== whole-cube trace: %.0fs, %zu sheets / %lld seed candidates ===\n",
                std::chrono::duration<double>(t2 - t1).count(), R.sheets.size(), (long long)R.seed_candidates);
    std::printf("total valid %lld  area ~%.3g vx^2  occupied 2vx-bins %lld\n", (long long)tot_valid, tot_area, (long long)R.occupied_bins);
    std::printf("AGGREGATE QUALITY (valid-weighted): spacing_cv %.3f  normal_smooth %.1fdeg  boundary %.1f%%  bad_angle %.2f%%  distant_fold %.2f%%  coverage %.3f  data_fidelity %.1f/255\n",
                wcv * iv, wns * iv, 100 * wbf * iv, 100 * wba * iv, 100 * wfold * iv, wcov * iv, wdf * iv);
    FILE* mf = std::fopen((outdir + "/meta.txt").c_str(), "w"); std::fprintf(mf, "%lld %zu\n", (long long)npts, R.sheets.size()); std::fclose(mf);
    std::printf("dumped %lld vertices -> %s/pts.f32 + ids.u16\n", (long long)npts, outdir.c_str());
    return 0;
}
