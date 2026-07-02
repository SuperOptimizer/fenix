// test_codec_bench.cpp — codec benchmark on a real CT volume. Tiles the volume into 64^3
// blocks, encodes/decodes each (parallel), and reports compression ratio, encode/decode
// throughput + per-block latency, and quality (PSNR, block-SSIM, MAE, 90/95/99/max abs error).
// Standalone main (NOT a unit test). Usage: test_codec_bench <vol.nrrd> [q1 q2 ...]
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

#include "codec/dct_block.hpp"
#include "core/core.hpp"
#include "core/parallel.hpp"
#include "io/nrrd.hpp"
#include "io/zarr.hpp"

using namespace fenix;
using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "data/crop512.nrrd";
    std::vector<f32> qs;
    for (int i = 2; i < argc; ++i)
        qs.push_back(static_cast<f32>(std::atof(argv[i])));
    if (qs.empty())
        qs = {1, 2, 4, 8, 16};

    // Input is a NRRD file or a local OME-Zarr directory (.zarr → read level 0 in full).
    const bool is_zarr = path.size() > 5 && path.substr(path.size() - 5) == ".zarr";
    Expected<Volume<f32>> volr = is_zarr ? [&]() -> Expected<Volume<f32>> {
        auto mr = io::read_zarray(path + "/0");
        if (!mr)
            return std::unexpected(mr.error());
        return io::read_zarr_region(path + "/0", {0, 0, 0}, mr->shape);
    }()
        : io::read_nrrd(path);
    if (!volr) {
        std::printf("codec-bench: skip (no %s): %s\n", path.c_str(), volr.error().message.c_str());
        return 0;
    }
    Volume<f32> vol = std::move(*volr);
    const Extent3 d = vol.dims();
    const auto vv = vol.view();
    const s64 S = 64;
    const s64 nb = d.z / S; // assume cube divisible by 64
    const s64 nblocks = nb * nb * nb;
    const s64 voxels = d.z * d.y * d.x;
    std::printf(
        "volume %lldx%lldx%lld  (%lld blocks of 64^3)  source=uint8-CT\n",
        (long long)d.z,
        (long long)d.y,
        (long long)d.x,
        (long long)nblocks
    );

    // gather a 64^3 block (contiguous) at block coord (bz,by,bx)
    auto gather = [&](s64 bz, s64 by, s64 bx, std::vector<f32>& out) {
        out.resize(static_cast<usize>(S * S * S));
        for (s64 z = 0; z < S; ++z)
            for (s64 y = 0; y < S; ++y)
                for (s64 x = 0; x < S; ++x)
                    out[static_cast<usize>((z * S + y) * S + x)] = vv(bz * S + z, by * S + y, bx * S + x);
    };

    // ---- DCT-16 codec, TILE model: a 64^3 tile = BPA^3 DCT blocks that SHARE one set of rANS category
    // tables (JPEG XL "group" model). The sole transform codec (the wavelet was retired, ADR 0005).
    // hf_exp/dz_frac overridable via env (FENIX_DCT_HF / FENIX_DCT_DZ) for RD sweeps without rebuild.
    const f32 hf = std::getenv("FENIX_DCT_HF") ? static_cast<f32>(std::atof(std::getenv("FENIX_DCT_HF")))
                                               : codec::DctParams{}.hf_exp;
    const f32 dz = std::getenv("FENIX_DCT_DZ") ? static_cast<f32>(std::atof(std::getenv("FENIX_DCT_DZ")))
                                               : codec::DctParams{}.dz_frac;
    const bool rdoq =
        std::getenv("FENIX_DCT_RDOQ") ? std::atoi(std::getenv("FENIX_DCT_RDOQ")) != 0 : codec::DctParams{}.rdoq;
    const f32 lam = std::getenv("FENIX_DCT_LAMBDA") ? static_cast<f32>(std::atof(std::getenv("FENIX_DCT_LAMBDA")))
                                                    : codec::DctParams{}.rdoq_lambda;
    const bool deblk = std::getenv("FENIX_DCT_DEBLOCK") ? std::atoi(std::getenv("FENIX_DCT_DEBLOCK")) != 0
                                                        : codec::DctParams{}.deblock;
    const f32 dbeta = std::getenv("FENIX_DCT_DBETA") ? static_cast<f32>(std::atof(std::getenv("FENIX_DCT_DBETA")))
                                                     : codec::DctParams{}.deblock_beta;
    const f32 dtc = std::getenv("FENIX_DCT_DTC") ? static_cast<f32>(std::atof(std::getenv("FENIX_DCT_DTC")))
                                                 : codec::DctParams{}.deblock_tc;
    const bool dcpred = std::getenv("FENIX_DCT_DCPRED") ? std::atoi(std::getenv("FENIX_DCT_DCPRED")) != 0
                                                        : codec::DctParams{}.dc_predict;
    const s64 BPA = S / codec::kDctN, nblk = nb * nb * nb; // S=64 tile → 4^3=64 DCT blocks/tile
    std::printf(
        "\nDCT-16 (%lld 64^3 tiles x %lld blocks, shared tables, hf_exp=%.2f dz_frac=%.2f):\n%-6s %9s %9s  %8s %8s  "
        "%7s %7s\n",
        (long long)nblk,
        (long long)(BPA * BPA * BPA),
        hf,
        dz,
        "q",
        "ratio8",
        "ratioF32",
        "encMB/s",
        "decMB/s",
        "PSNR",
        "MAE"
    );
    for (f32 q : qs) {
        std::vector<std::vector<u8>> pl(static_cast<usize>(nblk));
        codec::detail::g_plane_hdr_bytes.store(0);
        codec::detail::g_plane_enc_bytes.store(0);
        codec::detail::g_plane_count.store(0);
        codec::detail::g_dc_bytes.store(0);
        codec::detail::g_nsig_bytes.store(0);
        codec::detail::g_map_bytes.store(0);
        codec::detail::g_bits_bytes.store(0);
        auto t0 = clk::now();
        parallel_for(0, nblk, [&](s64 i) {
            const s64 bz = i / (nb * nb), by = (i / nb) % nb, bx = i % nb;
            std::vector<f32> tile;
            gather(bz, by, bx, tile);
            pl[static_cast<usize>(i)] = codec::encode_tile_dct<f32>(
                tile,
                BPA,
                {.q = q,
                 .hf_exp = hf,
                 .dz_frac = dz,
                 .rdoq = rdoq,
                 .rdoq_lambda = lam,
                 .deblock = deblk,
                 .deblock_beta = dbeta,
                 .deblock_tc = dtc,
                 .dc_predict = dcpred}
            );
        });
        auto t1 = clk::now();
        std::atomic<u64> cbytes{0};
        std::atomic<double> sse{0}, sae{0};
        auto t2 = clk::now();
        parallel_for(0, nblk, [&](s64 i) {
            const s64 bz = i / (nb * nb), by = (i / nb) % nb, bx = i % nb;
            auto dec_r = codec::decode_tile_dct<f32>(
                pl[static_cast<usize>(i)],
                BPA,
                {.q = q,
                 .hf_exp = hf,
                 .dz_frac = dz,
                 .rdoq = rdoq,
                 .rdoq_lambda = lam,
                 .deblock = deblk,
                 .deblock_beta = dbeta,
                 .deblock_tc = dtc,
                 .dc_predict = dcpred}
            );
            if (!dec_r) {
                std::fprintf(stderr, "codec-bench: decode failed: %s\n", dec_r.error().message.c_str());
                std::abort();
            }
            const std::vector<f32>& dec = *dec_r;
            std::vector<f32> orig;
            gather(bz, by, bx, orig);
            cbytes.fetch_add(pl[static_cast<usize>(i)].size(), std::memory_order_relaxed);
            double bsse = 0, bsae = 0;
            for (usize k = 0; k < dec.size(); ++k) {
                const double e = std::abs(static_cast<double>(orig[k]) - dec[k]);
                bsse += e * e;
                bsae += e;
            }
            double cur = sse.load(std::memory_order_relaxed);
            while (!sse.compare_exchange_weak(cur, cur + bsse)) {
            }
            cur = sae.load(std::memory_order_relaxed);
            while (!sae.compare_exchange_weak(cur, cur + bsae)) {
            }
        });
        auto t3 = clk::now();
        const double mse = sse.load() / static_cast<double>(voxels);
        const double mae = sae.load() / static_cast<double>(voxels);
        const double psnr = mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
        const double cb = static_cast<double>(cbytes.load());
        const double encMBs = (static_cast<double>(voxels) * 4.0 / 1e6) / (ms(t0, t1) / 1000.0);
        const double decMBs = (static_cast<double>(voxels) * 4.0 / 1e6) / (ms(t2, t3) / 1000.0);
        std::printf(
            "%-6.1f %9.1f %9.1f  %8.0f %8.0f  %7.2f %7.3f\n",
            q,
            static_cast<double>(voxels) / cb,
            static_cast<double>(voxels) * 4.0 / cb,
            encMBs,
            decMBs,
            psnr,
            mae
        );
        if (std::getenv("FENIX_DCT_STATS")) {
            const double hb = static_cast<double>(codec::detail::g_plane_hdr_bytes.load());
            const double eb = static_cast<double>(codec::detail::g_plane_enc_bytes.load());
            const double pc = static_cast<double>(codec::detail::g_plane_count.load());
            const double db = static_cast<double>(codec::detail::g_dc_bytes.load());
            const double nb2 = static_cast<double>(codec::detail::g_nsig_bytes.load());
            const double mb = static_cast<double>(codec::detail::g_map_bytes.load());
            const double bb = static_cast<double>(codec::detail::g_bits_bytes.load());
            std::printf(
                "   stats: cat-hdr %.1f%% | cat-enc %.1f%% | mantissa+sign %.1f%% | DC %.1f%% | nsig %.1f%% | ctxmap "
                "%.1f%%  (%.0f B/plane)\n",
                100.0 * hb / cb,
                100.0 * eb / cb,
                100.0 * bb / cb,
                100.0 * db / cb,
                100.0 * nb2 / cb,
                100.0 * mb / cb,
                pc > 0 ? hb / pc : 0.0
            );
        }
    }
    return 0;
}
