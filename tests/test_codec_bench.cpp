// test_codec_bench.cpp — codec benchmark on a real CT volume. Tiles the volume into 64^3
// blocks, encodes/decodes each (parallel), and reports compression ratio, encode/decode
// throughput + per-block latency, and quality (PSNR, block-SSIM, MAE, 90/95/99/max abs error).
// Standalone main (NOT a unit test). Usage: test_codec_bench <vol.nrrd> [q1 q2 ...]
#include "core/core.hpp"
#include "core/parallel.hpp"
#include "io/nrrd.hpp"
#include "codec/block.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

using namespace fenix;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "data/crop512.nrrd";
    std::vector<f32> qs;
    for (int i = 2; i < argc; ++i) qs.push_back(static_cast<f32>(std::atof(argv[i])));
    if (qs.empty()) qs = {1, 2, 4, 8, 16};

    auto volr = io::read_nrrd(path);
    if (!volr) { std::printf("codec-bench: skip (no %s): %s\n", path.c_str(), volr.error().message.c_str()); return 0; }
    Volume<f32> vol = std::move(*volr);
    const Extent3 d = vol.dims();
    const auto vv = vol.view();
    const s64 S = 64;
    const s64 nb = d.z / S;  // assume cube divisible by 64
    const s64 nblocks = nb * nb * nb;
    const s64 voxels = d.z * d.y * d.x;
    std::printf("volume %lldx%lldx%lld  (%lld blocks of 64^3)  source=uint8-CT\n",
                (long long)d.z, (long long)d.y, (long long)d.x, (long long)nblocks);
    std::printf("%-6s %9s %9s  %8s %8s  %7s %7s %7s  %8s %7s %7s %7s\n",
                "q", "ratio8", "ratioF32", "encMB/s", "decMB/s", "PSNR", "SSIM", "MAE",
                "p90", "p95", "p99", "max");

    // gather a 64^3 block (contiguous) at block coord (bz,by,bx)
    auto gather = [&](s64 bz, s64 by, s64 bx, std::vector<f32>& out) {
        out.resize(static_cast<usize>(S * S * S));
        for (s64 z = 0; z < S; ++z)
            for (s64 y = 0; y < S; ++y)
                for (s64 x = 0; x < S; ++x)
                    out[static_cast<usize>((z * S + y) * S + x)] = vv(bz * S + z, by * S + y, bx * S + x);
    };

    for (f32 q : qs) {
        std::vector<std::vector<u8>> payloads(static_cast<usize>(nblocks));
        codec::BlockParams bp{.q = q, .levels = 4};

        // ---- encode (parallel) ----
        auto t0 = clk::now();
        parallel_for(0, nblocks, [&](s64 i) {
            const s64 bz = i / (nb * nb), by = (i / nb) % nb, bx = i % nb;
            std::vector<f32> blk;
            gather(bz, by, bx, blk);
            payloads[static_cast<usize>(i)] = codec::encode_block(blk, S, bp);
        });
        auto t1 = clk::now();

        // ---- decode (parallel) + error accumulation ----
        std::atomic<u64> comp_bytes{0};
        std::atomic<double> sse{0}, sae{0};
        std::atomic<u64> maxe_bits{0};                    // max abs err via atomic max on bit-cast
        std::vector<u64> hist(1024, 0);                   // abs-err histogram (0..>255), 0.25 bins
        std::vector<std::array<double, 5>> ssim_acc(static_cast<usize>(nblocks));  // num,den per block
        auto t2 = clk::now();
        parallel_for(0, nblocks, [&](s64 i) {
            const s64 bz = i / (nb * nb), by = (i / nb) % nb, bx = i % nb;
            auto dec = codec::decode_block(payloads[static_cast<usize>(i)]);
            std::vector<f32> orig;
            gather(bz, by, bx, orig);
            comp_bytes.fetch_add(payloads[static_cast<usize>(i)].size(), std::memory_order_relaxed);
            double bsse = 0, bsae = 0, bmax = 0;
            // block stats for SSIM (global single-window per 64^3 block)
            double mx = 0, my = 0;
            for (usize k = 0; k < dec.size(); ++k) { mx += orig[k]; my += dec[k]; }
            mx /= static_cast<double>(dec.size()); my /= static_cast<double>(dec.size());
            double vx = 0, vy = 0, cxy = 0;
            for (usize k = 0; k < dec.size(); ++k) {
                const double e = std::abs(static_cast<double>(orig[k]) - dec[k]);
                bsse += e * e; bsae += e; bmax = std::max(bmax, e);
                const double dx = orig[k] - mx, dy = dec[k] - my;
                vx += dx * dx; vy += dy * dy; cxy += dx * dy;
            }
            const double nrec = static_cast<double>(dec.size());
            vx /= nrec; vy /= nrec; cxy /= nrec;
            const double C1 = 6.5025, C2 = 58.5225;  // (0.01*255)^2, (0.03*255)^2
            ssim_acc[static_cast<usize>(i)][0] = (2 * mx * my + C1) * (2 * cxy + C2);
            ssim_acc[static_cast<usize>(i)][1] = (mx * mx + my * my + C1) * (vx + vy + C2);
            // reduce errors
            double cur = sse.load(std::memory_order_relaxed);
            while (!sse.compare_exchange_weak(cur, cur + bsse)) {}
            cur = sae.load(std::memory_order_relaxed);
            while (!sae.compare_exchange_weak(cur, cur + bsae)) {}
            u64 mb = std::bit_cast<u64>(bmax), om = maxe_bits.load();
            while (mb > om && !maxe_bits.compare_exchange_weak(om, mb)) {}
            // local histogram merge under no lock: accumulate into a thread-private then merge — simple: atomic per bin is heavy; do a coarse merge
        });
        auto t3 = clk::now();

        // histogram pass (serial, cheap relative): recompute abs errs into hist for percentiles
        for (s64 i = 0; i < nblocks; ++i) {
            const s64 bz = i / (nb * nb), by = (i / nb) % nb, bx = i % nb;
            auto dec = codec::decode_block(payloads[static_cast<usize>(i)]);
            std::vector<f32> orig; gather(bz, by, bx, orig);
            for (usize k = 0; k < dec.size(); ++k) {
                double e = std::abs(static_cast<double>(orig[k]) - dec[k]);
                usize bin = static_cast<usize>(std::min(1023.0, e * 4.0));  // 0.25 per bin
                hist[bin]++;
            }
        }

        // metrics
        const double mse = sse.load() / static_cast<double>(voxels);
        const double mae = sae.load() / static_cast<double>(voxels);
        const double psnr = mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
        const double maxe = std::bit_cast<double>(maxe_bits.load());
        double snum = 0, sden = 0;
        for (auto& a : ssim_acc) { snum += a[0]; sden += a[1]; }
        const double ssim = sden > 0 ? snum / sden : 1.0;
        auto pct = [&](double frac) {
            const u64 target = static_cast<u64>(frac * static_cast<double>(voxels));
            u64 acc = 0;
            for (usize b = 0; b < hist.size(); ++b) { acc += hist[b]; if (acc >= target) return static_cast<double>(b) / 4.0; }
            return 255.0;
        };
        const double cb = static_cast<double>(comp_bytes.load());
        const double ratio8 = static_cast<double>(voxels) / cb;          // vs 1 byte/voxel (CT is 8-bit)
        const double ratioF = static_cast<double>(voxels) * 4.0 / cb;    // vs f32 input
        const double encMBs = (static_cast<double>(voxels) * 4.0 / 1e6) / (ms(t0, t1) / 1000.0);
        const double decMBs = (static_cast<double>(voxels) * 4.0 / 1e6) / (ms(t2, t3) / 1000.0);
        std::printf("%-6.1f %9.1f %9.1f  %8.0f %8.0f  %7.2f %7.4f %7.3f  %8.2f %7.2f %7.2f %7.2f\n",
                    q, ratio8, ratioF, encMBs, decMBs, psnr, ssim, mae, pct(0.90), pct(0.95), pct(0.99), maxe);
    }
    return 0;
}
