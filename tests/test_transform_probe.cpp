// test_transform_probe.cpp — RD probe for plan 2.2 (per-block transform choice). For every 16^3 block,
// compute the rate-distortion cost J = D + lambda*R under three transforms — DCT-II (current), DST-VII
// (ramp blocks), and IDENTITY/transform-skip (noise blocks) — pick the per-block winner, and report how
// often each wins + the total RD ceiling of per-block selection (with and without a mode-flag cost).
// Faithful to the codec: same dead-zone quant (dz=0.80, centroid 0.40), same freq-weighted step for the
// frequency transforms (flat step for identity), same magnitude-category+mantissa+sign rate model.
// Standalone main. Usage: test_transform_probe <vol.fxvol|.zarr> [q1 q2 ...]   (default q=8,32)
#include "core/core.hpp"
#include "core/parallel.hpp"
#include "bench_vol.hpp"
#include "io/zarr.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

using namespace fenix;
constexpr int N = 16, V = N * N * N;
using Mat = std::array<std::array<f32, N>, N>;

static Mat dct2_basis() {  // orthonormal DCT-II: m[k][n] = c_k cos(pi(2n+1)k/2N)
    Mat m{};
    const double pi = std::numbers::pi, c0 = std::sqrt(1.0 / N), ck = std::sqrt(2.0 / N);
    for (int k = 0; k < N; ++k)
        for (int n = 0; n < N; ++n) m[k][n] = static_cast<f32>((k == 0 ? c0 : ck) * std::cos(pi * (2 * n + 1) * k / (2.0 * N)));
    return m;
}
static Mat dst7_basis() {  // orthonormal DST-VII: s[k][n] = sqrt(4/(2N+1)) sin(pi(2n+1)(k+1)/(2N+1))
    Mat s{};
    const double pi = std::numbers::pi, c = std::sqrt(4.0 / (2 * N + 1));
    for (int k = 0; k < N; ++k)
        for (int n = 0; n < N; ++n) s[k][n] = static_cast<f32>(c * std::sin(pi * (2 * n + 1) * (k + 1) / (2.0 * N + 1)));
    return s;
}
static double max_offdiag(const Mat& B) {  // ||B B^T - I||_inf, to confirm orthonormality (Parseval)
    double mx = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double d = 0;
            for (int n = 0; n < N; ++n) d += static_cast<double>(B[i][n]) * B[j][n];
            mx = std::max(mx, std::abs(d - (i == j ? 1.0 : 0.0)));
        }
    return mx;
}
static void xform3d(const Mat& B, f32* blk) {  // separable: same 1-D basis along x, then y, then z
    f32 line[N], out[N];
    for (int z = 0; z < N; ++z)  // x-lines
        for (int y = 0; y < N; ++y) {
            f32* p = blk + (z * N + y) * N;
            for (int i = 0; i < N; ++i) line[i] = p[i];
            for (int k = 0; k < N; ++k) { f32 a = 0; for (int n = 0; n < N; ++n) a += B[k][n] * line[n]; out[k] = a; }
            for (int i = 0; i < N; ++i) p[i] = out[i];
        }
    for (int z = 0; z < N; ++z)  // y-lines
        for (int x = 0; x < N; ++x) {
            f32* p = blk + z * N * N + x;
            for (int i = 0; i < N; ++i) line[i] = p[i * N];
            for (int k = 0; k < N; ++k) { f32 a = 0; for (int n = 0; n < N; ++n) a += B[k][n] * line[n]; out[k] = a; }
            for (int i = 0; i < N; ++i) p[i * N] = out[i];
        }
    for (int y = 0; y < N; ++y)  // z-lines
        for (int x = 0; x < N; ++x) {
            f32* p = blk + y * N + x;
            for (int i = 0; i < N; ++i) line[i] = p[i * N * N];
            for (int k = 0; k < N; ++k) { f32 a = 0; for (int n = 0; n < N; ++n) a += B[k][n] * line[n]; out[k] = a; }
            for (int i = 0; i < N; ++i) p[i * N * N] = out[i];
        }
}

// Dead-zone quantize a coefficient array against a per-index step; return (distortion, rate-bits).
static void rd_cost(const f32* coef, const f32* step, double& D, double& R) {
    D = 0;
    R = 0;
    for (int i = 0; i < V; ++i) {
        const f32 s = step[i], a = std::abs(coef[i]);
        int level = 0;
        if (a >= 0.80f * s) level = static_cast<int>(std::floor(a / s + 0.20f));  // dz_frac=0.80 -> (1-dz)=0.20
        const f32 dq = level ? s * (static_cast<f32>(level) - 1.0f + 0.80f + 0.40f) : 0.0f;  // centroid 0.40
        const double e = static_cast<double>(a) - dq;
        D += e * e;
        if (level) { const int cat = std::bit_width(static_cast<unsigned>(level)); R += cat + (cat - 1) + 1; }  // category + mantissa + sign
    }
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "data/crop512.fxvol";
    std::vector<f32> qs;
    for (int i = 2; i < argc; ++i) qs.push_back(static_cast<f32>(std::atof(argv[i])));
    if (qs.empty()) qs = {8, 32};

    Expected<Volume<f32>> volr = bench::load_f32(path);
    if (!volr) { std::printf("transform-probe: skip (no %s)\n", path.c_str()); return 0; }
    Volume<f32> vol = std::move(*volr);
    const Extent3 d = vol.dims();
    const auto vv = vol.view();
    const s64 nbx = d.x / N, nby = d.y / N, nbz = d.z / N, nblk = nbz * nby * nbx;

    const Mat DCT = dct2_basis(), DST = dst7_basis();
    std::printf("transform-probe %s  %lldx%lldx%lld  (%lld 16^3 blocks)\n", path.c_str(),
                (long long)d.z, (long long)d.y, (long long)d.x, (long long)nblk);
    std::printf("orthonormality ||BB^T-I||: DCT %.2e  DST-VII %.2e  (Parseval ok if ~0)\n", max_offdiag(DCT), max_offdiag(DST));

    // per-index frequency-weighted step (DCT/DST) and flat step (identity), built per q below.
    const f32 hf = 0.65f;
    std::array<int, V> fsum{};
    for (int z = 0; z < N; ++z) for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) fsum[(z * N + y) * N + x] = z + y + x;

    for (f32 q : qs) {
        std::array<f32, V> stepF{}, stepI{};
        for (int i = 0; i < V; ++i) { stepF[i] = q * std::pow(1.0f + static_cast<f32>(fsum[i]), hf); stepI[i] = q; }
        const double lambda = 0.15 * q * q;       // RDOQ lambda at the base step
        const double flag_bits = 1.0;             // marginal cost of signalling a non-DCT mode per block

        std::vector<double> Jd(static_cast<usize>(nblk)), Jbest(static_cast<usize>(nblk)), Jbest_nf(static_cast<usize>(nblk));
        std::vector<u8> win(static_cast<usize>(nblk));  // 0=DCT 1=DST 2=identity (with flag cost)
        parallel_for(0, nblk, [&](s64 b) {
            const s64 bz = b / (nby * nbx), by = (b / nbx) % nby, bx = b % nbx;
            f32 blk[V];
            double mean = 0;
            for (int z = 0; z < N; ++z) for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
                const f32 v = vv(bz * N + z, by * N + y, bx * N + x);
                blk[(z * N + y) * N + x] = v;
                mean += v;
            }
            mean /= V;
            f32 resid[V], cdct[V], cdst[V];
            for (int i = 0; i < V; ++i) resid[i] = blk[i] - static_cast<f32>(mean);  // DC removed (same side-channel for all)
            for (int i = 0; i < V; ++i) { cdct[i] = resid[i]; cdst[i] = resid[i]; }
            xform3d(DCT, cdct);
            xform3d(DST, cdst);
            double Ddct, Rdct, Ddst, Rdst, Did, Rid;
            rd_cost(cdct, stepF.data(), Ddct, Rdct);
            rd_cost(cdst, stepF.data(), Ddst, Rdst);
            rd_cost(resid, stepI.data(), Did, Rid);   // identity: code the spatial residual, flat step
            const double jd = Ddct + lambda * Rdct;
            const double js = Ddst + lambda * Rdst, ji = Did + lambda * Rid;
            const double js_f = js + lambda * flag_bits, ji_f = ji + lambda * flag_bits;
            u8 w = 0; double bestf = jd, bestnf = std::min({jd, js, ji});
            if (js_f < bestf) { bestf = js_f; w = 1; }
            if (ji_f < bestf) { bestf = ji_f; w = 2; }
            Jd[static_cast<usize>(b)] = jd; Jbest[static_cast<usize>(b)] = bestf; Jbest_nf[static_cast<usize>(b)] = bestnf; win[static_cast<usize>(b)] = w;
        });
        u64 cnt[3] = {0, 0, 0};
        double sJd = 0, sJb = 0, sJbnf = 0;
        for (s64 b = 0; b < nblk; ++b) { cnt[win[static_cast<usize>(b)]]++; sJd += Jd[static_cast<usize>(b)]; sJb += Jbest[static_cast<usize>(b)]; sJbnf += Jbest_nf[static_cast<usize>(b)]; }
        const double pct = sJd > 0 ? 100.0 * (sJd - sJb) / sJd : 0.0;
        const double pct_nf = sJd > 0 ? 100.0 * (sJd - sJbnf) / sJd : 0.0;
        std::printf("\nq=%.0f  lambda=%.1f\n", q, lambda);
        std::printf("  win-rate (with 1-bit mode flag): DCT %.1f%%  DST-VII %.1f%%  identity %.1f%%\n",
                    100.0 * cnt[0] / nblk, 100.0 * cnt[1] / nblk, 100.0 * cnt[2] / nblk);
        std::printf("  RD ceiling of per-block selection: %.2f%% lower total J (%.2f%% if the flag were free)\n", pct, pct_nf);
    }
    return 0;
}
