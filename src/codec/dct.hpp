// codec/dct.hpp — separable ALL-FLOAT DCT-16 (the second transform codec's core; the first is the
// CDF 9/7 wavelet in wavelet.hpp). Orthonormal DCT-II forward / DCT-III inverse on 16-point lines,
// applied separably across the 3 axes of a 16³ block (and the 2 axes of a 16² block). A rewrite of
// matter-compressor's mc_codec — theirs is an INTEGER DCT; ours is float, per the codec's
// tolerance-only / fast-math rule (correctness = PSNR/τ, never bit-exact). Orthonormal ⇒ the inverse
// is the transpose ⇒ the round-trip is exact to fp error. See codec/CLAUDE.md, docs/research/research-mc.md.
#pragma once

#include "core/types.hpp"

#include <cmath>
#include <numbers>
#include <span>

namespace fenix::codec {

inline constexpr int kDctN = 16;  // fixed 16-point ("DCT-16"); blocks are 16³ / 16².

// Orthonormal DCT-II basis: m[k][n] = c_k · cos(π(2n+1)k / 2N), c_0 = √(1/N), c_{k>0} = √(2/N).
// We exploit the even/odd symmetry m[k][N-1-n] = (-1)^k m[k][n] to halve the work (partial butterfly):
// even coefficients depend only on the sums s[n]=x[n]+x[N-1-n], odd only on the diffs d[n]=x[n]-x[N-1-n].
// `me`/`mo` are the even/odd half-bases (H×H, H=N/2). Forward X[k]=Σ m[k][n]x[n]; inverse (DCT-III)
// x[n]=Σ_k m[k][n]X[k] — orthonormal, so the round-trip is exact to fp error.
inline constexpr int kDctH = kDctN / 2;
struct Dct16Basis {
    f32 me[kDctH][kDctH];  // me[ke][n] = m[2ke][n]   (even rows)
    f32 mo[kDctH][kDctH];  // mo[ko][n] = m[2ko+1][n] (odd rows)
};
inline const Dct16Basis& dct16_basis() {
    static const Dct16Basis b = [] {
        Dct16Basis d{};
        const f32 pi = std::numbers::pi_v<f32>;
        const f32 c0 = std::sqrt(1.0f / kDctN), ck = std::sqrt(2.0f / kDctN);
        auto m = [&](int k, int n) { return (k == 0 ? c0 : ck) * std::cos(pi * static_cast<f32>(2 * n + 1) * static_cast<f32>(k) / (2.0f * kDctN)); };
        for (int j = 0; j < kDctH; ++j)
            for (int n = 0; n < kDctH; ++n) {
                d.me[j][n] = m(2 * j, n);
                d.mo[j][n] = m(2 * j + 1, n);
            }
        return d;
    }();
    return b;
}

// Forward DCT-II via the even/odd partial butterfly (N²/2 MACs instead of N²).
inline void dct16_1d_fwd(const f32* x, f32* X) {
    const Dct16Basis& b = dct16_basis();
    f32 s[kDctH], dd[kDctH];
    for (int n = 0; n < kDctH; ++n) {
        s[n] = x[n] + x[kDctN - 1 - n];
        dd[n] = x[n] - x[kDctN - 1 - n];
    }
    for (int k = 0; k < kDctH; ++k) {
        f32 e = 0, o = 0;
        for (int n = 0; n < kDctH; ++n) {
            e += b.me[k][n] * s[n];
            o += b.mo[k][n] * dd[n];
        }
        X[2 * k] = e;
        X[2 * k + 1] = o;
    }
}

// Inverse (DCT-III) via the same symmetry — branchless SAXPY accumulation (half the MACs of the naive
// transpose-matmul; the per-coefficient skip-zero `if` was measured SLOWER — it breaks vectorization and
// the dense low-q case dominates, so we stay branchless).
inline void dct16_1d_inv(const f32* X, f32* x) {
    const Dct16Basis& b = dct16_basis();
    f32 E[kDctH] = {}, O[kDctH] = {};
    for (int k = 0; k < kDctH; ++k) {
        const f32 ev = X[2 * k], od = X[2 * k + 1];
        for (int n = 0; n < kDctH; ++n) {
            E[n] += b.me[k][n] * ev;
            O[n] += b.mo[k][n] * od;
        }
    }
    for (int n = 0; n < kDctH; ++n) {
        x[n] = E[n] + O[n];
        x[kDctN - 1 - n] = E[n] - O[n];
    }
}

namespace detail {
// Transform every length-16 line along one axis of a 16ᴰ block in place (stride between line elements
// = `stride`; `nlines` lines, each starting `step` apart in the flat array). One1d = dct16_1d_fwd/inv.
template <class One1d>
inline void dct16_lines(std::span<f32> b, s64 nlines, s64 step, s64 stride, One1d&& one) {
    f32 line[kDctN], out[kDctN];
    for (s64 l = 0; l < nlines; ++l) {
        f32* base = b.data() + l * step;  // NB: caller picks (step,stride) so lines never alias
        for (int i = 0; i < kDctN; ++i) line[i] = base[i * stride];
        one(line, out);
        for (int i = 0; i < kDctN; ++i) base[i * stride] = out[i];
    }
}
}  // namespace detail

// 3D 16³ block (ZYX contiguous, x fastest). Separable: transform x-lines, then y-lines, then z-lines.
// Order is irrelevant (separable); the inverse uses the same axis sweeps with the 1D inverse.
inline void dct16_3d_fwd(std::span<f32> blk) {
    constexpr s64 N = kDctN, NN = N * N;
    // x-lines: contiguous runs of 16, one per (z,y) -> step=16, stride=1.
    detail::dct16_lines(blk, NN, N, 1, dct16_1d_fwd);
    // y-lines: stride N. There are N*N lines indexed (z,x); a line starts at z*NN + x, stride N.
    {
        f32 line[N], out[N];
        for (s64 z = 0; z < N; ++z)
            for (s64 x = 0; x < N; ++x) {
                f32* p = blk.data() + z * NN + x;
                for (int i = 0; i < N; ++i) line[i] = p[i * N];
                dct16_1d_fwd(line, out);
                for (int i = 0; i < N; ++i) p[i * N] = out[i];
            }
    }
    // z-lines: stride NN, starting at y*N + x.
    {
        f32 line[N], out[N];
        for (s64 y = 0; y < N; ++y)
            for (s64 x = 0; x < N; ++x) {
                f32* p = blk.data() + y * N + x;
                for (int i = 0; i < N; ++i) line[i] = p[i * NN];
                dct16_1d_fwd(line, out);
                for (int i = 0; i < N; ++i) p[i * NN] = out[i];
            }
    }
}

inline void dct16_3d_inv(std::span<f32> blk) {
    constexpr s64 N = kDctN, NN = N * N;
    detail::dct16_lines(blk, NN, N, 1, dct16_1d_inv);
    {
        f32 line[N], out[N];
        for (s64 z = 0; z < N; ++z)
            for (s64 x = 0; x < N; ++x) {
                f32* p = blk.data() + z * NN + x;
                for (int i = 0; i < N; ++i) line[i] = p[i * N];
                dct16_1d_inv(line, out);
                for (int i = 0; i < N; ++i) p[i * N] = out[i];
            }
    }
    {
        f32 line[N], out[N];
        for (s64 y = 0; y < N; ++y)
            for (s64 x = 0; x < N; ++x) {
                f32* p = blk.data() + y * N + x;
                for (int i = 0; i < N; ++i) line[i] = p[i * NN];
                dct16_1d_inv(line, out);
                for (int i = 0; i < N; ++i) p[i * NN] = out[i];
            }
    }
}

// 2D 16² block (YX contiguous, x fastest) — for the image / parametric-surface / texture-layer path.
inline void dct16_2d_fwd(std::span<f32> blk) {
    detail::dct16_lines(blk, kDctN, kDctN, 1, dct16_1d_fwd);  // x-lines (rows)
    f32 line[kDctN], out[kDctN];
    for (s64 x = 0; x < kDctN; ++x) {  // y-lines (columns), stride 16
        f32* p = blk.data() + x;
        for (int i = 0; i < kDctN; ++i) line[i] = p[i * kDctN];
        dct16_1d_fwd(line, out);
        for (int i = 0; i < kDctN; ++i) p[i * kDctN] = out[i];
    }
}
inline void dct16_2d_inv(std::span<f32> blk) {
    detail::dct16_lines(blk, kDctN, kDctN, 1, dct16_1d_inv);
    f32 line[kDctN], out[kDctN];
    for (s64 x = 0; x < kDctN; ++x) {
        f32* p = blk.data() + x;
        for (int i = 0; i < kDctN; ++i) line[i] = p[i * kDctN];
        dct16_1d_inv(line, out);
        for (int i = 0; i < kDctN; ++i) p[i * kDctN] = out[i];
    }
}

}  // namespace fenix::codec
