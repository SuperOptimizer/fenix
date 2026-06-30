// codec/dct.hpp — separable ALL-FLOAT DCT-16, the core of fenix's sole transform codec (the CDF 9/7
// wavelet was retired, ADR 0005). Orthonormal DCT-II forward / DCT-III inverse on 16-point lines,
// applied separably across the 3 axes of a 16³ block (and the 2 axes of a 16² block). A rewrite of
// matter-compressor's mc_codec — theirs is an INTEGER DCT; ours is float, per the codec's
// tolerance-only / fast-math rule (correctness = PSNR/τ, never bit-exact). Orthonormal ⇒ the inverse
// is the transpose ⇒ the round-trip is exact to fp error. See codec/CLAUDE.md, docs/research/research-mc.md.
#pragma once

#include "core/types.hpp"

#include <cmath>
#include <cstring>
#include <numbers>
#include <span>

namespace fenix::codec {

inline constexpr int kDctN = 16;  // fixed 16-point ("DCT-16"); blocks are 16³ / 16².

// One panel row = the 16 contiguous columns (the x-axis lane) as a SIMD vector. A Clang vector extension
// (the project is Clang-only) — guaranteed to lower to AVX-512/AVX2/NEON, no std::simd dependency (libc++
// here has no <experimental/simd>). vload/vstore go through memcpy so any alignment is safe (the panel and
// the block buffer aren't guaranteed 64-byte aligned).
using DctVec = f32 __attribute__((ext_vector_type(kDctN)));
inline DctVec vload(const f32* p) { DctVec v; std::memcpy(&v, p, sizeof v); return v; }
inline void vstore(f32* p, DctVec v) { std::memcpy(p, &v, sizeof v); }

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

// Vectorized 16-point DCT on a 16×16 PANEL: transform runs down the ROW index (0..15), with the 16
// COLUMNS processed in parallel — the inner `for c` loops are 16 contiguous f32, which clang vectorizes
// to AVX2/NEON at -march=native -ffast-math (no gathers, no intrinsics). The 3D transform feeds each
// separable axis through this by choosing the panel layout so the column is the contiguous x-axis (so
// every load/store the butterfly itself does is unit-stride). Even/odd partial butterfly as in the 1-D
// scalar form. No aliasing: forward reads all rows into s/d before writing; inverse accumulates into
// E/O before writing.
inline void dct16_fwd_panel(f32 p[kDctN][kDctN]) {
    const Dct16Basis& b = dct16_basis();
    DctVec row[kDctN];
    for (int i = 0; i < kDctN; ++i) row[i] = vload(&p[i][0]);
    DctVec s[kDctH], d[kDctH];
    for (int n = 0; n < kDctH; ++n) { s[n] = row[n] + row[kDctN - 1 - n]; d[n] = row[n] - row[kDctN - 1 - n]; }
    for (int k = 0; k < kDctH; ++k) {
        DctVec e = 0.0f, o = 0.0f;
        for (int n = 0; n < kDctH; ++n) { e += s[n] * b.me[k][n]; o += d[n] * b.mo[k][n]; }
        vstore(&p[2 * k][0], e);
        vstore(&p[2 * k + 1][0], o);
    }
}
inline void dct16_inv_panel(f32 p[kDctN][kDctN]) {
    const Dct16Basis& b = dct16_basis();
    DctVec ev[kDctH], od[kDctH];
    for (int k = 0; k < kDctH; ++k) { ev[k] = vload(&p[2 * k][0]); od[k] = vload(&p[2 * k + 1][0]); }
    DctVec E[kDctH], O[kDctH];
    for (int n = 0; n < kDctH; ++n) { E[n] = 0.0f; O[n] = 0.0f; }
    for (int k = 0; k < kDctH; ++k)
        for (int n = 0; n < kDctH; ++n) { E[n] += ev[k] * b.me[k][n]; O[n] += od[k] * b.mo[k][n]; }
    for (int n = 0; n < kDctH; ++n) {
        vstore(&p[n][0], E[n] + O[n]);
        vstore(&p[kDctN - 1 - n][0], E[n] - O[n]);
    }
}

// 3D 16³ block (ZYX contiguous, x fastest). Separable; each pass uses the vectorized panel with x as the
// column (contiguous) lane. y-pass is in place (a z-slab IS a contiguous 16×16 [y][x] panel); x-pass and
// z-pass gather a local panel (transpose / strided rows) so the column stays the contiguous x. Order is
// irrelevant (separable).
inline void dct16_3d_fwd(std::span<f32> blk) {
    constexpr s64 N = kDctN, NN = N * N;
    f32* d = blk.data();
    f32 panel[kDctN][kDctN];
    for (s64 z = 0; z < N; ++z) {  // x-pass: panel[x][y] = slab[y][x] (transpose), transform along x
        f32* slab = d + z * NN;
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y) panel[x][y] = slab[y * N + x];
        dct16_fwd_panel(panel);
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y) slab[y * N + x] = panel[x][y];
    }
    for (s64 z = 0; z < N; ++z)  // y-pass: the z-slab is a contiguous [y][x] panel → in place
        dct16_fwd_panel(reinterpret_cast<f32(*)[kDctN]>(d + z * NN));
    for (s64 y = 0; y < N; ++y) {  // z-pass: panel[z][x] = blk[z][y][x] (strided rows), transform along z
        for (int z = 0; z < N; ++z)
            for (int x = 0; x < N; ++x) panel[z][x] = d[z * NN + y * N + x];
        dct16_fwd_panel(panel);
        for (int z = 0; z < N; ++z)
            for (int x = 0; x < N; ++x) d[z * NN + y * N + x] = panel[z][x];
    }
}

inline void dct16_3d_inv(std::span<f32> blk) {
    constexpr s64 N = kDctN, NN = N * N;
    f32* d = blk.data();
    f32 panel[kDctN][kDctN];
    for (s64 z = 0; z < N; ++z) {
        f32* slab = d + z * NN;
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y) panel[x][y] = slab[y * N + x];
        dct16_inv_panel(panel);
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y) slab[y * N + x] = panel[x][y];
    }
    for (s64 z = 0; z < N; ++z)
        dct16_inv_panel(reinterpret_cast<f32(*)[kDctN]>(d + z * NN));
    for (s64 y = 0; y < N; ++y) {
        for (int z = 0; z < N; ++z)
            for (int x = 0; x < N; ++x) panel[z][x] = d[z * NN + y * N + x];
        dct16_inv_panel(panel);
        for (int z = 0; z < N; ++z)
            for (int x = 0; x < N; ++x) d[z * NN + y * N + x] = panel[z][x];
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
