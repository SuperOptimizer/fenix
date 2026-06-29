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
// Forward:  X[k] = Σ_n m[k][n] x[n].   Inverse (DCT-III): x[n] = Σ_k m[k][n] X[k] (exact transpose).
struct Dct16Basis {
    f32 m[kDctN][kDctN];
};
inline const Dct16Basis& dct16_basis() {
    static const Dct16Basis b = [] {
        Dct16Basis d{};
        const f32 pi = std::numbers::pi_v<f32>;
        const f32 c0 = std::sqrt(1.0f / kDctN), ck = std::sqrt(2.0f / kDctN);
        for (int k = 0; k < kDctN; ++k)
            for (int n = 0; n < kDctN; ++n)
                d.m[k][n] = (k == 0 ? c0 : ck) * std::cos(pi * static_cast<f32>(2 * n + 1) * static_cast<f32>(k) / (2.0f * kDctN));
        return d;
    }();
    return b;
}

inline void dct16_1d_fwd(const f32* in, f32* out) {
    const Dct16Basis& b = dct16_basis();
    for (int k = 0; k < kDctN; ++k) {
        f32 s = 0;
        for (int n = 0; n < kDctN; ++n) s += b.m[k][n] * in[n];
        out[k] = s;
    }
}
inline void dct16_1d_inv(const f32* in, f32* out) {
    const Dct16Basis& b = dct16_basis();
    for (int n = 0; n < kDctN; ++n) {
        f32 s = 0;
        for (int k = 0; k < kDctN; ++k) s += b.m[k][n] * in[k];  // skip-zero is a decode optimization; correctness first
        out[n] = s;
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
