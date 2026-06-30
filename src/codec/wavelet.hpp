// codec/wavelet.hpp — CDF 9/7 irreversible wavelet (the shared 2D/3D codec core).
// Lifting scheme (JPEG2000 Annex H), whole-sample symmetric (mirror) boundary, float.
// Separable N-D: one level transforms each axis then recurses on the LL(L) corner.
// This is the transform stage; quantization + bitplane + rANS sit on top (see CLAUDE.md).
#pragma once

#include "core/types.hpp"

#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace fenix::codec {

// CDF 9/7 lifting coefficients (c3d / JPEG2000 Part 1 Annex H).
inline constexpr f32 w97_alpha = -1.586134342059924f;
inline constexpr f32 w97_beta = -0.052980118572961f;
inline constexpr f32 w97_gamma = 0.882911075530934f;
inline constexpr f32 w97_delta = 0.443506852043971f;
inline constexpr f32 w97_k = 1.230174104914001f;
inline constexpr f32 w97_inv_k = 0.812893066115961f;

namespace detail {

// Mirror index for whole-sample symmetric extension: a[-1]=a[1], a[n]=a[n-2].
inline s64 mirror(s64 i, s64 n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

// Forward 9/7 on a contiguous line, then deinterleave to [lows | highs] in place. `tmp` is a
// caller-owned scratch buffer (reused across lines to avoid per-line allocation).
inline void fwd1d(std::span<f32> a, std::vector<f32>& tmp) {
    const s64 n = static_cast<s64>(a.size());
    if (n < 2) return;
    auto at = [&](s64 i) -> f32 { return a[static_cast<usize>(mirror(i, n))]; };
    for (s64 i = 1; i < n; i += 2) a[static_cast<usize>(i)] += w97_alpha * (at(i - 1) + at(i + 1));
    for (s64 i = 0; i < n; i += 2) a[static_cast<usize>(i)] += w97_beta * (at(i - 1) + at(i + 1));
    for (s64 i = 1; i < n; i += 2) a[static_cast<usize>(i)] += w97_gamma * (at(i - 1) + at(i + 1));
    for (s64 i = 0; i < n; i += 2) a[static_cast<usize>(i)] += w97_delta * (at(i - 1) + at(i + 1));
    for (s64 i = 0; i < n; i += 2) a[static_cast<usize>(i)] *= w97_inv_k;  // low band
    for (s64 i = 1; i < n; i += 2) a[static_cast<usize>(i)] *= w97_k;      // high band
    // Deinterleave: evens -> [0, nlow), odds -> [nlow, n).
    const s64 nlow = (n + 1) / 2;
    tmp.resize(static_cast<usize>(n));
    s64 lo = 0, hi = nlow;
    for (s64 i = 0; i < n; ++i) tmp[static_cast<usize>((i & 1) ? hi++ : lo++)] = a[static_cast<usize>(i)];
    for (s64 i = 0; i < n; ++i) a[static_cast<usize>(i)] = tmp[static_cast<usize>(i)];
}

// Inverse: interleave [lows|highs] back, unscale, undo lifting in reverse.
inline void inv1d(std::span<f32> a, std::vector<f32>& tmp) {
    const s64 n = static_cast<s64>(a.size());
    if (n < 2) return;
    const s64 nlow = (n + 1) / 2;
    tmp.resize(static_cast<usize>(n));
    s64 lo = 0, hi = nlow;
    for (s64 i = 0; i < n; ++i) tmp[static_cast<usize>(i)] = a[static_cast<usize>((i & 1) ? hi++ : lo++)];
    for (s64 i = 0; i < n; ++i) a[static_cast<usize>(i)] = tmp[static_cast<usize>(i)];
    auto at = [&](s64 i) -> f32 { return a[static_cast<usize>(mirror(i, n))]; };
    for (s64 i = 0; i < n; i += 2) a[static_cast<usize>(i)] *= w97_k;      // undo low scale
    for (s64 i = 1; i < n; i += 2) a[static_cast<usize>(i)] *= w97_inv_k;  // undo high scale
    for (s64 i = 0; i < n; i += 2) a[static_cast<usize>(i)] -= w97_delta * (at(i - 1) + at(i + 1));
    for (s64 i = 1; i < n; i += 2) a[static_cast<usize>(i)] -= w97_gamma * (at(i - 1) + at(i + 1));
    for (s64 i = 0; i < n; i += 2) a[static_cast<usize>(i)] -= w97_beta * (at(i - 1) + at(i + 1));
    for (s64 i = 1; i < n; i += 2) a[static_cast<usize>(i)] -= w97_alpha * (at(i - 1) + at(i + 1));
}

// Vectorized 9/7 along a NON-contiguous axis: transform `n` rows (consecutive rows `as` apart along the
// axis) for `nx` CONTIGUOUS lanes (the unit-stride x dimension). The lifting/scale/(de)interleave are
// applied as loops over x — independent, unit-stride => the compiler autovectorizes them — with the
// (mirrored) neighbour-row pointers hoisted out of the x-loop. Mathematically identical to running
// fwd1d/inv1d per line, but it processes all x lanes together and needs no per-line gather/scatter. This
// is the wavelet's hot path (the y/z passes were strided-gather + scalar lifting on decode).
template <bool Forward>
inline void axis_vec(f32* base, s64 n, s64 as, s64 nx, std::vector<f32>& tmp) {
    if (n < 2) return;
    const s64 nlow = (n + 1) / 2;
    auto rowp = [&](s64 i) { return base + mirror(i, n) * as; };
    auto lift = [&](s64 i0, f32 coef) {
        for (s64 i = i0; i < n; i += 2) {
            f32* c = base + i * as;
            const f32* a = rowp(i - 1);
            const f32* b = rowp(i + 1);
            for (s64 x = 0; x < nx; ++x) c[x] += coef * (a[x] + b[x]);
        }
    };
    auto scale = [&](s64 i0, f32 s) {
        for (s64 i = i0; i < n; i += 2) {
            f32* c = base + i * as;
            for (s64 x = 0; x < nx; ++x) c[x] *= s;
        }
    };
    const usize rb = static_cast<usize>(nx) * sizeof(f32);
    auto deinterleave = [&]() {  // even rows -> [0,nlow), odd rows -> [nlow,n)
        tmp.resize(static_cast<usize>(n * nx));
        s64 lo = 0, hi = nlow;
        for (s64 i = 0; i < n; ++i) std::memcpy(tmp.data() + ((i & 1) ? hi++ : lo++) * nx, base + i * as, rb);
        for (s64 i = 0; i < n; ++i) std::memcpy(base + i * as, tmp.data() + i * nx, rb);
    };
    auto interleave = [&]() {  // inverse of deinterleave
        tmp.resize(static_cast<usize>(n * nx));
        s64 lo = 0, hi = nlow;
        for (s64 i = 0; i < n; ++i) std::memcpy(tmp.data() + i * nx, base + ((i & 1) ? hi++ : lo++) * as, rb);
        for (s64 i = 0; i < n; ++i) std::memcpy(base + i * as, tmp.data() + i * nx, rb);
    };
    if constexpr (Forward) {
        lift(1, w97_alpha); lift(0, w97_beta); lift(1, w97_gamma); lift(0, w97_delta);
        scale(0, w97_inv_k); scale(1, w97_k);
        deinterleave();
    } else {
        interleave();
        scale(0, w97_k); scale(1, w97_inv_k);
        lift(0, -w97_delta); lift(1, -w97_gamma); lift(0, -w97_beta); lift(1, -w97_alpha);
    }
}

// One 3D level. x is contiguous (scalar fwd1d/inv1d in place — no gather); y and z go through axis_vec,
// vectorized across the contiguous x lanes. Axis order x,y,z for both directions (separable => commutes).
template <bool Forward>
void transform_lines(std::span<f32> buf, Index3 strides, Extent3 ext) {
    std::vector<f32> tmp;
    f32* d = buf.data();
    for (s64 z = 0; z < ext.z; ++z)  // along x (contiguous lines)
        for (s64 y = 0; y < ext.y; ++y) {
            const std::span<f32> l = buf.subspan(static_cast<usize>(z * strides.z + y * strides.y), static_cast<usize>(ext.x));
            if constexpr (Forward) fwd1d(l, tmp); else inv1d(l, tmp);
        }
    for (s64 z = 0; z < ext.z; ++z)  // along y, vectorized over x
        axis_vec<Forward>(d + z * strides.z, ext.y, strides.y, ext.x, tmp);
    for (s64 y = 0; y < ext.y; ++y)  // along z, vectorized over x
        axis_vec<Forward>(d + y * strides.y, ext.z, strides.z, ext.x, tmp);
}

inline Extent3 halve(Extent3 e) { return {(e.z + 1) / 2, (e.y + 1) / 2, (e.x + 1) / 2}; }

// One 2D level over a (side x side) sub-block: x contiguous (fwd1d/inv1d), y via axis_vec across x.
template <bool Forward>
void transform_lines_2d(std::span<f32> buf, s64 stride_y, s64 ext_y, s64 ext_x) {
    std::vector<f32> tmp;
    for (s64 yy = 0; yy < ext_y; ++yy) {  // along x
        const std::span<f32> l = buf.subspan(static_cast<usize>(yy * stride_y), static_cast<usize>(ext_x));
        if constexpr (Forward) fwd1d(l, tmp); else inv1d(l, tmp);
    }
    axis_vec<Forward>(buf.data(), ext_y, stride_y, ext_x, tmp);  // along y, vectorized over x
}

}  // namespace detail

// Multi-level separable 2D forward DWT on a contiguous side*side block (row-major, y-major).
inline void dwt2_forward(std::span<f32> block, s64 side, int levels) {
    s64 ey = side, ex = side;
    for (int l = 0; l < levels && ey >= 2 && ex >= 2; ++l) {
        detail::transform_lines_2d<true>(block, side, ey, ex);
        ey = (ey + 1) / 2;
        ex = (ex + 1) / 2;
    }
}

inline void dwt2_inverse(std::span<f32> block, s64 side, int levels) {
    std::vector<std::pair<s64, s64>> exts;
    s64 ey = side, ex = side;
    for (int l = 0; l < levels && ey >= 2 && ex >= 2; ++l) {
        exts.emplace_back(ey, ex);
        ey = (ey + 1) / 2;
        ex = (ex + 1) / 2;
    }
    for (auto it = exts.rbegin(); it != exts.rend(); ++it)
        detail::transform_lines_2d<false>(block, side, it->first, it->second);
}

// Multi-level separable 3D forward DWT on a contiguous side^3 block (ZYX). `levels`
// decomposition levels; coarsest LLL ends up in the [0:h,0:h,0:h] corner.
inline void dwt3_forward(std::span<f32> block, s64 side, int levels) {
    const Index3 strides = {side * side, side, 1};
    Extent3 ext = {side, side, side};
    for (int l = 0; l < levels && ext.z >= 2 && ext.y >= 2 && ext.x >= 2; ++l) {
        detail::transform_lines<true>(block, strides, ext);
        ext = detail::halve(ext);
    }
}

// Inverse of dwt3_forward (same `levels`).
inline void dwt3_inverse(std::span<f32> block, s64 side, int levels) {
    const Index3 strides = {side * side, side, 1};
    // Reconstruct the extent at each level, then invert coarsest-first.
    std::vector<Extent3> exts;
    Extent3 ext = {side, side, side};
    for (int l = 0; l < levels && ext.z >= 2 && ext.y >= 2 && ext.x >= 2; ++l) {
        exts.push_back(ext);
        ext = detail::halve(ext);
    }
    for (auto it = exts.rbegin(); it != exts.rend(); ++it)
        detail::transform_lines<false>(block, strides, *it);
}

}  // namespace fenix::codec
