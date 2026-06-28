// codec/wavelet.hpp — CDF 9/7 irreversible wavelet (the shared 2D/3D codec core).
// Lifting scheme (JPEG2000 Annex H), whole-sample symmetric (mirror) boundary, float.
// Separable N-D: one level transforms each axis then recurses on the LL(L) corner.
// This is the transform stage; quantization + bitplane + rANS sit on top (see CLAUDE.md).
#pragma once

#include "core/types.hpp"

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

// Gather/transform/scatter one axis line of a strided 3D sub-block.
template <bool Forward>
void transform_lines(std::span<f32> buf, Index3 strides, Extent3 ext) {
    std::vector<f32> line, tmp;
    auto run = [&](std::span<f32> l) { if constexpr (Forward) fwd1d(l, tmp); else inv1d(l, tmp); };
    // along x
    for (s64 z = 0; z < ext.z; ++z)
        for (s64 y = 0; y < ext.y; ++y) {
            line.resize(static_cast<usize>(ext.x));
            const s64 base = z * strides.z + y * strides.y;
            for (s64 x = 0; x < ext.x; ++x) line[static_cast<usize>(x)] = buf[static_cast<usize>(base + x * strides.x)];
            run(line);
            for (s64 x = 0; x < ext.x; ++x) buf[static_cast<usize>(base + x * strides.x)] = line[static_cast<usize>(x)];
        }
    // along y
    for (s64 z = 0; z < ext.z; ++z)
        for (s64 x = 0; x < ext.x; ++x) {
            line.resize(static_cast<usize>(ext.y));
            const s64 base = z * strides.z + x * strides.x;
            for (s64 y = 0; y < ext.y; ++y) line[static_cast<usize>(y)] = buf[static_cast<usize>(base + y * strides.y)];
            run(line);
            for (s64 y = 0; y < ext.y; ++y) buf[static_cast<usize>(base + y * strides.y)] = line[static_cast<usize>(y)];
        }
    // along z
    for (s64 y = 0; y < ext.y; ++y)
        for (s64 x = 0; x < ext.x; ++x) {
            line.resize(static_cast<usize>(ext.z));
            const s64 base = y * strides.y + x * strides.x;
            for (s64 z = 0; z < ext.z; ++z) line[static_cast<usize>(z)] = buf[static_cast<usize>(base + z * strides.z)];
            run(line);
            for (s64 z = 0; z < ext.z; ++z) buf[static_cast<usize>(base + z * strides.z)] = line[static_cast<usize>(z)];
        }
}

inline Extent3 halve(Extent3 e) { return {(e.z + 1) / 2, (e.y + 1) / 2, (e.x + 1) / 2}; }

// One 2D level over a (side x side) sub-block (rows of length ext_x, ext_y rows), strides.
template <bool Forward>
void transform_lines_2d(std::span<f32> buf, s64 stride_y, s64 ext_y, s64 ext_x) {
    std::vector<f32> line, tmp;
    auto run = [&](std::span<f32> l) { if constexpr (Forward) fwd1d(l, tmp); else inv1d(l, tmp); };
    for (s64 yy = 0; yy < ext_y; ++yy) {  // along x
        line.resize(static_cast<usize>(ext_x));
        const s64 base = yy * stride_y;
        for (s64 xx = 0; xx < ext_x; ++xx) line[static_cast<usize>(xx)] = buf[static_cast<usize>(base + xx)];
        run(line);
        for (s64 xx = 0; xx < ext_x; ++xx) buf[static_cast<usize>(base + xx)] = line[static_cast<usize>(xx)];
    }
    for (s64 xx = 0; xx < ext_x; ++xx) {  // along y
        line.resize(static_cast<usize>(ext_y));
        for (s64 yy = 0; yy < ext_y; ++yy) line[static_cast<usize>(yy)] = buf[static_cast<usize>(yy * stride_y + xx)];
        run(line);
        for (s64 yy = 0; yy < ext_y; ++yy) buf[static_cast<usize>(yy * stride_y + xx)] = line[static_cast<usize>(yy)];
    }
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
