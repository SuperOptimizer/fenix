// preprocess/fft.hpp — first-party FFT (radix-2 iterative Cooley-Tukey), no FFTW (the
// fysics dependency-freedom goal). Foundation for Paganin/Wiener deconvolution and phase
// correlation. Sizes must be powers of two (callers pad; radix-3 / 3*2^k + 3D land later).
// See preprocess/CLAUDE.md, docs/research/research-fysics.md.
#pragma once

#include "core/core.hpp"

#include <bit>
#include <cmath>
#include <complex>
#include <numbers>
#include <span>

namespace fenix::preprocess {

using cf32 = std::complex<f32>;

// In-place radix-2 FFT. inverse=false: forward (no scaling). inverse=true: 1/n scaled.
inline void fft1d(std::span<cf32> a, bool inverse) {
    const usize n = a.size();
    if (n < 2) return;
    FENIX_ASSERT(std::has_single_bit(n));  // power of two
    // Decimation-in-time bit-reversal permutation.
    for (usize i = 1, j = 0; i < n; ++i) {
        usize bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const f32 sign = inverse ? 1.0f : -1.0f;
    for (usize len = 2; len <= n; len <<= 1) {
        const f32 ang = sign * 2.0f * std::numbers::pi_v<f32> / static_cast<f32>(len);
        const cf32 wlen(std::cos(ang), std::sin(ang));
        for (usize i = 0; i < n; i += len) {
            cf32 w(1.0f, 0.0f);
            for (usize k = 0; k < len / 2; ++k) {
                const cf32 u = a[i + k];
                const cf32 v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        const f32 inv = 1.0f / static_cast<f32>(n);
        for (cf32& x : a) x *= inv;
    }
}

// Smallest power of two >= n (for padding real inputs to a valid FFT length).
inline usize next_pow2(usize n) { return n <= 1 ? 1 : std::bit_ceil(n); }

// In-place separable 3D FFT over a ZYX cf32 volume (each axis dim a power of two).
inline void fft3d(std::span<cf32> vol, Extent3 d, bool inverse) {
    std::vector<cf32> line;
    auto run = [&](int axis) {
        const s64 n = (axis == 0) ? d.z : (axis == 1) ? d.y : d.x;
        const s64 sz = d.y * d.x, sy = d.x, sx = 1;
        const s64 stride = (axis == 0) ? sz : (axis == 1) ? sy : sx;
        const s64 o1 = (axis == 0) ? d.y : d.z;
        const s64 o2 = (axis == 0) ? d.x : (axis == 1) ? d.x : d.y;
        line.resize(static_cast<usize>(n));
        for (s64 a = 0; a < o1; ++a)
            for (s64 b = 0; b < o2; ++b) {
                s64 base;
                if (axis == 0) base = a * sy + b * sx;
                else if (axis == 1) base = a * sz + b * sx;
                else base = a * sz + b * sy;
                for (s64 t = 0; t < n; ++t) line[static_cast<usize>(t)] = vol[static_cast<usize>(base + t * stride)];
                fft1d(line, inverse);
                for (s64 t = 0; t < n; ++t) vol[static_cast<usize>(base + t * stride)] = line[static_cast<usize>(t)];
            }
    };
    run(2);
    run(1);
    run(0);
}

}  // namespace fenix::preprocess
