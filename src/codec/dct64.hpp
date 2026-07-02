// codec/dct64.hpp — separable all-float 64-point DCT-II (orthonormal), the 2D companion of
// the volume codec's DCT-16: 64×64 tiles for surface/image fields (codec/tile2d.hpp). Basis
// is a function-local static (no load-bearing global constructors); plain matrix-vector per
// line — 64² MACs, vectorized fine by the compiler under fast-math.
#pragma once

#include "core/core.hpp"

#include <cmath>
#include <numbers>

namespace fenix::codec {

inline constexpr int kDct64N = 64;

namespace detail {
struct Dct64Basis {
    // b[k][n] = s(k) * cos(pi/N * (n + 1/2) * k), orthonormal (s(0)=sqrt(1/N), else sqrt(2/N)).
    f32 b[kDct64N][kDct64N];
};

inline const Dct64Basis& dct64_basis() {
    static const Dct64Basis basis = [] {
        Dct64Basis t;
        constexpr f64 N = kDct64N;
        for (int k = 0; k < kDct64N; ++k) {
            const f64 s = std::sqrt((k == 0 ? 1.0 : 2.0) / N);
            for (int n = 0; n < kDct64N; ++n)
                t.b[k][n] = static_cast<f32>(s * std::cos(std::numbers::pi / N * (n + 0.5) * k));
        }
        return t;
    }();
    return basis;
}
}  // namespace detail

inline void dct64_1d_fwd(const f32* x, f32* X) {
    const auto& b = detail::dct64_basis();
    for (int k = 0; k < kDct64N; ++k) {
        f32 acc = 0;
        for (int n = 0; n < kDct64N; ++n) acc += b.b[k][n] * x[n];
        X[k] = acc;
    }
}

inline void dct64_1d_inv(const f32* X, f32* x) {
    const auto& b = detail::dct64_basis();
    for (int n = 0; n < kDct64N; ++n) {
        f32 acc = 0;
        for (int k = 0; k < kDct64N; ++k) acc += b.b[k][n] * X[k];
        x[n] = acc;
    }
}

// In-place 2D transform of a 64×64 tile (row-major t[v][u]): rows then columns.
inline void dct64_fwd_tile(f32* t) {
    f32 tmp[kDct64N];
    for (int v = 0; v < kDct64N; ++v) {
        dct64_1d_fwd(t + v * kDct64N, tmp);
        for (int u = 0; u < kDct64N; ++u) t[v * kDct64N + u] = tmp[u];
    }
    f32 col[kDct64N];
    for (int u = 0; u < kDct64N; ++u) {
        for (int v = 0; v < kDct64N; ++v) col[v] = t[v * kDct64N + u];
        dct64_1d_fwd(col, tmp);
        for (int v = 0; v < kDct64N; ++v) t[v * kDct64N + u] = tmp[v];
    }
}

inline void dct64_inv_tile(f32* t) {
    f32 tmp[kDct64N];
    f32 col[kDct64N];
    for (int u = 0; u < kDct64N; ++u) {
        for (int v = 0; v < kDct64N; ++v) col[v] = t[v * kDct64N + u];
        dct64_1d_inv(col, tmp);
        for (int v = 0; v < kDct64N; ++v) t[v * kDct64N + u] = tmp[v];
    }
    for (int v = 0; v < kDct64N; ++v) {
        dct64_1d_inv(t + v * kDct64N, tmp);
        for (int u = 0; u < kDct64N; ++u) t[v * kDct64N + u] = tmp[u];
    }
}

}  // namespace fenix::codec
