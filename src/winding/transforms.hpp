// winding/transforms.hpp — invertible-by-construction factors of the compositional
// diffeomorphism (spiral-v2): a per-slice 2x2 affine via matrix exponential (det = e^tr > 0
// always) and a monotonic radial gap-expander (searchsorted inverse). Composed with the
// SVF flow (flow.hpp) + umbilicus, these form the global fold-free map. See winding/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <cmath>
#include <vector>

namespace fenix::winding {

// 2x2 matrix exponential, closed form. L = [[a,b],[c,d]].
// expm(L) = e^tau ( cosh(s) I + sinh(s)/s * (L - tau I) ),  s^2 = ((a-d)/2)^2 + b c.
struct Mat2 {
    f32 m00, m01, m10, m11;
    [[nodiscard]] f32 det() const { return m00 * m11 - m01 * m10; }
};

inline Mat2 expm2(f32 a, f32 b, f32 c, f32 d) {
    const f32 tau = 0.5f * (a + d);
    const f32 s2 = 0.25f * (a - d) * (a - d) + b * c;
    f32 ch, sh_over_s;
    if (s2 > 1e-12f) {
        const f32 s = std::sqrt(s2);
        ch = std::cosh(s);
        sh_over_s = std::sinh(s) / s;
    } else if (s2 < -1e-12f) {
        const f32 s = std::sqrt(-s2);
        ch = std::cos(s);
        sh_over_s = std::sin(s) / s;
    } else {
        ch = 1.0f;
        sh_over_s = 1.0f;  // limit sinh(s)/s -> 1
    }
    const f32 e = std::exp(tau);
    // M' = L - tau I
    const f32 p00 = a - tau, p01 = b, p10 = c, p11 = d - tau;
    return {e * (ch + sh_over_s * p00), e * (sh_over_s * p01), e * (sh_over_s * p10),
            e * (ch + sh_over_s * p11)};
}

// Per-slice affine on (y,x): M = expm(L), with translation. Always orientation-preserving
// (det M = e^{trace L} > 0), so it can never fold. Inverse uses expm(-L).
struct AffineYX {
    f32 a = 0, b = 0, c = 0, d = 0;  // the log-matrix L (a,b;c,d)
    f32 ty = 0, tx = 0;              // translation

    [[nodiscard]] Vec3f apply(Vec3f p) const {
        const Mat2 m = expm2(a, b, c, d);
        return {p.z, m.m00 * p.y + m.m01 * p.x + ty, m.m10 * p.y + m.m11 * p.x + tx};
    }
    [[nodiscard]] Vec3f inverse(Vec3f p) const {
        const Mat2 mi = expm2(-a, -b, -c, -d);  // exp(-L) = inv(exp(L))
        const f32 y = p.y - ty, x = p.x - tx;
        return {p.z, mi.m00 * y + mi.m01 * x, mi.m10 * y + mi.m11 * x};
    }
};

// Per-z-band affine stack (spiral-v2's per-slice affine, banded): piecewise-constant
// AffineYX over z — band k covers [z0 + k*dz, z0 + (k+1)*dz). Each band is expm-based so
// every slice stays orientation-preserving/fold-free; capacity scales with band count.
// Empty bands => identity (the single-affine model is bands.size()==1 with any z0/dz).
struct AffineStack {
    f32 z0 = 0, dz = 1;
    std::vector<AffineYX> bands;

    [[nodiscard]] bool empty() const { return bands.empty(); }
    [[nodiscard]] s64 band_of(f32 z) const {
        if (bands.empty()) return -1;
        const s64 k = static_cast<s64>((z - z0) / dz);
        return std::clamp<s64>(k, 0, static_cast<s64>(bands.size()) - 1);
    }
    [[nodiscard]] Vec3f apply(Vec3f p) const {
        const s64 k = band_of(p.z);
        return k < 0 ? p : bands[static_cast<usize>(k)].apply(p);
    }
    [[nodiscard]] Vec3f inverse(Vec3f p) const {
        const s64 k = band_of(p.z);
        return k < 0 ? p : bands[static_cast<usize>(k)].inverse(p);
    }
};

// Monotonic radial gap-expander: per-winding positive scale (exp of logits); the winding
// boundary radii are the cumulative sum, so the remap is strictly increasing -> invertible.
// Models real non-uniform inter-winding spacing while keeping ideal-space uniform.
struct GapExpander {
    f32 dr = 1.0f;               // ideal radius per winding
    std::vector<f32> logits;     // per-winding log-scale (size = #windings)

    // Forward: ideal radius r -> scrolled radius. r in ideal units; winding = floor(r/dr).
    [[nodiscard]] f32 forward(f32 r_ideal) const {
        if (logits.empty()) return r_ideal;
        const f32 w = r_ideal / dr;
        s64 k = static_cast<s64>(w);
        f32 acc = 0;
        for (s64 i = 0; i < k && i < static_cast<s64>(logits.size()); ++i)
            acc += dr * std::exp(logits[static_cast<usize>(i)]);
        const f32 frac = w - static_cast<f32>(k);
        const f32 scale = (k < static_cast<s64>(logits.size())) ? std::exp(logits[static_cast<usize>(k)]) : 1.0f;
        return acc + frac * dr * scale;
    }

    // Inverse via searchsorted over the cumulative boundary radii (monotone -> well-defined).
    [[nodiscard]] f32 inverse(f32 r_scroll) const {
        if (logits.empty()) return r_scroll;
        f32 acc = 0;
        for (usize i = 0; i < logits.size(); ++i) {
            const f32 seg = dr * std::exp(logits[i]);
            if (r_scroll < acc + seg) return (static_cast<f32>(i) + (r_scroll - acc) / seg) * dr;
            acc += seg;
        }
        return static_cast<f32>(logits.size()) * dr + (r_scroll - acc);
    }
};

}  // namespace fenix::winding
