// preprocess/musica.hpp — MUSICA multiscale contrast amplification (Vuylsteke & Schoeters, Agfa),
// TRUE 3D. Build a Laplacian-style pyramid (mask-aware normalized convolution) with a separable 3D
// 5-tap binomial blur, apply a SUBLINEAR gain to each level's detail coefficients (boosts small/
// low-contrast coeffs more than large ones), rebuild:
//   y = a·sign(x)·|x/a|^p,  p<1     (per level, on detail coeffs; optional soft "coring" of noise)
// 3D (not per-slice) so the boost is z-consistent — the sheets are 3D structures. Masked (==0) and
// clipped (==max) voxels pass through unmodified (normalized convolution → no halos across a mask
// boundary). fysics lineage; the upstream is 2D-per-slice, this is the volumetric generalization.
// Memory: 8 full-volume f32 buffers (crop-scale; apron-tile for out-of-core later). See CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace fenix::preprocess {
namespace detail {

// One separable 5-tap binomial (1 4 6 4 1)/16 pass along `axis` (0=z,1=y,2=x), clamp boundary; in->out.
inline void blur_axis(const std::vector<f32>& in, std::vector<f32>& out, const Extent3& d, int axis) {
    constexpr f32 k0 = 6.0f / 16, k1 = 4.0f / 16, k2 = 1.0f / 16;
    const s64 Z = d.z, Y = d.y, X = d.x;
    auto cl = [](s64 i, s64 n) -> s64 { return i < 0 ? 0 : (i >= n ? n - 1 : i); };
    auto at = [&](s64 z, s64 y, s64 x) -> f32 { return in[static_cast<usize>((z * Y + y) * X + x)]; };
    if (axis == 2) {  // x (fastest, contiguous)
        parallel_for_z(d, [&](s64 z) {
            for (s64 y = 0; y < Y; ++y) {
                const f32* r = in.data() + static_cast<usize>((z * Y + y) * X);
                f32* o = out.data() + static_cast<usize>((z * Y + y) * X);
                for (s64 x = 0; x < X; ++x)
                    o[x] = k2 * r[cl(x - 2, X)] + k1 * r[cl(x - 1, X)] + k0 * r[x] + k1 * r[cl(x + 1, X)] +
                           k2 * r[cl(x + 2, X)];
            }
        });
    } else if (axis == 1) {  // y
        parallel_for_z(d, [&](s64 z) {
            for (s64 y = 0; y < Y; ++y)
                for (s64 x = 0; x < X; ++x)
                    out[static_cast<usize>((z * Y + y) * X + x)] =
                        k2 * at(z, cl(y - 2, Y), x) + k1 * at(z, cl(y - 1, Y), x) + k0 * at(z, y, x) +
                        k1 * at(z, cl(y + 1, Y), x) + k2 * at(z, cl(y + 2, Y), x);
        });
    } else {  // z
        parallel_for_z(d, [&](s64 z) {
            for (s64 y = 0; y < Y; ++y)
                for (s64 x = 0; x < X; ++x)
                    out[static_cast<usize>((z * Y + y) * X + x)] =
                        k2 * at(cl(z - 2, Z), y, x) + k1 * at(cl(z - 1, Z), y, x) + k0 * at(z, y, x) +
                        k1 * at(cl(z + 1, Z), y, x) + k2 * at(cl(z + 2, Z), y, x);
        });
    }
}

// Separable 3D binomial blur: in -> out (x then y then z), using `scr` as scratch.
inline void blur5_3d(const std::vector<f32>& in, std::vector<f32>& out, const Extent3& d, std::vector<f32>& scr) {
    blur_axis(in, out, d, 2);
    blur_axis(out, scr, d, 1);
    blur_axis(scr, out, d, 0);
}

// Sublinear gain on a detail coefficient (with soft noise coring below `core`).
inline f32 musica_gain(f32 x, f32 a, f32 p, f32 core) {
    const f32 ax = std::fabs(x);
    if (core > 0 && ax < core) return x * (ax / (core + 1e-8f));
    const f32 s = x < 0 ? -1.0f : 1.0f;
    return a * s * std::pow(ax / a, p);
}

}  // namespace detail

// TRUE 3D MUSICA on a volume with values in [0, vmax] (e.g. 0..255), in place. levels = pyramid depth,
// p = gain exponent (<1 boosts faint detail), core = noise coring threshold (in [0,1] units; 0 disables).
// Masked (==0) and clipped (==vmax) voxels pass through unmodified.
inline void musica_inplace(VolumeView<f32> vol, s32 levels = 4, f32 p = 0.7f, f32 core = 0.0f,
                           f32 vmax = 255.0f) {
    if (levels < 1) levels = 4;
    if (p <= 0) p = 0.7f;
    const Extent3 d = vol.dims();
    const usize n = static_cast<usize>(d.count());
    const f32 inv = 1.0f / vmax;
    std::vector<f32> V(n), W(n), base(n), Vn(n), Wn(n), tmp(n), scr(n), acc(n, 0.0f);
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const usize i = static_cast<usize>((z * d.y + y) * d.x + x);
                const f32 v = vol(z, y, x) * inv;
                const bool valid = v > 0.0f && v < 1.0f;
                W[i] = valid ? 1.0f : 0.0f;
                V[i] = valid ? v : 0.0f;
                base[i] = v;
            }
    });
    Vn = V;
    Wn = W;
    constexpr f32 a = 0.5f;  // gain normalization point
    for (s32 l = 0; l < levels; ++l) {
        const s32 iters = 1 << l;  // cumulative octave blur (1,3,7,15 passes)
        for (s32 it = 0; it < iters; ++it) {
            detail::blur5_3d(Vn, tmp, d, scr);
            std::swap(Vn, tmp);
            detail::blur5_3d(Wn, tmp, d, scr);
            std::swap(Wn, tmp);
        }
        for (usize i = 0; i < n; ++i) {
            const f32 nb = (Wn[i] > 1e-6f) ? Vn[i] / Wn[i] : base[i];  // normalized convolution
            acc[i] += detail::musica_gain(base[i] - nb, a, p, core);
            base[i] = nb;
        }
    }
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const usize i = static_cast<usize>((z * d.y + y) * d.x + x);
                const f32 v0 = vol(z, y, x) * inv;
                if (!(v0 > 0.0f && v0 < 1.0f)) continue;  // masked/clipped pass through
                const f32 v = std::clamp(base[i] + acc[i], 0.0f, 1.0f);
                vol(z, y, x) = v * vmax;
            }
    });
}

}  // namespace fenix::preprocess
