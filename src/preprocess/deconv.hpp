// preprocess/deconv.hpp — Wiener deconvolution of a Gaussian-like PSF via 3D FFT. This is
// the mechanism behind fysics's Paganin/matched deconvolution (restore contrast/sharpness
// the reconstruction's low-pass removed): F_out = F_in * H / (H^2 + reg). H is the PSF's
// frequency response. The full Paganin path adds the TIE-Hom transfer function + the matched
// nabu operator + per-volume regime gating; this is the core invert. See preprocess/CLAUDE.md.
#pragma once

#include "core/core.hpp"
#include "core/filter.hpp"
#include "preprocess/fft.hpp"

#include <cmath>
#include <numbers>
#include <vector>

namespace fenix::preprocess {

// Spatial unsharp mask: out = in + amount*(in - blur(in)). The cheap, out-of-core-friendly
// (separable Gaussian, no FFT) sharpener — restores the high frequencies the Paganin phase
// retrieval low-passed away. `sigma`/`amount` should be MATCHED to the recon's unsharp params
// from metadata.json (it records the exact blur the reconstruction left). In place.
inline void unsharp_mask(VolumeView<f32> v, f32 sigma, f32 amount) {
    if (amount <= 0.0f || sigma <= 0.0f) return;
    const Extent3 d = v.dims();
    Volume<f32> blur(d);
    for (s64 i = 0; i < d.count(); ++i) blur.flat()[static_cast<usize>(i)] = v.flat()[static_cast<usize>(i)];
    gaussian_blur(blur.view(), sigma);
    auto bv = blur.view();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) v(z, y, x) = v(z, y, x) + amount * (v(z, y, x) - bv(z, y, x));
    });
}

// Gaussian PSF transfer at normalized frequency f (cycles/sample): exp(-2 pi^2 sigma^2 f^2).
inline f32 gaussian_transfer(f32 f, f32 sigma) {
    const f32 two_pi2 = 2.0f * std::numbers::pi_v<f32> * std::numbers::pi_v<f32>;
    return std::exp(-two_pi2 * sigma * sigma * f * f);
}

// Apply a separable Gaussian PSF in the frequency domain (used to simulate blur / as the
// forward operator). dims must be powers of two.
inline Volume<f32> apply_psf(VolumeView<const f32> in, f32 sigma) {
    const Extent3 d = in.dims();
    std::vector<cf32> sp(static_cast<usize>(d.count()));
    for (s64 i = 0; i < d.count(); ++i) sp[static_cast<usize>(i)] = cf32(in.flat()[static_cast<usize>(i)], 0.0f);
    fft3d(sp, d, false);
    auto freq = [](s64 k, s64 n) { return static_cast<f32>(k <= n / 2 ? k : k - n) / static_cast<f32>(n); };
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const f32 h = gaussian_transfer(freq(z, d.z), sigma) *
                              gaussian_transfer(freq(y, d.y), sigma) *
                              gaussian_transfer(freq(x, d.x), sigma);
                sp[static_cast<usize>((z * d.y + y) * d.x + x)] *= h;
            }
    fft3d(sp, d, true);
    Volume<f32> out(d);
    for (s64 i = 0; i < d.count(); ++i) out.flat()[static_cast<usize>(i)] = sp[static_cast<usize>(i)].real();
    return out;
}

// Wiener deconvolution of a Gaussian PSF. `reg` (Tikhonov) caps the high-frequency gain.
inline Volume<f32> wiener_deconvolve(VolumeView<const f32> in, f32 sigma, f32 reg = 0.015f) {
    const Extent3 d = in.dims();
    std::vector<cf32> sp(static_cast<usize>(d.count()));
    for (s64 i = 0; i < d.count(); ++i) sp[static_cast<usize>(i)] = cf32(in.flat()[static_cast<usize>(i)], 0.0f);
    fft3d(sp, d, false);
    auto freq = [](s64 k, s64 n) { return static_cast<f32>(k <= n / 2 ? k : k - n) / static_cast<f32>(n); };
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const f32 h = gaussian_transfer(freq(z, d.z), sigma) *
                              gaussian_transfer(freq(y, d.y), sigma) *
                              gaussian_transfer(freq(x, d.x), sigma);
                const f32 inv = h / (h * h + reg);  // Wiener
                sp[static_cast<usize>((z * d.y + y) * d.x + x)] *= inv;
            }
    fft3d(sp, d, true);
    Volume<f32> out(d);
    for (s64 i = 0; i < d.count(); ++i) out.flat()[static_cast<usize>(i)] = sp[static_cast<usize>(i)].real();
    return out;
}

}  // namespace fenix::preprocess
