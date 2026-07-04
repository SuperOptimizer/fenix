// preprocess/phasecorr.hpp — phase-correlation translation estimation (fysics registration
// primitive). Cross-power spectrum conj(FA)*FB/|.| -> inverse FFT peak = the shift. Phase-
// only -> robust to brightness/contrast. Integer-peak here; sub-voxel (Foroosh) + the full
// affine/Demons stack layer on top. See preprocess/CLAUDE.md, docs/research/research-fysics.md.
#pragma once

#include "core/core.hpp"
#include "preprocess/fft.hpp"

#include <cmath>
#include <vector>

namespace fenix::preprocess {

// Returns the integer translation s (ZYX) such that b(x) ~= a(x - s). Dims must match and
// each be a power of two.
inline Vec3f phase_correlate(VolumeView<const f32> a, VolumeView<const f32> b, f32* confidence = nullptr) {
    const Extent3 d = a.dims();
    const usize n = static_cast<usize>(d.count());
    std::vector<cf32> fa(n), fb(n);
    for (s64 i = 0; i < d.count(); ++i) {
        fa[static_cast<usize>(i)] = cf32(a.flat()[static_cast<usize>(i)], 0.0f);
        fb[static_cast<usize>(i)] = cf32(b.flat()[static_cast<usize>(i)], 0.0f);
    }
    fft3d(fa, d, false);
    fft3d(fb, d, false);
    usize live = 0;  // bins with real energy — the confidence normalizer
    for (usize i = 0; i < n; ++i) {
        const cf32 cross = std::conj(fa[i]) * fb[i];
        const f32 mag = std::abs(cross);
        if (mag > 1e-12f) {
            fa[i] = cross / mag;  // phase only
            ++live;
        } else {
            fa[i] = cf32(0.0f, 0.0f);
        }
    }
    fft3d(fa, d, true);

    // Peak of the correlation surface.
    s64 best = 0;
    f32 bestv = -1.0f;
    for (s64 i = 0; i < d.count(); ++i) {
        const f32 v = fa[static_cast<usize>(i)].real();
        if (v > bestv) {
            bestv = v;
            best = i;
        }
    }
    // peak / (live-bin fraction): a perfect translation-only match scores ~1.0 even on
    // sparse content (dead spectrum bins would otherwise dilute the delta peak)
    if (confidence) *confidence = live ? bestv * static_cast<f32>(n) / static_cast<f32>(live) : 0.0f;
    s64 pz = best / (d.y * d.x), rem = best % (d.y * d.x), py = rem / d.x, px = rem % d.x;
    // Wrap to signed shift.
    auto wrap = [](s64 k, s64 nn) { return k > nn / 2 ? k - nn : k; };
    return {static_cast<f32>(wrap(pz, d.z)), static_cast<f32>(wrap(py, d.y)),
            static_cast<f32>(wrap(px, d.x))};
}

}  // namespace fenix::preprocess
