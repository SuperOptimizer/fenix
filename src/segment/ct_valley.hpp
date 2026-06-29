// segment/ct_valley.hpp — the CT inter-wrap separation oracle. The ML surface PREDICTION fuses
// adjacent papyrus wraps wherever they touch (very common), which collapses the geometric inter-wrap
// gap the winding step relies on. But the RAW CT keeps the two wraps as distinct density PEAKS even
// where they touch — only the saddle between them gets shallow (measured on paris4: the dip between
// consecutive wraps ranges from full air, min/H≈0, down to a 4% dip, min/H≈0.96). So we count physical
// wrap boundaries as the number of PROMINENT SADDLES along the short segment between two on-papyrus
// points = the number of wraps crossed = |Δwrap|. Prominence (a drop then a rise, each ≥ a fraction of
// the local papyrus height) — NOT absolute depth — is what survives touching: it keys on the two peaks
// being distinct, not on the gap being deep. Touch-proof AND spacing-free (no global wrap-spacing).
// Shared by the winding step (patch_graph.hpp) and the tracer growth barrier (grow.hpp). See CLAUDE.md.
#pragma once

#include "core/sampling.hpp"
#include "core/vec.hpp"

#include <algorithm>
#include <cmath>

namespace fenix::segment {

// Number of inter-wrap saddles (=|Δwrap|) strictly between two ON-PAPYRUS points pa,pb (ZYX voxel
// coords; both sit on traced surfaces => the profile starts and ends near a papyrus peak).
//   - sample the CT profile along pa->pb (trilinear, sub-voxel `sample_step` to catch ~1-voxel slivers),
//     lightly [1 2 1]/4 smoothed to kill single-sample noise.
//   - prominence saddle-count: in RISE, track the running peak; when the profile has dropped >= `prom_abs`
//     below that peak, switch to FALL and track the running valley; when it has risen >= `prom_abs` above
//     that valley, count ONE saddle and switch back to RISE. So a saddle counts only if BOTH the descent
//     into it and the climb out of it clear the prominence — rejecting noise wiggles and shallow texture.
//   - `prom_abs` is in CT units; callers pass `prom_frac * H` where H = local papyrus height (the
//     endpoint mean), so the test scales with local density and a shallow-but-real touch-gap still counts.
// Same wrap (no saddle) => 0; adjacent wraps (one saddle, even a shallow touch) => 1; k apart => k.
template <class T>
inline int count_air_valleys(VolumeView<const T> ct, Vec3f pa, Vec3f pb, f32 prom_frac, f32 sample_step = 0.5f) {
    const Vec3f d = pb - pa;
    const f32 L = norm(d);
    if (L < 1e-3f) return 0;
    const int N = std::max(3, static_cast<int>(std::ceil(L / std::max(0.1f, sample_step))) + 1);
    const Vec3f u = d / static_cast<f32>(N - 1);
    const f32 e0 = sample_trilinear(ct, pa), e1 = sample_trilinear(ct, pb);
    const f32 H = 0.5f * (e0 + e1);
    if (H < 1e-3f) return 0;  // endpoints not on papyrus => the profile is meaningless
    const f32 prom = std::max(1.0f, prom_frac * H);  // >=1 CT unit so quantization noise can't trip it
    int count = 0;
    bool falling = false;
    f32 peak = e0, valley = e0;
    f32 r0 = e0, r1 = sample_trilinear(ct, pa + u);  // raw[i-1], raw[i] for centred smoothing
    for (int i = 1; i < N - 1; ++i) {
        const f32 r2 = sample_trilinear(ct, pa + u * static_cast<f32>(i + 1));
        const f32 sm = 0.25f * (r0 + 2.0f * r1 + r2);
        if (!falling) {
            if (sm > peak) peak = sm;
            else if (peak - sm >= prom) { falling = true; valley = sm; }  // descended into a gap
        } else {
            if (sm < valley) valley = sm;
            else if (sm - valley >= prom) { ++count; falling = false; peak = sm; }  // climbed back out
        }
        r0 = r1;
        r1 = r2;
    }
    return count;
}

// Boolean early-out form for the tracer hot loop: true once the profile from pa has descended by
// >= prom_frac*H below its running peak — i.e. growth is about to cross into the next wrap's gap.
template <class T>
inline bool crosses_valley(VolumeView<const T> ct, Vec3f pa, Vec3f pb, f32 prom_frac, f32 sample_step = 0.5f) {
    const Vec3f d = pb - pa;
    const f32 L = norm(d);
    if (L < 1e-3f) return false;
    const int N = std::max(3, static_cast<int>(std::ceil(L / std::max(0.1f, sample_step))) + 1);
    const Vec3f u = d / static_cast<f32>(N - 1);
    const f32 e0 = sample_trilinear(ct, pa), e1 = sample_trilinear(ct, pb);
    const f32 H = 0.5f * (e0 + e1);
    if (H < 1e-3f) return false;
    const f32 prom = std::max(1.0f, prom_frac * H);
    f32 peak = e0, r0 = e0, r1 = sample_trilinear(ct, pa + u);
    for (int i = 1; i < N - 1; ++i) {
        const f32 r2 = sample_trilinear(ct, pa + u * static_cast<f32>(i + 1));
        const f32 sm = 0.25f * (r0 + 2.0f * r1 + r2);
        if (sm > peak) peak = sm;
        else if (peak - sm >= prom) return true;
        r0 = r1;
        r1 = r2;
    }
    return false;
}

}  // namespace fenix::segment
