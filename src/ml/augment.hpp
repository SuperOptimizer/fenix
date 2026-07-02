// ml/augment.hpp — train-time data augmentation for the surface/ink segmentation nets. Torch-free,
// header-only, deterministic (every transform is a pure function of an explicit seed, so a training run
// is reproducible and the same patch augments identically across restarts). Operates on Volume<f32>
// (post-decode, pre-normalize) plus a paired label volume where a transform is geometric.
//
// The augmentation set is chosen from the TTA ablation (docs/design/ml-accel-and-distillation.md §7),
// which is a DIAGNOSIS of what the original teacher's training under-covered:
//   - the model was strongly orientation-variant (flip-agreement 0.49) -> ORIENTATION augmentation is
//     the highest-value gap: full 3D flips + octahedral rotations + modest in-plane arbitrary rotation.
//   - it was already noise-robust (noise-TTA added nothing) -> intensity augmentation is SUFFICIENT;
//     keep a standard set, don't over-invest.
//   - scroll-specific modes generic nnU-Net augmentation misses: elastic sheet deformation, CT-realistic
//     degradation (rings/gradients — the fysics failure modes), and COMPRESSION-STATE (the student runs
//     on lossy .fxvol, so it must be invariant to DCT-quant artifacts).
// Respect the anisotropy: the z (scroll/beam) axis is physically distinct — arbitrary rotation is about
// z only (in-plane), never a full-SO(3) tumble.
#pragma once

#include "core/core.hpp"
#include "core/hash.hpp"
#include "core/rng.hpp"
#include "core/sampling.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace fenix::ml::aug {

// A geometric transform must move the label field identically to the image; an intensity/degradation
// transform touches only the image. `label` may be an empty Volume (no paired labels).
struct Sample {
    Volume<f32> image;
    Volume<u8> label;  // empty() when no labels are paired
    [[nodiscard]] bool has_label() const { return label.dims().count() > 0; }
};

// Standard-normal draw (Box-Muller) from a Pcg32 stream.
inline f32 randn(Pcg32& rng) {
    const f32 u1 = std::max(rng.next_f32(), 1e-7f), u2 = rng.next_f32();
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
}
inline f32 uniform(Pcg32& rng, f32 lo, f32 hi) { return lo + (hi - lo) * rng.next_f32(); }

// --- geometric: flips + octahedral rotation (exact, no interpolation) -------------------------------
// Apply one of the 48 octahedral symmetries (index 0..47: perm*8 + flipmask) to image (+label). Exact
// voxel remap — the orientation lever the ablation flagged as most valuable, and interpolation-free.
inline void octahedral(Sample& s, int sym) {
    static const int kPerm[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    const int* p = kPerm[(sym / 8) % 6];
    const int fm = sym % 8;
    auto remap = [&](auto src) {
        using T = std::remove_cv_t<std::remove_reference_t<decltype(src(0, 0, 0))>>;
        const Extent3 d = src.dims();
        const std::array<s64, 3> od{d.z, d.y, d.x};
        const Extent3 nd{od[static_cast<usize>(p[0])], od[static_cast<usize>(p[1])], od[static_cast<usize>(p[2])]};
        Volume<T> out(nd);
        auto ov = out.view();
        parallel_for_z(nd, [&](s64 z) {
            for (s64 y = 0; y < nd.y; ++y)
                for (s64 x = 0; x < nd.x; ++x) {
                    std::array<s64, 3> o{z, y, x};  // output coord in permuted axes
                    std::array<s64, 3> i{};         // source coord in original axes
                    i[static_cast<usize>(p[0])] = o[0]; i[static_cast<usize>(p[1])] = o[1]; i[static_cast<usize>(p[2])] = o[2];
                    if (fm & 1) i[0] = d.z - 1 - i[0];
                    if (fm & 2) i[1] = d.y - 1 - i[1];
                    if (fm & 4) i[2] = d.x - 1 - i[2];
                    ov(z, y, x) = src(i[0], i[1], i[2]);
                }
        });
        return out;
    };
    s.image = remap(s.image.view());
    if (s.has_label()) s.label = remap(s.label.view());
}

// --- geometric: arbitrary in-plane (about z) rotation, trilinear ------------------------------------
// Rotate about z by `deg` (image trilinear, label nearest). Off-axis orientations the octahedral group
// can't reach — a TRAIN-time positive (the interpolation blur that made this bad at test time forces
// invariance here). z stays distinguished. Out-of-frame reads edge-clamp.
inline void rotate_z(Sample& s, f32 deg) {
    const f32 r = deg * 3.14159265358979f / 180.0f, c = std::cos(r), sn = std::sin(r);
    const Extent3 d = s.image.dims();
    const f32 cy = (d.y - 1) * 0.5f, cx = (d.x - 1) * 0.5f;
    Volume<f32> oi(d);
    auto oiv = oi.view();
    VolumeView<const f32> iv = s.image.view();
    Volume<u8> ol;
    if (s.has_label()) ol = Volume<u8>(d);
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const f32 dy = static_cast<f32>(y) - cy, dx = static_cast<f32>(x) - cx;
                const f32 sy = cy + c * dy - sn * dx, sx = cx + sn * dy + c * dx;  // inverse map
                oiv(z, y, x) = sample_trilinear(iv, Vec3f{static_cast<f32>(z), sy, sx});
                if (s.has_label()) {
                    const s64 ny = static_cast<s64>(std::lround(sy)), nx = static_cast<s64>(std::lround(sx));
                    ol.view()(z, y, x) = s.label.view().at_clamped(z, ny, nx);
                }
            }
    });
    s.image = std::move(oi);
    if (s.has_label()) s.label = std::move(ol);
}

// --- geometric: elastic deformation (the top scroll-specific augmentation) --------------------------
// Displace each voxel by a smooth random field (coarse grid of Gaussian displacements, trilinearly
// upsampled), magnitude `alpha` voxels, control spacing `sigma` voxels. Sheets bend/delaminate/compress;
// training on locally-warped patches generalizes across that geometry. Image trilinear, label nearest.
inline void elastic(Sample& s, u64 seed, f32 alpha, f32 sigma) {
    const Extent3 d = s.image.dims();
    const s64 gz = std::max<s64>(2, static_cast<s64>(d.z / std::max(1.0f, sigma)) + 2);
    const s64 gy = std::max<s64>(2, static_cast<s64>(d.y / std::max(1.0f, sigma)) + 2);
    const s64 gx = std::max<s64>(2, static_cast<s64>(d.x / std::max(1.0f, sigma)) + 2);
    // Coarse displacement fields (z,y,x components), one Gaussian draw per control point.
    Volume<f32> cz(Extent3{gz, gy, gx}), cy(Extent3{gz, gy, gx}), cx(Extent3{gz, gy, gx});
    Pcg32 rng(seed);
    for (s64 i = 0; i < gz * gy * gx; ++i) {
        cz.flat()[static_cast<usize>(i)] = alpha * randn(rng);
        cy.flat()[static_cast<usize>(i)] = alpha * randn(rng);
        cx.flat()[static_cast<usize>(i)] = alpha * randn(rng);
    }
    auto czv = cz.view(), cyv = cy.view(), cxv = cx.view();
    VolumeView<const f32> iv = s.image.view();
    Volume<f32> oi(d);
    auto oiv = oi.view();
    Volume<u8> ol;
    if (s.has_label()) ol = Volume<u8>(d);
    // Map an output voxel to the coarse grid's continuous coords (control point k sits at k*spacing).
    const f32 fz = static_cast<f32>(gz - 1) / std::max<f32>(1, d.z - 1);
    const f32 fy = static_cast<f32>(gy - 1) / std::max<f32>(1, d.y - 1);
    const f32 fx = static_cast<f32>(gx - 1) / std::max<f32>(1, d.x - 1);
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const Vec3f g{static_cast<f32>(z) * fz, static_cast<f32>(y) * fy, static_cast<f32>(x) * fx};
                const f32 ddz = sample_trilinear(czv, g), ddy = sample_trilinear(cyv, g), ddx = sample_trilinear(cxv, g);
                const Vec3f sp{static_cast<f32>(z) + ddz, static_cast<f32>(y) + ddy, static_cast<f32>(x) + ddx};
                oiv(z, y, x) = sample_trilinear(iv, sp);
                if (s.has_label()) {
                    const s64 nz = static_cast<s64>(std::lround(sp.z)), ny = static_cast<s64>(std::lround(sp.y)), nx = static_cast<s64>(std::lround(sp.x));
                    ol.view()(z, y, x) = s.label.view().at_clamped(nz, ny, nx);
                }
            }
    });
    s.image = std::move(oi);
    if (s.has_label()) s.label = std::move(ol);
}

// --- intensity (image only; keep standard, per the ablation) ----------------------------------------
// Additive Gaussian noise (sigma in raw intensity units) + brightness (add) + contrast (mul about mean)
// + gamma (on a [0,1]-normalized copy). Small ranges — the model is already noise-robust.
struct IntensityParams {
    f32 noise_sigma = 3.0f;   // ~0..255 data
    f32 brightness = 6.0f;    // ± additive
    f32 contrast = 0.15f;     // ± fractional about the patch mean
    f32 gamma = 0.2f;         // ± exponent about 1
};
inline void intensity(Sample& s, u64 seed, const IntensityParams& p = {}) {
    const Extent3 d = s.image.dims();
    const s64 n = d.count();
    // Patch mean for the contrast pivot.
    f64 sum = 0;
    for (s64 i = 0; i < n; ++i) sum += s.image.flat()[static_cast<usize>(i)];
    const f32 mean = static_cast<f32>(sum / std::max<s64>(1, n));
    Pcg32 rng(seed);
    const f32 br = uniform(rng, -p.brightness, p.brightness);
    const f32 ct = 1.0f + uniform(rng, -p.contrast, p.contrast);
    const f32 gm = std::exp(uniform(rng, -p.gamma, p.gamma));  // log-uniform about 1
    f32 mn = s.image.flat()[0], mx = mn;
    for (s64 i = 0; i < n; ++i) { const f32 v = s.image.flat()[static_cast<usize>(i)]; mn = std::min(mn, v); mx = std::max(mx, v); }
    const f32 rng_inv = 1.0f / std::max(1e-6f, mx - mn);
    parallel_for(0, n, [&](s64 i) {
        Pcg32 lr(seed ^ hash_value(static_cast<u64>(i)));
        f32 v = s.image.flat()[static_cast<usize>(i)];
        v = (v - mean) * ct + mean + br;               // contrast + brightness
        f32 t = std::clamp((v - mn) * rng_inv, 0.0f, 1.0f);
        t = std::pow(t, gm);                            // gamma on normalized
        v = mn + t / rng_inv;
        v += p.noise_sigma * randn(lr);                 // additive noise
        s.image.flat()[static_cast<usize>(i)] = v;
    });
}

// --- CT-realistic degradation (the fenix-specific advantage) ----------------------------------------
// Ring artifact (concentric intensity ripple about the slice center — the classic tomography failure)
// + a smooth low-frequency intensity gradient (beam-hardening cupping). These are the real scanner
// modes the fysics dering/deconv stage removes; augmenting with them teaches robustness. Image only.
struct CtDegradeParams {
    f32 ring_amp = 4.0f;       // ± ring ripple amplitude (intensity units)
    f32 ring_period = 18.0f;   // radial period (voxels)
    f32 gradient_amp = 8.0f;   // ± cupping amplitude across the patch
};
inline void ct_degrade(Sample& s, u64 seed, const CtDegradeParams& p = {}) {
    const Extent3 d = s.image.dims();
    Pcg32 rng(seed);
    const f32 ra = uniform(rng, 0.0f, p.ring_amp), phase = uniform(rng, 0.0f, 6.2832f);
    const f32 period = std::max(2.0f, p.ring_period * uniform(rng, 0.7f, 1.3f));
    const f32 ga = uniform(rng, -p.gradient_amp, p.gradient_amp);
    // Random gradient direction in-plane.
    const f32 gdir = uniform(rng, 0.0f, 6.2832f), gcy = std::cos(gdir), gcx = std::sin(gdir);
    const f32 cy = (d.y - 1) * 0.5f, cx = (d.x - 1) * 0.5f;
    const f32 half = 0.5f * std::sqrt(static_cast<f32>(d.y) * d.y + static_cast<f32>(d.x) * d.x);
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const f32 dy = static_cast<f32>(y) - cy, dx = static_cast<f32>(x) - cx;
                const f32 rr = std::sqrt(dy * dy + dx * dx);
                const f32 ring = ra * std::sin(6.2832f * rr / period + phase);
                const f32 grad = ga * ((gcy * dy + gcx * dx) / std::max(1.0f, half));  // ±1 across the patch
                s.image.view()(z, y, x) += ring + grad;
            }
    });
}

// --- compression-state augmentation (student runs on lossy .fxvol) ----------------------------------
// Emulate DCT-quantization by block-mean quantization on 8^3 blocks blended with the original: dead-zone
// rounding of each block's DC toward a quantization grid + light per-block high-freq attenuation. A cheap
// stand-in for a full codec round-trip that captures the artifact class (blockiness + DC banding + HF
// loss) the student must be invariant to. Image only. strength in [0,1].
inline void compression(Sample& s, u64 seed, f32 strength) {
    if (strength <= 0.0f) return;
    const Extent3 d = s.image.dims();
    const s64 B = 8;
    const f32 q = 4.0f + 28.0f * strength;   // coarser DC grid as strength rises
    const f32 hf = 1.0f - 0.6f * strength;   // high-freq retention
    Pcg32 rng(seed);
    const f32 blend = uniform(rng, 0.5f, 1.0f) * strength;
    parallel_for(0, (d.z + B - 1) / B, [&](s64 bz) {
        for (s64 by = 0; by * B < d.y; ++by)
            for (s64 bx = 0; bx * B < d.x; ++bx) {
                const s64 z0 = bz * B, y0 = by * B, x0 = bx * B;
                const s64 z1 = std::min(z0 + B, d.z), y1 = std::min(y0 + B, d.y), x1 = std::min(x0 + B, d.x);
                f64 sum = 0; s64 cnt = 0;
                for (s64 z = z0; z < z1; ++z) for (s64 y = y0; y < y1; ++y) for (s64 x = x0; x < x1; ++x) { sum += s.image.view()(z, y, x); ++cnt; }
                const f32 dc = static_cast<f32>(sum / std::max<s64>(1, cnt));
                const f32 dcq = std::round(dc / q) * q;  // dead-zone-quantized block DC
                for (s64 z = z0; z < z1; ++z) for (s64 y = y0; y < y1; ++y) for (s64 x = x0; x < x1; ++x) {
                    const f32 v = s.image.view()(z, y, x);
                    const f32 comp = dcq + hf * (v - dc);  // attenuate HF, snap DC
                    s.image.view()(z, y, x) = v + blend * (comp - v);
                }
            }
    });
}

// --- policy: sample a full augmentation chain for one training patch --------------------------------
// Probabilities/ranges tuned to the ablation: orientation ALWAYS (the big gap), elastic often, CT/
// compression moderately, intensity light. Each stage draws from a distinct sub-seed so toggling one
// doesn't reshuffle the others. Geometric first (on image+label), then image-only corruptions.
struct Policy {
    f32 p_rotate = 0.5f;   f32 rot_max_deg = 20.0f;
    f32 p_elastic = 0.5f;  f32 elastic_alpha = 3.0f;  f32 elastic_sigma = 24.0f;
    f32 p_intensity = 0.8f;
    f32 p_ct = 0.4f;
    f32 p_compress = 0.5f; f32 compress_max = 0.8f;
};
inline void augment(Sample& s, u64 seed, const Policy& pol = {}) {
    auto sub = [&](u32 k) { return seed ^ hash_value(static_cast<u64>(0xA5A5'0000u + k)); };
    auto chance = [&](u32 k, f32 p) { Pcg32 r(sub(k)); return r.next_f32() < p; };
    // Orientation: octahedral always (exact), + optional small in-plane rotation.
    { Pcg32 r(sub(1)); octahedral(s, static_cast<int>(r.bounded(48))); }
    if (chance(2, pol.p_rotate)) { Pcg32 r(sub(3)); rotate_z(s, uniform(r, -pol.rot_max_deg, pol.rot_max_deg)); }
    if (chance(4, pol.p_elastic)) elastic(s, sub(5), pol.elastic_alpha, pol.elastic_sigma);
    if (chance(6, pol.p_ct)) ct_degrade(s, sub(7));
    if (chance(8, pol.p_compress)) { Pcg32 r(sub(9)); compression(s, sub(10), uniform(r, 0.2f, pol.compress_max)); }
    if (chance(11, pol.p_intensity)) intensity(s, sub(12));
}

}  // namespace fenix::ml::aug
