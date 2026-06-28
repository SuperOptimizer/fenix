// segment/structure_tensor.hpp — structure-tensor sheet detection (the classical front
// end). Produces per-voxel sheetness (scalar) + normal (across-sheet direction) for the
// unrolling fit's data terms. See segment/CLAUDE.md, docs/research/research-core.md.
#pragma once

#include "core/core.hpp"
#include "segment/sheet_field.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace fenix::segment {

struct StParams {
    f32 sigma_grad = 1.0f;    // pre-smoothing before the gradient (config; no baked default policy)
    f32 sigma_tensor = 2.0f;  // integration scale of the tensor
};

// Sheetness only (the (l0-l1)/l0 planarity scalar), computed TILED with a halo so peak RAM is
// O(tile^3), not O(volume). The whole-volume `structure_tensor` below holds 6 component volumes +
// a per-voxel normal field at once (~10x the volume) -> OOMs at 1024^3. This one is the data-term
// path: it never materializes the normal field and never holds the full tensor. Interior results
// match the whole-volume version up to the gaussian's 3-sigma tail truncation (within tolerance).
//
// Templated on input `T` (u8/u16/f32 — converted to f32 per tile) and output `Out`: with Out=f32 the
// raw 0..1 sheetness; with an integer Out the sheetness is scaled to the type's full range (e.g. u8 ->
// 0..255). Keeping resident volumes u8 cuts RAM 4x vs f32. The Paganin/unsharp deconv is FOLDED in
// per-tile (`unsharp_sigma`/`unsharp_amount`) so no full-volume blur copy is ever allocated.
template <class T = f32, class Out = f32>
inline Volume<Out> structure_tensor_sheetness(VolumeView<const T> ct, StParams p, s64 tile = 256,
                                              f32 unsharp_sigma = 0.0f, f32 unsharp_amount = 0.0f) {
    const Extent3 d = ct.dims();
    Volume<Out> out = Volume<Out>::zeros(d);
    VolumeView<Out> ov = out.view();
    constexpr f32 kOutScale = std::is_same_v<Out, f32> ? 1.0f : static_cast<f32>(std::numeric_limits<Out>::max());
    // Halo must cover the (optional) unsharp blur, the gradient pre-smooth and the tensor integration.
    const f32 sig_sum = p.sigma_grad + p.sigma_tensor + std::max(0.0f, unsharp_sigma);
    const s64 halo = std::max<s64>(4, static_cast<s64>(std::ceil(3.0f * sig_sum)) + 1);

    // SERIAL over tiles, but every step INSIDE a tile is parallel (blur + gradient + eig all use the
    // cores). One tile resident at a time => peak RAM ~ 8 * pad-tile^3 * 4B (bounded by `tile`, NOT the
    // volume). Crucially this avoids nesting a parallel region inside a parallel one: parallelizing the
    // tile loop instead would nest gaussian_blur's parallel_for and oversubscribe the cores (measured
    // ~10x slower). Tiles touch disjoint interior regions of `out`.
    for (s64 tz = 0; tz < d.z; tz += tile)
        for (s64 ty = 0; ty < d.y; ty += tile)
            for (s64 tx = 0; tx < d.x; tx += tile) {
                // Interior block [t*, t*1) and its padded extent [p*0, p*1) clamped to the volume.
                const s64 iz1 = std::min(tz + tile, d.z), iy1 = std::min(ty + tile, d.y), ix1 = std::min(tx + tile, d.x);
                const s64 pz0 = std::max<s64>(0, tz - halo), py0 = std::max<s64>(0, ty - halo), px0 = std::max<s64>(0, tx - halo);
                const s64 pz1 = std::min(d.z, iz1 + halo), py1 = std::min(d.y, iy1 + halo), px1 = std::min(d.x, ix1 + halo);
                const Extent3 pd{pz1 - pz0, py1 - py0, px1 - px0};

                Volume<f32> work(pd);
                VolumeView<f32> wv = work.view();
                parallel_for_z(pd, [&](s64 z) {
                    for (s64 y = 0; y < pd.y; ++y)
                        for (s64 x = 0; x < pd.x; ++x) wv(z, y, x) = static_cast<f32>(ct(pz0 + z, py0 + y, px0 + x));
                });
                // Folded deconv: unsharp = work + amount*(work - blur(work)), on this tile's f32 copy.
                if (unsharp_amount > 0.0f && unsharp_sigma > 0.0f) {
                    Volume<f32> bl(pd);
                    parallel_for(0, pd.count(), [&](s64 i) { bl.flat()[static_cast<usize>(i)] = wv.flat()[static_cast<usize>(i)]; });
                    gaussian_blur(bl.view(), unsharp_sigma);
                    auto bv = bl.view();
                    parallel_for(0, pd.count(), [&](s64 i) {
                        wv.flat()[static_cast<usize>(i)] += unsharp_amount * (wv.flat()[static_cast<usize>(i)] - bv.flat()[static_cast<usize>(i)]);
                    });
                }
                gaussian_blur(wv, p.sigma_grad);

                Volume<f32> jzz(pd), jyy(pd), jxx(pd), jzy(pd), jzx(pd), jyx(pd);
                VolumeView<const f32> wc = work.view();
                parallel_for_z(pd, [&](s64 z) {
                    for (s64 y = 0; y < pd.y; ++y)
                        for (s64 x = 0; x < pd.x; ++x) {
                            Vec3f g = gradient_at(wc, z, y, x);
                            jzz(z, y, x) = g.z * g.z; jyy(z, y, x) = g.y * g.y; jxx(z, y, x) = g.x * g.x;
                            jzy(z, y, x) = g.z * g.y; jzx(z, y, x) = g.z * g.x; jyx(z, y, x) = g.y * g.x;
                        }
                });
                for (Volume<f32>* c : {&jzz, &jyy, &jxx, &jzy, &jzx, &jyx}) gaussian_blur(c->view(), p.sigma_tensor);

                auto svz = jzz.view(); auto svy = jyy.view(); auto svx = jxx.view();
                auto vzy = jzy.view(); auto vzx = jzx.view(); auto vyx = jyx.view();
                // Write only the interior (the halo ring is discarded — it exists to make the smoothing
                // correct up to the tile boundary).
                const s64 oz0 = tz - pz0, oy0 = ty - py0, ox0 = tx - px0;
                const s64 ny = iy1 - ty, nx = ix1 - tx;
                parallel_for_z(Extent3{iz1 - tz, ny, nx}, [&](s64 lz) {
                    const s64 z = oz0 + lz;
                    for (s64 ly = 0; ly < ny; ++ly) {
                        const s64 y = oy0 + ly;
                        for (s64 lx = 0; lx < nx; ++lx) {
                            const s64 x = ox0 + lx;
                            auto e = sym_eig3<f32>(svz(z, y, x), svy(z, y, x), svx(z, y, x), vzy(z, y, x), vzx(z, y, x), vyx(z, y, x));
                            const f32 s = (e.values[0] - e.values[1]) / (e.values[0] + tol::eps);
                            if constexpr (std::is_same_v<Out, f32>)
                                ov(tz + lz, ty + ly, tx + lx) = s;
                            else
                                ov(tz + lz, ty + ly, tx + lx) = static_cast<Out>(std::clamp(s, 0.0f, 1.0f) * kOutScale + 0.5f);
                        }
                    }
                });
            }
    return out;
}

// Build the resident CT sheetness data term (full-res u8) the fast way: mean-downsample the CT by
// `ds`, compute the (tiled) sheetness on that ~ds^3-smaller volume with the Paganin deconv folded in,
// then upsample back to full res and gate by the air-cut threshold. The CT term is a COARSE fallback
// (it supplements the prediction in cracks, snapped along the normal) so full-res sheetness is
// unnecessary — ds=2 cuts the structure-tensor cost ~8x. ds=1 == full-res sheetness.
template <class T>
inline Volume<u8> ct_sheetness_term(VolumeView<const T> ct, f32 air_cut_thresh, f32 unsharp_sigma, int ds = 2) {
    const Extent3 d = ct.dims();
    if (ds < 1) ds = 1;
    const Extent3 dd{std::max<s64>(1, d.z / ds), std::max<s64>(1, d.y / ds), std::max<s64>(1, d.x / ds)};

    // Mean-downsample to f32.
    Volume<f32> small(dd);
    VolumeView<f32> sv = small.view();
    const f32 invc = 1.0f / static_cast<f32>(ds * ds * ds);
    parallel_for_z(dd, [&](s64 z) {
        for (s64 y = 0; y < dd.y; ++y)
            for (s64 x = 0; x < dd.x; ++x) {
                f32 acc = 0;
                for (int dz = 0; dz < ds; ++dz)
                    for (int dy = 0; dy < ds; ++dy)
                        for (int dx = 0; dx < ds; ++dx)
                            acc += static_cast<f32>(ct(z * ds + dz, y * ds + dy, x * ds + dx));
                sv(z, y, x) = acc * invc;
            }
    });

    // Sheetness on the small volume (deconv sigma is in downsampled voxels -> divide by ds).
    const f32 usig = unsharp_sigma > 0.0f ? unsharp_sigma / static_cast<f32>(ds) : 0.0f;
    Volume<f32> sh = structure_tensor_sheetness<f32, f32>(small.view(), StParams{1.0f, 2.0f}, 256, usig,
                                                          usig > 0.0f ? 1.0f : 0.0f);
    VolumeView<const f32> shv = sh.view();

    // Upsample (trilinear) to full-res u8, gated by the air-cut so only papyrus density contributes.
    Volume<u8> out(d);
    VolumeView<u8> ov = out.view();
    const f32 invds = 1.0f / static_cast<f32>(ds);
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                if (static_cast<f32>(ct(z, y, x)) < air_cut_thresh) { ov(z, y, x) = 0; continue; }
                const Vec3f p{(static_cast<f32>(z) + 0.5f) * invds - 0.5f, (static_cast<f32>(y) + 0.5f) * invds - 0.5f,
                              (static_cast<f32>(x) + 0.5f) * invds - 0.5f};
                const f32 s = std::clamp(sample_trilinear(shv, p), 0.0f, 1.0f);
                ov(z, y, x) = static_cast<u8>(s * 255.0f + 0.5f);
            }
    });
    return out;
}

// Compute the structure tensor of `ct` and extract sheetness + normal.
inline SheetField structure_tensor(VolumeView<const f32> ct, StParams p) {
    const Extent3 d = ct.dims();
    const usize n = static_cast<usize>(d.count());

    // Pre-smooth a working copy.
    Volume<f32> work(d);
    for (usize i = 0; i < n; ++i) work.flat()[i] = ct.flat()[i];
    gaussian_blur(work.view(), p.sigma_grad);

    // Tensor components Jzz,Jyy,Jxx,Jzy,Jzx,Jyx from the outer product of the gradient.
    Volume<f32> jzz(d), jyy(d), jxx(d), jzy(d), jzx(d), jyx(d);
    VolumeView<const f32> wv = work.view();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                Vec3f g = gradient_at(wv, z, y, x);
                jzz(z, y, x) = g.z * g.z;
                jyy(z, y, x) = g.y * g.y;
                jxx(z, y, x) = g.x * g.x;
                jzy(z, y, x) = g.z * g.y;
                jzx(z, y, x) = g.z * g.x;
                jyx(z, y, x) = g.y * g.x;
            }
    });

    // Integrate (smooth) each component at the tensor scale.
    for (Volume<f32>* c : {&jzz, &jyy, &jxx, &jzy, &jzx, &jyx}) gaussian_blur(c->view(), p.sigma_tensor);

    SheetField out{Volume<f32>::zeros(d), std::vector<Vec3f>(n)};
    auto svz = jzz.view();
    auto svy = jyy.view();
    auto svx = jxx.view();
    auto vzy = jzy.view();
    auto vzx = jzx.view();
    auto vyx = jyx.view();
    VolumeView<f32> sheet = out.sheetness.view();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                auto e = sym_eig3<f32>(svz(z, y, x), svy(z, y, x), svx(z, y, x), vzy(z, y, x),
                                       vzx(z, y, x), vyx(z, y, x));
                f32 l0 = e.values[0], l1 = e.values[1];
                sheet(z, y, x) = (l0 - l1) / (l0 + tol::eps);
                out.normal[static_cast<usize>((z * d.y + y) * d.x + x)] = e.vectors[0];
            }
    });
    return out;
}

}  // namespace fenix::segment
