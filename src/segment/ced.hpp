// segment/ced.hpp — coherence-enhancing (sheet) diffusion (taberna `ced.c`). Diffuse ALONG confident
// sheets — closing porosity / the soft-prediction "fog" — while diffusing weakly ACROSS them so
// adjacent wraps stay separate. Anisotropic diffusion driven by the structure-tensor sheetness+normal
// (computed ONCE from the initial field):
//     u += dt·div(D∇u),  D = c_perp·I − (c_perp−c_norm)·n nᵀ,  c_perp = c_norm + (1−c_norm)·sheetness
// so the diffusivity is c_norm across the sheet-normal n and c_perp (→1 on confident sheet) in-plane.
// dt ≤ 1/6 for 3D explicit stability. Applies to a noisy CT *or* a foggy ML prediction. See CLAUDE.md.
//
// MEMORY: uses the whole-volume `structure_tensor` (materializes 6 tensor + a normal field ⇒ ~10× the
// input) → CROP-SCALE (≤ ~256³ comfortably). Tile with a halo for larger volumes (TODO, mirrors
// structure_tensor_sheetness). Peak during iteration is ~8× (u + sheetness + 3×normal + 3×flux).
#pragma once

#include "core/core.hpp"
#include "segment/structure_tensor.hpp"

#include <algorithm>
#include <vector>

namespace fenix::segment {

struct CedParams {
    f32 sigma_grad = 0.5f;    // pre-smooth before the tensor gradient
    f32 sigma_tensor = 1.0f;  // tensor integration scale (keep < inter-wrap gap)
    f32 c_norm = 0.05f;       // across-sheet diffusivity floor
    f32 dt = 0.12f;           // time step (≤ 1/6 for 3D explicit stability)
    int iters = 12;
};

inline void ced_inplace(VolumeView<f32> vol, CedParams p = {}) {
    const Extent3 d = vol.dims();
    if (p.dt > 1.0f / 6.0f) p.dt = 1.0f / 6.0f;  // clamp to the stability bound

    // Sheetness + normal from the initial field (fixed across the diffusion).
    SheetField sf = structure_tensor(vol, StParams{p.sigma_grad, p.sigma_tensor});
    VolumeView<const f32> sh = sf.sheetness.view();
    const std::vector<Vec3f>& nrm = sf.normal;

    Volume<f32> jz(d), jy(d), jx(d);
    VolumeView<f32> jzv = jz.view(), jyv = jy.view(), jxv = jx.view();
    VolumeView<const f32> vc = vol;  // const alias for reads (writes happen only in the divergence pass)
    auto cl = [](s64 i, s64 n) -> s64 { return i < 0 ? 0 : (i >= n ? n - 1 : i); };

    for (int it = 0; it < p.iters; ++it) {
        // Pass 1: flux j = D·∇u  (reads u, writes j).
        parallel_for_z(d, [&](s64 z) {
            for (s64 y = 0; y < d.y; ++y)
                for (s64 x = 0; x < d.x; ++x) {
                    const Vec3f g = gradient_at(vc, z, y, x);
                    const Vec3f nn = nrm[static_cast<usize>((z * d.y + y) * d.x + x)];
                    const f32 s = std::clamp(sh(z, y, x), 0.0f, 1.0f);
                    const f32 c_perp = p.c_norm + (1.0f - p.c_norm) * s;
                    const f32 coef = c_perp - p.c_norm;
                    const f32 ndotg = nn.z * g.z + nn.y * g.y + nn.x * g.x;
                    jzv(z, y, x) = c_perp * g.z - coef * ndotg * nn.z;
                    jyv(z, y, x) = c_perp * g.y - coef * ndotg * nn.y;
                    jxv(z, y, x) = c_perp * g.x - coef * ndotg * nn.x;
                }
        });
        // Pass 2: u += dt·div(j)  (reads j, writes u).
        parallel_for_z(d, [&](s64 z) {
            for (s64 y = 0; y < d.y; ++y)
                for (s64 x = 0; x < d.x; ++x) {
                    const f32 dvz = 0.5f * (jzv(cl(z + 1, d.z), y, x) - jzv(cl(z - 1, d.z), y, x));
                    const f32 dvy = 0.5f * (jyv(z, cl(y + 1, d.y), x) - jyv(z, cl(y - 1, d.y), x));
                    const f32 dvx = 0.5f * (jxv(z, y, cl(x + 1, d.x)) - jxv(z, y, cl(x - 1, d.x)));
                    vol(z, y, x) += p.dt * (dvz + dvy + dvx);
                }
        });
    }
}

}  // namespace fenix::segment
