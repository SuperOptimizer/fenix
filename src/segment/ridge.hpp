// segment/ridge.hpp — non-maximum-suppression centerline extraction (taberna `ridge.c`). Collapse a
// thick sheetness band to a ~1-voxel ridge: keep a voxel iff its sheetness ≥ s_min, its field value
// ≥ i_min, and the value is a local MAX along ±step·n (the sheet normal) — i.e. the bright crest of
// the CT (or the prediction) across the sheet. Matches the thin-centerline GT-label convention.
// Sub-voxel parabola check rejects voxels whose true peak sits in a neighbour. See segment/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <cmath>
#include <vector>

namespace fenix::segment {

// `val` = the field to ridge (raw CT, or the (CED'd) prediction); `sheet`+`normal` from the structure
// tensor of that field. Returns a u8 binary ridge mask.
inline Volume<u8> ridge_nms(VolumeView<const f32> val, VolumeView<const f32> sheet,
                            const std::vector<Vec3f>& normal, f32 s_min, f32 i_min, f32 step = 1.0f) {
    const Extent3 d = val.dims();
    Volume<u8> out = Volume<u8>::zeros(d);
    VolumeView<u8> ov = out.view();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                if (sheet(z, y, x) < s_min) continue;
                const f32 c = val(z, y, x);
                if (c < i_min) continue;
                const Vec3f n = normal[static_cast<usize>((z * d.y + y) * d.x + x)];
                const Vec3f pf{static_cast<f32>(z) + step * n.z, static_cast<f32>(y) + step * n.y, static_cast<f32>(x) + step * n.x};
                const Vec3f pb{static_cast<f32>(z) - step * n.z, static_cast<f32>(y) - step * n.y, static_cast<f32>(x) - step * n.x};
                const f32 fwd = sample_trilinear(val, pf);
                const f32 bwd = sample_trilinear(val, pb);
                if (c < fwd || c < bwd) continue;  // not a local max across the normal
                const f32 denom = bwd - 2.0f * c + fwd;
                if (denom < -1e-3f) {  // strictly concave: reject if the sub-voxel peak is in a neighbour
                    const f32 tstar = 0.5f * (bwd - fwd) / denom;
                    if (std::fabs(tstar) > 0.5f) continue;
                }
                ov(z, y, x) = 1;  // (near-flat plateau kept)
            }
    });
    return out;
}

}  // namespace fenix::segment
