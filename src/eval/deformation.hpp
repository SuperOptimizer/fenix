// eval/deformation.hpp — invertibility metrics for a deformation field. The Jacobian fold
// fraction (det(I + grad disp) <= floor) is the strongest GT-free guard against wrap merges
// / fold-over in the unrolling (taberna jacobian_fold_fraction, spiral-v2 no-fold). The
// diffeomorphic fit guarantees this structurally; we still measure it. See eval/CLAUDE.md.
#pragma once

#include "core/core.hpp"

namespace fenix::eval {

// Displacement field as three ZYX-aligned component volumes (dz, dy, dx per voxel).
// Returns the fraction of interior voxels whose deformation Jacobian determinant is
// <= `det_floor` (folded / non-invertible). 0 == everywhere orientation-preserving.
inline f64 jacobian_fold_fraction(VolumeView<const f32> dz, VolumeView<const f32> dy,
                                  VolumeView<const f32> dx, f32 det_floor = tol::fold_det_floor) {
    const Extent3 d = dz.dims();
    if (d.z < 3 || d.y < 3 || d.x < 3) return 0.0;
    s64 total = 0, folded = 0;
    // Central differences of each displacement component along each axis -> J = I + grad d.
    for (s64 z = 1; z < d.z - 1; ++z)
        for (s64 y = 1; y < d.y - 1; ++y)
            for (s64 x = 1; x < d.x - 1; ++x) {
                const f32 dzz = 0.5f * (dz(z + 1, y, x) - dz(z - 1, y, x));
                const f32 dzy = 0.5f * (dz(z, y + 1, x) - dz(z, y - 1, x));
                const f32 dzx = 0.5f * (dz(z, y, x + 1) - dz(z, y, x - 1));
                const f32 dyz = 0.5f * (dy(z + 1, y, x) - dy(z - 1, y, x));
                const f32 dyy = 0.5f * (dy(z, y + 1, x) - dy(z, y - 1, x));
                const f32 dyx = 0.5f * (dy(z, y, x + 1) - dy(z, y, x - 1));
                const f32 dxz = 0.5f * (dx(z + 1, y, x) - dx(z - 1, y, x));
                const f32 dxy = 0.5f * (dx(z, y + 1, x) - dx(z, y - 1, x));
                const f32 dxx = 0.5f * (dx(z, y, x + 1) - dx(z, y, x - 1));
                // J = I + grad(disp), rows/cols in (z,y,x).
                const f32 a = 1.0f + dzz, b = dzy, c = dzx;
                const f32 e = dyz, f = 1.0f + dyy, g = dyx;
                const f32 h = dxz, i = dxy, j = 1.0f + dxx;
                const f32 det = a * (f * j - g * i) - b * (e * j - g * h) + c * (e * i - f * h);
                ++total;
                if (det <= det_floor) ++folded;
            }
    return total ? static_cast<f64>(folded) / static_cast<f64>(total) : 0.0;
}

}  // namespace fenix::eval
