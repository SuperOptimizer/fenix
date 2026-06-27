// eval/nsd.hpp — Normalized Surface Dice (DeepMind surface_distance, spacing 1). Fraction
// of each surface lying within tolerance tau of the other, symmetric. Built on geom EDT.
// The gold-standard surface metric (plan flagged it as missing in villa). See eval/CLAUDE.md.
#pragma once

#include "core/core.hpp"
#include "geom/edt.hpp"

#include <cmath>

namespace fenix::eval {

// Boundary voxels of a binary mask: a foreground voxel with a background (or out-of-bounds)
// 6-neighbour. Border-touching foreground counts as surface.
inline Volume<u8> surface_voxels(VolumeView<const u8> mask) {
    const Extent3 d = mask.dims();
    Volume<u8> s = Volume<u8>::zeros(d);
    VolumeView<u8> sv = s.view();
    auto bg = [&](s64 z, s64 y, s64 x) {
        return z < 0 || z >= d.z || y < 0 || y >= d.y || x < 0 || x >= d.x || mask(z, y, x) == 0;
    };
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                if (!mask(z, y, x)) continue;
                if (bg(z - 1, y, x) || bg(z + 1, y, x) || bg(z, y - 1, x) || bg(z, y + 1, x) ||
                    bg(z, y, x - 1) || bg(z, y, x + 1))
                    sv(z, y, x) = 1;
            }
    });
    return s;
}

// NSD@tau in [0,1]: 1 == surfaces coincide within tau everywhere. Symmetric.
inline f64 nsd(VolumeView<const u8> pred, VolumeView<const u8> gt, f32 tau) {
    Volume<u8> ps = surface_voxels(pred);
    Volume<u8> gs = surface_voxels(gt);
    Volume<f32> d_to_gt = geom::edt_squared(gs.view());  // squared dist to gt surface
    Volume<f32> d_to_ps = geom::edt_squared(ps.view());  // squared dist to pred surface
    const f32 tau2 = tau * tau;
    const s64 n = pred.size();
    s64 np = 0, ng = 0, near_p = 0, near_g = 0;
    for (s64 i = 0; i < n; ++i) {
        if (ps.flat()[static_cast<usize>(i)]) {
            ++np;
            if (d_to_gt.flat()[static_cast<usize>(i)] <= tau2) ++near_p;
        }
        if (gs.flat()[static_cast<usize>(i)]) {
            ++ng;
            if (d_to_ps.flat()[static_cast<usize>(i)] <= tau2) ++near_g;
        }
    }
    if (np + ng == 0) return 1.0;
    return static_cast<f64>(near_p + near_g) / static_cast<f64>(np + ng);
}

}  // namespace fenix::eval
