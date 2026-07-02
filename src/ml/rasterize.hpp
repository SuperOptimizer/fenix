// ml/rasterize.hpp — train-time GT label rasterization (torch-free, always built): turn the
// part of a core::Surface intersecting a patch into a sheet-band mask. Per the training
// pipeline design (docs/design/training-pipeline.md): labels are rasterized PER PATCH at
// train time (thickness is a hyperparameter; geometric augments transform coords first, so
// labels are never interpolated through a warp). Method: every valid quad cell overlapping
// the (band-expanded) patch bbox is sampled as a bilinear surface patch at ≤`step`-voxel
// spacing, and each sample stamps a Euclidean sphere of radius thickness/2 — cheap, exact
// to ~step/2, no full-patch EDT needed.
#pragma once

#include "core/core.hpp"
#include "core/surface.hpp"
#include "core/vec.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace fenix::ml {

struct RasterParams {
    f32 thickness = 2.0f;  // full band width in voxels (label = within thickness/2 of the sheet)
    f32 step = 0.5f;       // max surface-sample spacing in voxels
};

namespace detail {
// Precomputed sphere-stamp offsets for radius r (voxel centers within Euclidean r).
struct Stamp {
    std::vector<Index3> off;
    explicit Stamp(f32 r) {
        const s64 ir = static_cast<s64>(std::ceil(r));
        for (s64 z = -ir; z <= ir; ++z)
            for (s64 y = -ir; y <= ir; ++y)
                for (s64 x = -ir; x <= ir; ++x)
                    if (static_cast<f32>(z * z + y * y + x * x) <= r * r) off.push_back({z, y, x});
    }
};
}  // namespace detail

// Rasterize the sheet band of `s` (coords in the SAME voxel space as the patch) into a
// dense u8 mask over [origin, origin+extent): 255 = within thickness/2 of the surface.
inline Volume<u8> rasterize_band(const Surface& s, Index3 origin, Extent3 extent, RasterParams p = {}) {
    Volume<u8> out(extent);  // zero-initialized
    const f32 r = std::max(0.5f, p.thickness * 0.5f);
    const detail::Stamp stamp(r);
    const f32 lo_z = static_cast<f32>(origin.z) - r, hi_z = static_cast<f32>(origin.z + extent.z) + r;
    const f32 lo_y = static_cast<f32>(origin.y) - r, hi_y = static_cast<f32>(origin.y + extent.y) + r;
    const f32 lo_x = static_cast<f32>(origin.x) - r, hi_x = static_cast<f32>(origin.x + extent.x) + r;
    auto ov = out.view();

    auto stamp_at = [&](Vec3f q) {
        const s64 cz = static_cast<s64>(std::lround(q.z)) - origin.z;
        const s64 cy = static_cast<s64>(std::lround(q.y)) - origin.y;
        const s64 cx = static_cast<s64>(std::lround(q.x)) - origin.x;
        for (const Index3& o : stamp.off) {
            const s64 z = cz + o.z, y = cy + o.y, x = cx + o.x;
            if (z < 0 || y < 0 || x < 0 || z >= extent.z || y >= extent.y || x >= extent.x) continue;
            ov(z, y, x) = 255;
        }
    };

    // Quad cells (u,v)-(u+1,v+1); grid step is scale_u/scale_v voxels, so subdivide each cell
    // by ceil(scale/step) to hit the sample-spacing target.
    const int su = std::max(1, static_cast<int>(std::ceil(s.scale_u / p.step)));
    const int sv = std::max(1, static_cast<int>(std::ceil(s.scale_v / p.step)));
    for (s64 v = 0; v + 1 < s.nv; ++v)
        for (s64 u = 0; u + 1 < s.nu; ++u) {
            if (!s.is_valid(u, v) || !s.is_valid(u + 1, v) || !s.is_valid(u, v + 1) || !s.is_valid(u + 1, v + 1))
                continue;
            const Vec3f c00 = s.at(u, v), c10 = s.at(u + 1, v), c01 = s.at(u, v + 1), c11 = s.at(u + 1, v + 1);
            // cheap cell-bbox vs patch-bbox rejection
            const f32 bzl = std::min(std::min(c00.z, c10.z), std::min(c01.z, c11.z));
            const f32 bzh = std::max(std::max(c00.z, c10.z), std::max(c01.z, c11.z));
            const f32 byl = std::min(std::min(c00.y, c10.y), std::min(c01.y, c11.y));
            const f32 byh = std::max(std::max(c00.y, c10.y), std::max(c01.y, c11.y));
            const f32 bxl = std::min(std::min(c00.x, c10.x), std::min(c01.x, c11.x));
            const f32 bxh = std::max(std::max(c00.x, c10.x), std::max(c01.x, c11.x));
            if (bzh < lo_z || bzl > hi_z || byh < lo_y || byl > hi_y || bxh < lo_x || bxl > hi_x) continue;
            for (int j = 0; j <= sv; ++j)
                for (int i = 0; i <= su; ++i) {
                    const f32 a = static_cast<f32>(i) / static_cast<f32>(su);
                    const f32 b = static_cast<f32>(j) / static_cast<f32>(sv);
                    const Vec3f q =
                        c00 * ((1 - a) * (1 - b)) + c10 * (a * (1 - b)) + c01 * ((1 - a) * b) + c11 * (a * b);
                    stamp_at(q);
                }
        }
    return out;
}

}  // namespace fenix::ml
