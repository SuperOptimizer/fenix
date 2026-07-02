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
#include "ml/surface_index.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace fenix::ml {

struct RasterParams {
    f32 thickness = 2.0f;  // full band width in voxels (label = within thickness/2 of the sheet)
    f32 shell = 12.0f;     // trusted-background shell full width (0 = binary band, no shell)
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

namespace detail {
// Stamp every valid quad cell of `s` overlapping the (r-expanded) patch bbox with `value`,
// skipping voxels already labeled with a HIGHER value (sheet beats shell beats unlabeled).
inline void raster_pass(const Surface& s,
                        Index3 origin,
                        Extent3 extent,
                        f32 r,
                        u8 value,
                        VolumeView<u8> ov,
                        f32 step,
                        std::span<const UvRect> rects) {
    const Stamp stamp(r);
    const f32 lo_z = static_cast<f32>(origin.z) - r, hi_z = static_cast<f32>(origin.z + extent.z) + r;
    const f32 lo_y = static_cast<f32>(origin.y) - r, hi_y = static_cast<f32>(origin.y + extent.y) + r;
    const f32 lo_x = static_cast<f32>(origin.x) - r, hi_x = static_cast<f32>(origin.x + extent.x) + r;

    auto stamp_at = [&](Vec3f q) {
        const s64 cz = static_cast<s64>(std::lround(q.z)) - origin.z;
        const s64 cy = static_cast<s64>(std::lround(q.y)) - origin.y;
        const s64 cx = static_cast<s64>(std::lround(q.x)) - origin.x;
        for (const Index3& o : stamp.off) {
            const s64 z = cz + o.z, y = cy + o.y, x = cx + o.x;
            if (z < 0 || y < 0 || x < 0 || z >= extent.z || y >= extent.y || x >= extent.x) continue;
            u8& v = ov(z, y, x);
            if (v < value) v = value;
        }
    };

    // Quad cells (u,v)-(u+1,v+1); grid step is scale_u/scale_v voxels, so subdivide each cell
    // by ceil(scale/step) to hit the sample-spacing target.
    const int su = std::max(1, static_cast<int>(std::ceil(s.scale_u / step)));
    const int sv = std::max(1, static_cast<int>(std::ceil(s.scale_v / step)));
    for (const UvRect& rc : rects)
        for (s64 v = rc.v0; v < rc.v1 && v + 1 < s.nv; ++v)
            for (s64 u = rc.u0; u < rc.u1 && u + 1 < s.nu; ++u) {
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
}
}  // namespace detail

// Label semantics (docs/design/training-pipeline.md, multi-segment supervision):
//   255 = sheet (within thickness/2 of ANY mesh)
//   128 = trusted background (within shell/2 of a labeled sheet but not on one — that
//         segment's own accuracy vouches for "no sheet here")
//   0   = unlabeled — a sheet no segment covers may exist; the hard loss must IGNORE it
//         (the dense KD teacher term covers these voxels instead).
inline constexpr u8 kLabelSheet = 255, kLabelBackground = 128, kLabelUnknown = 0;

// Rasterize the UNION of all `meshes` intersecting the patch (multiple segments routinely
// share a training chunk — labeling only one would mark the others' true sheets as
// background). Coords must be in the same voxel space as the patch.
inline Volume<u8> rasterize_band_multi(std::span<const Surface* const> meshes,
                                       Index3 origin,
                                       Extent3 extent,
                                       RasterParams p = {},
                                       const VolumeSurfaceIndex* index = nullptr) {
    Volume<u8> out = Volume<u8>::zeros(extent);  // Volume(Extent3) is for-overwrite: NOT zeroed
    const f32 rb = std::max(0.5f, p.thickness * 0.5f);
    const f32 rs = std::max(rb, p.shell * 0.5f);
    auto ov = out.view();
    // With an index: query once (expanded by the largest radius) and visit only the returned
    // uv-tile rects. Without: every mesh's full cell range (fine for 1-2 meshes / tests).
    std::vector<std::vector<UvRect>> per_mesh(meshes.size());
    if (index) {
        const geom::Box3f q{static_cast<f32>(origin.z) - rs,
                            static_cast<f32>(origin.z + extent.z) + rs,
                            static_cast<f32>(origin.y) - rs,
                            static_cast<f32>(origin.y + extent.y) + rs,
                            static_cast<f32>(origin.x) - rs,
                            static_cast<f32>(origin.x + extent.x) + rs};
        for (auto& h : index->query(q)) per_mesh[h.mesh] = std::move(h.rects);
    } else {
        for (usize m = 0; m < meshes.size(); ++m) per_mesh[m] = {UvRect{0, 0, meshes[m]->nu - 1, meshes[m]->nv - 1}};
    }
    // shell first (bigger radius), then sheets overwrite via the value-precedence stamp
    if (p.shell > 0)
        for (usize m = 0; m < meshes.size(); ++m)
            if (!per_mesh[m].empty())
                detail::raster_pass(*meshes[m], origin, extent, rs, kLabelBackground, ov, p.step * 2, per_mesh[m]);
    for (usize m = 0; m < meshes.size(); ++m)
        if (!per_mesh[m].empty())
            detail::raster_pass(*meshes[m], origin, extent, rb, kLabelSheet, ov, p.step, per_mesh[m]);
    return out;
}

// Single-mesh convenience (tests, tools): binary band, no shell.
inline Volume<u8> rasterize_band(const Surface& s, Index3 origin, Extent3 extent, RasterParams p = {}) {
    const Surface* one[] = {&s};
    RasterParams q = p;
    q.shell = 0;
    return rasterize_band_multi(one, origin, extent, q);
}

}  // namespace fenix::ml
