// geom/morphology.hpp — binary morphology: majority filter (= discrete curvature flow,
// the single biggest Kaggle post-proc gain), dilate, erode. Shared by postproc/segment.
#pragma once

#include "core/core.hpp"
#include "geom/connected_components.hpp"  // Conn

#include <utility>  // std::swap

namespace fenix::geom {

// 27-neighbourhood majority vote, iterated. `thresh` fg neighbours (of 27, incl. self)
// to set a voxel. thresh=14 -> strict majority (smooths boundaries, fills pinholes).
inline Volume<u8> majority_filter(VolumeView<const u8> mask, int iters = 1, int thresh = 14) {
    const Extent3 d = mask.dims();
    Volume<u8> a(d), b(d);
    for (s64 i = 0; i < d.count(); ++i) a.flat()[static_cast<usize>(i)] = mask.flat()[static_cast<usize>(i)];
    for (int it = 0; it < iters; ++it) {
        VolumeView<const u8> av = a.view();
        VolumeView<u8> bv = b.view();
        parallel_for_z(d, [&](s64 z) {
            for (s64 y = 0; y < d.y; ++y)
                for (s64 x = 0; x < d.x; ++x) {
                    int cnt = 0;
                    for (s64 dz = -1; dz <= 1; ++dz)
                        for (s64 dy = -1; dy <= 1; ++dy)
                            for (s64 dx = -1; dx <= 1; ++dx)
                                cnt += av.at_clamped(z + dz, y + dy, x + dx) ? 1 : 0;
                    bv(z, y, x) = cnt >= thresh ? u8{1} : u8{0};
                }
        });
        std::swap(a, b);
    }
    return a;
}

// Dilate / erode over 6- or 26-connectivity.
inline Volume<u8> dilate(VolumeView<const u8> mask, Conn conn = Conn::Six) {
    const Extent3 d = mask.dims();
    Volume<u8> out(d);
    VolumeView<u8> ov = out.view();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                u8 v = mask.at_clamped(z, y, x);
                if (!v) {
                    for (s64 dz = -1; dz <= 1 && !v; ++dz)
                        for (s64 dy = -1; dy <= 1 && !v; ++dy)
                            for (s64 dx = -1; dx <= 1 && !v; ++dx) {
                                if (conn == Conn::Six && ((dz != 0) + (dy != 0) + (dx != 0)) != 1) continue;
                                if (mask.at_clamped(z + dz, y + dy, x + dx)) v = 1;
                            }
                }
                ov(z, y, x) = v;
            }
    });
    return out;
}

inline Volume<u8> erode(VolumeView<const u8> mask, Conn conn = Conn::Six) {
    const Extent3 d = mask.dims();
    Volume<u8> out(d);
    VolumeView<u8> ov = out.view();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                u8 v = mask.at_clamped(z, y, x);
                if (v) {
                    for (s64 dz = -1; dz <= 1 && v; ++dz)
                        for (s64 dy = -1; dy <= 1 && v; ++dy)
                            for (s64 dx = -1; dx <= 1 && v; ++dx) {
                                if (conn == Conn::Six && ((dz != 0) + (dy != 0) + (dx != 0)) != 1) continue;
                                if (!mask.at_clamped(z + dz, y + dy, x + dx)) v = 0;
                            }
                }
                ov(z, y, x) = v;
            }
    });
    return out;
}

}  // namespace fenix::geom
