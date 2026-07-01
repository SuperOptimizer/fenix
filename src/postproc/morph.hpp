// postproc/morph.hpp — the Kaggle-winning binary-mask morphology (taberna `morph.c` lineage).
// Complements cleanup.hpp (remove_small_components, fill_holes). Everything double-buffered /
// alias-safe; out-of-bounds neighbours = background. See postproc/CLAUDE.md.
//   - majority_filter : iterated 27-neighbour median = discrete surface-tension flow (biggest LB gain)
//   - connect_fragments: bridge VOI-splitting gaps between distinct components WITHOUT thickening
//   - plug_pinholes    : fill 1-voxel background holes surrounded on all 6 faces
//   - ball_{dilate,erode,close,open} : Euclidean-ball morphology
#pragma once

#include "core/core.hpp"
#include "geom/connected_components.hpp"

#include <utility>
#include <vector>

namespace fenix::postproc {

// A voxel becomes fg iff ≥ `thresh` of its 27 neighbours (incl. itself) are fg. thresh=14 = strict
// majority = a discrete mean-curvature flow that smooths the boundary. Iterate 6–10× (~+0.01 LB).
inline Volume<u8> majority_filter(VolumeView<const u8> mask, int iters = 1, int thresh = 14) {
    const Extent3 d = mask.dims();
    Volume<u8> a(d), b(d);
    for (s64 i = 0; i < d.count(); ++i) a.flat()[static_cast<usize>(i)] = mask.flat()[static_cast<usize>(i)] ? u8{1} : u8{0};
    VolumeView<u8> cur = a.view(), nxt = b.view();
    for (int it = 0; it < iters; ++it) {
        parallel_for_z(d, [&](s64 z) {
            for (s64 y = 0; y < d.y; ++y)
                for (s64 x = 0; x < d.x; ++x) {
                    int c = 0;
                    for (s64 dz = -1; dz <= 1; ++dz)
                        for (s64 dy = -1; dy <= 1; ++dy)
                            for (s64 dx = -1; dx <= 1; ++dx) {
                                const s64 zz = z + dz, yy = y + dy, xx = x + dx;
                                if (zz >= 0 && zz < d.z && yy >= 0 && yy < d.y && xx >= 0 && xx < d.x && cur(zz, yy, xx)) ++c;
                            }
                    nxt(z, y, x) = (c >= thresh) ? u8{1} : u8{0};
                }
        });
        std::swap(cur, nxt);
    }
    Volume<u8> out(d);
    for (s64 i = 0; i < d.count(); ++i) out.flat()[static_cast<usize>(i)] = cur.flat()[static_cast<usize>(i)];
    return out;
}

// Background voxel with all 6 face-neighbours foreground -> foreground (kills 1-voxel pinholes).
inline Volume<u8> plug_pinholes(VolumeView<const u8> mask) {
    const Extent3 d = mask.dims();
    Volume<u8> out(d);
    VolumeView<u8> ov = out.view();
    parallel_for_z(d, [&](s64 z) {
        auto fg = [&](s64 zz, s64 yy, s64 xx) {
            return zz >= 0 && zz < d.z && yy >= 0 && yy < d.y && xx >= 0 && xx < d.x && mask(zz, yy, xx) != 0;
        };
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                if (mask(z, y, x)) { ov(z, y, x) = 1; continue; }
                ov(z, y, x) = (fg(z - 1, y, x) && fg(z + 1, y, x) && fg(z, y - 1, x) && fg(z, y + 1, x) &&
                               fg(z, y, x - 1) && fg(z, y, x + 1)) ? u8{1} : u8{0};
            }
    });
    return out;
}

// A background voxel that "sees" ≥2 DISTINCT foreground components within Euclidean radius `r`
// becomes foreground. Bridges VOI-splitting gaps between wrap fragments WITHOUT the SurfaceDice
// cost of blanket dilation — it only fills where two distinct pieces face off across a thin gap.
inline Volume<u8> connect_fragments(VolumeView<const u8> mask, int r,
                                    geom::Conn conn = geom::Conn::TwentySix) {
    const Extent3 d = mask.dims();
    auto cc = geom::connected_components(mask, conn);
    VolumeView<s32> lbl = cc.labels.view();
    const s64 r2 = static_cast<s64>(r) * r;
    Volume<u8> out(d);
    VolumeView<u8> ov = out.view();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                if (mask(z, y, x)) { ov(z, y, x) = 1; continue; }
                s32 first = 0;
                bool two = false;
                for (s64 dz = -r; dz <= r && !two; ++dz)
                    for (s64 dy = -r; dy <= r && !two; ++dy)
                        for (s64 dx = -r; dx <= r; ++dx) {
                            if (dz * dz + dy * dy + dx * dx > r2) continue;
                            const s64 zz = z + dz, yy = y + dy, xx = x + dx;
                            if (zz < 0 || zz >= d.z || yy < 0 || yy >= d.y || xx < 0 || xx >= d.x) continue;
                            const s32 l = lbl(zz, yy, xx);
                            if (!l) continue;
                            if (!first) first = l;
                            else if (l != first) { two = true; break; }
                        }
                ov(z, y, x) = two ? u8{1} : u8{0};
            }
    });
    return out;
}

namespace detail {
inline Volume<u8> ball_morph(VolumeView<const u8> mask, int r, bool dilate) {
    const Extent3 d = mask.dims();
    const s64 r2 = static_cast<s64>(r) * r;
    Volume<u8> out(d);
    VolumeView<u8> ov = out.view();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                bool hit = false;  // dilate: any fg in ball; erode: any bg/OOB in ball
                for (s64 dz = -r; dz <= r && !hit; ++dz)
                    for (s64 dy = -r; dy <= r && !hit; ++dy)
                        for (s64 dx = -r; dx <= r; ++dx) {
                            if (dz * dz + dy * dy + dx * dx > r2) continue;
                            const s64 zz = z + dz, yy = y + dy, xx = x + dx;
                            const bool oob = zz < 0 || zz >= d.z || yy < 0 || yy >= d.y || xx < 0 || xx >= d.x;
                            const bool fg = !oob && mask(zz, yy, xx) != 0;
                            if (dilate ? fg : (!fg)) { hit = true; break; }
                        }
                ov(z, y, x) = (dilate ? hit : !hit) ? u8{1} : u8{0};
            }
    });
    return out;
}
}  // namespace detail

inline Volume<u8> ball_dilate(VolumeView<const u8> mask, int r) { return detail::ball_morph(mask, r, true); }
inline Volume<u8> ball_erode(VolumeView<const u8> mask, int r) { return detail::ball_morph(mask, r, false); }
inline Volume<u8> ball_close(VolumeView<const u8> mask, int r) { return ball_erode(ball_dilate(mask, r).view(), r); }
inline Volume<u8> ball_open(VolumeView<const u8> mask, int r) { return ball_dilate(ball_erode(mask, r).view(), r); }

}  // namespace fenix::postproc
