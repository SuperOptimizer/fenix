// postproc/cleanup.hpp — binary mask cleanup: drop small components (dust removal) +
// fill enclosed holes (cavity fill). Built on geom CC. The high-value Kaggle post-proc.
// See postproc/CLAUDE.md.
#pragma once

#include "core/core.hpp"
#include "geom/connected_components.hpp"

#include <vector>

namespace fenix::postproc {

// Remove foreground components smaller than `min_size` voxels.
inline Volume<u8> remove_small_components(VolumeView<const u8> mask, s64 min_size,
                                          geom::Conn conn = geom::Conn::TwentySix) {
    auto cc = geom::connected_components(mask, conn);
    std::vector<s64> size(static_cast<usize>(cc.count + 1), 0);
    VolumeView<s32> lbl = cc.labels.view();
    const Extent3 d = mask.dims();
    for (s64 i = 0; i < d.count(); ++i) size[static_cast<usize>(lbl.flat()[static_cast<usize>(i)])]++;
    Volume<u8> out = Volume<u8>::zeros(d);
    for (s64 i = 0; i < d.count(); ++i) {
        const s32 l = lbl.flat()[static_cast<usize>(i)];
        if (l != 0 && size[static_cast<usize>(l)] >= min_size) out.flat()[static_cast<usize>(i)] = 1;
    }
    return out;
}

// Fill background regions fully enclosed by foreground (cavities) -> foreground.
// Background connectivity 6-conn (the dual of fg 26-conn). Removes b2 cavities.
inline Volume<u8> fill_holes(VolumeView<const u8> mask) {
    const Extent3 d = mask.dims();
    Volume<u8> bg(d);
    for (s64 i = 0; i < d.count(); ++i) bg.flat()[static_cast<usize>(i)] = mask.flat()[static_cast<usize>(i)] ? u8{0} : u8{1};
    auto cc = geom::connected_components(bg.view(), geom::Conn::Six);
    std::vector<u8> border(static_cast<usize>(cc.count + 1), 0);
    VolumeView<s32> lbl = cc.labels.view();
    auto mark = [&](s64 z, s64 y, s64 x) {
        if (s32 l = lbl(z, y, x)) border[static_cast<usize>(l)] = 1;
    };
    for (s64 y = 0; y < d.y; ++y)
        for (s64 x = 0; x < d.x; ++x) { mark(0, y, x); mark(d.z - 1, y, x); }
    for (s64 z = 0; z < d.z; ++z)
        for (s64 x = 0; x < d.x; ++x) { mark(z, 0, x); mark(z, d.y - 1, x); }
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y) { mark(z, y, 0); mark(z, y, d.x - 1); }

    Volume<u8> out(d);
    for (s64 i = 0; i < d.count(); ++i) {
        if (mask.flat()[static_cast<usize>(i)]) {
            out.flat()[static_cast<usize>(i)] = 1;
        } else {
            const s32 l = lbl.flat()[static_cast<usize>(i)];
            out.flat()[static_cast<usize>(i)] = (l != 0 && !border[static_cast<usize>(l)]) ? u8{1} : u8{0};
        }
    }
    return out;
}

}  // namespace fenix::postproc
