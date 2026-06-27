// geom/connected_components.hpp — 3D connected-component labeling (union-find), 6- or
// 26-connectivity. Compact 1-based labels. Shared by eval/topo/postproc/segment.
// Default connectivity convention: fg 6-connected (docs/conventions.md).
#pragma once

#include "core/core.hpp"

#include <vector>

namespace fenix::geom {

enum class Conn { Six = 6, TwentySix = 26 };

struct CcResult {
    Volume<s32> labels;  // 0 = background; 1..count = components
    s32 count = 0;
};

// Label connected components of the foreground (mask != 0).
inline CcResult connected_components(VolumeView<const u8> mask, Conn conn = Conn::Six) {
    const Extent3 d = mask.dims();
    const s64 n = d.count();
    std::vector<s64> parent(static_cast<usize>(n));
    for (s64 i = 0; i < n; ++i) parent[static_cast<usize>(i)] = i;

    auto find = [&](s64 i) {
        while (parent[static_cast<usize>(i)] != i) {
            parent[static_cast<usize>(i)] = parent[static_cast<usize>(parent[static_cast<usize>(i)])];
            i = parent[static_cast<usize>(i)];
        }
        return i;
    };
    auto unite = [&](s64 a, s64 b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        // Attach the larger-index root under the smaller (deterministic roots).
        if (a < b) parent[static_cast<usize>(b)] = a;
        else parent[static_cast<usize>(a)] = b;
    };

    // Lower-neighbor offsets (only neighbors already visited in z,y,x scan order).
    struct Off {
        s64 dz, dy, dx;
    };
    // Neighbors already visited in z,y,x scan order (lexicographically before the origin).
    std::vector<Off> offs;
    for (s64 dz = -1; dz <= 1; ++dz)
        for (s64 dy = -1; dy <= 1; ++dy)
            for (s64 dx = -1; dx <= 1; ++dx) {
                const bool visited_before =
                    dz < 0 || (dz == 0 && dy < 0) || (dz == 0 && dy == 0 && dx < 0);
                if (!visited_before) continue;
                if (conn == Conn::Six && ((dz != 0) + (dy != 0) + (dx != 0)) != 1) continue;
                offs.push_back({dz, dy, dx});
            }

    auto lin = [&](s64 z, s64 y, s64 x) { return (z * d.y + y) * d.x + x; };
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                if (!mask(z, y, x)) continue;
                const s64 i = lin(z, y, x);
                for (const Off& o : offs) {
                    const s64 nz = z + o.dz, ny = y + o.dy, nx = x + o.dx;
                    if (nz < 0 || ny < 0 || nx < 0 || ny >= d.y || nx >= d.x) continue;
                    if (mask(nz, ny, nx)) unite(i, lin(nz, ny, nx));
                }
            }

    CcResult r{Volume<s32>::zeros(d), 0};
    VolumeView<s32> lbl = r.labels.view();
    std::vector<s32> root_label(static_cast<usize>(n), 0);
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                if (!mask(z, y, x)) continue;
                const s64 root = find(lin(z, y, x));
                s32& rl = root_label[static_cast<usize>(root)];
                if (rl == 0) rl = ++r.count;
                lbl(z, y, x) = rl;
            }
    return r;
}

}  // namespace fenix::geom
