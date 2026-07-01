// geom/connected_components.hpp — 3D connected-component labeling (union-find), 6- or
// 26-connectivity. Compact 1-based labels. Shared by eval/topo/postproc/segment.
// Default connectivity convention: fg 6-connected (docs/conventions.md).
#pragma once

#include "core/core.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace fenix::geom {

enum class Conn { Six = 6, TwentySix = 26 };

struct CcResult {
    Volume<s32> labels;  // 0 = background; 1..count = components
    s32 count = 0;
};

// Label connected components of the foreground (mask != 0).
// Parallel: z-slab-local union-find (each slab's unions touch only its own parent entries),
// then a serial merge over the slab-boundary planes, then a parallel read-only relabel.
// Labels are IDENTICAL to the serial scan: unite keeps the smaller index as root, so a
// component's root is its first voxel in z,y,x scan order, and compact ids are assigned in
// ascending-root order == first-occurrence order.
inline CcResult connected_components(VolumeView<const u8> mask, Conn conn = Conn::Six) {
    const Extent3 d = mask.dims();
    const s64 n = d.count();
    std::vector<s64> parent(static_cast<usize>(n));
    parallel_for(0, n, [&](s64 i) { parent[static_cast<usize>(i)] = i; });

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

    // Lower-neighbor offsets (only neighbors already visited in z,y,x scan order); dz ∈ {-1,0}.
    struct Off {
        s64 dz, dy, dx;
    };
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

    const s64 nslabs = std::max<s64>(1, std::min<s64>(cpu_budget(), d.z));
    auto slab_begin = [&](s64 s) { return d.z * s / nslabs; };

    // Phase 1 (parallel): unions strictly inside each slab — find/unite only walk indices whose
    // voxels lie in the slab, so slabs never race on parent entries.
    parallel_for(0, nslabs, [&](s64 s) {
        const s64 z0 = slab_begin(s), z1 = slab_begin(s + 1);
        for (s64 z = z0; z < z1; ++z)
            for (s64 y = 0; y < d.y; ++y)
                for (s64 x = 0; x < d.x; ++x) {
                    if (!mask(z, y, x)) continue;
                    const s64 i = lin(z, y, x);
                    for (const Off& o : offs) {
                        const s64 nz = z + o.dz, ny = y + o.dy, nx = x + o.dx;
                        if (nz < z0 || ny < 0 || nx < 0 || ny >= d.y || nx >= d.x) continue;
                        if (mask(nz, ny, nx)) unite(i, lin(nz, ny, nx));
                    }
                }
    });

    // Phase 2 (serial): the deferred cross-boundary unions (nz landed in the previous slab).
    for (s64 s = 1; s < nslabs; ++s) {
        const s64 z = slab_begin(s);
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                if (!mask(z, y, x)) continue;
                const s64 i = lin(z, y, x);
                for (const Off& o : offs) {
                    if (o.dz == 0) continue;
                    const s64 ny = y + o.dy, nx = x + o.dx;
                    if (ny < 0 || nx < 0 || ny >= d.y || nx >= d.x) continue;
                    if (mask(z - 1, ny, nx)) unite(i, lin(z - 1, ny, nx));
                }
            }
    }

    // Phase 3: read-only find (no path compression — all writes are done, threads may share paths).
    auto find_ro = [&](s64 i) {
        while (parent[static_cast<usize>(i)] != i) i = parent[static_cast<usize>(i)];
        return i;
    };

    // Collect the distinct roots (parallel, per-slab sets), assign ids in ascending-root order.
    std::vector<std::vector<s64>> slab_roots(static_cast<usize>(nslabs));
    parallel_for(0, nslabs, [&](s64 s) {
        std::vector<s64>& out = slab_roots[static_cast<usize>(s)];
        s64 last = -1;
        const s64 z0 = slab_begin(s), z1 = slab_begin(s + 1);
        for (s64 z = z0; z < z1; ++z)
            for (s64 y = 0; y < d.y; ++y)
                for (s64 x = 0; x < d.x; ++x) {
                    if (!mask(z, y, x)) continue;
                    const s64 root = find_ro(lin(z, y, x));
                    if (root != last) { out.push_back(root); last = root; }
                }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    });
    std::vector<s64> roots;
    for (auto& sr : slab_roots) roots.insert(roots.end(), sr.begin(), sr.end());
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());

    CcResult r{Volume<s32>::zeros(d), static_cast<s32>(roots.size())};
    VolumeView<s32> lbl = r.labels.view();
    // root -> compact id; only root slots are written/read, so no zero-fill of the n-array needed.
    std::unique_ptr<s32[]> root_label(new s32[static_cast<usize>(n)]);
    for (usize k = 0; k < roots.size(); ++k) root_label[static_cast<usize>(roots[k])] = static_cast<s32>(k) + 1;

    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                if (!mask(z, y, x)) continue;
                lbl(z, y, x) = root_label[static_cast<usize>(find_ro(lin(z, y, x)))];
            }
    });
    return r;
}

}  // namespace fenix::geom
