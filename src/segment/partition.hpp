// segment/partition.hpp — Mutex Watershed (Wolf et al. 2018) on a signed graph (taberna `partition.c`).
// Parameter-free signed-graph partitioning: process edges by descending |w|; an attractive edge (w>0)
// merges its two clusters unless a mutex (cannot-link) forbids it; a repulsive edge (w<0) plants a
// mutex between the clusters. The repulsion is what keeps TOUCHING wraps in separate instances — the
// collapse-to-one-label failure that unsigned agglomeration can't prevent. Union-find + per-root mutex
// lists (stale entries resolved via find). See segment/CLAUDE.md.
#pragma once

#include "core/core.hpp"
#include "segment/affinity.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace fenix::segment {

// Partition `g` into clusters; fills `labels` (size g.nnodes) with a dense cluster id per node.
// Returns the number of clusters.
inline s64 mws_partition(const SignedGraph& g, std::vector<u32>& labels) {
    const s64 nn = g.nnodes;
    labels.assign(static_cast<usize>(nn < 0 ? 0 : nn), 0);
    if (nn <= 0) return 0;

    std::vector<s32> parent(static_cast<usize>(nn)), sz(static_cast<usize>(nn), 1);
    for (s64 i = 0; i < nn; ++i) parent[static_cast<usize>(i)] = static_cast<s32>(i);
    auto find = [&](s32 x) {
        while (parent[static_cast<usize>(x)] != x) {
            parent[static_cast<usize>(x)] = parent[static_cast<usize>(parent[static_cast<usize>(x)])];
            x = parent[static_cast<usize>(x)];
        }
        return x;
    };
    std::vector<std::vector<s32>> mx(static_cast<usize>(nn));  // per-root mutex lists
    auto blocked = [&](s32 ra, s32 rb) {
        for (s32 v : mx[static_cast<usize>(ra)]) if (find(v) == rb) return true;
        for (s32 v : mx[static_cast<usize>(rb)]) if (find(v) == ra) return true;
        return false;
    };

    std::vector<SignedEdge> e = g.edges;
    std::sort(e.begin(), e.end(), [](const SignedEdge& a, const SignedEdge& b) {
        return std::fabs(a.w) > std::fabs(b.w);  // descending |w|
    });

    for (const SignedEdge& ed : e) {
        s32 ra = find(static_cast<s32>(ed.a)), rb = find(static_cast<s32>(ed.b));
        if (ra == rb) continue;
        if (ed.w > 0.0f) {  // attractive: merge unless mutexed
            if (blocked(ra, rb)) continue;
            s32 big = ra, small = rb;
            if (sz[static_cast<usize>(big)] < sz[static_cast<usize>(small)]) std::swap(big, small);
            parent[static_cast<usize>(small)] = big;
            sz[static_cast<usize>(big)] += sz[static_cast<usize>(small)];
            auto& ms = mx[static_cast<usize>(small)];
            auto& mb = mx[static_cast<usize>(big)];
            mb.insert(mb.end(), ms.begin(), ms.end());
            ms.clear();
            ms.shrink_to_fit();
        } else if (ed.w < 0.0f) {  // repulsive: plant a mutex both ways
            mx[static_cast<usize>(ra)].push_back(rb);
            mx[static_cast<usize>(rb)].push_back(ra);
        }
    }

    std::vector<s32> remap(static_cast<usize>(nn), -1);
    s64 ncluster = 0;
    for (s64 i = 0; i < nn; ++i) {
        const s32 r = find(static_cast<s32>(i));
        if (remap[static_cast<usize>(r)] < 0) remap[static_cast<usize>(r)] = static_cast<s32>(ncluster++);
        labels[static_cast<usize>(i)] = static_cast<u32>(remap[static_cast<usize>(r)]);
    }
    return ncluster;
}

}  // namespace fenix::segment
