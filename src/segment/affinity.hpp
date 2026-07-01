// segment/affinity.hpp — signed region-adjacency graph from sheet detection (taberna `affinity.c`),
// for signed-graph partitioning into per-wrap instances. Signed-edge convention: w>0 attractive
// ("same wrap"), w<0 repulsive ("different wraps, must not merge"). Between two adjacent foreground
// voxels along axis `d`:  across = ½(|n_i·d|+|n_j·d|),  sb = ½(sheet_i+sheet_j),
//   w = k_attract·(1−across) − k_repel·(across·sb).
// In-plane contact (edge ⟂ normal) attracts; a contact ALONG the normal through high-sheetness
// material repels — the signature of crossing into a TOUCHING next wrap. Voxel-resolution; supervoxel
// coarsening (snic) slots in behind the same graph later. See segment/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <cmath>
#include <vector>

namespace fenix::segment {

struct SignedEdge {
    u32 a, b;
    f32 w;  // >0 attractive, <0 repulsive
};
struct SignedGraph {
    std::vector<SignedEdge> edges;
    s64 nnodes = 0;
    std::vector<s32> node_of;   // voxel linear index -> node id, or -1 (background)
    std::vector<s32> voxel_of;  // node id -> voxel linear index
};
struct AffinityParams {
    f32 k_attract = 1.0f;
    f32 k_repel = 1.0f;
};

// Build a voxel-level signed RAG over the foreground (mask != 0). `normal` is ZYX per-voxel.
inline SignedGraph build_signed_affinity(VolumeView<const u8> mask, VolumeView<const f32> sheet,
                                         const std::vector<Vec3f>& normal, AffinityParams p = {}) {
    const Extent3 d = mask.dims();
    const usize n = static_cast<usize>(d.count());
    SignedGraph g;
    g.node_of.assign(n, -1);
    for (usize i = 0; i < n; ++i)
        if (mask.flat()[i]) g.node_of[i] = static_cast<s32>(g.nnodes++);
    g.voxel_of.assign(static_cast<usize>(g.nnodes ? g.nnodes : 1), 0);
    for (usize i = 0; i < n; ++i)
        if (g.node_of[i] >= 0) g.voxel_of[static_cast<usize>(g.node_of[i])] = static_cast<s32>(i);

    auto comp = [](const Vec3f& v, int ax) { return ax == 0 ? v.x : (ax == 1 ? v.y : v.z); };
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const usize i = static_cast<usize>((z * d.y + y) * d.x + x);
                if (g.node_of[i] < 0) continue;
                for (int ax = 0; ax < 3; ++ax) {  // +x, +y, +z contacts
                    const s64 zz = z + (ax == 2), yy = y + (ax == 1), xx = x + (ax == 0);
                    if (xx >= d.x || yy >= d.y || zz >= d.z) continue;
                    const usize j = static_cast<usize>((zz * d.y + yy) * d.x + xx);
                    if (g.node_of[j] < 0) continue;
                    const f32 across = 0.5f * (std::fabs(comp(normal[i], ax)) + std::fabs(comp(normal[j], ax)));
                    const f32 sb = 0.5f * (sheet.flat()[i] + sheet.flat()[j]);
                    const f32 w = p.k_attract * (1.0f - across) - p.k_repel * (across * sb);
                    g.edges.push_back({static_cast<u32>(g.node_of[i]), static_cast<u32>(g.node_of[j]), w});
                }
            }
    return g;
}

}  // namespace fenix::segment
