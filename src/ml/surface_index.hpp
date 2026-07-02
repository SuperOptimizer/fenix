// ml/surface_index.hpp — the two-level spatial index over segment surfaces (torch-free):
// answers "which surfaces exist in this box, and WHERE" in ~log time instead of scanning
// every cell of every mesh (81 PHercParis4 meshes ≈ 10^9 cell checks per patch without it).
// Level 1: an R-tree over whole-mesh bboxes. Level 2: per mesh, an R-tree over 16×16-cell
// uv-tile bboxes. A box query returns, per intersecting mesh, the uv-tile rects that touch
// the box — the rasterizer then visits only those cells. Mirrors VC3D's surface-patch
// R-tree concept (geom/rtree.hpp is the index structure itself).
#pragma once

#include "core/core.hpp"
#include "core/surface.hpp"
#include "geom/rtree.hpp"

#include <span>
#include <vector>

namespace fenix::ml {

struct UvRect {
    s64 u0 = 0, v0 = 0, u1 = 0, v1 = 0;  // half-open cell range [u0,u1) x [v0,v1)
};

// Per-mesh index: uv-tile (16x16 cells) bounding boxes in an R-tree.
class SurfaceIndex {
  public:
    static constexpr s64 kTile = 16;

    SurfaceIndex() = default;
    explicit SurfaceIndex(const Surface& s) {
        nu_ = s.nu;
        nv_ = s.nv;
        ntx_ = (std::max<s64>(nu_ - 1, 0) + kTile - 1) / kTile;
        nty_ = (std::max<s64>(nv_ - 1, 0) + kTile - 1) / kTile;
        std::vector<std::pair<geom::Box3f, u32>> items;
        for (s64 tv = 0; tv < nty_; ++tv)
            for (s64 tu = 0; tu < ntx_; ++tu) {
                geom::Box3f b;
                // cells [u,u+1) need corner u+1: include the +1 rim, clamp to grid
                const s64 u1 = std::min(nu_, (tu + 1) * kTile + 1), v1 = std::min(nv_, (tv + 1) * kTile + 1);
                for (s64 v = tv * kTile; v < v1; ++v)
                    for (s64 u = tu * kTile; u < u1; ++u) {
                        if (!s.is_valid(u, v)) continue;
                        const Vec3f c = s.at(u, v);
                        b.expand({c.z, c.z, c.y, c.y, c.x, c.x});
                    }
                if (b.zhi >= b.zlo) items.push_back({b, static_cast<u32>(tv * ntx_ + tu)});
            }
        tree_ = geom::BoxRTree::build(std::move(items));
    }

    [[nodiscard]] geom::Box3f bounds() const { return tree_.bounds(); }

    // uv-tile rects whose surface bboxes intersect `q` (already expanded by any band radius).
    [[nodiscard]] std::vector<UvRect> query(const geom::Box3f& q) const {
        std::vector<u32> ids;
        tree_.query(q, ids);
        std::vector<UvRect> out;
        out.reserve(ids.size());
        for (u32 id : ids) {
            const s64 tu = id % ntx_, tv = id / ntx_;
            out.push_back(
                {tu * kTile, tv * kTile, std::min(nu_ - 1, (tu + 1) * kTile), std::min(nv_ - 1, (tv + 1) * kTile)});
        }
        return out;
    }

  private:
    geom::BoxRTree tree_;
    s64 nu_ = 0, nv_ = 0, ntx_ = 0, nty_ = 0;
};

// Volume-level index: which meshes exist in a box + their per-mesh tile rects.
class VolumeSurfaceIndex {
  public:
    VolumeSurfaceIndex() = default;
    explicit VolumeSurfaceIndex(std::span<const Surface* const> meshes) {
        idx_.reserve(meshes.size());
        std::vector<std::pair<geom::Box3f, u32>> items;
        for (usize m = 0; m < meshes.size(); ++m) {
            idx_.emplace_back(*meshes[m]);
            const geom::Box3f b = idx_.back().bounds();
            if (b.zhi >= b.zlo) items.push_back({b, static_cast<u32>(m)});
        }
        tree_ = geom::BoxRTree::build(std::move(items));
    }

    struct Hit {
        u32 mesh = 0;
        std::vector<UvRect> rects;  // where (in uv-cells) the mesh touches the box
    };
    [[nodiscard]] std::vector<Hit> query(const geom::Box3f& q) const {
        std::vector<u32> ms;
        tree_.query(q, ms);
        std::vector<Hit> out;
        for (u32 m : ms) {
            Hit h{m, idx_[m].query(q)};
            if (!h.rects.empty()) out.push_back(std::move(h));
        }
        return out;
    }
    [[nodiscard]] const SurfaceIndex& mesh_index(usize m) const { return idx_[m]; }

  private:
    std::vector<SurfaceIndex> idx_;
    geom::BoxRTree tree_;
};

}  // namespace fenix::ml
