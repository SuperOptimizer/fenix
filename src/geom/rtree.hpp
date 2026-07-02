// geom/rtree.hpp — packed static R-tree over 3D boxes (STR bulk-load: sort-tile-recursive),
// the spatial index VC3D keeps over its surface patches, rebuilt fresh. Build once from
// (box, id) pairs, then query(box) -> ids whose boxes intersect. No insert/delete — our
// surfaces are immutable once loaded; bulk-load beats dynamic R-trees on both build and
// query for that shape. ZYX boxes, f32 bounds, ids are caller-defined u32s.
#pragma once

#include "core/core.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace fenix::geom {

struct Box3f {
    f32 zlo = 0, zhi = -1, ylo = 0, yhi = -1, xlo = 0, xhi = -1;  // empty by default
    [[nodiscard]] bool intersects(const Box3f& o) const {
        return zlo <= o.zhi && o.zlo <= zhi && ylo <= o.yhi && o.ylo <= yhi && xlo <= o.xhi && o.xlo <= xhi;
    }
    void expand(const Box3f& o) {
        if (o.zhi < o.zlo) return;
        if (zhi < zlo) {
            *this = o;
            return;
        }
        zlo = std::min(zlo, o.zlo);
        zhi = std::max(zhi, o.zhi);
        ylo = std::min(ylo, o.ylo);
        yhi = std::max(yhi, o.yhi);
        xlo = std::min(xlo, o.xlo);
        xhi = std::max(xhi, o.xhi);
    }
};

class BoxRTree {
  public:
    static constexpr int kFan = 16;  // node fan-out

    BoxRTree() = default;

    // STR bulk load: sort leaves into z-slabs, each slab into y-runs, each run by x; pack
    // kFan-at-a-time bottom-up. O(n log n) build, ~log_16(n) node visits per query.
    static BoxRTree build(std::vector<std::pair<Box3f, u32>> items) {
        BoxRTree t;
        const usize n = items.size();
        if (n == 0) return t;
        const usize nleaf = (n + kFan - 1) / kFan;
        const auto slab = static_cast<usize>(std::ceil(std::cbrt(static_cast<f64>(nleaf))));
        auto center = [](const Box3f& b, int ax) {
            return ax == 0 ? b.zlo + b.zhi : ax == 1 ? b.ylo + b.yhi : b.xlo + b.xhi;
        };
        std::sort(items.begin(), items.end(), [&](const auto& a, const auto& b) {
            return center(a.first, 0) < center(b.first, 0);
        });
        const usize zchunk = (n + slab - 1) / slab;
        for (usize z0 = 0; z0 < n; z0 += zchunk) {
            const usize z1 = std::min(n, z0 + zchunk);
            std::sort(items.begin() + static_cast<std::ptrdiff_t>(z0),
                      items.begin() + static_cast<std::ptrdiff_t>(z1),
                      [&](const auto& a, const auto& b) { return center(a.first, 1) < center(b.first, 1); });
            const usize ychunk = (z1 - z0 + slab - 1) / slab;
            for (usize y0 = z0; y0 < z1; y0 += ychunk) {
                const usize y1 = std::min(z1, y0 + ychunk);
                std::sort(items.begin() + static_cast<std::ptrdiff_t>(y0),
                          items.begin() + static_cast<std::ptrdiff_t>(y1),
                          [&](const auto& a, const auto& b) { return center(a.first, 2) < center(b.first, 2); });
            }
        }
        // level 0: leaves hold items directly
        t.items_ = std::move(items);
        std::vector<Node> cur;
        for (usize i = 0; i < n; i += kFan) {
            Node nd;
            nd.first = static_cast<u32>(i);
            nd.count = static_cast<u32>(std::min<usize>(kFan, n - i));
            nd.leaf = 1;
            for (u32 k = 0; k < nd.count; ++k) nd.box.expand(t.items_[i + k].first);
            cur.push_back(nd);
        }
        // pack upper levels until one root
        while (cur.size() > 1) {
            const u32 base = static_cast<u32>(t.nodes_.size());
            for (const Node& nd : cur) t.nodes_.push_back(nd);
            std::vector<Node> up;
            for (usize i = 0; i < cur.size(); i += kFan) {
                Node nd;
                nd.first = base + static_cast<u32>(i);
                nd.count = static_cast<u32>(std::min<usize>(kFan, cur.size() - i));
                nd.leaf = 0;
                for (u32 k = 0; k < nd.count; ++k) nd.box.expand(cur[i + k].box);
                up.push_back(nd);
            }
            cur = std::move(up);
        }
        t.root_ = cur[0];
        t.has_root_ = true;
        return t;
    }

    // Append every id whose box intersects q.
    void query(const Box3f& q, std::vector<u32>& out) const {
        if (!has_root_ || !root_.box.intersects(q)) return;
        query_(root_, q, out);
    }
    [[nodiscard]] Box3f bounds() const { return has_root_ ? root_.box : Box3f{}; }
    [[nodiscard]] usize size() const { return items_.size(); }

  private:
    struct Node {
        Box3f box;
        u32 first = 0, count = 0;
        u32 leaf = 0;
    };
    void query_(const Node& nd, const Box3f& q, std::vector<u32>& out) const {
        if (nd.leaf) {
            for (u32 k = 0; k < nd.count; ++k)
                if (items_[nd.first + k].first.intersects(q)) out.push_back(items_[nd.first + k].second);
        } else {
            for (u32 k = 0; k < nd.count; ++k)
                if (nodes_[nd.first + k].box.intersects(q)) query_(nodes_[nd.first + k], q, out);
        }
    }
    std::vector<std::pair<Box3f, u32>> items_;
    std::vector<Node> nodes_;
    Node root_;
    bool has_root_ = false;
};

}  // namespace fenix::geom
