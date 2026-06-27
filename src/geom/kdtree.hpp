// geom/kdtree.hpp — 3D KD-tree for nearest-neighbour queries (mesh cleanup, point-cloud
// dedup, constraint association). Build O(n log n), query ~O(log n) average. See geom/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <algorithm>
#include <limits>
#include <span>
#include <vector>

namespace fenix::geom {

class KdTree {
public:
    KdTree() = default;
    explicit KdTree(std::span<const Vec3f> points) : pts_(points.begin(), points.end()) {
        idx_.resize(pts_.size());
        for (usize i = 0; i < idx_.size(); ++i) idx_[i] = static_cast<s64>(i);
        if (!idx_.empty()) build(0, static_cast<s64>(idx_.size()), 0);
    }

    // Index of the nearest stored point to q (-1 if empty).
    [[nodiscard]] s64 nearest(Vec3f q) const {
        if (idx_.empty()) return -1;
        s64 best = -1;
        f32 best_d2 = std::numeric_limits<f32>::max();
        search(0, static_cast<s64>(idx_.size()), 0, q, best, best_d2);
        return best;
    }

    [[nodiscard]] const Vec3f& point(s64 i) const { return pts_[static_cast<usize>(i)]; }

private:
    static f32 axis(const Vec3f& p, int a) { return a == 0 ? p.z : (a == 1 ? p.y : p.x); }
    static f32 dist2(const Vec3f& a, const Vec3f& b) {
        const Vec3f d = a - b;
        return d.z * d.z + d.y * d.y + d.x * d.x;
    }

    // Build over idx_[lo,hi) splitting on `depth%3` axis; node = median at mid.
    void build(s64 lo, s64 hi, int depth) {
        if (hi - lo <= 1) return;
        const int a = depth % 3;
        const s64 mid = (lo + hi) / 2;
        std::nth_element(idx_.begin() + lo, idx_.begin() + mid, idx_.begin() + hi,
                         [&](s64 i, s64 j) { return axis(pts_[static_cast<usize>(i)], a) <
                                                    axis(pts_[static_cast<usize>(j)], a); });
        build(lo, mid, depth + 1);
        build(mid + 1, hi, depth + 1);
    }

    void search(s64 lo, s64 hi, int depth, Vec3f q, s64& best, f32& best_d2) const {
        if (hi - lo <= 0) return;
        const s64 mid = (lo + hi) / 2;
        const s64 pi = idx_[static_cast<usize>(mid)];
        const f32 d2 = dist2(pts_[static_cast<usize>(pi)], q);
        if (d2 < best_d2) {
            best_d2 = d2;
            best = pi;
        }
        if (hi - lo == 1) return;
        const int a = depth % 3;
        const f32 diff = axis(q, a) - axis(pts_[static_cast<usize>(pi)], a);
        const bool go_left = diff < 0;
        if (go_left) search(lo, mid, depth + 1, q, best, best_d2);
        else search(mid + 1, hi, depth + 1, q, best, best_d2);
        // Backtrack into the far branch if the splitting plane is within the best radius.
        if (diff * diff < best_d2) {
            if (go_left) search(mid + 1, hi, depth + 1, q, best, best_d2);
            else search(lo, mid, depth + 1, q, best, best_d2);
        }
    }

    std::vector<Vec3f> pts_;
    std::vector<s64> idx_;
};

}  // namespace fenix::geom
