// test_kdtree.cpp — KD-tree nearest-neighbour vs brute force.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "geom/kdtree.hpp"

#include <vector>

using namespace fenix;
using namespace fenix::geom;

static s64 brute_nearest(const std::vector<Vec3f>& pts, Vec3f q) {
    s64 best = -1;
    f32 bd = 1e30f;
    for (usize i = 0; i < pts.size(); ++i) {
        Vec3f d = pts[i] - q;
        f32 d2 = d.z * d.z + d.y * d.y + d.x * d.x;
        if (d2 < bd) {
            bd = d2;
            best = static_cast<s64>(i);
        }
    }
    return best;
}

TEST(kdtree_matches_brute_force) {
    Pcg32 rng{17};
    std::vector<Vec3f> pts;
    for (int i = 0; i < 2000; ++i)
        pts.push_back(Vec3f{rng.next_f32() * 100, rng.next_f32() * 100, rng.next_f32() * 100});
    KdTree tree(pts);
    for (int q = 0; q < 200; ++q) {
        Vec3f query{rng.next_f32() * 100, rng.next_f32() * 100, rng.next_f32() * 100};
        s64 kd = tree.nearest(query);
        s64 bf = brute_nearest(pts, query);
        // Compare by distance (ties allowed): the KD result must be as close as brute force.
        Vec3f a = tree.point(kd) - query, b = pts[static_cast<usize>(bf)] - query;
        f32 da = a.z * a.z + a.y * a.y + a.x * a.x, db = b.z * b.z + b.y * b.y + b.x * b.x;
        CHECK(std::abs(da - db) < 1e-3f);
    }
}

TEST(kdtree_empty_and_single) {
    KdTree empty{std::span<const Vec3f>{}};
    CHECK(empty.nearest(Vec3f{0, 0, 0}) == -1);
    std::vector<Vec3f> one{Vec3f{1, 2, 3}};
    KdTree t(one);
    CHECK(t.nearest(Vec3f{9, 9, 9}) == 0);
}
