// test_eval_geom.cpp — morphology + eval metrics (Dice/IoU/VOI).
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "eval/metrics.hpp"
#include "eval/score.hpp"
#include "geom/morphology.hpp"

#include <cmath>

using namespace fenix;

TEST(majority_filter_removes_speckle) {
    const s64 s = 16;
    Volume<u8> m = Volume<u8>::zeros({s, s, s});
    // a solid block
    for (s64 z = 4; z < 12; ++z)
        for (s64 y = 4; y < 12; ++y)
            for (s64 x = 4; x < 12; ++x) m(z, y, x) = 1;
    // an isolated speckle
    m(1, 1, 1) = 1;
    auto f = geom::majority_filter(m.view(), 1, 14);
    CHECK(f(1, 1, 1) == 0);  // lone speckle removed
    CHECK(f(8, 8, 8) == 1);  // solid interior kept
}

TEST(dilate_erode_grow_shrink) {
    const s64 s = 16;
    Volume<u8> m = Volume<u8>::zeros({s, s, s});
    m(8, 8, 8) = 1;
    auto d = geom::dilate(m.view(), geom::Conn::Six);
    CHECK(d(8, 8, 9) == 1);  // grew to 6-neighbours
    CHECK(d(8, 8, 8) == 1);
    auto e = geom::erode(d.view(), geom::Conn::Six);
    CHECK(e(8, 8, 8) == 1);  // erosion of the dilated single point restores center
    CHECK(e(8, 8, 9) == 0);
}

TEST(dice_iou_basic) {
    const s64 s = 8;
    Volume<u8> a = Volume<u8>::zeros({s, s, s});
    Volume<u8> b = Volume<u8>::zeros({s, s, s});
    for (s64 i = 0; i < 100; ++i) a.flat()[static_cast<usize>(i)] = 1;        // first 100
    for (s64 i = 50; i < 150; ++i) b.flat()[static_cast<usize>(i)] = 1;       // overlap 50
    // |A|=100,|B|=100, inter=50 -> dice=100/200=0.5, iou=50/150
    CHECK(std::abs(eval::dice(a.view(), b.view()) - 0.5) < 1e-9);
    CHECK(std::abs(eval::iou(a.view(), b.view()) - (50.0 / 150.0)) < 1e-9);
    CHECK(std::abs(eval::dice(a.view(), a.view()) - 1.0) < 1e-9);  // identical
}

TEST(voi_zero_for_identical_partition) {
    const s64 s = 8;
    Volume<s32> seg = Volume<s32>::zeros({s, s, s});
    for (s64 i = 0; i < seg.size(); ++i)
        seg.flat()[static_cast<usize>(i)] = static_cast<s32>(1 + (i / 64));  // blocks of 64
    auto v_same = eval::voi(seg.view(), seg.view());
    CHECK(v_same.total() < 1e-9);  // identical partition -> VOI 0

    // Merge two gt labels in seg -> nonzero (under-segmentation: merge term up).
    Volume<s32> merged = Volume<s32>::zeros({s, s, s});
    for (s64 i = 0; i < merged.size(); ++i) merged.flat()[static_cast<usize>(i)] = 1;  // all one label
    auto v_merge = eval::voi(merged.view(), seg.view());
    CHECK(v_merge.total() > 0.1);
}

// Directional check for the merge/split swap fix (metrics.hpp:72-73): seg = a single label covering
// gt's whole footprint is a pure MERGE (seg under-segments gt into one blob) -> merge > 0, split ~ 0.
// The transposed call (seg=many labels vs gt=one) is a pure SPLIT -> split > 0, merge ~ 0.
TEST(voi_split_merge_direction) {
    const s64 s = 8;
    Volume<s32> many = Volume<s32>::zeros({s, s, s});
    for (s64 i = 0; i < many.size(); ++i) many.flat()[static_cast<usize>(i)] = static_cast<s32>(1 + (i / 64));
    Volume<s32> one = Volume<s32>::zeros({s, s, s});
    for (s64 i = 0; i < one.size(); ++i) one.flat()[static_cast<usize>(i)] = 1;

    // seg=one (a single blob) vs gt=many (several true labels): seg under-segments -> pure merge.
    auto v_merge = eval::voi(one.view(), many.view());
    CHECK(v_merge.merge > 0.1);
    CHECK(v_merge.split < 1e-9);

    // Transposed: seg=many vs gt=one: seg over-segments -> pure split.
    auto v_split = eval::voi(many.view(), one.view());
    CHECK(v_split.split > 0.1);
    CHECK(v_split.merge < 1e-9);
}

// Same directional check for score.hpp's voi_union (union-foreground variant): pred=one label vs
// gt=many is a pure merge; the transpose is a pure split.
TEST(voi_union_split_merge_direction) {
    const s64 s = 8;
    Volume<u8> many = Volume<u8>::zeros({s, s, s});
    for (s64 i = 0; i < many.size(); ++i) many.flat()[static_cast<usize>(i)] = 1;  // foreground everywhere
    // Carve `many`'s underlying label structure via disjoint blocks separated by a background gap so
    // connected_components sees multiple components; `one` is fully connected (one component).
    Volume<u8> one = Volume<u8>::zeros({s, s, s});
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x) {
                one.view()(z, y, x) = 1;
                many.view()(z, y, x) = (x < 3 || x > 4) ? 1 : 0;  // gap at x=3,4 -> 2 components
            }

    auto vu_merge = eval::voi_union(one.view(), many.view());  // pred=one vs gt=many -> merge
    CHECK(vu_merge.merge > 1e-9);
    CHECK(vu_merge.merge > vu_merge.split);

    auto vu_split = eval::voi_union(many.view(), one.view());  // transpose -> split
    CHECK(vu_split.split > 1e-9);
    CHECK(vu_split.split > vu_split.merge);
}
