// test_nsd.cpp — Normalized Surface Dice.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "eval/nsd.hpp"

using namespace fenix;

static Volume<u8> solid_box(s64 s, s64 lo, s64 hi) {
    Volume<u8> m = Volume<u8>::zeros({s, s, s});
    for (s64 z = lo; z < hi; ++z)
        for (s64 y = lo; y < hi; ++y)
            for (s64 x = lo; x < hi; ++x) m(z, y, x) = 1;
    return m;
}

TEST(nsd_identical_is_one) {
    auto a = solid_box(24, 6, 18);
    CHECK(std::abs(eval::nsd(a.view(), a.view(), 0.0f) - 1.0) < 1e-9);
    CHECK(std::abs(eval::nsd(a.view(), a.view(), 2.0f) - 1.0) < 1e-9);
}

TEST(nsd_shifted_box_tolerance_curve) {
    auto a = solid_box(24, 6, 18);
    auto b = solid_box(24, 7, 19);  // shifted by 1 voxel
    f64 n0 = eval::nsd(a.view(), b.view(), 0.0f);
    f64 n1 = eval::nsd(a.view(), b.view(), 1.5f);
    f64 n3 = eval::nsd(a.view(), b.view(), 3.0f);
    CHECK(n0 < 1.0);          // exact match imperfect under a 1-voxel shift
    CHECK(n1 > n0);           // looser tolerance -> higher NSD
    CHECK(n3 >= n1);
    CHECK(n3 > 0.9);          // within 3 voxels nearly all surface matches
}

TEST(nsd_disjoint_is_low) {
    auto a = solid_box(40, 2, 8);
    auto b = solid_box(40, 30, 36);  // far apart
    CHECK(eval::nsd(a.view(), b.view(), 1.0f) < 0.05);
}
