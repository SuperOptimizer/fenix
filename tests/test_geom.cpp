// test_geom.cpp — exact EDT + connected components.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "geom/connected_components.hpp"
#include "geom/edt.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::geom;

TEST(edt_single_seed_is_exact_squared_distance) {
    const s64 s = 16;
    Volume<u8> seed = Volume<u8>::zeros({s, s, s});
    seed(8, 8, 8) = 1;
    auto d = edt_squared(seed.view());
    // squared distance to (8,8,8)
    CHECK(d(8, 8, 8) == 0.0f);
    CHECK(std::abs(d(8, 8, 10) - 4.0f) < 1e-3f);          // dx=2 -> 4
    CHECK(std::abs(d(5, 8, 8) - 9.0f) < 1e-3f);           // dz=3 -> 9
    CHECK(std::abs(d(8, 11, 12) - (9.0f + 16.0f)) < 1e-3f);  // dy=3,dx=4 -> 25
}

TEST(edt_two_seeds_takes_nearest) {
    const s64 s = 20;
    Volume<u8> seed = Volume<u8>::zeros({s, s, s});
    seed(0, 0, 0) = 1;
    seed(0, 0, 19) = 1;
    auto d = edt_squared(seed.view());
    // voxel near the right seed -> small distance to it, not the left one
    CHECK(std::abs(d(0, 0, 17) - 4.0f) < 1e-3f);  // 2 from x=19
}

TEST(cc_counts_separate_blobs) {
    const s64 s = 16;
    Volume<u8> m = Volume<u8>::zeros({s, s, s});
    // blob A
    for (s64 z = 1; z < 4; ++z)
        for (s64 y = 1; y < 4; ++y)
            for (s64 x = 1; x < 4; ++x) m(z, y, x) = 1;
    // blob B, far away
    for (s64 z = 10; z < 13; ++z)
        for (s64 y = 10; y < 13; ++y)
            for (s64 x = 10; x < 13; ++x) m(z, y, x) = 1;
    auto r = connected_components(m.view(), Conn::Six);
    CHECK(r.count == 2);
    CHECK(r.labels(2, 2, 2) != 0);
    CHECK(r.labels(11, 11, 11) != 0);
    CHECK(r.labels(2, 2, 2) != r.labels(11, 11, 11));  // distinct components
    CHECK(r.labels(7, 7, 7) == 0);                      // background
}

TEST(cc_diagonal_only_joins_under_26) {
    const s64 s = 8;
    Volume<u8> m = Volume<u8>::zeros({s, s, s});
    m(2, 2, 2) = 1;
    m(3, 3, 3) = 1;  // touches only at a corner
    CHECK(connected_components(m.view(), Conn::Six).count == 2);
    CHECK(connected_components(m.view(), Conn::TwentySix).count == 1);
}
