// test_postproc.cpp — dust removal + hole filling (verified via topology).
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "postproc/cleanup.hpp"
#include "topo/betti.hpp"

using namespace fenix;

TEST(remove_small_components_drops_dust) {
    const s64 s = 20;
    Volume<u8> m = Volume<u8>::zeros({s, s, s});
    for (s64 z = 4; z < 14; ++z)
        for (s64 y = 4; y < 14; ++y)
            for (s64 x = 4; x < 14; ++x) m(z, y, x) = 1;  // big blob (1000 voxels)
    m(0, 0, 0) = 1;                                       // dust (1 voxel)
    m(0, 0, 2) = 1;                                       // dust (1 voxel)
    auto cleaned = postproc::remove_small_components(m.view(), 50);
    CHECK(cleaned(8, 8, 8) == 1);   // blob kept
    CHECK(cleaned(0, 0, 0) == 0);   // dust gone
    CHECK(cleaned(0, 0, 2) == 0);
}

TEST(fill_holes_removes_cavity) {
    const s64 s = 28;
    Volume<u8> m = Volume<u8>::zeros({s, s, s});
    const f32 c = 14.0f;
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x) {
                f32 r2 = (static_cast<f32>(z) - c) * (static_cast<f32>(z) - c) +
                         (static_cast<f32>(y) - c) * (static_cast<f32>(y) - c) +
                         (static_cast<f32>(x) - c) * (static_cast<f32>(x) - c);
                if (r2 <= 121.0f && r2 >= 49.0f) m(z, y, x) = 1;  // hollow shell -> cavity
            }
    CHECK(topo::betti_numbers(m.view()).b2 == 1);  // has a cavity
    auto filled = postproc::fill_holes(m.view());
    topo::Betti b = topo::betti_numbers(filled.view());
    CHECK(b.b2 == 0);  // cavity filled
    CHECK(b.b0 == 1);  // still one component
    CHECK(filled(14, 14, 14) == 1);  // center now solid
}
