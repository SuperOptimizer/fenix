// test_marching.cpp — marching tetrahedra isosurface on a sphere SDF.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "geom/marching.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::geom;

TEST(marching_tetrahedra_sphere) {
    const s64 s = 32;
    const f32 c = 16.0f, R = 10.0f;
    // Signed-distance-ish field: value = radius - R, iso = 0 -> sphere of radius R.
    Volume<f32> f(Extent3{s, s, s});
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x) {
                f32 r = std::sqrt((static_cast<f32>(z) - c) * (static_cast<f32>(z) - c) +
                                  (static_cast<f32>(y) - c) * (static_cast<f32>(y) - c) +
                                  (static_cast<f32>(x) - c) * (static_cast<f32>(x) - c));
                f(z, y, x) = r - R;
            }
    Mesh m = marching_tetrahedra(f.view(), 0.0f);
    REQUIRE(m.tri_count() > 100);  // a sphere produces many triangles

    // Every vertex lies (near) on the sphere of radius R about the centre.
    f32 max_dev = 0;
    for (const Vec3f& v : m.vertices) {
        f32 r = std::sqrt((v.z - c) * (v.z - c) + (v.y - c) * (v.y - c) + (v.x - c) * (v.x - c));
        max_dev = std::max(max_dev, std::abs(r - R));
    }
    CHECK(max_dev < 1.0f);  // on the iso surface (sub-voxel)
}

TEST(marching_empty_when_iso_outside_range) {
    const s64 s = 16;
    Volume<f32> f(Extent3{s, s, s});
    for (s64 i = 0; i < f.size(); ++i) f.flat()[static_cast<usize>(i)] = 5.0f;  // all 5
    Mesh m = marching_tetrahedra(f.view(), 100.0f);  // iso never crossed
    CHECK(m.tri_count() == 0);
}
