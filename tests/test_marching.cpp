// test_marching.cpp — marching tetrahedra isosurface on a sphere SDF.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "geom/marching.hpp"

#include <cmath>
#include <map>
#include <unordered_map>

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

// The header's headline claim ("watertight by construction"): extracting a closed implicit
// surface (a sphere SDF, fully interior to the volume so no boundary cuts) must produce a mesh
// where every (undirected) edge is shared by EXACTLY 2 triangles — never 1 (a hole/crack) and
// never >2 (a double-cover/bowtie). This is exactly the case the wrong-diagonal bug (marching.hpp:
// emit(c0,c1,c2)+emit(c0,c2,c3) instead of the true diagonal c0-c3) broke on every 2-2 sign-split
// tet: 24-38% hole area + 12-26% double-covered area per affected tet (per the review). NOTE:
// per-triangle winding consistency (a separate, lower-severity concern the review flagged as a
// follow-up — neither the pre-fix nor post-fix code orients triangles by field sign) is NOT
// asserted here; this test only locks the watertightness property the diagonal fix restores.
TEST(marching_tetrahedra_sphere_is_watertight) {
    const s64 s = 24;
    const f32 c = 12.0f, R = 8.0f;  // fully interior: R+1 < c, so no boundary tets are cut
    Volume<f32> f(Extent3{s, s, s});
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x) {
                const f32 r = std::sqrt((static_cast<f32>(z) - c) * (static_cast<f32>(z) - c) +
                                        (static_cast<f32>(y) - c) * (static_cast<f32>(y) - c) +
                                        (static_cast<f32>(x) - c) * (static_cast<f32>(x) - c));
                f(z, y, x) = r - R;
            }
    Mesh m = marching_tetrahedra(f.view(), 0.0f);
    REQUIRE(m.tri_count() > 100);

    // Weld coincident vertices (position-keyed, tolerance-quantized per tests/CLAUDE.md — never
    // bit-exact under fast-math) so shared crossings on shared tet edges collapse to one vertex id.
    auto qkey = [](Vec3f v) -> std::tuple<s64, s64, s64> {
        constexpr f32 q = 1024.0f;  // quantize to ~1/1024 voxel — far finer than any crossing jitter
        return {static_cast<s64>(std::lround(v.z * q)), static_cast<s64>(std::lround(v.y * q)),
                static_cast<s64>(std::lround(v.x * q))};
    };
    std::map<std::tuple<s64, s64, s64>, s32> weld;
    std::vector<s32> remap(m.vertices.size());
    for (usize i = 0; i < m.vertices.size(); ++i) {
        auto key = qkey(m.vertices[i]);
        auto [it, inserted] = weld.try_emplace(key, static_cast<s32>(i));
        remap[i] = it->second;
    }

    // Count each UNDIRECTED edge across all triangles (canonicalized a<b so both orientations of
    // the same edge collide). Marching tetrahedra can (independent of the diagonal-choice bug this
    // test targets) emit a rare zero-area triangle when a crossing lands exactly on a shared tet
    // vertex/edge — skip those rather than asserting they can't occur; they contribute no edges
    // either way so they don't affect the watertightness count.
    std::unordered_map<u64, int> edge_count;
    auto ekey = [](s32 a, s32 b) -> u64 {
        if (a > b) std::swap(a, b);
        return (static_cast<u64>(static_cast<u32>(a)) << 32) | static_cast<u32>(b);
    };
    s64 degenerate = 0;
    for (const auto& t : m.tris) {
        const s32 a = remap[static_cast<usize>(t[0])], b = remap[static_cast<usize>(t[1])],
                  cc = remap[static_cast<usize>(t[2])];
        if (a == b || b == cc || a == cc) { ++degenerate; continue; }
        ++edge_count[ekey(a, b)];
        ++edge_count[ekey(b, cc)];
        ++edge_count[ekey(cc, a)];
    }
    CHECK(static_cast<f64>(degenerate) < 0.01 * static_cast<f64>(m.tri_count()));  // rare, not systemic
    s64 bad = 0;
    for (const auto& [key, cnt] : edge_count)
        if (cnt != 2) ++bad;  // closed 2-manifold: every edge borders exactly 2 triangles
    CHECK(bad == 0);
}
