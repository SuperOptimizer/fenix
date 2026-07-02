// tests/test_train_substrate.cpp — the torch-free training data plane: patch sampler
// (deterministic, mesh-weighted, valid-only) + GT band rasterizer (thickness/coverage).
#define FENIX_TEST_MAIN
#include "core/test.hpp"
#include "ml/rasterize.hpp"
#include "ml/sampler.hpp"

#include <cmath>

using namespace fenix;

namespace {
// Flat-ish surface at z ~= zc across a [0,D)^3 volume, grid step 20 vox.
Surface make_plane(s64 nu, s64 nv, f32 zc) {
    Surface s(nu, nv);
    s.scale_u = 20.0f;
    s.scale_v = 20.0f;
    for (s64 v = 0; v < nv; ++v)
        for (s64 u = 0; u < nu; ++u)
            s.set(u,
                  v,
                  Vec3f{zc + 2.0f * std::sin(static_cast<f32>(u) * 0.3f),
                        static_cast<f32>(v) * 20.0f,
                        static_cast<f32>(u) * 20.0f});
    return s;
}
}  // namespace

TEST(sampler_is_deterministic_and_valid_only) {
    Surface a = make_plane(32, 32, 100.0f);
    Surface b = make_plane(8, 8, 50.0f);
    for (s64 u = 0; u < 32; ++u)
        for (s64 v = 10; v < 20; ++v) a.valid[a.idx(u, v)] = 0;  // hole
    const Surface* meshes[] = {&a, &b};
    ml::PatchSampler smp(meshes, 42, 8.0f);
    ml::PatchSampler smp2(meshes, 42, 8.0f);
    CHECK(smp.total_weight() == a.valid_count() + b.valid_count());
    int hits_a = 0, hits_b = 0;
    for (u64 i = 0; i < 500; ++i) {
        const auto d = smp.draw(i);
        const auto d2 = smp2.draw(i);
        REQUIRE(d.mesh >= 0);
        CHECK(d.mesh == d2.mesh);
        CHECK(d.center.z == d2.center.z);  // same seed+i => identical draw
        (d.mesh == 0 ? hits_a : hits_b)++;
        // center must be within jitter of SOME valid cell of the drawn mesh
        const Surface& m = d.mesh == 0 ? a : b;
        f32 best = 1e30f;
        for (s64 v = 0; v < m.nv; ++v)
            for (s64 u = 0; u < m.nu; ++u) {
                if (!m.is_valid(u, v)) continue;
                best = std::min(best, norm(m.at(u, v) - d.center));
            }
        CHECK(best <= 8.0f * 1.8f);  // within the jitter ball (sqrt(3)*8, slack)
    }
    // weighting: mesh a has ~14x mesh b's valid cells
    CHECK(hits_a > hits_b * 4);
}

TEST(sampler_different_seed_differs) {
    Surface a = make_plane(16, 16, 64.0f);
    const Surface* meshes[] = {&a};
    ml::PatchSampler s1(meshes, 1), s2(meshes, 2);
    int same = 0;
    for (u64 i = 0; i < 50; ++i)
        if (norm(s1.draw(i).center - s2.draw(i).center) < 1e-3f) ++same;
    CHECK(same < 5);
}

TEST(rasterize_band_thickness_and_coverage) {
    Surface s = make_plane(16, 16, 64.0f);  // spans z~62-66, y/x 0..300
    const Index3 org{32, 40, 40};
    const Extent3 ext{64, 64, 64};
    ml::RasterParams rp;
    rp.thickness = 4.0f;
    Volume<u8> band = ml::rasterize_band(s, org, ext, rp);
    auto bv = band.view();
    s64 on = 0;
    for (s64 z = 0; z < ext.z; ++z)
        for (s64 y = 0; y < ext.y; ++y)
            for (s64 x = 0; x < ext.x; ++x) on += bv(z, y, x) != 0;
    REQUIRE(on > 0);
    // every ON voxel is within thickness/2 (+sampling slack) of the analytic surface
    for (s64 z = 0; z < ext.z; ++z)
        for (s64 y = 0; y < ext.y; ++y)
            for (s64 x = 0; x < ext.x; ++x) {
                if (!bv(z, y, x)) continue;
                const f32 gu = static_cast<f32>(x + org.x) / 20.0f;
                const f32 zs = 64.0f + 2.0f * std::sin(gu * 0.3f);
                CHECK(std::abs(static_cast<f32>(z + org.z) - zs) <= rp.thickness * 0.5f + 1.5f);
            }
    // the surface line through the patch center is covered (no holes along the sheet)
    for (s64 y = 8; y < 56; y += 4)
        for (s64 x = 8; x < 56; x += 4) {
            const f32 gu = static_cast<f32>(x + org.x) / 20.0f;
            const s64 zs = static_cast<s64>(std::lround(64.0f + 2.0f * std::sin(gu * 0.3f))) - org.z;
            bool covered = false;
            for (s64 dz = -1; dz <= 1; ++dz)
                if (zs + dz >= 0 && zs + dz < ext.z && bv(zs + dz, y, x)) covered = true;
            CHECK(covered);
        }
}

TEST(rasterize_respects_validity_and_patch_bounds) {
    Surface s = make_plane(16, 16, 64.0f);
    for (s64 u = 0; u < 16; ++u)
        for (s64 v = 0; v < 16; ++v)
            if (u >= 8) s.valid[s.idx(u, v)] = 0;  // right half invalid (x >= 160)
    Volume<u8> band = ml::rasterize_band(s, {32, 0, 0}, {64, 320, 320}, {.thickness = 2.0f});
    auto bv = band.view();
    s64 on_left = 0, on_right = 0;
    for (s64 z = 0; z < 64; ++z)
        for (s64 y = 0; y < 320; ++y)
            for (s64 x = 0; x < 320; ++x) {
                if (!bv(z, y, x)) continue;
                (x < 145 ? on_left : on_right) += x < 145 || x > 165;  // skip the boundary strip
            }
    CHECK(on_left > 0);
    CHECK(on_right == 0);  // invalid half must stay empty
}
