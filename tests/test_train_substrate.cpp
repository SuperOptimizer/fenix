// tests/test_train_substrate.cpp — the torch-free training data plane: patch sampler
// (deterministic, mesh-weighted, valid-only) + GT band rasterizer (thickness/coverage).
#define FENIX_TEST_MAIN
#include "core/test.hpp"
#include "geom/rtree.hpp"
#include "ml/rasterize.hpp"
#include "ml/sampler.hpp"
#include "ml/surface_index.hpp"

#include <cmath>
#include <cstring>
#include <random>

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

TEST(rtree_matches_brute_force) {
    std::vector<std::pair<geom::Box3f, u32>> items;
    std::mt19937_64 rng(9);
    auto rf = [&](f32 lo, f32 hi) { return lo + (hi - lo) * static_cast<f32>(rng() % 10000) / 10000.0f; };
    for (u32 i = 0; i < 500; ++i) {
        const f32 z = rf(0, 900), y = rf(0, 900), x = rf(0, 900);
        items.push_back({{z, z + rf(1, 60), y, y + rf(1, 60), x, x + rf(1, 60)}, i});
    }
    auto all = items;
    auto tree = geom::BoxRTree::build(std::move(items));
    for (int t = 0; t < 30; ++t) {
        const f32 z = rf(0, 900), y = rf(0, 900), x = rf(0, 900);
        const geom::Box3f q{z, z + 120, y, y + 120, x, x + 120};
        std::vector<u32> got;
        tree.query(q, got);
        std::sort(got.begin(), got.end());
        std::vector<u32> want;
        for (auto& [b, id] : all)
            if (b.intersects(q)) want.push_back(id);
        std::sort(want.begin(), want.end());
        REQUIRE(got == want);
    }
}

TEST(surface_index_finds_meshes_and_rects) {
    Surface a = make_plane(32, 32, 100.0f);  // z~98-102, y/x 0..620
    Surface b = make_plane(32, 32, 400.0f);  // far away in z
    const Surface* ptrs[] = {&a, &b};
    ml::VolumeSurfaceIndex idx(ptrs);
    // box around mesh a only
    auto hits = idx.query({90, 110, 100, 200, 100, 200});
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].mesh == 0u);
    REQUIRE(!hits[0].rects.empty());
    // every returned rect must be near the queried y/x window (cells are 20 vox apart)
    for (auto& r : hits[0].rects) {
        CHECK(r.u1 * 20 >= 100 - ml::SurfaceIndex::kTile * 20);
        CHECK(r.u0 * 20 <= 200 + ml::SurfaceIndex::kTile * 20);
    }
    // box around nothing
    CHECK(idx.query({700, 800, 700, 800, 700, 800}).empty());
    // box catching both meshes
    CHECK(idx.query({0, 500, 0, 620, 0, 620}).size() == 2);
}

TEST(rasterize_multi_union_and_index_equivalence) {
    Surface a = make_plane(16, 16, 60.0f);
    Surface b = make_plane(16, 16, 80.0f);  // second sheet 20 vox deeper
    const Surface* ptrs[] = {&a, &b};
    ml::VolumeSurfaceIndex idx(ptrs);
    const Index3 org{40, 40, 40};
    const Extent3 ext{64, 64, 64};
    const ml::RasterParams rp{.thickness = 3.0f, .shell = 12.0f};
    Volume<u8> with_idx = ml::rasterize_band_multi(ptrs, org, ext, rp, &idx);
    Volume<u8> without = ml::rasterize_band_multi(ptrs, org, ext, rp, nullptr);
    // The index is an accelerator, not a semantic. Fast-math means band-EDGE voxels can
    // round differently between the two call sites (bit-exactness is explicitly not a
    // project guarantee), so compare label-class COUNTS with a tolerance instead of
    // voxelwise equality, plus absolute semantic probes on the indexed output.
    auto count = [&](const Volume<u8>& g, u8 val) {
        s64 n = 0;
        for (u8 x : g.flat()) n += x == val;
        return n;
    };
    const s64 sheet_i = count(with_idx, ml::kLabelSheet), sheet_o = count(without, ml::kLabelSheet);
    const s64 bg_i = count(with_idx, ml::kLabelBackground), bg_o = count(without, ml::kLabelBackground);
    REQUIRE(sheet_o > 0);
    REQUIRE(bg_o > 0);

    CHECK(std::abs(sheet_i - sheet_o) * 20 < sheet_o);  // within 5%
    CHECK(std::abs(bg_i - bg_o) * 20 < bg_o);
    // both sheets present in the indexed output: z=60 -> local 20 (sheet z wobbles ±2 with
    // the sine term, sample the exact analytic z), z=80 -> local 40
    auto v = with_idx.view();
    auto zs = [&](f32 zc, s64 x) {
        return static_cast<s64>(std::lround(zc + 2.0f * std::sin(static_cast<f32>(x + 40) / 20.0f * 0.3f))) - 40;
    };
    CHECK(v(zs(60.0f, 16), 16, 16) == ml::kLabelSheet);
    CHECK(v(zs(80.0f, 16), 16, 16) == ml::kLabelSheet);
    CHECK(v(zs(60.0f, 16) + 4, 16, 16) == ml::kLabelBackground);  // 4 vox off-sheet: inside shell/2=6
    s64 unl = 0;
    for (u8 x : with_idx.flat()) unl += x == ml::kLabelUnknown;
    CHECK(unl > 0);  // voxels far from both sheets stay unlabeled (e.g. local z~30, 10 from each)
}

TEST(sampler_locality_clusters_share_neighborhood) {
    Surface a = make_plane(64, 64, 100.0f);  // spans ~1260^2 in y/x
    const Surface* meshes[] = {&a};
    ml::PatchSampler smp(meshes, 7, 32.0f, /*locality=*/8, /*spread=*/192.0f);
    ml::PatchSampler smp2(meshes, 7, 32.0f, 8, 192.0f);
    // determinism holds with locality on
    for (u64 i = 0; i < 40; ++i) CHECK(norm(smp.draw(i).center - smp2.draw(i).center) < 1e-3f);
    // members of one cluster stay within spread*sqrt(3)+slack of each other...
    for (u64 c = 0; c < 5; ++c) {
        const Vec3f c0 = smp.draw(c * 8).center;
        for (u64 k = 1; k < 8; ++k)
            CHECK(norm(smp.draw(c * 8 + k).center - c0) <= 2.0f * 192.0f * 1.8f);
    }
    // ...while distinct clusters spread across the mesh (not all in one spot)
    f32 maxd = 0;
    for (u64 c = 1; c < 12; ++c) maxd = std::max(maxd, norm(smp.draw(c * 8).center - smp.draw(0).center));
    CHECK(maxd > 300.0f);
}

TEST(trust_grid_roundtrip_and_gating) {
    // write a trust file by hand (the surf-qc regions= format), read it back
    const std::string tp = "/tmp/fenix_test_trust.txt";
    {
        std::FILE* f = std::fopen(tp.c_str(), "w");
        REQUIRE(f != nullptr);
        // 16x16 uv grid, tile=8 -> 2x2 tiles; fail the (ti=1, tj=0) tile (u>=8, v<8)
        std::fprintf(f, "fxtrust1 16 16 8\nPF\nP?\n");
        std::fclose(f);
    }
    auto g = ml::read_trust(tp);
    REQUIRE(g.has_value());
    CHECK(g->tu == 2);
    CHECK(g->tv == 2);
    CHECK(!g->untrusted(0, 0));
    CHECK(g->untrusted(9, 3));    // 'F' tile
    CHECK(!g->untrusted(3, 9));   // 'P'
    CHECK(!g->untrusted(12, 12)); // '?' counts as trusted (unknown must not shrink coverage)

    // gating: same plane rasterized with/without the trust grid — the failed tile's
    // voxels must stay unlabeled, trusted tiles unchanged
    Surface s = make_plane(16, 16, 64.0f);
    const Surface* one[] = {&s};
    const ml::TrustGrid* tg[] = {&*g};
    const Index3 org{32, 0, 0};
    const Extent3 ext{64, 320, 320};
    Volume<u8> plain = ml::rasterize_band_multi(one, org, ext, {.thickness = 4.0f, .shell = 0}, nullptr);
    Volume<u8> gated = ml::rasterize_band_multi(one, org, ext, {.thickness = 4.0f, .shell = 0}, nullptr, {}, tg);
    auto pv = plain.view();
    auto gv = gated.view();
    s64 n_plain = 0, n_gated = 0, lost_in_fail = 0, lost_in_pass = 0;
    for (s64 z = 0; z < ext.z; ++z)
        for (s64 y = 0; y < ext.y; ++y)
            for (s64 x = 0; x < ext.x; ++x) {
                n_plain += pv(z, y, x) == ml::kLabelSheet;
                n_gated += gv(z, y, x) == ml::kLabelSheet;
                if (pv(z, y, x) == ml::kLabelSheet && gv(z, y, x) != ml::kLabelSheet) {
                    // grid step 20 vox: u ~ x/20, v ~ y/20; the failed tile is u in [8,16), v in [0,8)
                    const bool in_fail_tile = x >= 8 * 20 - 4 && y < 8 * 20 + 4;
                    (in_fail_tile ? lost_in_fail : lost_in_pass)++;
                }
            }
    REQUIRE(n_plain > 0);
    CHECK(n_gated < n_plain);       // something was gated out
    CHECK(lost_in_fail > 0);        // and it was the failed tile
    CHECK(lost_in_pass == 0);       // trusted tiles untouched
    std::remove(tp.c_str());
}
