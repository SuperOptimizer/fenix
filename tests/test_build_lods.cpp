// test_build_lods.cpp — out-of-core LOD pyramid (VolumeArchive::build_pyramid_ooc) vs the in-core
// write_volume reference. A LOD0-only sparse archive (air chunks left ABSENT, like export-scroll) +
// build_pyramid_ooc must reproduce write_volume's pyramid: same #LODs, same per-LOD dims, and each level
// close in value (build_pyramid_ooc cascades on the DECODED level below, a small extra q16 loss).
#define FENIX_TEST_MAIN
#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

#include <cmath>
#include <filesystem>
#include <vector>

using namespace fenix;
using namespace fenix::codec;

static Volume<u8> gradient_sphere(s64 N) {  // gradient sphere (never 0) in air → absent corners + averaging
    Volume<u8> V = Volume<u8>::zeros({N, N, N});
    auto v = V.view();
    const f64 c = static_cast<f64>(N) / 2.0, R = static_cast<f64>(N) * 0.36;
    for (s64 z = 0; z < N; ++z)
        for (s64 y = 0; y < N; ++y)
            for (s64 x = 0; x < N; ++x)
                if (std::sqrt((z - c) * (z - c) + (y - c) * (y - c) + (x - c) * (x - c)) < R)
                    v(z, y, x) = static_cast<u8>(30 + ((x + 2 * y + 3 * z) % 200));
    return V;
}

static f64 psnr_between(const Volume<f32>& a, const Volume<f32>& b) {
    const Extent3 d = a.dims();
    auto av = a.view(), bv = b.view();
    f64 sse = 0;
    s64 n = 0;
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const f64 e = static_cast<f64>(av(z, y, x)) - bv(z, y, x);
                sse += e * e;
                ++n;
            }
    const f64 mse = sse / static_cast<f64>(n ? n : 1);
    return mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
}

TEST(build_lods_matches_write_volume) {
    const s64 N = 200, cs = fxvol_chunk_side;
    const Extent3 D{N, N, N};
    Volume<u8> V = gradient_sphere(N);
    auto vv = V.view();
    const auto dir = std::filesystem::temp_directory_path();
    const std::string refp = (dir / "fenix_bl_ref.fxvol").string();
    const std::string tstp = (dir / "fenix_bl_test.fxvol").string();

    // REF: full in-core pyramid.
    {
        auto a = VolumeArchive::create(refp, D, DctParams{.q = 16.0f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(vv).has_value());
        REQUIRE(a->close().has_value());
    }
    // TEST: LOD0 only, air chunks left ABSENT (mimics export-scroll), then out-of-core pyramid.
    {
        auto ar = VolumeArchive::create(tstp, D, DctParams{.q = 16.0f});
        REQUIRE(ar.has_value());
        VolumeArchive a = std::move(*ar);
        const ChunkCoord ce = a.chunk_extent(0);
        std::vector<u8> blk(static_cast<usize>(cs * cs * cs));
        for (s64 cz = 0; cz < ce.z; ++cz)
            for (s64 cy = 0; cy < ce.y; ++cy)
                for (s64 cx = 0; cx < ce.x; ++cx) {
                    bool air = true;
                    for (s64 z = 0; z < cs; ++z)
                        for (s64 y = 0; y < cs; ++y)
                            for (s64 x = 0; x < cs; ++x) {
                                const s64 vz = std::min(cz * cs + z, N - 1), vy = std::min(cy * cs + y, N - 1),
                                          vx = std::min(cx * cs + x, N - 1);
                                const u8 val = vv(vz, vy, vx);
                                blk[static_cast<usize>((z * cs + y) * cs + x)] = val;
                                if (val != 0) air = false;
                            }
                    if (air) continue;  // leave ABSENT
                    REQUIRE(a.write_chunk(0, {cz, cy, cx}, std::span<const u8>(blk)).has_value());
                }
        REQUIRE(a.commit().has_value());
        REQUIRE(a.build_pyramid_ooc(64, [](s64, s64, s64) {}).has_value());
        REQUIRE(a.close().has_value());
    }

    auto ref = VolumeArchive::open(refp);
    auto tst = VolumeArchive::open(tstp);
    REQUIRE(ref.has_value());
    REQUIRE(tst.has_value());
    CHECK(ref->nlods() == tst->nlods());
    CHECK(tst->nlods() > 1);  // pyramid was actually built
    for (s64 lod = 0; lod < static_cast<s64>(tst->nlods()); ++lod) {
        auto ra = ref->read_volume(lod), ta = tst->read_volume(lod);
        REQUIRE(ra.has_value());
        REQUIRE(ta.has_value());
        CHECK(ra->dims() == ta->dims());
        const f64 p = psnr_between(*ra, *ta);
        CHECK(p > (lod == 0 ? 40.0 : 30.0));  // LOD0 identical encode; coarser levels within cascade tolerance
    }
    // a coarse LOD must carry real (non-absent) data, not be all-skipped
    CHECK(tst->coverage(tst->nlods() - 1, {0, 0, 0}) != Coverage::Absent);

    std::error_code ec;
    std::filesystem::remove(refp, ec);
    std::filesystem::remove(tstp, ec);
}
