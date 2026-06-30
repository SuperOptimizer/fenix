// test_fxvol.cpp — .fxvol v4 container (mmap'd 3-level Morton radix page-table). Exercises multi-chunk
// addressing across many radix leaves/nodes, the sparse ABSENT/ZERO/REAL tri-state, persistence across
// close/reopen, full write_volume/read_volume, and robustness on garbage/truncated files (no crash —
// fuzzed-in-spirit; the hard "no UB on any bytes" rule). See ADR 0006 + docs/design/fxvol-v4-layout.md.
#define FENIX_TEST_MAIN
#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

#include <cmath>
#include <filesystem>
#include <vector>

using namespace fenix;
using namespace fenix::codec;

// A coord-seeded smooth+noise chunk_side³ block (distinct per chunk, so collisions would show as errors).
static std::vector<f32> chunk_pattern(u32 seed) {
    const s64 s = fxvol_chunk_side;
    std::vector<f32> b(static_cast<usize>(s * s * s));
    Pcg32 rng{seed};
    const f32 phase = static_cast<f32>(seed % 17);
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x)
                b[static_cast<usize>((z * s + y) * s + x)] =
                    60.0f + 50.0f * std::sin(0.05f * static_cast<f32>(x) + phase) + 4.0f * rng.next_f32();
    return b;
}

TEST(fxvol_multichunk_morton_sparse) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_fxvol_mc.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);

    Extent3 dims{128, 192, 256};  // chunk extent 2×3×4 = 24 chunks → multiple radix leaves
    const ChunkCoord ce{2, 3, 4};

    // REAL at a scattered subset, ZERO at another, the rest left ABSENT. Seed = a unique chunk id.
    auto cid = [&](s64 z, s64 y, s64 x) { return static_cast<u32>((z * ce.y + y) * ce.x + x + 1); };
    std::vector<ChunkCoord> reals{{0, 0, 0}, {1, 2, 3}, {0, 1, 2}, {1, 0, 0}, {0, 2, 3}};
    std::vector<ChunkCoord> zeros{{1, 1, 1}, {0, 0, 3}};

    {
        auto a = VolumeArchive::create(path, dims, {.q = 4.0f});
        REQUIRE(a.has_value());
        for (auto c : reals) {
            auto blk = chunk_pattern(cid(c.z, c.y, c.x));
            REQUIRE(a->write_chunk(c, blk).has_value());
        }
        std::vector<f32> air(static_cast<usize>(fxvol_chunk_side * fxvol_chunk_side * fxvol_chunk_side), 0.0f);
        for (auto c : zeros) REQUIRE(a->write_chunk(c, air).has_value());
        REQUIRE(a->close().has_value());
    }

    auto a = VolumeArchive::open(path);
    REQUIRE(a.has_value());
    CHECK(a->dims() == dims);
    CHECK(a->chunk_extent() == ce);

    // coverage tri-state is correct per chunk (no Morton collisions)
    for (auto c : reals) CHECK(a->coverage(c) == Coverage::Real);
    for (auto c : zeros) CHECK(a->coverage(c) == Coverage::Zero);
    CHECK(a->coverage({1, 2, 2}) == Coverage::Absent);
    CHECK(a->coverage({0, 1, 1}) == Coverage::Absent);

    // each REAL chunk round-trips to ITS OWN pattern (a collision would blow up max_err)
    for (auto c : reals) {
        auto got = a->read_chunk(c);
        REQUIRE(got.has_value());
        auto want = chunk_pattern(cid(c.z, c.y, c.x));
        f32 me = 0;
        for (usize i = 0; i < want.size(); ++i) me = std::max(me, std::abs((*got)[i] - want[i]));
        CHECK(me < 20.0f);
    }
    auto z = a->read_chunk(zeros[0]);
    REQUIRE(z.has_value());
    CHECK((*z)[1000] == 0.0f);
    auto ab = a->read_chunk({1, 2, 2});
    REQUIRE(ab.has_value());
    CHECK((*ab)[0] == 0.0f);

    std::filesystem::remove(path);
}

TEST(fxvol_write_read_volume) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_fxvol_vol.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);

    Extent3 dims{96, 128, 64};  // not chunk-aligned in z (96) → edge replication exercised
    Volume<f32> vol = Volume<f32>::zeros(dims);
    VolumeView<f32> vv = vol.view();
    for (s64 zz = 0; zz < dims.z; ++zz)
        for (s64 yy = 0; yy < dims.y; ++yy)
            for (s64 xx = 0; xx < dims.x; ++xx)
                vv(zz, yy, xx) = 50.0f + 40.0f * std::sin(0.03f * static_cast<f32>(xx + yy));

    {
        auto a = VolumeArchive::create(path, dims, {.q = 2.0f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume(vol.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    auto a = VolumeArchive::open(path);
    REQUIRE(a.has_value());
    auto rt = a->read_volume();
    REQUIRE(rt.has_value());
    VolumeView<const f32> rv = rt->view();
    f64 sse = 0;
    f32 me = 0;
    for (s64 zz = 0; zz < dims.z; ++zz)
        for (s64 yy = 0; yy < dims.y; ++yy)
            for (s64 xx = 0; xx < dims.x; ++xx) {
                const f32 e = std::abs(rv(zz, yy, xx) - vv(zz, yy, xx));
                sse += static_cast<f64>(e) * e;
                me = std::max(me, e);
            }
    const f64 psnr = 10.0 * std::log10(255.0 * 255.0 / (sse / static_cast<f64>(dims.z * dims.y * dims.x)));
    CHECK(psnr > 40.0);  // q=2 high fidelity
    CHECK(me < 15.0f);
    std::filesystem::remove(path);
}

TEST(fxvol_robust_open_bad_bytes) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_fxvol_bad.fxvol";
    const std::string path = tmp.string();
    // garbage (bad magic) → clean error, no crash
    {
        std::vector<u8> junk(4096);
        Pcg32 rng{99};
        for (auto& b : junk) b = static_cast<u8>(rng.next_u32());
        FILE* f = std::fopen(path.c_str(), "wb");
        REQUIRE(f != nullptr);
        std::fwrite(junk.data(), 1, junk.size(), f);
        std::fclose(f);
    }
    CHECK(!VolumeArchive::open(path).has_value());  // bad magic rejected
    // truncated (< superblock) → clean error
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        REQUIRE(f != nullptr);
        u8 tiny[16] = {};
        std::fwrite(tiny, 1, sizeof tiny, f);
        std::fclose(f);
    }
    CHECK(!VolumeArchive::open(path).has_value());
    std::filesystem::remove(path);
}
