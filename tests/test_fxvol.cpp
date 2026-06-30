// test_fxvol.cpp — .fxvol v4 container (mmap'd 3-level Morton radix page-table). Exercises multi-chunk
// addressing across many radix leaves/nodes, the sparse ABSENT/ZERO/REAL tri-state, persistence across
// close/reopen, full write_volume/read_volume, and robustness on garbage/truncated files (no crash —
// fuzzed-in-spirit; the hard "no UB on any bytes" rule). See ADR 0006 + docs/design/fxvol-v4-layout.md.
#define FENIX_TEST_MAIN
#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

using namespace fenix;
using namespace fenix::codec;

static void poke(const std::string& path, u64 off, const void* data, usize n) {
    FILE* f = std::fopen(path.c_str(), "r+b");
    if (!f) return;
    std::fseek(f, static_cast<long>(off), SEEK_SET);
    std::fwrite(data, 1, n, f);
    std::fclose(f);
}
static u64 peek_u64(const std::string& path, u64 off) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return 0;
    std::fseek(f, static_cast<long>(off), SEEK_SET);
    u64 v = 0;
    (void)std::fread(&v, 1, 8, f);
    std::fclose(f);
    return v;
}

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

TEST(fxvol_commit_checkpoint_persist) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_fxvol_commit.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);
    Extent3 dims{64, 64, 128};  // chunk extent 1×1×2
    auto x = chunk_pattern(11), y = chunk_pattern(22);
    {
        auto a = VolumeArchive::create(path, dims, {.q = 4.0f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_chunk({0, 0, 0}, x).has_value());
        REQUIRE(a->commit().has_value());  // mid-session checkpoint
        REQUIRE(a->write_chunk({0, 0, 1}, y).has_value());
        REQUIRE(a->close().has_value());
    }
    auto a = VolumeArchive::open(path);
    REQUIRE(a.has_value());
    CHECK(a->coverage({0, 0, 0}) == Coverage::Real);
    CHECK(a->coverage({0, 0, 1}) == Coverage::Real);  // committed after the checkpoint, still durable
    std::filesystem::remove(path);
}

TEST(fxvol_double_buffer_recovery) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_fxvol_recover.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);
    Extent3 dims{64, 64, 64};
    auto x = chunk_pattern(7);
    {
        auto a = VolumeArchive::create(path, dims, {.q = 4.0f});  // → superblock seq 1 in slot A (off 0)
        REQUIRE(a.has_value());
        REQUIRE(a->write_chunk({0, 0, 0}, x).has_value());
        REQUIRE(a->close().has_value());  // commit seq 2 in slot B (off 4096)
    }
    // normal reopen sees the latest commit (slot B)
    {
        auto a = VolumeArchive::open(path);
        REQUIRE(a.has_value());
        CHECK(a->coverage({0, 0, 0}) == Coverage::Real);
    }
    // corrupt the LATEST superblock slot's crc (slot B @ 4096+68) → recovery falls back to slot A (seq 1,
    // the empty pre-write commit): reopen succeeds, dims valid, the chunk reads ABSENT, NO crash.
    {
        const u32 garbage = 0xdeadbeefu;
        poke(path, 4096 + 68, &garbage, 4);
        auto a = VolumeArchive::open(path);
        REQUIRE(a.has_value());
        CHECK(a->dims() == dims);
        CHECK(a->coverage({0, 0, 0}) == Coverage::Absent);
    }
    // corrupt slot A too → no valid superblock → clean error (no crash)
    {
        const u32 garbage = 0xdeadbeefu;
        poke(path, 0 + 68, &garbage, 4);
        CHECK(!VolumeArchive::open(path).has_value());
    }
    std::filesystem::remove(path);
}

TEST(fxvol_blob_crc_detects_corruption) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_fxvol_blobcrc.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);
    Extent3 dims{64, 64, 64};
    auto x = chunk_pattern(3);
    {
        auto a = VolumeArchive::create(path, dims, {.q = 4.0f});  // seq 1 → slot A
        REQUIRE(a.has_value());
        REQUIRE(a->write_chunk({0, 0, 0}, x).has_value());
        REQUIRE(a->close().has_value());  // seq 2 → slot B (latest); committed_eof at 4096+16
    }
    // the single chunk's blob is the last allocation; its payload sits just before committed_eof (then a
    // u32 crc). Flip a payload byte 8 below committed_eof → the per-blob crc32c must reject the read.
    const u64 ceof = peek_u64(path, 4096 + 16);
    REQUIRE(ceof > 4096 + 8);
    u8 b = 0;
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        std::fseek(f, static_cast<long>(ceof - 8), SEEK_SET);
        (void)std::fread(&b, 1, 1, f);
        std::fclose(f);
    }
    b ^= 0xffu;
    poke(path, ceof - 8, &b, 1);
    auto a = VolumeArchive::open(path);
    REQUIRE(a.has_value());
    CHECK(a->coverage({0, 0, 0}) == Coverage::Real);   // the slot still says REAL
    CHECK(!a->read_chunk({0, 0, 0}).has_value());       // but the blob crc mismatch is caught (no garbage/crash)
    std::filesystem::remove(path);
}

TEST(fxvol_block_cache) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_fxvol_cache.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);
    Extent3 dims{128, 128, 128};  // 2³ = 8 tiles of 64³ → 8³ = 512 decoded 16³ chunks
    Volume<f32> vol = Volume<f32>::zeros(dims);
    VolumeView<f32> vv = vol.view();
    for (s64 z = 0; z < dims.z; ++z)
        for (s64 y = 0; y < dims.y; ++y)
            for (s64 x = 0; x < dims.x; ++x)
                vv(z, y, x) = 50.0f + 40.0f * std::sin(0.02f * static_cast<f32>(x + 2 * y + 3 * z));
    {
        auto a = VolumeArchive::create(path, dims, {.q = 2.0f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume(vol.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    auto a = VolumeArchive::open(path);
    REQUIRE(a.has_value());
    auto full = a->read_volume();
    REQUIRE(full.has_value());
    VolumeView<const f32> fv = full->view();

    a->reserve_cache(2u << 20, 16);  // 2 MiB budget << 512×16 KiB = 8 MiB of chunks → forces eviction

    // voxel view is EXACT vs read_volume (same deterministic decode path), even with eviction churn
    Pcg32 rng{1};
    for (int i = 0; i < 300; ++i) {
        const s64 z = rng.next_u32() % 128, y = rng.next_u32() % 128, x = rng.next_u32() % 128;
        auto v = a->voxel(z, y, x);
        REQUIRE(v.has_value());
        CHECK(*v == fv(z, y, x));
    }
    // amortization: a tile-mate access hits the chunk populated by its 64³ tile's decode
    a->block16({0, 0, 0});  // miss → decodes the tile, caches its 64 chunks
    const u64 m_after = a->cache_misses();
    auto mate = a->block16({0, 0, 1});  // same 64³ tile → cache hit
    REQUIRE(mate.has_value());
    CHECK(a->cache_misses() == m_after);  // no new decode
    CHECK(a->cache_hits() > 0);
    // byte budget is respected after touching every chunk
    for (s64 bz = 0; bz < 8; ++bz)
        for (s64 by = 0; by < 8; ++by)
            for (s64 bx = 0; bx < 8; ++bx) (void)a->block16({bz, by, bx});
    CHECK(a->cache_bytes() > 0);
    CHECK(a->cache_bytes() <= (2u << 20));
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
