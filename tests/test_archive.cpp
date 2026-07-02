// test_archive.cpp — .fxvol create/write/close/open/read roundtrip + coverage tri-state.
#define FENIX_TEST_MAIN
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/parallel.hpp"
#include "core/test.hpp"

using namespace fenix;
using namespace fenix::codec;

static std::vector<f32> structured_chunk() {
    const s64 s = fxvol_chunk_side;
    std::vector<f32> b(static_cast<usize>(s * s * s));
    Pcg32 rng{5};
    const Index3 st{s * s, s, 1};
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x)
                b[static_cast<usize>(z * st.z + y * st.y + x * st.x)] =
                    40.0f + 80.0f * std::exp(-0.1f * static_cast<f32>((x - 30) * (x - 30))) + rng.next_f32();
    return b;
}

TEST(archive_roundtrip_and_coverage) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_test.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);

    Extent3 dims{64, 128, 64}; // chunk extent: z=1, y=2, x=1
    auto data = structured_chunk();

    {
        auto a = VolumeArchive::create(path, dims, {.q = 4.0f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_chunk({0, 0, 0}, data).has_value());
        // second chunk left all-zero -> recorded ZERO
        std::vector<f32> air(data.size(), 0.0f);
        REQUIRE(a->write_chunk({0, 1, 0}, air).has_value());
        REQUIRE(a->close().has_value());
    }

    auto a = VolumeArchive::open(path);
    REQUIRE(a.has_value());
    CHECK(a->dims() == dims);
    CHECK(a->chunk_extent() == ChunkCoord{1, 2, 1});
    CHECK(a->coverage({0, 0, 0}) == Coverage::Real);
    CHECK(a->coverage({0, 1, 0}) == Coverage::Zero);
    CHECK(a->coverage({0, 5, 0}) == Coverage::Absent);

    auto got = a->read_chunk({0, 0, 0});
    REQUIRE(got.has_value());
    REQUIRE(got->size() == data.size());
    f32 max_err = 0;
    for (usize i = 0; i < data.size(); ++i)
        max_err = std::max(max_err, std::abs((*got)[i] - data[i]));
    CHECK(max_err < 20.0f); // q=4 lossy reconstruction

    auto air = a->read_chunk({0, 1, 0});
    REQUIRE(air.has_value());
    CHECK((*air)[123] == 0.0f); // ZERO chunk -> zeros
    auto absent = a->read_chunk({0, 5, 0});
    REQUIRE(absent.has_value());
    CHECK((*absent)[0] == 0.0f); // ABSENT -> fill

    std::filesystem::remove(path);
}

// Regression for the confirmed finding: read_chunk_as's blob-bounds check overflowed on a corrupt slot
// length (s.len near u64 max wrapped `s.len + 4` and `hz - s.len - 4` back into range), driving crc32c()
// ~2^64 bytes past the mapping. Craft a REAL archive, then poke a corrupt FxSlot.len directly into the
// mmap'd leaf (page-table leaves carry no CRC, so this is exactly the class of corruption a crafted file
// can reach) and confirm the read is rejected instead of segfaulting.
TEST(archive_rejects_corrupt_slot_length) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_test_corrupt_slot.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);

    Extent3 dims{64, 64, 64};
    auto data = structured_chunk();
    {
        auto a = VolumeArchive::create(path, dims, {.q = 4.0f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_chunk({0, 0, 0}, data).has_value());
        REQUIRE(a->close().has_value());
    }

    // Reopen read-write so we can poke the mmap directly, locate the one leaf slot, corrupt its length to
    // near-u64-max (the exact wraparound case from the finding), then verify read_chunk rejects it.
    auto a = VolumeArchive::open(path, /*writable=*/true);
    REQUIRE(a.has_value());
    CHECK(a->coverage({0, 0, 0}) == Coverage::Real);
    const u64 root = a->lod_root_offset(0);
    REQUIRE(root != 0);

    // Re-derive the leaf offset the same way slot_read_ does (public accessors don't expose it directly,
    // so walk the file bytes ourselves via a second read-only mmap-free path: reopen and binary-search is
    // overkill — instead corrupt via the known layout: L0 node -> L1 node -> leaf, all indexed by the
    // Morton key of chunk (0,0,0), which is 0, so index 0 at every level).
    // We can't reach base_ from the test (private), so instead corrupt via the file directly: the leaf for
    // Morton-index 0 sits at a fixed, deterministic offset chain we can recompute: root[0] -> node[0] ->
    // leaf[0]. Read those offsets from the file directly (same layout the archive itself uses).
    std::vector<u8> raw(std::filesystem::file_size(path));
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        REQUIRE(f != nullptr);
        REQUIRE(std::fread(raw.data(), 1, raw.size(), f) == raw.size());
        std::fclose(f);
    }
    u64 l1_off, leaf_off;
    std::memcpy(&l1_off, raw.data() + root, 8); // node[0] entry 0
    REQUIRE(l1_off != 0);
    std::memcpy(&leaf_off, raw.data() + l1_off, 8); // leaf-node entry 0
    REQUIRE(leaf_off != 0);
    // leaf[0] = FxSlot{off, len} for chunk (0,0,0) (Morton(0,0,0) = 0 -> index 0 at every level).
    const u64 slot_off_field = leaf_off;     // FxSlot.off
    const u64 slot_len_field = leaf_off + 8; // FxSlot.len

    u64 corrupt_len = ~0ull - 3; // exact wraparound case from the finding: len+4 wraps to 0
    FILE* fw = std::fopen(path.c_str(), "r+b");
    REQUIRE(fw != nullptr);
    REQUIRE(std::fseek(fw, static_cast<long>(slot_len_field), SEEK_SET) == 0);
    REQUIRE(std::fwrite(&corrupt_len, 1, 8, fw) == 8);
    std::fclose(fw);
    (void)slot_off_field;

    auto a2 = VolumeArchive::open(path);
    REQUIRE(a2.has_value());
    auto got = a2->read_chunk({0, 0, 0});
    CHECK(!got.has_value()); // corrupt length -> rejected, never a crash

    std::filesystem::remove(path);
}

// Regression for the "unverified" companion finding on the same page: open() validated nlods_/dtype/eof/
// roots but not dims_/params_.q, so a crafted (CRC-valid) superblock with an absurd dims_ could later
// drive read_volume_as's Volume<T>::zeros(vd) into an unbounded/negative-arithmetic allocation, and a
// non-finite/non-positive q would corrupt every dequant downstream. Craft a real archive, corrupt dims_.z
// to an out-of-envelope value, recompute the superblock CRC (so the corruption isn't masked by the
// existing CRC check — this isolates the new range-validation path specifically), and confirm open()
// rejects it.
TEST(archive_rejects_out_of_range_dims) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_test_bad_dims.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);
    {
        auto a = VolumeArchive::create(path, Extent3{64, 64, 64}, {.q = 4.0f});
        REQUIRE(a.has_value());
        REQUIRE(a->close().has_value());
    }
    std::vector<u8> raw(std::filesystem::file_size(path));
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        REQUIRE(f != nullptr);
        REQUIRE(std::fread(raw.data(), 1, raw.size(), f) == raw.size());
        std::fclose(f);
    }
    // The active superblock slot is whichever of the two has the higher commit_seq: create() writes seq=1
    // into slot 0 ((1-1)&1==0), then close()'s commit() bumps to seq=2 and writes slot 1 ((2-1)&1==1) —
    // so after create()+close() the ACTIVE slot is 1, not 0. dims_.z sits at +32 within a slot.
    const u64 slot_base = codec::detail::kFxSuper;
    const u64 dims_z_off = slot_base + 32;
    const s64 huge = (1ll << 18) + 1; // one past the documented envelope
    std::memcpy(raw.data() + dims_z_off, &huge, 8);
    const u32 crc = codec::detail::crc32c(raw.data() + slot_base, codec::detail::kFxSbCrcLen);
    std::memcpy(raw.data() + slot_base + codec::detail::kFxSbCrcLen, &crc, 4);
    FILE* fw = std::fopen(path.c_str(), "wb");
    REQUIRE(fw != nullptr);
    REQUIRE(std::fwrite(raw.data(), 1, raw.size(), fw) == raw.size());
    std::fclose(fw);

    CHECK(!VolumeArchive::open(path).has_value()); // out-of-range dims_, CRC-consistent -> still rejected
    std::filesystem::remove(path);
}

// block16()/gather_box_f32() are documented thread-safe (see archive.hpp). Regression for the confirmed
// race: the block cache used to be lazily constructed on first block16() call with an unsynchronized
// test-and-assign on a `mutable unique_ptr`. Hammer block16 from many threads on a freshly-opened archive
// (nothing calls reserve_cache() first) and confirm every result is valid — this alone doesn't prove the
// absence of a race (TSan would), but it exercises exactly the path the finding was about and would
// reliably crash/UAF on the old lazy-init code under contention.
TEST(archive_block16_concurrent_use_is_safe) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_test_concurrent_block16.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);

    Extent3 dims{128, 128, 128};
    {
        auto a = VolumeArchive::create(path, dims, {.q = 4.0f});
        REQUIRE(a.has_value());
        for (s64 cz = 0; cz < 2; ++cz)
            for (s64 cy = 0; cy < 2; ++cy)
                for (s64 cx = 0; cx < 2; ++cx) {
                    auto data = structured_chunk();
                    REQUIRE(a->write_chunk({cz, cy, cx}, data).has_value());
                }
        REQUIRE(a->close().has_value());
    }

    auto a = VolumeArchive::open(path); // no reserve_cache() — relies on create/open's eager default cache
    REQUIRE(a.has_value());
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            Pcg32 rng{static_cast<u32>(1000 + t)};
            for (int i = 0; i < 200; ++i) {
                const s64 bz = static_cast<s64>(rng.next_u32() % 8), by = static_cast<s64>(rng.next_u32() % 8),
                          bx = static_cast<s64>(rng.next_u32() % 8);
                auto r = a->block16({bz, by, bx});
                if (!r)
                    failures.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads)
        th.join();
    CHECK(failures.load() == 0);

    std::filesystem::remove(path);
}

// Regression for the confirmed low/hygiene finding: sample_f32 computed a block coord via truncating `/`
// and an offset via truncating `%` on the SAME negative z/y/x — those disagree in sign (z=-1 gives
// z/16==0, a valid-looking block, but z%16==-1), so `off` wrapped to a huge usize -> OOB read of the 4
// KiB cache block. sample_f32 now rejects negative coordinates explicitly instead of reading garbage.
TEST(archive_sample_f32_rejects_negative_coords) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_test_neg_coord.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);
    {
        auto a = VolumeArchive::create(path, Extent3{64, 64, 64}, {.q = 4.0f});
        REQUIRE(a.has_value());
        auto data = structured_chunk();
        REQUIRE(a->write_chunk({0, 0, 0}, data).has_value());
        REQUIRE(a->close().has_value());
    }
    auto a = VolumeArchive::open(path);
    REQUIRE(a.has_value());
    CHECK(!a->sample_f32(-1, 0, 0).has_value());
    CHECK(!a->sample_f32(0, -1, 0).has_value());
    CHECK(!a->sample_f32(0, 0, -1).has_value());
    CHECK(a->sample_f32(0, 0, 0).has_value()); // still works for valid coordinates
    std::filesystem::remove(path);
}
