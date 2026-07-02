// tests/test_feed.cpp — the train-feed data plane end-to-end: synthetic CT .fxvol + .fxsurf
// -> run_train_feed(count=slots, no consumer needed) -> validate ring protocol + patch contents.
#define FENIX_TEST_MAIN
#include "core/test.hpp"
#include "ml/feed.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>

using namespace fenix;
namespace fs = std::filesystem;

TEST(train_feed_end_to_end_ring) {
    const fs::path dir = fs::temp_directory_path() / "fenix_feed_test";
    fs::create_directories(dir);
    const Extent3 vd{128, 128, 128};

    // CT: a gradient volume (value = x&0xff) so patch content is position-checkable.
    Volume<u8> ct(vd);
    auto cv = ct.view();
    for (s64 z = 0; z < vd.z; ++z)
        for (s64 y = 0; y < vd.y; ++y)
            for (s64 x = 0; x < vd.x; ++x) cv(z, y, x) = static_cast<u8>((x * 2) & 0xff);
    const std::string ctp = (dir / "ct.fxvol").string();
    {
        auto a = codec::VolumeArchive::create(ctp, vd, codec::DctParams{.q = 0.5f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(ct.view()).has_value());
        REQUIRE(a->close().has_value());
    }

    // Surface: flat sheet at z=64 spanning the volume, grid step 16.
    Surface s(9, 9);
    s.scale_u = 16.0f;
    s.scale_v = 16.0f;
    for (s64 v = 0; v < 9; ++v)
        for (s64 u = 0; u < 9; ++u) s.set(u, v, Vec3f{64.0f, static_cast<f32>(v) * 16.0f, static_cast<f32>(u) * 16.0f});
    const std::string sp = (dir / "seg.fxsurf").string();
    REQUIRE(io::write_fxsurf(sp, s).has_value());

    const std::string pairs = (dir / "pairs.txt").string();
    {
        std::ofstream f(pairs);
        f << "# test corpus\n" << sp << " " << ctp << "\n";
    }

    // count == slots: producers fill every slot and exit without needing a consumer.
    const std::string ring = (dir / "ring.bin").string();
    Context ctx;
    const std::string_view args[] = {pairs, ring, "patch=64", "slots=6", "count=6", "threads=2", "octa=0", "seed=7"};
    auto r = ml::run_train_feed(args, ctx);
    REQUIRE(r.has_value());

    // Validate the ring: header, then every slot READY with coherent contents.
    std::ifstream f(ring, std::ios::binary);
    REQUIRE(static_cast<bool>(f));
    std::vector<u8> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto* hdr = reinterpret_cast<const ml::detail::RingHeader*>(buf.data());
    REQUIRE(std::memcmp(hdr->magic, "FXRING1\0", 8) == 0);
    CHECK(hdr->nslots == 6u);
    CHECK(hdr->patch == 64u);
    CHECK(hdr->channels == 2u);
    const u64 tensor = 64 * 64 * 64;
    CHECK(hdr->slot_bytes == ml::detail::kSlotHdr + tensor * 2);

    int ready = 0;
    for (u32 sl = 0; sl < hdr->nslots; ++sl) {
        const u8* base = buf.data() + ml::detail::kRingHdr + sl * hdr->slot_bytes;
        const auto* sh = reinterpret_cast<const ml::detail::SlotHeader*>(base);
        if (sh->state != ml::detail::kReady) continue;
        ++ready;
        CHECK(sh->mesh == 0u);
        // CT channel matches the source gradient at the recorded origin
        const u8* ctd = base + ml::detail::kSlotHdr;
        bool ct_ok = true;
        for (s64 x = 0; x < 64; x += 7)
            if (ctd[static_cast<usize>(x)] != static_cast<u8>(((sh->origin[2] + x) * 2) & 0xff)) ct_ok = false;
        CHECK(ct_ok);
        // GT channel: the sheet plane must appear iff z=64 is inside the patch (it always is —
        // patches are sampled on the surface), and its band must sit at the right local z.
        const u8* gtd = ctd + tensor;
        const s64 lz = 64 - sh->origin[0];
        REQUIRE(lz >= 0);
        REQUIRE(lz < 64);
        s64 on = 0, on_plane = 0;
        for (u64 k = 0; k < tensor; ++k) on += gtd[k] != 0;
        for (s64 y = 8; y < 56; ++y)
            for (s64 x = 8; x < 56; ++x) on_plane += gtd[static_cast<usize>((lz * 64 + y) * 64 + x)] != 0;
        CHECK(on > 0);
        // interior of the plane through the patch must be densely labeled (surface spans 0..128)
        CHECK(on_plane > 40 * 40 / 2);
    }
    CHECK(ready == 6);
    fs::remove_all(dir);
}

TEST(train_feed_resamples_to_canonical_2p4um) {
    const fs::path dir = fs::temp_directory_path() / "fenix_feed_um";
    fs::create_directories(dir);
    const Extent3 vd{128, 128, 128};  // source volume at 4.8 um -> canonical dims 64^3
    Volume<u8> ct(vd);
    auto cv = ct.view();
    for (s64 z = 0; z < vd.z; ++z)
        for (s64 y = 0; y < vd.y; ++y)
            for (s64 x = 0; x < vd.x; ++x) cv(z, y, x) = static_cast<u8>(x);  // gradient along x
    const std::string ctp = (dir / "ct.fxvol").string();
    {
        auto a = codec::VolumeArchive::create(ctp, vd, codec::DctParams{.q = 0.5f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(ct.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    // flat sheet at SOURCE z=64 (canonical z=32), spanning the source volume, grid step 16 src vox
    Surface s(9, 9);
    s.scale_u = 16.0f;
    s.scale_v = 16.0f;
    for (s64 v = 0; v < 9; ++v)
        for (s64 u = 0; u < 9; ++u)
            s.set(u, v, Vec3f{64.0f, static_cast<f32>(v) * 16.0f, static_cast<f32>(u) * 16.0f});
    const std::string sp = (dir / "seg.fxsurf").string();
    REQUIRE(io::write_fxsurf(sp, s).has_value());
    const std::string pairs = (dir / "pairs.txt").string();
    { std::ofstream f(pairs); f << sp << " " << ctp << " um=4.8\n"; }

    const std::string ring = (dir / "ring.bin").string();
    Context ctx;
    const std::string_view args[] = {pairs, ring, "patch=48", "slots=4", "count=4",
                                     "threads=2", "octa=0", "seed=3"};
    REQUIRE(ml::run_train_feed(args, ctx).has_value());

    std::ifstream f(ring, std::ios::binary);
    std::vector<u8> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto* hdr = reinterpret_cast<const ml::detail::RingHeader*>(buf.data());
    const u64 tensor = 48 * 48 * 48;
    int checked = 0;
    for (u32 sl = 0; sl < hdr->nslots; ++sl) {
        const u8* base = buf.data() + ml::detail::kRingHdr + sl * hdr->slot_bytes;
        const auto* sh = reinterpret_cast<const ml::detail::SlotHeader*>(base);
        if (sh->state != ml::detail::kReady) continue;
        ++checked;
        // origins are CANONICAL: must fit the 64^3 canonical volume
        REQUIRE(sh->origin[0] >= 0);
        REQUIRE(sh->origin[0] + 48 <= 64);
        const u8* ctd = base + ml::detail::kSlotHdr;
        // CT: canonical voxel x maps to source x*2 -> gradient value ~ 2*(origin.x + x) (+/- interp+codec)
        for (s64 x = 4; x < 44; x += 8) {
            const f32 want = 2.0f * (static_cast<f32>(sh->origin[2]) + static_cast<f32>(x) + 0.5f) - 0.5f;
            const f32 got = static_cast<f32>(ctd[static_cast<usize>(x)]);
            CHECK(std::abs(got - want) <= 3.0f);
        }
        // GT: sheet at canonical z=32 -> local z = 32 - origin.z must carry sheet labels
        const u8* gtd = ctd + tensor;
        const s64 lz = 32 - sh->origin[0];
        REQUIRE(lz >= 0);
        REQUIRE(lz < 48);
        s64 on = 0;
        for (s64 y = 8; y < 40; ++y)
            for (s64 x = 8; x < 40; ++x)
                for (s64 dz = -1; dz <= 1; ++dz)
                    on += gtd[static_cast<usize>(((lz + dz) * 48 + y) * 48 + x)] == ml::kLabelSheet;
        CHECK(on > 32 * 32 / 2);
    }
    REQUIRE(checked > 0);
    fs::remove_all(dir);
}
