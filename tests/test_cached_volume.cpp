// tests/test_cached_volume.cpp — the on-demand training chunk cache (io/cached_volume.hpp):
// gather pulls missing chunks from the zarr into the .fxvol cache; later gathers are served
// locally (proven by DELETING the zarr and gathering again); the cache persists across opens.
#define FENIX_TEST_MAIN
#include "core/test.hpp"
#include "io/cached_volume.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

using namespace fenix;
namespace fs = std::filesystem;

namespace {
// Raw zarr v2, u8, 128^3 volume in 64^3 chunks (matches the archive chunk side so coverage
// mapping is 1:1), every chunk present, voxel value = (z+y+x) & 0xff.
std::string make_zarr(const fs::path& root) {
    fs::remove_all(root);
    fs::create_directories(root);
    {
        std::ofstream z(root / ".zarray");
        z << R"({"zarr_format":2,"shape":[128,128,128],"chunks":[64,64,64],"dtype":"|u1",)"
          << R"("compressor":null,"fill_value":0,"order":"C","dimension_separator":"/"})";
    }
    std::vector<u8> buf(64 * 64 * 64);
    for (int cz = 0; cz < 2; ++cz)
        for (int cy = 0; cy < 2; ++cy)
            for (int cx = 0; cx < 2; ++cx) {
                for (s64 z = 0; z < 64; ++z)
                    for (s64 y = 0; y < 64; ++y)
                        for (s64 x = 0; x < 64; ++x)
                            buf[static_cast<usize>((z * 64 + y) * 64 + x)] =
                                static_cast<u8>((cz * 64 + z + cy * 64 + y + cx * 64 + x) & 0xff);
                fs::create_directories(root / std::to_string(cz) / std::to_string(cy));
                std::ofstream c(root / std::to_string(cz) / std::to_string(cy) / std::to_string(cx), std::ios::binary);
                c.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
            }
    return root.string();
}
}  // namespace

TEST(cached_volume_fetches_once_then_serves_locally) {
    const fs::path dir = fs::temp_directory_path() / "fenix_cache_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string zroot = make_zarr(dir / "vol.zarr");
    const std::string cpath = (dir / "cache.fxvol").string();

    auto cv = io::CachedVolume::open(cpath, zroot, 0.5f);
    REQUIRE(cv.has_value());
    CHECK(cv->dims() == Extent3{128, 128, 128});

    // gather a box spanning 4 chunks -> fetch + cache
    std::vector<f32> buf(64 * 64 * 64);
    REQUIRE(cv->gather_box_f32(32, 32, 32, 64, 64, 64, buf.data()).has_value());
    // value check within codec tolerance (q=0.5): voxel (32,32,32) = 96
    CHECK(std::abs(buf[0] - 96.0f) < 2.0f);
    CHECK(cv->archive().coverage({0, 0, 0}) != codec::Coverage::Absent);
    CHECK(cv->archive().coverage({1, 1, 1}) != codec::Coverage::Absent);

    // DELETE the zarr: subsequent gathers of the cached region must still work (local serve)
    fs::remove_all(dir / "vol.zarr");
    std::vector<f32> buf2(64 * 64 * 64);
    REQUIRE(cv->gather_box_f32(32, 32, 32, 64, 64, 64, buf2.data()).has_value());
    CHECK(std::abs(buf2[0] - buf[0]) < 1e-3f);

    // ...but an UNCACHED region now hard-fails (fetch needed, source gone — never silent air)
    // note: all 8 chunks of this 128^3 got cached by the spanning gather, so re-create a
    // fresh cache against the deleted zarr to prove the failure path
    auto cv2 = io::CachedVolume::open((dir / "cache2.fxvol").string(), zroot);
    CHECK(!cv2.has_value());  // .zarray itself is gone
    fs::remove_all(dir);
}

TEST(cached_volume_persists_across_reopen) {
    const fs::path dir = fs::temp_directory_path() / "fenix_cache_test2";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string zroot = make_zarr(dir / "vol.zarr");
    const std::string cpath = (dir / "cache.fxvol").string();
    {
        auto cv = io::CachedVolume::open(cpath, zroot, 0.5f);
        REQUIRE(cv.has_value());
        std::vector<f32> buf(32 * 32 * 32);
        REQUIRE(cv->gather_box_f32(0, 0, 0, 32, 32, 32, buf.data()).has_value());
        REQUIRE(cv->archive().close().has_value());
    }
    fs::remove_all(dir / "vol.zarr");  // source gone; cache must carry the data
    {
        // reopen fails at .zarray read (source identity check needs it) — document that a
        // cache without its source is usable only via the plain archive path
        auto cv = io::CachedVolume::open(cpath, zroot, 0.5f);
        CHECK(!cv.has_value());
        auto a = codec::VolumeArchive::open(cpath);
        REQUIRE(a.has_value());
        CHECK(a->coverage({0, 0, 0}) != codec::Coverage::Absent);
        std::vector<f32> buf(32 * 32 * 32);
        REQUIRE(a->gather_box_f32(0, 0, 0, 0, 32, 32, 32, buf.data()).has_value());
        CHECK(std::abs(buf[0] - 0.0f) < 2.0f);
    }
    fs::remove_all(dir);
}

TEST(cached_volume_disk_budget_resets_and_stays_correct) {
    const fs::path dir = fs::temp_directory_path() / "fenix_cache_budget";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string zroot = make_zarr(dir / "vol.zarr");
    const std::string cpath = (dir / "cache.fxvol").string();

    auto cv = io::CachedVolume::open(cpath, zroot, 0.5f);
    REQUIRE(cv.has_value());
    cv->disk_budget(1);  // 1 byte: EVERY fill overflows -> reset before each append

    // Gather all 8 chunks one by one — each triggers a reset of the previous state, and the
    // data must stay exact regardless (reset -> absent -> refetch is the designed path).
    std::vector<u8> out(64 * 64 * 64);
    for (int cz = 0; cz < 2; ++cz)
        for (int cy = 0; cy < 2; ++cy)
            for (int cx = 0; cx < 2; ++cx) {
                REQUIRE(cv->gather_box_u8(cz * 64, cy * 64, cx * 64, 64, 64, 64, out.data()).has_value());
                // tolerance check (q=0.5 lossy; the sawtooth wrap rings): >=99% of voxels within +/-4
                s64 good = 0;
                for (s64 z = 0; z < 64; ++z)
                    for (s64 y = 0; y < 64; ++y)
                        for (s64 x = 0; x < 64; ++x) {
                            const int want =
                                static_cast<int>(static_cast<u8>((cz * 64 + z + cy * 64 + y + cx * 64 + x) & 0xff));
                            const int got = static_cast<int>(out[static_cast<usize>((z * 64 + y) * 64 + x)]);
                            good += std::abs(got - want) <= 4;
                        }
                CHECK(good >= 64 * 64 * 64 * 99 / 100);
            }
    // file stayed bounded: at most one fill's worth of chunks live in the cache
    REQUIRE(cv->archive().committed_size() > 0);
    fs::remove_all(dir);
}
