// test_io.cpp — .fxvol volume roundtrip (the only volume container fenix reads/writes; NRRD was removed).
#define FENIX_TEST_MAIN
#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

#include <cmath>
#include <filesystem>

using namespace fenix;

TEST(fxvol_roundtrip_u8) {
    // Write a small u8 volume through the DCT codec archive and read it back within tolerance
    // (the codec is lossy — assert MAE, never bit-exact, per the fast-math/tolerance convention).
    Extent3 d{16, 24, 32};  // z,y,x
    Volume<u8> vol(d);
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x)
                vol.view()(z, y, x) = static_cast<u8>((z * 7 + y * 3 + x) & 0xff);

    auto tmp = std::filesystem::temp_directory_path() / "fenix_test_io.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);

    {
        auto a = codec::VolumeArchive::create(path, d, codec::DctParams{.q = 2.0f});
        REQUIRE(a.has_value());
        auto w = a->template write_volume<u8>(vol.view());
        REQUIRE(w.has_value());
        auto c = a->close();
        REQUIRE(c.has_value());
    }

    auto got = codec::VolumeArchive::open(path);
    REQUIRE(got.has_value());
    auto rv = got->read_volume(0);  // decode LOD0 dense f32
    REQUIRE(rv.has_value());
    REQUIRE(rv->dims() == d);

    f64 sae = 0;
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x)
                sae += std::abs(static_cast<f64>(rv->view()(z, y, x)) - static_cast<f64>(vol.view()(z, y, x)));
    const f64 mae = sae / static_cast<f64>(d.count());
    CHECK(mae < 8.0);  // low-q DCT on smooth-ish data stays well within a few levels
    std::filesystem::remove(path);
}

TEST(fxvol_missing_file_errors) {
    auto got = codec::VolumeArchive::open("/no/such/fenix.fxvol");
    REQUIRE(!got.has_value());
}

#include "io/io.hpp"

TEST(import_export_npy_roundtrip) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "fenix_npy_test";
    fs::create_directories(dir);
    const Extent3 d{64, 64, 64};
    Volume<u8> v(d);
    auto vv = v.view();
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x)
                vv(z, y, x) = static_cast<u8>((z * 3 + y * 5 + x * 7) & 0xff);
    const std::string fx0 = (dir / "a.fxvol").string(), npy = (dir / "a.npy").string(),
                      fx1 = (dir / "b.fxvol").string(), npy1 = (dir / "b.npy").string();
    {
        auto a = codec::VolumeArchive::create(fx0, d, codec::DctParams{.q = 1.0f});
        REQUIRE(a.has_value());
        REQUIRE(a->template write_volume<u8>(v.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    Context ctx;
    const std::string_view e_args[] = {fx0, npy};
    REQUIRE(io::export_npy(e_args, ctx).has_value());
    const std::string_view i_args[] = {npy, fx1, "q=1"};
    REQUIRE(io::import_npy(i_args, ctx).has_value());
    const std::string_view e2_args[] = {fx1, npy1};
    REQUIRE(io::export_npy(e2_args, ctx).has_value());
    // dims survive; content within codec tolerance at q=1
    auto b = codec::VolumeArchive::open(fx1);
    REQUIRE(b.has_value());
    CHECK(b->dims().z == d.z);
    auto rb = b->read_volume_as<u8>(0);
    REQUIRE(rb.has_value());
    f64 mae = 0;
    auto rv = rb->view();
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x)
                mae += std::abs(static_cast<f64>(rv(z, y, x)) - static_cast<f64>(vv(z, y, x)));
    mae /= static_cast<f64>(d.count());
    CHECK(mae < 2.0);
    fs::remove_all(dir);
}
