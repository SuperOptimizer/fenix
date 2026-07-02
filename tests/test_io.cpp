// test_io.cpp — NRRD READ (fenix imports foreign NRRD; it never writes it — see io/nrrd.hpp).
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "io/nrrd.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>

using namespace fenix;

TEST(nrrd_read_f32) {
    // Hand-write a minimal raw f32 NRRD (the writer was removed; read must still import foreign data).
    Extent3 d{5, 7, 9};  // z,y,x
    std::vector<f32> data(static_cast<usize>(d.count()));
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x)
                data[static_cast<usize>((z * d.y + y) * d.x + x)] = static_cast<f32>(z * 100 + y * 10 + x);

    auto tmp = std::filesystem::temp_directory_path() / "fenix_test_read.nrrd";
    const std::string path = tmp.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f << "NRRD0004\ntype: float\ndimension: 3\nsizes: " << d.x << " " << d.y << " " << d.z
          << "\nencoding: raw\nendian: little\n\n";
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size() * sizeof(f32)));
    }
    auto got = io::read_nrrd(path);
    REQUIRE(got.has_value());
    REQUIRE(got->dims() == d);
    CHECK((*got)(3, 4, 5) == 345.0f);
    std::filesystem::remove(path);
}

TEST(nrrd_missing_file_errors) {
    auto got = io::read_nrrd("/no/such/fenix.nrrd");
    REQUIRE(!got.has_value());
    CHECK(got.error().code == Errc::not_found);
}
