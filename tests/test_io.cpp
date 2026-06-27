// test_io.cpp — NRRD write/read roundtrip.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "io/nrrd.hpp"

#include <cmath>
#include <filesystem>

using namespace fenix;

TEST(nrrd_roundtrip) {
    Extent3 d{5, 7, 9};  // z,y,x
    Volume<f32> v(d);
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x)
                v(z, y, x) = static_cast<f32>(z * 100 + y * 10 + x);

    auto tmp = std::filesystem::temp_directory_path() / "fenix_test.nrrd";
    const std::string path = tmp.string();
    std::filesystem::remove(path);

    REQUIRE(io::write_nrrd(path, v.view()).has_value());
    auto got = io::read_nrrd(path);
    REQUIRE(got.has_value());
    REQUIRE(got->dims() == d);
    f32 max_err = 0;
    for (s64 i = 0; i < d.count(); ++i)
        max_err = std::max(max_err, std::abs(got->flat()[static_cast<usize>(i)] -
                                             v.flat()[static_cast<usize>(i)]));
    CHECK(max_err == 0.0f);  // raw f32 -> exact
    CHECK((*got)(3, 4, 5) == 345.0f);

    std::filesystem::remove(path);
}

TEST(nrrd_missing_file_errors) {
    auto got = io::read_nrrd("/no/such/fenix.nrrd");
    REQUIRE(!got.has_value());
    CHECK(got.error().code == Errc::not_found);
}
