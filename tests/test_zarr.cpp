// test_zarr.cpp — OME-Zarr v2 raw reader: region read + missing-chunk-as-fill.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "io/zarr.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

using namespace fenix;
namespace fs = std::filesystem;

// Write a tiny raw zarr v2 (u8, chunks 4^3, dim_separator '/') with one chunk present.
static std::string make_zarr() {
    fs::path root = fs::temp_directory_path() / "fenix_test.zarr";
    fs::remove_all(root);
    fs::create_directories(root);
    {
        std::ofstream z(root / ".zarray");
        z << R"({"zarr_format":2,"shape":[8,8,8],"chunks":[4,4,4],"dtype":"|u1",)"
          << R"("compressor":null,"fill_value":0,"order":"C","dimension_separator":"/"})";
    }
    // Chunk (0,0,0): 4^3 u8, value = local linear index.
    fs::create_directories(root / "0" / "0");
    std::ofstream c(root / "0" / "0" / "0", std::ios::binary);
    std::vector<u8> buf(64);
    for (usize i = 0; i < 64; ++i) buf[i] = static_cast<u8>(i);
    c.write(reinterpret_cast<const char*>(buf.data()), 64);
    return root.string();
}

TEST(zarr_reads_present_chunk_and_missing_fill) {
    std::string root = make_zarr();
    // Read the full 8^3 region; chunk (0,0,0) present, others missing -> fill 0.
    auto v = io::read_zarr_region(root, {0, 0, 0}, {8, 8, 8});
    REQUIRE(v.has_value());
    CHECK(v->dims() == Extent3{8, 8, 8});
    // Present chunk: voxel (1,2,3) -> local index (1*4+2)*4+3 = 27.
    CHECK((*v)(1, 2, 3) == 27.0f);
    CHECK((*v)(0, 0, 0) == 0.0f);
    // Missing chunk region (>=4 in any axis but chunk (1,*,*) absent) -> fill 0.
    CHECK((*v)(7, 7, 7) == 0.0f);
    fs::remove_all(root);
}

TEST(zarr_subregion_read) {
    std::string root = make_zarr();
    auto v = io::read_zarr_region(root, {0, 0, 0}, {2, 2, 4});
    REQUIRE(v.has_value());
    CHECK(v->dims() == Extent3{2, 2, 4});
    CHECK((*v)(1, 1, 3) == static_cast<f32>((1 * 4 + 1) * 4 + 3));  // 23
    fs::remove_all(root);
}

TEST(zarr_rejects_compressed) {
    fs::path root = fs::temp_directory_path() / "fenix_test_c.zarr";
    fs::remove_all(root);
    fs::create_directories(root);
    {
        std::ofstream z(root / ".zarray");
        z << R"({"shape":[4,4,4],"chunks":[4,4,4],"dtype":"|u1",)"
          << R"("compressor":{"id":"blosc"},"dimension_separator":"."})";
    }
    auto v = io::read_zarr_region(root, {0, 0, 0}, {4, 4, 4});
    CHECK(!v.has_value());  // blosc not supported yet
    fs::remove_all(root);
}
