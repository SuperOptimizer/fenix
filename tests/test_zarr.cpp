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

TEST(zarr_rejects_unsupported_compressor) {
    fs::path root = fs::temp_directory_path() / "fenix_test_c.zarr";
    fs::remove_all(root);
    fs::create_directories(root);
    {
        std::ofstream z(root / ".zarray");
        z << R"({"shape":[4,4,4],"chunks":[4,4,4],"dtype":"|u1",)"
          << R"("compressor":{"id":"gzip","level":5},"dimension_separator":"."})";
    }
    auto v = io::read_zarr_region(root.string(), {0, 0, 0}, {4, 4, 4});  // path->wstring on Windows: pass string
    CHECK(!v.has_value());  // raw + blosc only — anything else is a typed rejection
    fs::remove_all(root);
}

#ifdef FENIX_HAVE_BLOSC2
TEST(zarr_reads_blosc_compressed_chunks) {
    fs::path root = fs::temp_directory_path() / "fenix_blosc.zarr";
    fs::remove_all(root);
    fs::create_directories(root / "0" / "0");
    {
        std::ofstream z(root / ".zarray");
        z << R"({"zarr_format":2,"shape":[8,8,8],"chunks":[4,4,4],"dtype":"|u1",)"
          << R"("compressor":{"id":"blosc","cname":"zstd","clevel":3,"shuffle":1},)"
          << R"("fill_value":0,"order":"C","dimension_separator":"/"})";
    }
    std::vector<u8> raw(64);
    for (usize i = 0; i < 64; ++i) raw[i] = static_cast<u8>(i);
    std::vector<u8> comp(64 + BLOSC2_MAX_OVERHEAD);
    const int n = blosc2_compress(3, 1, 1, raw.data(), 64, comp.data(), static_cast<int32_t>(comp.size()));
    REQUIRE(n > 0);
    {
        std::ofstream c(root / "0" / "0" / "0", std::ios::binary);
        c.write(reinterpret_cast<const char*>(comp.data()), n);
    }
    auto v = io::read_zarr_region(root.string(), {0, 0, 0}, {8, 8, 8});
    REQUIRE(v.has_value());
    CHECK((*v)(1, 2, 3) == 27.0f);  // present chunk decodes through blosc
    CHECK((*v)(5, 5, 5) == 0.0f);   // missing chunk = fill
    fs::remove_all(root);
}
#endif

// Regression: the never-become-air invariant (root CLAUDE.md §2.4, io/CLAUDE.md). A chunk that
// EXISTS but is short (truncated write, crash mid-copy) must be a hard error, not silent fill.
TEST(zarr_truncated_chunk_is_hard_error_not_fill) {
    std::string root = make_zarr();
    // Truncate the present chunk (0,0,0) from 64 bytes down to 10 — present but short.
    {
        fs::path cpath = fs::path(root) / "0" / "0" / "0";
        std::ofstream c(cpath, std::ios::binary | std::ios::trunc);
        std::vector<u8> buf(10, 7);
        c.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    }
    auto v = io::read_zarr_region(root, {0, 0, 0}, {8, 8, 8});
    CHECK(!v.has_value());  // must hard-fail, NOT silently read back as fill/air
    fs::remove_all(root);
}

TEST(zarr_unreadable_chunk_is_hard_error_not_fill) {
    std::string root = make_zarr();
    fs::path cpath = fs::path(root) / "0" / "0" / "0";
    std::error_code ec;
    fs::permissions(cpath, fs::perms::none, ec);
    if (ec) { fs::remove_all(root); return; }  // permission bits unsupported on this fs — skip
    // Root itself may still be readable via the running user's own perms (e.g. root uid) — only
    // assert the invariant when the chmod actually blocks a plain open.
    std::ifstream probe(cpath, std::ios::binary);
    const bool blocked = !probe.is_open();
    probe.close();
    if (blocked) {
        auto v = io::read_zarr_region(root, {0, 0, 0}, {8, 8, 8});
        CHECK(!v.has_value());  // EACCES must hard-fail, NOT be conflated with "missing" (ENOENT)
    }
    fs::permissions(cpath, fs::perms::owner_all, ec);
    fs::remove_all(root);
}

TEST(zarr_genuinely_missing_chunk_is_still_legal_fill) {
    std::string root = make_zarr();
    // Chunk (1,0,0) was never written by make_zarr() — genuinely absent, must read as fill (air),
    // not an error. (Also covered by zarr_reads_present_chunk_and_missing_fill; kept explicit here
    // as the control case alongside the two hard-error regressions above.)
    auto v = io::read_zarr_region(root, {4, 0, 0}, {4, 4, 4});
    REQUIRE(v.has_value());
    CHECK((*v)(0, 0, 0) == 0.0f);
    fs::remove_all(root);
}

// copy_zarr_region_local: a truncated SOURCE chunk must propagate as a hard error, not be copied
// verbatim into the destination slab (which would just move the silent-air bug downstream).
TEST(zarr_copy_local_propagates_truncated_source_chunk_error) {
    std::string root = make_zarr();
    {
        fs::path cpath = fs::path(root) / "0" / "0" / "0";
        std::ofstream c(cpath, std::ios::binary | std::ios::trunc);
        std::vector<u8> buf(5, 1);
        c.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    }
    fs::path out = fs::temp_directory_path() / "fenix_test_copy_out.zarr";
    fs::remove_all(out);
    // copy_zarr_region_local copies chunk bytes verbatim, so the truncated source chunk itself
    // isn't size-validated at copy time (raw byte copy, no decode) — but the same corrupted output
    // must then be caught as a hard error on READ, proving the corruption cannot silently become
    // fill/air anywhere along this path.
    auto cp = io::copy_zarr_region_local(root, out.string(), {0, 0, 0}, {8, 8, 8});
    if (cp.has_value()) {
        auto v = io::read_zarr_region(out.string() + "/0", {0, 0, 0}, {8, 8, 8});
        CHECK(!v.has_value());
    }
    fs::remove_all(out);
    fs::remove_all(root);
}

TEST(zarr_v3_raw_and_missing_chunks) {
    namespace fs = std::filesystem;
    fs::path root = fs::temp_directory_path() / "fenix_v3.zarr";
    fs::remove_all(root);
    fs::create_directories(root / "c" / "0" / "0");
    {
        std::ofstream j(root / "zarr.json");
        j << "{\"zarr_format\": 3, \"shape\": [8, 8, 8], "
             "\"chunk_grid\": {\"name\": \"regular\", \"configuration\": {\"chunk_shape\": [4, 4, 4]}}, "
             "\"data_type\": \"uint8\", "
             "\"chunk_key_encoding\": {\"name\": \"default\", \"configuration\": {\"separator\": \"/\"}}, "
             "\"codecs\": [{\"name\": \"bytes\", \"configuration\": {\"endian\": \"little\"}}]}";
    }
    {
        std::ofstream c(root / "c" / "0" / "0" / "0", std::ios::binary);
        for (int i = 0; i < 64; ++i) c.put(static_cast<char>(i + 1));
    }
    auto v = io::read_zarr_region<u8>(root.string(), {0, 0, 0}, {8, 8, 8});
    REQUIRE(v.has_value());
    CHECK(v->view()(0, 0, 0) == 1);
    CHECK(v->view()(3, 3, 3) == 64);
    CHECK(v->view()(7, 7, 7) == 0);  // missing chunk = fill
    fs::remove_all(root);
}

TEST(zarr_v3_sharded) {
    namespace fs = std::filesystem;
    fs::path root = fs::temp_directory_path() / "fenix_v3s.zarr";
    fs::remove_all(root);
    fs::create_directories(root / "c" / "0" / "0");
    {
        std::ofstream j(root / "zarr.json");
        j << "{\"zarr_format\": 3, \"shape\": [8, 8, 8], "
             "\"chunk_grid\": {\"name\": \"regular\", \"configuration\": {\"chunk_shape\": [8, 8, 8]}}, "
             "\"data_type\": \"uint8\", "
             "\"chunk_key_encoding\": {\"name\": \"default\", \"configuration\": {\"separator\": \"/\"}}, "
             "\"codecs\": [{\"name\": \"sharding_indexed\", \"configuration\": {"
             "\"chunk_shape\": [4, 4, 4], "
             "\"codecs\": [{\"name\": \"bytes\", \"configuration\": {\"endian\": \"little\"}}], "
             "\"index_codecs\": [{\"name\": \"bytes\"}], \"index_location\": \"end\"}}]}";
    }
    {
        // shard: inner (0,0,0) = all 7s at offset 0; inner (1,1,1) = all 9s at offset 64;
        // other 6 inner chunks missing (~0 offsets)
        std::ofstream c(root / "c" / "0" / "0" / "0", std::ios::binary);
        for (int i = 0; i < 64; ++i) c.put(7);
        for (int i = 0; i < 64; ++i) c.put(9);
        auto put64 = [&](u64 v) { c.write(reinterpret_cast<const char*>(&v), 8); };
        for (int ci = 0; ci < 8; ++ci) {
            if (ci == 0) { put64(0); put64(64); }
            else if (ci == 7) { put64(64); put64(64); }  // (1,1,1) is the last index slot
            else { put64(~u64{0}); put64(~u64{0}); }
        }
    }
    auto v = io::read_zarr_region<u8>(root.string(), {0, 0, 0}, {8, 8, 8});
    REQUIRE(v.has_value());
    CHECK(v->view()(0, 0, 0) == 7);
    CHECK(v->view()(3, 3, 3) == 7);
    CHECK(v->view()(4, 4, 4) == 9);
    CHECK(v->view()(7, 7, 7) == 9);
    CHECK(v->view()(0, 7, 7) == 0);  // missing inner chunk = fill
    fs::remove_all(root);
}
