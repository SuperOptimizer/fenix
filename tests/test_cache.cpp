// tests/test_cache.cpp — the local artifact cache (io/cache.hpp): fetch-once tifxyz ->
// .fxsurf transcode (proven by deleting the source), the CachedPyramid streaming
// multi-LOD source over a zarr multiscale (gathers at every level, block16, offline
// reopen off the cache), and cache-key collision safety.
#define FENIX_TEST_MAIN
#include "core/test.hpp"
#include "io/cache.hpp"
#include "view/slice_engine.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

using namespace fenix;
namespace fs = std::filesystem;

namespace {

// Minimal classic-LE single-strip f32 TIFF (fixture only — fenix never writes TIFF).
void write_test_tiff(const std::string& path, s64 w, s64 h, const std::vector<f32>& pix) {
    std::ofstream f(path, std::ios::binary);
    auto u16w = [&](u16 v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    auto u32w = [&](u32 v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    const u32 data_bytes = static_cast<u32>(w * h * 4);
    f.write("II", 2);
    u16w(42);
    u32w(8 + data_bytes);
    f.write(reinterpret_cast<const char*>(pix.data()), data_bytes);
    u16w(9);
    auto ent = [&](u16 tag, u16 type, u32 cnt, u32 val) { u16w(tag); u16w(type); u32w(cnt); u32w(val); };
    ent(256, 3, 1, static_cast<u32>(w));
    ent(257, 3, 1, static_cast<u32>(h));
    ent(258, 3, 1, 32);
    ent(259, 3, 1, 1);
    ent(273, 4, 1, 8);
    ent(277, 3, 1, 1);
    ent(278, 3, 1, static_cast<u32>(h));
    ent(279, 4, 1, data_bytes);
    ent(339, 3, 1, 3);
    u32w(0);
}

std::string make_tifxyz(const fs::path& dir) {
    fs::remove_all(dir);
    fs::create_directories(dir);
    const s64 w = 24, h = 16;
    std::vector<f32> xs(static_cast<usize>(w * h)), ys(xs.size()), zs(xs.size());
    for (s64 v = 0; v < h; ++v)
        for (s64 u = 0; u < w; ++u) {
            const usize i = static_cast<usize>(v * w + u);
            xs[i] = 100.0f + static_cast<f32>(u) * 20.0f;
            ys[i] = 200.0f + static_cast<f32>(v) * 20.0f;
            zs[i] = 5000.0f + static_cast<f32>(u + v);
        }
    write_test_tiff((dir / "x.tif").string(), w, h, xs);
    write_test_tiff((dir / "y.tif").string(), w, h, ys);
    write_test_tiff((dir / "z.tif").string(), w, h, zs);
    std::ofstream m(dir / "meta.json");
    m << R"({"format":"tifxyz","scale":[0.05,0.05],"type":"seg"})";
    return dir.string();
}

// One raw zarr v2 u8 level: value = (z+y+x) & 0xff at LEVEL-0 SCALE (level k striding by
// 2^k), so cross-level values agree where they sample the same LOD-0 lattice point.
void make_zarr_level(const fs::path& root, s64 side, s64 chunk, int lod) {
    fs::create_directories(root);
    {
        std::ofstream z(root / ".zarray");
        z << R"({"zarr_format":2,"shape":[)" << side << "," << side << "," << side
          << R"(],"chunks":[)" << chunk << "," << chunk << "," << chunk
          << R"(],"dtype":"|u1","compressor":null,"fill_value":0,"order":"C","dimension_separator":"/"})";
    }
    const s64 nc = (side + chunk - 1) / chunk, stride = s64{1} << lod;
    std::vector<u8> buf(static_cast<usize>(chunk * chunk * chunk));
    for (s64 cz = 0; cz < nc; ++cz)
        for (s64 cy = 0; cy < nc; ++cy)
            for (s64 cx = 0; cx < nc; ++cx) {
                for (s64 z = 0; z < chunk; ++z)
                    for (s64 y = 0; y < chunk; ++y)
                        for (s64 x = 0; x < chunk; ++x)
                            buf[static_cast<usize>((z * chunk + y) * chunk + x)] = static_cast<u8>(
                                ((cz * chunk + z) + (cy * chunk + y) + (cx * chunk + x)) * stride & 0xff);
                fs::create_directories(root / std::to_string(cz) / std::to_string(cy));
                std::ofstream c(root / std::to_string(cz) / std::to_string(cy) / std::to_string(cx),
                                std::ios::binary);
                c.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
            }
}

std::string make_multiscale(const fs::path& root) {
    fs::remove_all(root);
    make_zarr_level(root / "0", 128, 64, 0);
    make_zarr_level(root / "1", 64, 64, 1);
    return root.string();
}

}  // namespace

TEST(cache_key_is_fs_safe_and_collision_free) {
    const std::string a = io::cache_key("https://host/scrollA/volumes/v.zarr");
    const std::string b = io::cache_key("https://host/scrollB/volumes/v.zarr");  // same tail
    CHECK(a != b);
    CHECK(a == io::cache_key("https://host/scrollA/volumes/v.zarr"));
    for (char c : a) CHECK(c != '/');
    CHECK(io::cache_key("https://host/x.zarr/") == io::cache_key("https://host/x.zarr"));
}

TEST(cached_surface_transcodes_once_then_serves_locally) {
    const fs::path dir = fs::temp_directory_path() / "fenix_surfcache_test";
    fs::remove_all(dir);
    const std::string seg = make_tifxyz(dir / "seg.tifxyz");
    const std::string cache = (dir / "cache").string();

    auto p1 = io::cached_surface(seg, cache);
    REQUIRE(p1.has_value());
    auto s1 = io::read_fxsurf(*p1);
    REQUIRE(s1.has_value());
    CHECK(s1->nu == 24);
    CHECK(std::abs(s1->scale_u - 20.0f) < 1e-4f);
    CHECK(std::abs(s1->at(10, 10).x - 300.0f) <= 0.25f);  // ZYX mapping survived the cache

    // DELETE the source: the cached entry must still serve (fetch-once achieved).
    fs::remove_all(dir / "seg.tifxyz");
    auto p2 = io::cached_surface(seg, cache);
    REQUIRE(p2.has_value());
    CHECK(*p2 == *p1);
    CHECK(io::read_fxsurf(*p2).has_value());

    // A different source string maps to a different entry (no reuse across sources).
    auto p3 = io::cached_surface(make_tifxyz(dir / "other.tifxyz"), cache);
    REQUIRE(p3.has_value());
    CHECK(*p3 != *p1);
    fs::remove_all(dir);
}

TEST(cached_pyramid_streams_all_levels_and_reopens_offline) {
    const fs::path dir = fs::temp_directory_path() / "fenix_pyr_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string zroot = make_multiscale(dir / "vol.zarr");
    const std::string cache = (dir / "cache").string();

    {
        auto p = io::CachedPyramid::open(zroot, cache, 0.5f);
        REQUIRE(p.has_value());
        CHECK(p->nlods() == 2u);
        CHECK(p->dims() == Extent3{128, 128, 128});
        CHECK(p->dims_at(1) == Extent3{64, 64, 64});
        CHECK(p->src_dtype() == codec::DType::u8);

        std::vector<f32> buf(32 * 32 * 32);
        REQUIRE(p->gather_box_f32(0, 32, 32, 32, 32, 32, 32, buf.data()).has_value());
        CHECK(std::abs(buf[0] - 96.0f) < 4.0f);  // (32+32+32)&0xff, codec tolerance
        std::vector<f32> buf1(16 * 16 * 16);
        REQUIRE(p->gather_box_f32(1, 16, 16, 16, 16, 16, 16, buf1.data()).has_value());
        CHECK(std::abs(buf1[0] - 96.0f) < 4.0f);  // same LOD-0 point (16*2 per axis)

        auto blk = p->block16(0, ChunkCoord{2, 2, 2});  // voxel (32,32,32)
        REQUIRE(blk.has_value());
        CHECK(std::abs(static_cast<f32>((**blk)[0]) - 96.0f) < 5.0f);
    }
    // OFFLINE: source gone, the caches carry the touched region; untouched hard-fails.
    fs::remove_all(dir / "vol.zarr");
    {
        auto p = io::CachedPyramid::open(zroot, cache, 0.5f);
        REQUIRE(p.has_value());
        CHECK(p->nlods() == 2u);
        std::vector<f32> buf(32 * 32 * 32);
        REQUIRE(p->gather_box_f32(0, 32, 32, 32, 32, 32, 32, buf.data()).has_value());
        CHECK(std::abs(buf[0] - 96.0f) < 4.0f);
        std::vector<f32> far(8 * 8 * 8);
        CHECK(!p->gather_box_f32(0, 96, 96, 96, 8, 8, 8, far.data()).has_value());  // never silent air
    }
    fs::remove_all(dir);
}

TEST(cached_pyramid_adaptive_render_never_blocks_then_converges) {
    const fs::path dir = fs::temp_directory_path() / "fenix_pyr_adaptive_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string zroot = make_multiscale(dir / "vol.zarr");

    auto p = io::CachedPyramid::open(zroot, (dir / "cache").string(), 0.5f);
    REQUIRE(p.has_value());
    // Warm ONLY the coarse level (the overview floor); level 0 stays cold.
    std::vector<f32> warm(64 * 64 * 64);
    REQUIRE(p->gather_box_f32(1, 0, 0, 0, 64, 64, 64, warm.data()).has_value());
    CHECK(p->chunk_state(0, ChunkCoord{0, 0, 0}) == codec::Coverage::Absent);

    view::SliceEngine engine(*p);
    view::SliceSpec s;
    s.axis = view::SliceAxis::z;
    s.slice = 32;
    s.center_u = 64;
    s.center_v = 64;
    s.zoom = 1.0f;  // LOD 0 — cold
    s.width = 64;
    s.height = 64;
    // Best-effort: returns immediately with coarse-fallback data + missing chunks
    // scheduled in the background; value comes from the warmed level-1 data.
    auto img = engine.render_available(s);
    REQUIRE(img.has_value());
    CHECK(img->lod == 0);
    CHECK(img->missing > 0);
    CHECK(std::abs(img->pix[static_cast<usize>(32 * 64 + 32)] - 160.0f) < 8.0f);  // coarse ≈ right

    // Eventual consistency: background fills land, generation bumps, re-render is sharp.
    const u64 g0 = p->ready_generation();
    for (int spin = 0; spin < 5000; ++spin) {
        auto again = engine.render_available(s);
        REQUIRE(again.has_value());
        if (again->missing == 0) {
            CHECK(p->ready_generation() != g0);
            CHECK(std::abs(again->pix[static_cast<usize>(32 * 64 + 32)] - 160.0f) < 5.0f);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        REQUIRE(spin < 4999);  // must converge
    }
    fs::remove_all(dir);
}

TEST(cached_pyramid_drives_slice_engine) {
    const fs::path dir = fs::temp_directory_path() / "fenix_pyr_engine_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string zroot = make_multiscale(dir / "vol.zarr");

    auto p = io::CachedPyramid::open(zroot, (dir / "cache").string(), 0.5f);
    REQUIRE(p.has_value());
    view::SliceEngine engine(*p);
    view::SliceSpec s;
    s.axis = view::SliceAxis::z;
    s.slice = 32;
    s.center_u = 64;
    s.center_v = 64;
    s.zoom = 1.0f;
    s.width = 64;
    s.height = 64;
    auto img = engine.render(s);
    REQUIRE(img.has_value());
    CHECK(img->lod == 0);
    // pixel at volume (32, 64, 64): value (32+64+64)&0xff = 160
    CHECK(std::abs(img->pix[static_cast<usize>(32 * 64 + 32)] - 160.0f) < 5.0f);

    s.zoom = 0.5f;  // zoomed out -> LOD 1 must serve (streamed from the level-1 zarr)
    auto img1 = engine.render(s);
    REQUIRE(img1.has_value());
    CHECK(img1->lod == 1);
    fs::remove_all(dir);
}
