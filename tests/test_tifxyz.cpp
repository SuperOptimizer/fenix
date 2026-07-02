// tests/test_tifxyz.cpp — the tifxyz importer (first-party TIFF reader + meta.json scale)
// and the .fxsurf v2 quantized/delta/rANS container it feeds.
#define FENIX_TEST_MAIN
#include "core/test.hpp"
#include "io/surface.hpp"
#include "io/tifxyz.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

using namespace fenix;
namespace fs = std::filesystem;

namespace {

// Minimal classic-LE single-strip f32 TIFF writer (test fixture only — fenix never writes TIFF).
void write_test_tiff(const std::string& path, s64 w, s64 h, const std::vector<f32>& pix) {
    std::ofstream f(path, std::ios::binary);
    auto u16w = [&](u16 v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    auto u32w = [&](u32 v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    const u32 data_bytes = static_cast<u32>(w * h * 4);
    const u32 ifd_off = 8 + data_bytes;
    f.write("II", 2);
    u16w(42);
    u32w(ifd_off);
    f.write(reinterpret_cast<const char*>(pix.data()), data_bytes);
    const u16 nent = 9;
    u16w(nent);
    auto ent = [&](u16 tag, u16 type, u32 cnt, u32 val) {
        u16w(tag);
        u16w(type);
        u32w(cnt);
        u32w(val);
    };
    ent(256, 3, 1, static_cast<u32>(w));  // width
    ent(257, 3, 1, static_cast<u32>(h));  // height
    ent(258, 3, 1, 32);                   // bits
    ent(259, 3, 1, 1);                    // uncompressed
    ent(273, 4, 1, 8);                    // strip offset
    ent(277, 3, 1, 1);                    // samples/pixel
    ent(278, 3, 1, static_cast<u32>(h));  // rows/strip
    ent(279, 4, 1, data_bytes);           // strip bytes
    ent(339, 3, 1, 3);                    // sampleformat = float
    u32w(0);                              // next IFD
}

Surface make_wavy(s64 nu, s64 nv) {
    Surface s(nu, nv);
    for (s64 v = 0; v < nv; ++v)
        for (s64 u = 0; u < nu; ++u) {
            const f32 fu = static_cast<f32>(u), fv = static_cast<f32>(v);
            s.set(u,
                  v,
                  Vec3f{100.0f + 3.0f * std::sin(fu * 0.1f) + fv * 0.5f,
                        2000.0f + fv * 20.0f,
                        500.0f + fu * 20.0f + 5.0f * std::cos(fv * 0.2f)});
        }
    return s;
}

}  // namespace

TEST(fxsurf_v2_roundtrip_quantized) {
    const std::string p = "test_fxsurf_v2.fxsurf";
    Surface s = make_wavy(64, 48);
    s.scale_u = 20.0f;
    s.scale_v = 20.0f;
    // punch some invalid holes
    for (s64 u = 10; u < 20; ++u)
        for (s64 v = 5; v < 30; ++v) s.valid[s.idx(u, v)] = 0;
    REQUIRE(io::write_fxsurf(p, s).has_value());
    auto r = io::read_fxsurf(p);
    REQUIRE(r.has_value());
    CHECK(r->nu == s.nu);
    CHECK(r->nv == s.nv);
    CHECK(r->scale_u == 20.0f);
    const f32 tol = 1.0f / 16.0f;  // the coord quant step
    for (s64 v = 0; v < s.nv; ++v)
        for (s64 u = 0; u < s.nu; ++u) {
            CHECK(r->is_valid(u, v) == s.is_valid(u, v));
            if (!s.is_valid(u, v)) continue;
            const Vec3f a = s.at(u, v), b = r->at(u, v);
            CHECK(std::abs(a.z - b.z) <= tol);
            CHECK(std::abs(a.y - b.y) <= tol);
            CHECK(std::abs(a.x - b.x) <= tol);
        }
    std::remove(p.c_str());
}

TEST(fxsurf_v2_channels_roundtrip) {
    const std::string p = "test_fxsurf_v2c.fxsurf";
    Surface s = make_wavy(32, 32);
    s.alloc_channels();
    for (usize i = 0; i < s.normal.size(); ++i) {
        s.normal[i] = Vec3f{0.6f, 0.64f, 0.48f};
        s.conf[i] = 1.5f;
    }
    REQUIRE(io::write_fxsurf(p, s).has_value());
    auto r = io::read_fxsurf(p);
    REQUIRE(r.has_value());
    REQUIRE(r->has_channels());
    CHECK(std::abs(r->normal[7].y - 0.64f) <= 1.0f / 16384.0f);
    CHECK(std::abs(r->conf[7] - 1.5f) <= 1.0f / 256.0f);
    std::remove(p.c_str());
}

TEST(fxsurf_v2_compresses_smooth_grids) {
    const std::string p = "test_fxsurf_ratio.fxsurf";
    Surface s = make_wavy(256, 256);
    REQUIRE(io::write_fxsurf(p, s).has_value());
    const auto sz = fs::file_size(p);
    const auto raw = static_cast<uintmax_t>(s.nu * s.nv) * (sizeof(Vec3f) + 1);
    CHECK(sz * 4 < raw);  // at least 4x smaller than the v1 raw dump on smooth data
    std::remove(p.c_str());
}

TEST(fxsurf_rejects_corrupt) {
    const std::string p = "test_fxsurf_bad.fxsurf";
    {
        std::ofstream f(p, std::ios::binary);
        f << "FXSF garbage that is not a real container";
    }
    CHECK(!io::read_fxsurf(p).has_value());
    std::remove(p.c_str());
}

TEST(tifxyz_import_end_to_end) {
    const fs::path dir = fs::temp_directory_path() / "fenix_tifxyz_test";
    fs::create_directories(dir);
    const s64 w = 40, h = 30;
    std::vector<f32> xs(static_cast<usize>(w * h)), ys(xs.size()), zs(xs.size());
    for (s64 v = 0; v < h; ++v)
        for (s64 u = 0; u < w; ++u) {
            const usize i = static_cast<usize>(v * w + u);
            if (u < 4 && v < 4) {
                xs[i] = ys[i] = zs[i] = -1.0f;
                continue;
            }  // invalid corner
            xs[i] = 100.0f + static_cast<f32>(u) * 20.0f;
            ys[i] = 200.0f + static_cast<f32>(v) * 20.0f;
            zs[i] = 5000.0f + static_cast<f32>(u + v);
        }
    write_test_tiff((dir / "x.tif").string(), w, h, xs);
    write_test_tiff((dir / "y.tif").string(), w, h, ys);
    write_test_tiff((dir / "z.tif").string(), w, h, zs);
    {
        std::ofstream m(dir / "meta.json");
        m << R"({"format":"tifxyz","scale":[0.05,0.05],"type":"seg"})";
    }

    auto s = io::read_tifxyz(dir.string());
    REQUIRE(s.has_value());
    CHECK(s->nu == w);
    CHECK(s->nv == h);
    CHECK(std::abs(s->scale_u - 20.0f) < 1e-4f);
    CHECK(!s->is_valid(0, 0));
    CHECK(s->is_valid(10, 10));
    // ZYX order: coord.z from z.tif, coord.x from x.tif
    const Vec3f c = s->at(10, 10);
    CHECK(std::abs(c.z - 5020.0f) < 1e-3f);
    CHECK(std::abs(c.y - 400.0f) < 1e-3f);
    CHECK(std::abs(c.x - 300.0f) < 1e-3f);

    // full pipe: -> .fxsurf -> read back within quant tolerance
    const std::string out = (dir / "seg.fxsurf").string();
    REQUIRE(io::write_fxsurf(out, *s).has_value());
    auto r = io::read_fxsurf(out);
    REQUIRE(r.has_value());
    CHECK(std::abs(r->at(10, 10).x - 300.0f) <= 1.0f / 16.0f);
    fs::remove_all(dir);
}

TEST(tiff_rejects_garbage) {
    const fs::path p = fs::temp_directory_path() / "fenix_bad.tif";
    {
        std::ofstream f(p, std::ios::binary);
        f << "MM not a le tiff";
    }
    CHECK(!io::read_tiff(p.string()).has_value());
    std::vector<u8> junk{'I', 'I', 42, 0, 200, 0, 0, 0};  // IFD offset past EOF
    CHECK(!io::decode_tiff(junk).has_value());
    fs::remove(p);
}
