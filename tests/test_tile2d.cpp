// tests/test_tile2d.cpp — the generic 64² DCT tile codec (codec/tile2d.hpp): scalar fields,
// RGB (YCoCg), ZYX coordinate surfaces (affine + tangent frame), tolerance verification,
// raw-tile fallback, and corrupt-input rejection.
#define FENIX_TEST_MAIN
#include "codec/tile2d.hpp"
#include "core/test.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace fenix;

namespace {
std::vector<f32> smooth_field(s64 w, s64 h, f32 amp, u64 seed) {
    std::vector<f32> f(static_cast<usize>(w * h));
    std::mt19937_64 rng(seed);
    const f32 pz = static_cast<f32>(rng() % 7 + 1) * 0.017f;
    for (s64 y = 0; y < h; ++y)
        for (s64 x = 0; x < w; ++x)
            f[static_cast<usize>(y * w + x)] = amp * (std::sin(static_cast<f32>(x) * pz) +
                                                      std::cos(static_cast<f32>(y) * 0.031f)) +
                                               0.2f * static_cast<f32>(x) + 0.1f * static_cast<f32>(y);
    return f;
}
}  // namespace

TEST(tile2d_scalar_roundtrip_within_tau) {
    const s64 w = 150, h = 100;  // non-multiple of 64 exercises edge tiles
    const f32 tau = 0.05f;
    auto f = smooth_field(w, h, 10.0f, 1);
    auto enc = codec::encode_field2d(f, w, h, tau, tau);
    auto dec = codec::decode_field2d(enc, w, h);
    REQUIRE(dec.has_value());
    f32 maxe = 0;
    for (usize i = 0; i < f.size(); ++i) maxe = std::max(maxe, std::abs((*dec)[i] - f[i]));
    CHECK(maxe <= tau);
    CHECK(enc.size() * 3 < f.size() * 4);  // beats raw f32 comfortably on smooth data
}

TEST(tile2d_scalar_respects_validity) {
    const s64 w = 64, h = 64;
    const f32 tau = 0.01f;
    auto f = smooth_field(w, h, 5.0f, 2);
    std::vector<u8> valid(f.size(), 1);
    for (usize i = 0; i < f.size(); i += 3) {
        f[i] = 1e9f;  // garbage in invalid cells must not poison the tile or the verify
        valid[i] = 0;
    }
    auto enc = codec::encode_field2d(f, w, h, tau, tau, valid);
    auto dec = codec::decode_field2d(enc, w, h);
    REQUIRE(dec.has_value());
    for (usize i = 0; i < f.size(); ++i)
        if (valid[i]) CHECK(std::abs((*dec)[i] - f[i]) <= tau);
}

TEST(tile2d_raw_fallback_on_noise) {
    // White noise defeats the DCT: the verify loop must fall back to raw and STILL meet tau.
    const s64 w = 64, h = 64;
    const f32 tau = 0.01f;
    std::vector<f32> f(static_cast<usize>(w * h));
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<f32> d(-100.0f, 100.0f);
    for (auto& v : f) v = d(rng);
    auto enc = codec::encode_field2d(f, w, h, tau, tau);
    auto dec = codec::decode_field2d(enc, w, h);
    REQUIRE(dec.has_value());
    f32 maxe = 0;
    for (usize i = 0; i < f.size(); ++i) maxe = std::max(maxe, std::abs((*dec)[i] - f[i]));
    CHECK(maxe <= tau);
}

TEST(tile2d_rgb_roundtrip) {
    const s64 w = 130, h = 70;
    const f32 tau = 0.5f;  // ~half an 8-bit level
    auto r = smooth_field(w, h, 40.0f, 4), g = smooth_field(w, h, 35.0f, 5), b = smooth_field(w, h, 30.0f, 6);
    auto enc = codec::encode_rgb2d(r, g, b, w, h, tau);
    auto dec = codec::decode_rgb2d(enc, w, h);
    REQUIRE(dec.has_value());
    f32 maxe = 0;
    for (usize i = 0; i < r.size(); ++i) {
        maxe = std::max(maxe, std::abs(dec->r[i] - r[i]));
        maxe = std::max(maxe, std::abs(dec->g[i] - g[i]));
        maxe = std::max(maxe, std::abs(dec->b[i] - b[i]));
    }
    CHECK(maxe <= tau);
}

TEST(tile2d_coords_roundtrip_3d_bound) {
    const s64 nu = 200, nv = 150;
    const f32 tau = 1.0f / 16.0f;
    std::vector<Vec3f> c(static_cast<usize>(nu * nv));
    std::vector<u8> valid(c.size(), 1);
    for (s64 v = 0; v < nv; ++v)
        for (s64 u = 0; u < nu; ++u) {
            const usize i = static_cast<usize>(v * nu + u);
            const f32 fu = static_cast<f32>(u), fv = static_cast<f32>(v);
            c[i] = Vec3f{4000.0f + 8.0f * std::sin(fu * 0.05f) + fv * 0.3f, 9000.0f + fv * 20.0f,
                         27000.0f + fu * 20.0f + 3.0f * std::cos(fv * 0.09f)};
        }
    for (s64 v = 40; v < 90; ++v)
        for (s64 u = 10; u < 60; ++u) {
            const usize i = static_cast<usize>(v * nu + u);
            valid[i] = 0;
            c[i] = Vec3f{-1, -1, -1};  // invalid garbage must not leak into valid cells
        }
    auto enc = codec::encode_coords2d(c, valid, nu, nv, tau);
    auto dec = codec::decode_coords2d(enc, nu, nv);
    REQUIRE(dec.has_value());
    f32 maxe = 0;
    for (usize i = 0; i < c.size(); ++i) {
        if (!valid[i]) continue;
        maxe = std::max(maxe, norm((*dec)[i] - c[i]));
    }
    CHECK(maxe <= tau);
    // decorrelation should crush this: raw is 12 B/cell
    CHECK(enc.size() * 8 < c.size() * 12);
}

TEST(tile2d_coords_degenerate_tiles) {
    // A tile with <3 valid cells (rank-deficient fit) must still roundtrip within tau.
    const s64 nu = 64, nv = 64;
    const f32 tau = 1.0f / 16.0f;
    std::vector<Vec3f> c(static_cast<usize>(nu * nv), Vec3f{0, 0, 0});
    std::vector<u8> valid(c.size(), 0);
    c[5] = Vec3f{123.4f, 567.8f, 9012.3f};
    valid[5] = 1;
    c[100] = Vec3f{124.0f, 568.0f, 9013.0f};
    valid[100] = 1;
    auto enc = codec::encode_coords2d(c, valid, nu, nv, tau);
    auto dec = codec::decode_coords2d(enc, nu, nv);
    REQUIRE(dec.has_value());
    CHECK(norm((*dec)[5] - c[5]) <= tau);
    CHECK(norm((*dec)[100] - c[100]) <= tau);
}

TEST(tile2d_rejects_corrupt) {
    const s64 w = 100, h = 80;
    auto f = smooth_field(w, h, 10.0f, 7);
    auto enc = codec::encode_field2d(f, w, h, 0.05f, 0.05f);
    CHECK(!codec::decode_field2d(std::span<const u8>(enc).first(6)).has_value());
    CHECK(!codec::decode_field2d(enc, w + 1, h).has_value());  // dims mismatch
    auto bad = enc;
    for (usize i = 20; i < bad.size() && i < 200; i += 7) bad[i] ^= 0x5a;
    auto r = codec::decode_field2d(bad, w, h);  // must not crash; error or garbage values both fine
    (void)r;
    std::vector<u8> junk(64, 0xab);
    CHECK(!codec::decode_coords2d(junk, 64, 64).has_value());
}
