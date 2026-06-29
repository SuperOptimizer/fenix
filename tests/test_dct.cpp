// test_dct.cpp — the separable float DCT-16 (codec/dct.hpp): exact (fp) round-trip, Parseval energy
// preservation (orthonormality), low-frequency energy compaction on a smooth block, and the DC term.
#define FENIX_TEST_MAIN
#include "codec/dct.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

#include <cmath>
#include <vector>

using namespace fenix;
using namespace fenix::codec;

namespace {
f64 sumsq(std::span<const f32> v) {
    f64 s = 0;
    for (f32 x : v) s += static_cast<f64>(x) * x;
    return s;
}
}  // namespace

TEST(dct16_3d_roundtrip_and_parseval) {
    constexpr s64 N = kDctN, V = N * N * N;
    std::vector<f32> blk(V), orig(V);
    for (s64 i = 0; i < V; ++i) orig[i] = blk[i] = 3.0f * std::sin(0.07f * static_cast<f32>(i)) + std::cos(0.013f * static_cast<f32>(i * i));
    const f64 e_in = sumsq(orig);
    dct16_3d_fwd(blk);
    const f64 e_freq = sumsq(blk);
    dct16_3d_inv(blk);
    f64 maxerr = 0;
    for (s64 i = 0; i < V; ++i) maxerr = std::max(maxerr, std::abs(static_cast<f64>(blk[i] - orig[i])));
    std::printf("  [dct3: maxerr %.2e | energy in %.4f freq %.4f]\n", maxerr, e_in, e_freq);
    CHECK(maxerr < 1e-2);                              // round-trip exact to fp accumulation
    CHECK(std::abs(e_freq - e_in) < 1e-3 * e_in);      // Parseval: orthonormal transform preserves energy
}

TEST(dct16_3d_energy_compaction) {
    constexpr s64 N = kDctN, V = N * N * N;
    std::vector<f32> blk(V);  // smooth low-frequency block -> energy should pack into the low-freq corner
    for (s64 z = 0; z < N; ++z)
        for (s64 y = 0; y < N; ++y)
            for (s64 x = 0; x < N; ++x)
                blk[(z * N + y) * N + x] = 100.0f + 20.0f * std::cos(0.2f * static_cast<f32>(x)) +
                                           15.0f * std::cos(0.15f * static_cast<f32>(y)) + 10.0f * static_cast<f32>(z) / N;
    const f64 total = sumsq(blk);
    dct16_3d_fwd(blk);
    f64 low = 0;  // energy in the 4x4x4 lowest-frequency corner
    for (s64 z = 0; z < 4; ++z)
        for (s64 y = 0; y < 4; ++y)
            for (s64 x = 0; x < 4; ++x) low += static_cast<f64>(blk[(z * N + y) * N + x]) * blk[(z * N + y) * N + x];
    const f64 frac = low / sumsq(blk);
    std::printf("  [dct3 compaction: %.4f of energy in 4^3 low corner]\n", frac);
    CHECK(frac > 0.99);
    CHECK(std::abs(sumsq(blk) - total) < 1e-3 * total);
}

TEST(dct16_dc_term) {
    constexpr s64 N = kDctN, V = N * N * N;
    std::vector<f32> blk(V, 7.0f);  // constant block -> all energy in the DC coefficient
    dct16_3d_fwd(blk);
    // orthonormal DCT-II DC gain per axis = sqrt(1/N); 3D DC = (1/sqrt(N))^3 * sum = 7 * N^3 / N^1.5 = 7*N^1.5
    const f32 expect = 7.0f * std::pow(static_cast<f32>(N), 1.5f);
    f64 rest = sumsq(blk) - static_cast<f64>(blk[0]) * blk[0];
    std::printf("  [dct3 DC: coef0 %.2f (expect %.2f) rest-energy %.2e]\n", blk[0], expect, rest);
    CHECK(std::abs(blk[0] - expect) < 1e-1);
    CHECK(rest < 1e-2);
}

TEST(dct16_2d_roundtrip) {
    constexpr s64 N = kDctN, A = N * N;
    std::vector<f32> blk(A), orig(A);
    for (s64 i = 0; i < A; ++i) orig[i] = blk[i] = std::sin(0.3f * static_cast<f32>(i)) - 0.5f * static_cast<f32>(i % 7);
    dct16_2d_fwd(blk);
    dct16_2d_inv(blk);
    f64 maxerr = 0;
    for (s64 i = 0; i < A; ++i) maxerr = std::max(maxerr, std::abs(static_cast<f64>(blk[i] - orig[i])));
    CHECK(maxerr < 1e-3);
}
