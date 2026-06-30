// test_dct_block.cpp — the DCT-16 lossy block codec (codec/dct_block.hpp) across ALL supported dtypes
// (u8/u16/u32/s8/s16/s32/f16/f32). For each: encode a smooth 16³ block, decode, and require the
// round-trip error to stay within the quant step and the payload to compress. Confirms the dtype I/O
// layer (widen→f32→DCT→quant→entropy→dequant→IDCT→narrow) round-trips every type.
#define FENIX_TEST_MAIN
#include "codec/dct_block.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

#include <cmath>
#include <vector>

using namespace fenix;
using namespace fenix::codec;

namespace {
struct Stat {
    f64 mae, maxerr, ratio;
};

// Build a smooth kDctN³ block of dtype T scaled around `mag`, encode at step q, decode, measure.
template <class T>
Stat run(f32 mag, f32 q) {
    constexpr s64 N = kDctN, V = N * N * N;
    std::vector<T> blk(static_cast<usize>(V));
    const bool sgn = std::is_signed_v<T> || std::same_as<T, f16> || std::same_as<T, f32>;
    for (s64 z = 0; z < N; ++z)
        for (s64 y = 0; y < N; ++y)
            for (s64 x = 0; x < N; ++x) {
                const f32 s = 0.35f * std::cos(0.2f * x) * std::cos(0.15f * y) + 0.1f * static_cast<f32>(z) / N;
                const f32 v = sgn ? mag * s : 0.5f * mag + 0.45f * mag * s;  // signed: centered; unsigned: [~0.05,0.95]*mag
                blk[static_cast<usize>((z * N + y) * N + x)] = from_f32_one<T>(v);
            }
    DctParams pr;
    pr.q = q;
    const std::vector<u8> payload = encode_block_dct<T>(blk, pr);
    const std::vector<T> dec = decode_block_dct_to<T>(payload, pr);
    f64 sae = 0, mx = 0;
    for (usize i = 0; i < blk.size(); ++i) {
        const f64 e = std::abs(static_cast<f64>(blk[i]) - static_cast<f64>(dec[i]));
        sae += e;
        mx = std::max(mx, e);
    }
    return {sae / static_cast<f64>(V), mx, static_cast<f64>(V * sizeof(T)) / static_cast<f64>(payload.size())};
}

template <class T>
void check(const char* name, f32 mag, f32 q) {
    const Stat s = run<T>(mag, q);
    std::printf("  [%-4s mag=%.0f q=%.2f: MAE %.3f  max %.3f  ratio %.2fx]\n", name, mag, q, s.mae, s.maxerr, s.ratio);
    CHECK(s.mae < q);          // mean error within the quant step (orthonormal transform => spatial≈coef error)
    CHECK(s.maxerr < 12.0 * q);// no wild outliers
    CHECK(s.ratio > 1.5);      // smooth block compresses
}
}  // namespace

TEST(dct_block_roundtrips_all_dtypes) {
    check<u8>("u8", 220.0f, 4.0f);
    check<u16>("u16", 40000.0f, 600.0f);
    check<u32>("u32", 1.0e6f, 15000.0f);
    check<s8>("s8", 110.0f, 2.0f);
    check<s16>("s16", 30000.0f, 500.0f);
    check<s32>("s32", 1.0e6f, 15000.0f);
    check<f16>("f16", 100.0f, 1.5f);
    check<f32>("f32", 500.0f, 8.0f);
}
