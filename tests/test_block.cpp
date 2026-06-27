// test_block.cpp — end-to-end lossy block codec (DWT+quant+rANS) roundtrip.
#define FENIX_TEST_MAIN
#include "codec/block.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

#include <cmath>
#include <vector>

using namespace fenix;
using namespace fenix::codec;

// Synthetic CT-like 64^3 block: smooth background + a wavy "sheet" + a little noise.
static std::vector<f32> make_block(s64 side) {
    std::vector<f32> b(static_cast<usize>(side * side * side));
    Pcg32 rng{11};
    const Index3 st{side * side, side, 1};
    for (s64 z = 0; z < side; ++z)
        for (s64 y = 0; y < side; ++y)
            for (s64 x = 0; x < side; ++x) {
                f32 sheet = 120.0f * std::exp(-0.5f * std::pow((static_cast<f32>(x) -
                                                                32.0f - 6.0f * std::sin(static_cast<f32>(z) * 0.2f)) /
                                                                   3.0f, 2.0f));
                f32 bg = 30.0f + 0.2f * static_cast<f32>(y);
                f32 noise = rng.next_f32() * 2.0f;
                b[static_cast<usize>(z * st.z + y * st.y + x * st.x)] = bg + sheet + noise;
            }
    return b;
}

static void stats(std::span<const f32> a, std::span<const f32> b, f32& max_err, f64& mse) {
    max_err = 0;
    mse = 0;
    for (usize i = 0; i < a.size(); ++i) {
        f32 d = std::abs(a[i] - b[i]);
        max_err = std::max(max_err, d);
        mse += static_cast<f64>(d) * static_cast<f64>(d);
    }
    mse /= static_cast<f64>(a.size());
}

TEST(block_roundtrip_near_lossless) {
    const s64 side = 64;
    auto orig = make_block(side);
    auto enc = encode_block(orig, side, {.q = 2.0f, .levels = 4});
    auto dec = decode_block(enc);
    REQUIRE(dec.size() == orig.size());
    f32 max_err;
    f64 mse;
    stats(orig, dec, max_err, mse);
    // Quantization step q=2 -> small reconstruction error; CDF 9/7 isn't orthonormal so
    // voxel error exceeds q/2 but stays modest.
    CHECK(max_err < 12.0f);
    CHECK(mse < 4.0);
    CHECK(enc.size() < orig.size() * sizeof(f32));  // compresses vs raw f32
}

TEST(block_higher_q_compresses_more_lower_fidelity) {
    const s64 side = 64;
    auto orig = make_block(side);
    auto lo_q = encode_block(orig, side, {.q = 2.0f, .levels = 4});
    auto hi_q = encode_block(orig, side, {.q = 32.0f, .levels = 4});
    CHECK(hi_q.size() < lo_q.size());  // coarser quant -> smaller

    f32 me_lo, me_hi;
    f64 mse_lo, mse_hi;
    stats(orig, decode_block(lo_q), me_lo, mse_lo);
    stats(orig, decode_block(hi_q), me_hi, mse_hi);
    CHECK(mse_hi > mse_lo);  // coarser quant -> more error
    // Even at q=32, a ~120-amplitude sheet survives recognizably.
    CHECK(me_hi < 120.0f);
}

TEST(block_flat_region_compresses_hard) {
    const s64 side = 64;
    std::vector<f32> flat(static_cast<usize>(side * side * side), 50.0f);
    auto enc = encode_block(flat, side, {.q = 8.0f, .levels = 4});
    auto dec = decode_block(enc);
    f32 max_err;
    f64 mse;
    stats(flat, dec, max_err, mse);
    CHECK(max_err < 3.0f);     // constant field: only the quantized DC contributes (~q/4)
    CHECK(enc.size() < 2000);  // 262144 voxels -> tiny (~900x)
}
