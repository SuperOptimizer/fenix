// test_fft.cpp — radix-2 FFT roundtrip + known transforms.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "preprocess/fft.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using namespace fenix;
using namespace fenix::preprocess;

TEST(fft_roundtrip) {
    std::vector<cf32> a(64);
    Pcg32 rng{31};
    for (auto& x : a) x = cf32(rng.next_f32() * 2 - 1, rng.next_f32() * 2 - 1);
    std::vector<cf32> orig = a;
    fft1d(a, false);
    fft1d(a, true);
    f32 max_err = 0;
    for (usize i = 0; i < a.size(); ++i) max_err = std::max(max_err, std::abs(a[i] - orig[i]));
    CHECK(max_err < 1e-3f);
}

TEST(fft_delta_is_flat) {
    std::vector<cf32> a(32, cf32(0, 0));
    a[0] = cf32(1, 0);  // delta at 0 -> all-ones spectrum
    fft1d(a, false);
    for (const cf32& x : a) {
        CHECK(std::abs(x.real() - 1.0f) < 1e-4f);
        CHECK(std::abs(x.imag()) < 1e-4f);
    }
}

TEST(fft_cosine_has_two_peaks) {
    const usize n = 64;
    const int k = 5;  // frequency
    std::vector<cf32> a(n);
    for (usize i = 0; i < n; ++i)
        a[i] = cf32(std::cos(2.0f * std::numbers::pi_v<f32> * static_cast<f32>(k) *
                             static_cast<f32>(i) / static_cast<f32>(n)),
                    0.0f);
    fft1d(a, false);
    // Real cosine -> peaks of magnitude n/2 at bins k and n-k.
    auto mag = [&](usize b) { return std::abs(a[b]); };
    CHECK(mag(static_cast<usize>(k)) > 0.4f * static_cast<f32>(n));
    CHECK(mag(n - static_cast<usize>(k)) > 0.4f * static_cast<f32>(n));
    CHECK(mag(static_cast<usize>(k) + 2) < 1.0f);  // other bins ~0
}

TEST(next_pow2_helper) {
    CHECK(next_pow2(1) == 1);
    CHECK(next_pow2(5) == 8);
    CHECK(next_pow2(64) == 64);
    CHECK(next_pow2(176) == 256);
}
