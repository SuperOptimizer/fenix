// test_wavelet.cpp — CDF 9/7 transform roundtrip (lossless up to fp error).
#define FENIX_TEST_MAIN
#include "codec/wavelet.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

#include <cmath>
#include <vector>

using namespace fenix;
using namespace fenix::codec;

static f32 max_abs_diff(std::span<const f32> a, std::span<const f32> b) {
    f32 m = 0;
    for (usize i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

TEST(dwt1d_roundtrip) {
    std::vector<f32> orig(64), work(64);
    Pcg32 rng{1};
    for (usize i = 0; i < orig.size(); ++i) orig[i] = rng.next_f32() * 255.0f;
    work = orig;
    std::vector<f32> scratch;
    fenix::codec::detail::fwd1d(work, scratch);
    fenix::codec::detail::inv1d(work, scratch);
    CHECK(max_abs_diff(orig, work) < 1e-3f);
}

TEST(dwt1d_odd_length_roundtrip) {
    std::vector<f32> orig(37), work;
    Pcg32 rng{2};
    for (auto& v : orig) v = rng.next_f32();
    work = orig;
    std::vector<f32> scratch;
    fenix::codec::detail::fwd1d(work, scratch);
    fenix::codec::detail::inv1d(work, scratch);
    CHECK(max_abs_diff(orig, work) < 1e-4f);
}

TEST(dwt3_block_roundtrip_multilevel) {
    const s64 side = 64;
    std::vector<f32> orig(static_cast<usize>(side * side * side)), work;
    Pcg32 rng{3};
    for (auto& v : orig) v = rng.next_f32() * 1000.0f;
    work = orig;
    dwt3_forward(work, side, 4);
    // Energy compaction sanity: the LLL corner (8^3) should hold large coefficients.
    f32 lll_energy = 0, total_energy = 0;
    const Index3 st{side * side, side, 1};
    for (s64 z = 0; z < side; ++z)
        for (s64 y = 0; y < side; ++y)
            for (s64 x = 0; x < side; ++x) {
                f32 c = work[static_cast<usize>(z * st.z + y * st.y + x * st.x)];
                total_energy += c * c;
                if (z < 8 && y < 8 && x < 8) lll_energy += c * c;
            }
    CHECK(total_energy > 0.0f);
    CHECK(lll_energy > 0.0f);
    dwt3_inverse(work, side, 4);
    CHECK(max_abs_diff(orig, work) < 1e-1f);  // 64^3 * 4 levels accumulates fp error
}

TEST(dwt2_roundtrip_and_compaction) {
    const s64 side = 64;
    std::vector<f32> orig(static_cast<usize>(side * side)), work;
    Pcg32 rng{8};
    for (auto& v : orig) v = rng.next_f32() * 255.0f;
    work = orig;
    dwt2_forward(work, side, 4);
    dwt2_inverse(work, side, 4);
    CHECK(max_abs_diff(orig, work) < 1e-2f);  // 2D roundtrip exact to fp error

    // Smooth 2D ramp -> energy compacts into the LL corner.
    std::vector<f32> ramp(static_cast<usize>(side * side));
    for (s64 y = 0; y < side; ++y)
        for (s64 x = 0; x < side; ++x) ramp[static_cast<usize>(y * side + x)] = static_cast<f32>(y + x);
    dwt2_forward(ramp, side, 3);
    f64 ll = 0, tot = 0;
    const s64 h = side >> 3;
    for (s64 y = 0; y < side; ++y)
        for (s64 x = 0; x < side; ++x) {
            f64 c = static_cast<f64>(ramp[static_cast<usize>(y * side + x)]);
            tot += c * c;
            if (y < h && x < h) ll += c * c;
        }
    CHECK(ll / tot > 0.99);
}

TEST(dwt3_smooth_block_compacts_energy) {
    // A linear ramp has no high-frequency content; 9/7's vanishing moments push (almost)
    // all energy into the coarse LLL corner. After `levels` on a side^3 block the LLL
    // corner is [0 : side>>levels]^3. (Tiny blocks are boundary-dominated; use 64^3.)
    const s64 side = 64;
    const int levels = 3;
    const s64 h = side >> levels;  // LLL corner side = 8
    std::vector<f32> work(static_cast<usize>(side * side * side));
    const Index3 st{side * side, side, 1};
    for (s64 z = 0; z < side; ++z)
        for (s64 y = 0; y < side; ++y)
            for (s64 x = 0; x < side; ++x)
                work[static_cast<usize>(z * st.z + y * st.y + x * st.x)] =
                    static_cast<f32>(z + y + x);
    dwt3_forward(work, side, levels);
    f64 lll_energy = 0, total_energy = 0;
    for (s64 z = 0; z < side; ++z)
        for (s64 y = 0; y < side; ++y)
            for (s64 x = 0; x < side; ++x) {
                f64 c = static_cast<f64>(work[static_cast<usize>(z * st.z + y * st.y + x * st.x)]);
                total_energy += c * c;
                if (z < h && y < h && x < h) lll_energy += c * c;
            }
    CHECK(lll_energy / total_energy > 0.99);  // smooth field -> energy compacted in LLL
}
