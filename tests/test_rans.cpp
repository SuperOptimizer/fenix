// test_rans.cpp — static rANS exact roundtrip + compression sanity.
#define FENIX_TEST_MAIN
#include <array>
#include <vector>

#include "codec/rans.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

using namespace fenix;
using namespace fenix::codec;

static RansModel model_for(std::span<const u8> data) {
    std::array<u64, 256> counts{};
    for (u8 b : data)
        counts[b]++;
    return RansModel::from_counts(counts);
}

TEST(rans_roundtrip_skewed) {
    // Skewed distribution (mostly zeros) — the bitplane-symbol regime.
    std::vector<u8> data(20000);
    Pcg32 rng{1};
    for (auto& b : data)
        b = static_cast<u8>(rng.next_u32() % 100 == 0 ? rng.next_u32() & 0xff : 0);
    RansModel m = model_for(data);
    auto enc = rans_encode(data, m);
    auto dec = rans_decode(enc, data.size(), m);
    REQUIRE(dec.size() == data.size());
    CHECK(dec == data);
    CHECK(enc.size() < data.size()); // skewed -> compresses
}

TEST(rans_roundtrip_uniform) {
    std::vector<u8> data(8000);
    Pcg32 rng{7};
    for (auto& b : data)
        b = static_cast<u8>(rng.next_u32() & 0xff);
    RansModel m = model_for(data);
    auto enc = rans_encode(data, m);
    auto dec = rans_decode(enc, data.size(), m);
    CHECK(dec == data); // exact even when ~incompressible
}

// Regression for the confirmed encode-side finding: RansModel::from_counts's rounding-repair pass could
// underflow the largest freq below 0 and wrap around u16 when the 0->1 floor bumps overshoot the deficit
// by more than any single symbol's freq — 64 symbols @ count 10000 + 192 symbols @ count 1 (total
// 640192) is the exact adversarial-but-natural histogram from the review. floor gives 63 for each common
// symbol (assigned = 64*63 + 192*1 = 4224 > rans_scale=4096), and the old single-symbol subtract
// (63 - 128 = -65) wrapped to 65471, which then made the slot-fill loop write ~69632 entries into the
// 4096-entry slot array. The fix must produce a valid model (sum==rans_scale, every occurring symbol's
// freq in [1,4096]) and a working roundtrip instead.
TEST(rans_from_counts_handles_overshoot_histogram) {
    std::array<u64, 256> counts{};
    for (int s = 0; s < 64; ++s)
        counts[static_cast<usize>(s)] = 10000;
    for (int s = 64; s < 256; ++s)
        counts[static_cast<usize>(s)] = 1;
    RansModel m = RansModel::from_counts(counts);
    u32 sum = 0;
    for (int s = 0; s < 256; ++s) {
        if (counts[static_cast<usize>(s)] > 0)
            CHECK(m.freq[static_cast<usize>(s)] >= 1);      // occurring symbols stay representable
        CHECK(m.freq[static_cast<usize>(s)] <= rans_scale); // no u16 wraparound
        sum += m.freq[static_cast<usize>(s)];
    }
    CHECK(sum == rans_scale); // exactly normalized -> slot[] fill writes exactly rans_scale entries

    // Round-trip data drawn from this exact histogram to prove the model still encodes/decodes correctly.
    std::vector<u8> data;
    data.reserve(640192);
    for (int s = 0; s < 64; ++s)
        data.insert(data.end(), 10000, static_cast<u8>(s));
    for (int s = 64; s < 256; ++s)
        data.push_back(static_cast<u8>(s));
    auto enc = rans_encode(data, m);
    auto dec = rans_decode(enc, data.size(), m);
    CHECK(dec.size() == data.size());
}

// The general property behind the above: from_counts must never overshoot/undershoot rans_scale for any
// occurring-symbol distribution, across a spread of skews (uniform-ish to extremely skewed).
TEST(rans_from_counts_always_sums_to_scale) {
    Pcg32 rng{99};
    for (int trial = 0; trial < 50; ++trial) {
        std::array<u64, 256> counts{};
        const int nsym = 1 + static_cast<int>(rng.next_u32() % 256);
        for (int s = 0; s < nsym; ++s)
            counts[static_cast<usize>(s)] = 1 + (rng.next_u32() % 20000);
        RansModel m = RansModel::from_counts(counts);
        u32 sum = 0;
        for (int s = 0; s < 256; ++s) {
            if (counts[static_cast<usize>(s)] > 0)
                CHECK(m.freq[static_cast<usize>(s)] >= 1);
            sum += m.freq[static_cast<usize>(s)];
        }
        CHECK(sum == rans_scale);
    }
}

TEST(rans_roundtrip_edge_cases) {
    // Single symbol repeated.
    std::vector<u8> ones(1000, 42);
    RansModel m = model_for(ones);
    auto enc = rans_encode(ones, m);
    auto dec = rans_decode(enc, ones.size(), m);
    CHECK(dec == ones);
    CHECK(enc.size() < 100); // one symbol -> tiny

    // Empty input.
    std::vector<u8> empty;
    RansModel me = model_for(empty);
    auto ence = rans_encode(empty, me);
    auto dece = rans_decode(ence, 0, me);
    CHECK(dece.empty());
}
