// test_rans.cpp — static rANS exact roundtrip + compression sanity.
#define FENIX_TEST_MAIN
#include "codec/rans.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

#include <array>
#include <vector>

using namespace fenix;
using namespace fenix::codec;

static RansModel model_for(std::span<const u8> data) {
    std::array<u64, 256> counts{};
    for (u8 b : data) counts[b]++;
    return RansModel::from_counts(counts);
}

TEST(rans_roundtrip_skewed) {
    // Skewed distribution (mostly zeros) — the bitplane-symbol regime.
    std::vector<u8> data(20000);
    Pcg32 rng{1};
    for (auto& b : data) b = static_cast<u8>(rng.next_u32() % 100 == 0 ? rng.next_u32() & 0xff : 0);
    RansModel m = model_for(data);
    auto enc = rans_encode(data, m);
    auto dec = rans_decode(enc, data.size(), m);
    REQUIRE(dec.size() == data.size());
    CHECK(dec == data);
    CHECK(enc.size() < data.size());  // skewed -> compresses
}

TEST(rans_roundtrip_uniform) {
    std::vector<u8> data(8000);
    Pcg32 rng{7};
    for (auto& b : data) b = static_cast<u8>(rng.next_u32() & 0xff);
    RansModel m = model_for(data);
    auto enc = rans_encode(data, m);
    auto dec = rans_decode(enc, data.size(), m);
    CHECK(dec == data);  // exact even when ~incompressible
}

TEST(rans_roundtrip_edge_cases) {
    // Single symbol repeated.
    std::vector<u8> ones(1000, 42);
    RansModel m = model_for(ones);
    auto enc = rans_encode(ones, m);
    auto dec = rans_decode(enc, ones.size(), m);
    CHECK(dec == ones);
    CHECK(enc.size() < 100);  // one symbol -> tiny

    // Empty input.
    std::vector<u8> empty;
    RansModel me = model_for(empty);
    auto ence = rans_encode(empty, me);
    auto dece = rans_decode(ence, 0, me);
    CHECK(dece.empty());
}
