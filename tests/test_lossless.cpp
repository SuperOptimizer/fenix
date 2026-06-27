// test_lossless.cpp — lossless label/integer codec (exact roundtrip + compression).
#define FENIX_TEST_MAIN
#include "codec/lossless.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

#include <vector>

using namespace fenix;
using namespace fenix::codec;

TEST(lossless_labels_exact_roundtrip) {
    // A blocky label volume (instance labels) — lots of runs of the same value.
    std::vector<s32> labels(40000);
    for (usize i = 0; i < labels.size(); ++i) labels[i] = static_cast<s32>(1 + (i / 500));
    auto enc = lossless_encode<s32>(labels);
    auto dec = lossless_decode<s32>(enc);
    REQUIRE(dec.size() == labels.size());
    CHECK(dec == labels);                       // exact
    CHECK(enc.size() < labels.size() * sizeof(s32) / 2);  // compresses repetitive labels
}

TEST(lossless_u8_mask_roundtrip) {
    std::vector<u8> mask(10000, 0);
    for (usize i = 3000; i < 6000; ++i) mask[i] = 1;
    auto enc = lossless_encode<u8>(mask);
    auto dec = lossless_decode<u8>(enc);
    CHECK(dec == mask);
    CHECK(enc.size() < mask.size());  // a few runs -> tiny
}

TEST(lossless_random_is_exact) {
    std::vector<u16> data(5000);
    Pcg32 rng{12};
    for (auto& v : data) v = static_cast<u16>(rng.next_u32());
    auto dec = lossless_decode<u16>(lossless_encode<u16>(data));
    CHECK(dec == data);  // exact even when incompressible
}
