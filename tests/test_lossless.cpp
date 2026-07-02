// test_lossless.cpp — lossless label/integer codec (exact roundtrip + compression).
#define FENIX_TEST_MAIN
#include <cstring>
#include <span>
#include <vector>

#include "codec/lossless.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

using namespace fenix;
using namespace fenix::codec;

TEST(lossless_labels_exact_roundtrip) {
    // A blocky label volume (instance labels) — lots of runs of the same value.
    std::vector<s32> labels(40000);
    for (usize i = 0; i < labels.size(); ++i)
        labels[i] = static_cast<s32>(1 + (i / 500));
    auto enc = lossless_encode<s32>(labels);
    auto dec = lossless_decode<s32>(enc);
    REQUIRE(dec.has_value());
    REQUIRE(dec->size() == labels.size());
    CHECK(*dec == labels);                               // exact
    CHECK(enc.size() < labels.size() * sizeof(s32) / 2); // compresses repetitive labels
}

TEST(lossless_u8_mask_roundtrip) {
    std::vector<u8> mask(10000, 0);
    for (usize i = 3000; i < 6000; ++i)
        mask[i] = 1;
    auto enc = lossless_encode<u8>(mask);
    auto dec = lossless_decode<u8>(enc);
    REQUIRE(dec.has_value());
    CHECK(*dec == mask);
    CHECK(enc.size() < mask.size()); // a few runs -> tiny
}

TEST(lossless_random_is_exact) {
    std::vector<u16> data(5000);
    Pcg32 rng{12};
    for (auto& v : data)
        v = static_cast<u16>(rng.next_u32());
    auto dec = lossless_decode<u16>(lossless_encode<u16>(data));
    REQUIRE(dec.has_value());
    CHECK(*dec == data); // exact even when incompressible
}

TEST(lossless_decode_rejects_truncated_payload) {
    std::vector<u32> data(2000);
    Pcg32 rng{5};
    for (auto& v : data)
        v = rng.next_u32();
    auto enc = lossless_encode<u32>(data);
    for (usize cut : {usize{0}, usize{1}, usize{7}, usize{16}, enc.size() / 2, enc.size() - 1}) {
        auto dec = lossless_decode<u32>(std::span<const u8>(enc.data(), cut));
        CHECK(!dec.has_value()); // truncated at any point -> error, never a crash
    }
}

TEST(lossless_decode_rejects_bad_freq_table) {
    std::vector<u8> data(1000, 7);
    auto enc = lossless_encode<u8>(data);
    // Header: n (8B) + B (8B), then per-plane [enc_size(8B) + 512B freq table + payload]. Corrupt the
    // first freq table's first entry to blow the sum way past rans_scale.
    REQUIRE(enc.size() > 16 + 8 + 2);
    u16 huge = 0xffff;
    std::memcpy(enc.data() + 16 + 8, &huge, 2);
    auto dec = lossless_decode<u8>(enc);
    CHECK(!dec.has_value()); // corrupt freq table -> rejected, not an OOB slot write
}

TEST(lossless_decode_rejects_wrong_element_size) {
    std::vector<u16> data(500);
    Pcg32 rng{3};
    for (auto& v : data)
        v = static_cast<u16>(rng.next_u32());
    auto enc = lossless_encode<u16>(data);
    // Decoding a u16-encoded stream as u64 must reject on B != sizeof(T), not silently read OOB.
    auto dec = lossless_decode<u64>(enc);
    CHECK(!dec.has_value());
}
