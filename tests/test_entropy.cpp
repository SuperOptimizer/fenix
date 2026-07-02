// test_entropy.cpp — codec/entropy.hpp: varint bounds-checking, decode_plane corrupt-input rejection,
// RansModel::from_freqs sum validation. Regression coverage for the confirmed decode-side finding: every
// byte-plane header field is untrusted (a crafted .fxvol tile need only pass its blob CRC), so a
// truncated or corrupt field must produce an Expected error, never an OOB read/write or shift UB.
#define FENIX_TEST_MAIN
#include <array>
#include <vector>

#include "codec/entropy.hpp"
#include "codec/rans.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

using namespace fenix;
using namespace fenix::codec;

TEST(get_var_roundtrip) {
    std::vector<u8> b;
    codec::detail::put_var(b, 0);
    codec::detail::put_var(b, 127);
    codec::detail::put_var(b, 128);
    codec::detail::put_var(b, 300000);
    codec::detail::put_var(b, 0xffffffffu);
    usize p = 0;
    auto v0 = codec::detail::get_var(b, p);
    REQUIRE(v0.has_value());
    CHECK(*v0 == 0u);
    auto v1 = codec::detail::get_var(b, p);
    REQUIRE(v1.has_value());
    CHECK(*v1 == 127u);
    auto v2 = codec::detail::get_var(b, p);
    REQUIRE(v2.has_value());
    CHECK(*v2 == 128u);
    auto v3 = codec::detail::get_var(b, p);
    REQUIRE(v3.has_value());
    CHECK(*v3 == 300000u);
    auto v4 = codec::detail::get_var(b, p);
    REQUIRE(v4.has_value());
    CHECK(*v4 == 0xffffffffu);
    CHECK(p == b.size());
}

TEST(get_var_rejects_truncated_and_unterminated) {
    // Empty buffer.
    {
        std::vector<u8> b;
        usize p = 0;
        CHECK(!codec::detail::get_var(b, p).has_value());
    }
    // Continuation bit set on the last byte -> would read past the end.
    {
        std::vector<u8> b{0x80};
        usize p = 0;
        CHECK(!codec::detail::get_var(b, p).has_value());
    }
    // A long run of continuation bytes (never terminates within the buffer) must not read OOB.
    {
        std::vector<u8> b(10, 0x80);
        usize p = 0;
        CHECK(!codec::detail::get_var(b, p).has_value());
    }
    // 6+ continuation bytes that DO terminate would drive sh>=32 (shift UB) if uncapped; must reject.
    {
        std::vector<u8> b{0x80, 0x80, 0x80, 0x80, 0x80, 0x01};
        usize p = 0;
        CHECK(!codec::detail::get_var(b, p).has_value());
    }
}

TEST(get_u32_bounds_checked) {
    std::vector<u8> b{1, 2, 3}; // only 3 bytes: any u32 read must fail
    usize p = 0;
    CHECK(!codec::detail::get_u32(b, p).has_value());
    std::vector<u8> b4{1, 2, 3, 4};
    usize p2 = 0;
    auto v = codec::detail::get_u32(b4, p2);
    REQUIRE(v.has_value());
    CHECK(p2 == 4);
}

static RansModel model_for(std::span<const u8> data) {
    std::array<u64, 256> counts{};
    for (u8 b : data)
        counts[b]++;
    return RansModel::from_counts(counts);
}

TEST(decode_plane_roundtrip) {
    std::vector<u8> plane(3000);
    Pcg32 rng{4};
    for (auto& b : plane)
        b = static_cast<u8>(rng.next_u32() % 40 == 0 ? rng.next_u32() & 0xff : 0);
    std::vector<u8> out;
    codec::detail::encode_plane(out, plane);
    usize p = 0;
    auto dec = codec::detail::decode_plane(out, p, static_cast<u32>(plane.size()));
    REQUIRE(dec.has_value());
    CHECK(*dec == plane);
    CHECK(p == out.size());
}

TEST(decode_plane_rejects_length_over_bound) {
    std::vector<u8> plane(100, 5);
    std::vector<u8> out;
    codec::detail::encode_plane(out, plane);
    usize p = 0;
    auto dec = codec::detail::decode_plane(out, p, 50); // max_n smaller than the encoded plane length
    CHECK(!dec.has_value());
}

TEST(decode_plane_rejects_truncated_payload) {
    std::vector<u8> plane(2000);
    Pcg32 rng{9};
    for (auto& b : plane)
        b = static_cast<u8>(rng.next_u32() & 0xff);
    std::vector<u8> out;
    codec::detail::encode_plane(out, plane);
    for (usize cut : {usize{0}, usize{1}, usize{3}, out.size() / 2, out.size() - 1}) {
        usize p = 0;
        auto dec = codec::detail::decode_plane(std::span<const u8>(out.data(), cut), p, static_cast<u32>(plane.size()));
        CHECK(!dec.has_value());
    }
}

TEST(decode_plane_rejects_corrupt_freq_table) {
    std::vector<u8> plane(500, 3);
    std::vector<u8> out;
    codec::detail::encode_plane(out, plane);
    // Layout: var n | var enc_size | var nused | nused*(u8 sym, var freq) | enc bytes. Corrupt the
    // single freq entry's value byte to something that blows the sum past rans_scale.
    usize p = 0;
    auto n_r = codec::detail::get_var(out, p);
    REQUIRE(n_r.has_value());
    auto es_r = codec::detail::get_var(out, p);
    REQUIRE(es_r.has_value());
    auto nused_r = codec::detail::get_var(out, p);
    REQUIRE(nused_r.has_value());
    REQUIRE(*nused_r >= 1);
    p += 1; // skip the symbol byte
    REQUIRE(p < out.size());
    out[p] |= 0x80u; // force the freq varint to keep extending -> a much larger decoded freq
    if (p + 1 < out.size())
        out[p + 1] = 0x7f;
    usize p2 = 0;
    auto dec = codec::detail::decode_plane(out, p2, static_cast<u32>(plane.size()) + 4096);
    CHECK(!dec.has_value()); // corrupt table must be rejected, never an OOB slot write
}

TEST(rans_from_freqs_rejects_bad_sum) {
    std::array<u16, 256> freq{};
    freq[0] = 4096; // valid: sums to rans_scale
    CHECK(RansModel::valid_freqs(freq));
    freq[1] = 1; // now sums to 4097 -> invalid
    CHECK(!RansModel::valid_freqs(freq));
    // from_freqs must not crash and must fall back to a valid (uniform) model rather than overrun slot[].
    RansModel m = RansModel::from_freqs(freq);
    u32 sum = 0;
    for (u16 f : m.freq)
        sum += f;
    CHECK(sum == rans_scale);
}

TEST(rans_from_freqs_accepts_valid_sum) {
    std::vector<u8> data(4000);
    Pcg32 rng{2};
    for (auto& b : data)
        b = static_cast<u8>(rng.next_u32() % 10);
    RansModel m0 = model_for(data);
    RansModel m1 = RansModel::from_freqs(m0.freq);
    auto enc = rans_encode(data, m1);
    auto dec = rans_decode(enc, data.size(), m1);
    CHECK(dec == data);
}

TEST(bitreader_rejects_out_of_range_bits) {
    std::vector<u8> bytes(8, 0xff);
    codec::detail::BitReader r{bytes.data(), bytes.size()};
    CHECK(r.get(-1) == 0);
    CHECK(r.get(0) == 0);
    CHECK(r.get(200) == 0);  // must not shift-UB or corrupt internal state
    CHECK(r.get(8) == 0xff); // still usable afterward
}
