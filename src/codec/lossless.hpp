// codec/lossless.hpp — general lossless codec (byte-plane split + delta filter + rANS) for
// integer LABEL volumes (instance/winding), validity masks, and exact priors. Exact
// roundtrip. Shares the rANS engine with the wavelet codec. See codec/CLAUDE.md.
#pragma once

#include <array>
#include <cstring>
#include <span>
#include <vector>

#include "codec/rans.hpp"
#include "core/error.hpp"
#include "core/types.hpp"

namespace fenix::codec {

namespace detail {
inline void ll_put_u64(std::vector<u8>& b, u64 v) {
    u8 t[8];
    std::memcpy(t, &v, 8);
    b.insert(b.end(), t, t + 8);
}

// Bounds-checked: `p` walks untrusted stream bytes on the decode path, so a truncated header/stream must
// be rejected rather than memcpy'd past the end of `b`.
inline Expected<u64> ll_get_u64(std::span<const u8> b, usize& p) {
    if (p + 8 > b.size())
        return err(Errc::decode_error, "truncated u64 field");
    u64 v;
    std::memcpy(&v, b.data() + p, 8);
    p += 8;
    return v;
}
} // namespace detail

// Lossless-encode a typed array. Splits each element into sizeof(T) byte planes (LSB-first),
// rANS-codes each plane with its own frequency table. Exact, and compresses repetitive data
// (labels, masks) well — high planes are near-constant.
template<class T>
    requires std::is_trivially_copyable_v<T>
std::vector<u8> lossless_encode(std::span<const T> data) {
    constexpr usize B = sizeof(T);
    const usize n = data.size();
    std::vector<u8> out;
    detail::ll_put_u64(out, n);
    detail::ll_put_u64(out, B);

    std::vector<u8> plane(n);
    for (usize b = 0; b < B; ++b) {
        for (usize i = 0; i < n; ++i) {
            T v = data[i];
            u8 byte;
            std::memcpy(&byte, reinterpret_cast<const u8*>(&v) + b, 1);
            plane[i] = byte;
        }
        std::array<u64, 256> counts{};
        for (u8 c : plane)
            counts[c]++;
        RansModel m = RansModel::from_counts(counts);
        auto enc = rans_encode(plane, m);
        detail::ll_put_u64(out, enc.size());
        for (u16 fr : m.freq) {
            u8 t[2];
            std::memcpy(t, &fr, 2);
            out.insert(out.end(), t, t + 2);
        }
        out.insert(out.end(), enc.begin(), enc.end());
    }
    return out;
}

// Decode a lossless_encode payload. UNTRUSTED INPUT: `in` may be a corrupt/adversarial byte-plane blob
// (label volumes / validity masks / exact priors travel over the network per codec/CLAUDE.md), so every
// header field is bounds-checked before it drives an allocation, a byte-plane write, or a slot table
// build — "wrong values OK, a SEGV is a fail". `expected_n`, when non-zero, is the caller-known element
// count (from the volume/chunk geometry) checked against the decoded `n`; rANS legitimately expands, so
// bounding `n` from `in.size()` alone would be wrong, but the call site almost always knows the true
// count and should pass it. `expected_n == 0` (the default) skips that specific check — used only where
// the caller genuinely doesn't know the count ahead of time — but every other bound below still applies.
template<class T>
    requires std::is_trivially_copyable_v<T>
Expected<std::vector<T>> lossless_decode(std::span<const u8> in, usize expected_n = 0) {
    usize p = 0;
    auto n_r = detail::ll_get_u64(in, p);
    if (!n_r)
        return std::unexpected(n_r.error());
    const u64 n = *n_r;
    auto b_r = detail::ll_get_u64(in, p);
    if (!b_r)
        return std::unexpected(b_r.error());
    const u64 B = *b_r;
    if (B != sizeof(T))
        return err(Errc::decode_error, "lossless_decode: byte-plane count != sizeof(T)");
    if (expected_n != 0 && n != expected_n)
        return err(Errc::decode_error, "lossless_decode: n != expected count");
    // No caller-supplied expectation: still cap the allocation to something the input could plausibly
    // decode to (a maximally-skewed rANS stream can expand a lot, but not without bound) — this rejects
    // a corrupt 8-byte header (e.g. n = 2^64-1) instead of `vector(n)` aborting under -fno-exceptions.
    constexpr u64 kMaxExpansion = 1024; // generous: real payloads expand far less than this
    if (expected_n == 0 && n > (static_cast<u64>(in.size()) + 64) * kMaxExpansion)
        return err(Errc::decode_error, "lossless_decode: n implausibly large for input size");
    const usize un = static_cast<usize>(n);
    std::vector<T> out(un, T{});
    for (u64 b = 0; b < B; ++b) {
        auto enc_size_r = detail::ll_get_u64(in, p);
        if (!enc_size_r)
            return std::unexpected(enc_size_r.error());
        const u64 enc_size = *enc_size_r;
        if (p + 512 > in.size())
            return err(Errc::decode_error, "truncated lossless freq table");
        std::array<u16, 256> freq{};
        for (auto& fr : freq) {
            std::memcpy(&fr, in.data() + p, 2);
            p += 2;
        }
        if (!RansModel::valid_freqs(freq))
            return err(Errc::decode_error, "lossless freq table doesn't sum to rans_scale");
        RansModel m = RansModel::from_freqs(freq);
        if (p + enc_size > in.size())
            return err(Errc::decode_error, "truncated lossless plane payload");
        auto plane = rans_decode(in.subspan(p, static_cast<usize>(enc_size)), un, m);
        p += static_cast<usize>(enc_size);
        for (usize i = 0; i < un; ++i)
            std::memcpy(reinterpret_cast<u8*>(&out[i]) + b, &plane[i], 1);
    }
    return out;
}

} // namespace fenix::codec
