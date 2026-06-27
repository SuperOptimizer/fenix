// codec/lossless.hpp — general lossless codec (byte-plane split + delta filter + rANS) for
// integer LABEL volumes (instance/winding), validity masks, and exact priors. Exact
// roundtrip. Shares the rANS engine with the wavelet codec. See codec/CLAUDE.md.
#pragma once

#include "codec/rans.hpp"
#include "core/types.hpp"

#include <array>
#include <cstring>
#include <span>
#include <vector>

namespace fenix::codec {

namespace detail {
inline void ll_put_u64(std::vector<u8>& b, u64 v) {
    u8 t[8];
    std::memcpy(t, &v, 8);
    b.insert(b.end(), t, t + 8);
}
inline u64 ll_get_u64(std::span<const u8> b, usize& p) {
    u64 v;
    std::memcpy(&v, b.data() + p, 8);
    p += 8;
    return v;
}
}  // namespace detail

// Lossless-encode a typed array. Splits each element into sizeof(T) byte planes (LSB-first),
// rANS-codes each plane with its own frequency table. Exact, and compresses repetitive data
// (labels, masks) well — high planes are near-constant.
template <class T>
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
        for (u8 c : plane) counts[c]++;
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

template <class T>
    requires std::is_trivially_copyable_v<T>
std::vector<T> lossless_decode(std::span<const u8> in) {
    usize p = 0;
    const u64 n = detail::ll_get_u64(in, p);
    const u64 B = detail::ll_get_u64(in, p);
    std::vector<T> out(static_cast<usize>(n), T{});
    for (u64 b = 0; b < B; ++b) {
        const u64 enc_size = detail::ll_get_u64(in, p);
        std::array<u16, 256> freq{};
        for (auto& fr : freq) {
            std::memcpy(&fr, in.data() + p, 2);
            p += 2;
        }
        RansModel m = RansModel::from_freqs(freq);
        auto plane = rans_decode(in.subspan(p, static_cast<usize>(enc_size)), static_cast<usize>(n), m);
        p += static_cast<usize>(enc_size);
        for (usize i = 0; i < static_cast<usize>(n); ++i)
            std::memcpy(reinterpret_cast<u8*>(&out[i]) + b, &plane[i], 1);
    }
    return out;
}

}  // namespace fenix::codec
