// codec/block.hpp — the end-to-end lossy block codec for one side^3 f32 block:
//   forward DWT -> dead-zone quantize -> zigzag byte-split -> rANS.
// Round-trips within ~q/2 max error per coefficient (the near-lossless knob). This is the
// correct reference; bitplane-progressive encoding (LOD + quality truncation) layers on
// top of the same DWT+quant stages next.
#pragma once

#include "codec/rans.hpp"
#include "codec/wavelet.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <span>
#include <vector>

namespace fenix::codec {

struct BlockParams {
    f32 q = 8.0f;   // quantization step (smaller = higher fidelity / larger). max err ~q/2.
    int levels = 4; // DWT decomposition levels (config; see codec/CLAUDE.md)
};

namespace detail {

inline u16 zigzag(s16 v) {
    return static_cast<u16>((static_cast<u16>(v) << 1) ^ static_cast<u16>(v >> 15));
}
inline s16 unzigzag(u16 z) { return static_cast<s16>((z >> 1) ^ (~(z & 1) + 1)); }

inline void put_u32(std::vector<u8>& b, u32 v) {
    u8 t[4];
    std::memcpy(t, &v, 4);
    b.insert(b.end(), t, t + 4);
}
inline u32 get_u32(std::span<const u8> b, usize& p) {
    u32 v;
    std::memcpy(&v, b.data() + p, 4);
    p += 4;
    return v;
}

// rANS-encode a byte plane, prefixed with its 256x u16 freq table + symbol count.
inline void encode_plane(std::vector<u8>& out, std::span<const u8> plane) {
    std::array<u64, 256> counts{};
    for (u8 b : plane) counts[b]++;
    RansModel m = RansModel::from_counts(counts);
    auto enc = rans_encode(plane, m);
    put_u32(out, static_cast<u32>(plane.size()));
    put_u32(out, static_cast<u32>(enc.size()));
    for (u16 f : m.freq) {
        u8 t[2];
        std::memcpy(t, &f, 2);
        out.insert(out.end(), t, t + 2);
    }
    out.insert(out.end(), enc.begin(), enc.end());
}

inline std::vector<u8> decode_plane(std::span<const u8> in, usize& p) {
    const u32 n = get_u32(in, p);
    const u32 enc_size = get_u32(in, p);
    std::array<u16, 256> freq{};
    for (auto& f : freq) {
        std::memcpy(&f, in.data() + p, 2);
        p += 2;
    }
    RansModel m = RansModel::from_freqs(freq);
    auto dec = rans_decode(in.subspan(p, enc_size), n, m);
    p += enc_size;
    return dec;
}

}  // namespace detail

// Encode a contiguous side^3 f32 block. Returns the self-contained block payload.
inline std::vector<u8> encode_block(std::span<const f32> block, s64 side, BlockParams params) {
    std::vector<f32> coef(block.begin(), block.end());
    dwt3_forward(coef, side, params.levels);

    const usize n = coef.size();
    const f32 inv_q = 1.0f / params.q;
    std::vector<u8> lo(n), hi(n);
    for (usize i = 0; i < n; ++i) {
        s32 level = static_cast<s32>(std::lround(coef[i] * inv_q));
        level = std::clamp(level, -32768, 32767);
        u16 z = detail::zigzag(static_cast<s16>(level));
        lo[i] = static_cast<u8>(z & 0xffu);
        hi[i] = static_cast<u8>(z >> 8);
    }

    std::vector<u8> out;
    detail::put_u32(out, static_cast<u32>(side));
    detail::put_u32(out, static_cast<u32>(params.levels));
    detail::put_u32(out, std::bit_cast<u32>(params.q));
    detail::encode_plane(out, lo);
    detail::encode_plane(out, hi);
    return out;
}

// Decode a block payload back to a side^3 f32 block (lossy, within ~q/2 per coefficient).
inline std::vector<f32> decode_block(std::span<const u8> payload) {
    usize p = 0;
    const s64 side = static_cast<s64>(detail::get_u32(payload, p));
    const int levels = static_cast<int>(detail::get_u32(payload, p));
    const f32 q = std::bit_cast<f32>(detail::get_u32(payload, p));
    auto lo = detail::decode_plane(payload, p);
    auto hi = detail::decode_plane(payload, p);

    const usize n = lo.size();
    std::vector<f32> coef(n);
    for (usize i = 0; i < n; ++i) {
        u16 z = static_cast<u16>(lo[i] | (static_cast<u16>(hi[i]) << 8));
        s16 level = detail::unzigzag(z);
        coef[i] = static_cast<f32>(level) * q;
    }
    dwt3_inverse(coef, side, levels);
    return coef;
}

}  // namespace fenix::codec
