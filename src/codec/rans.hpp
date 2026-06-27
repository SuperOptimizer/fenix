// codec/rans.hpp — static range-ANS entropy coder (byte alphabet). The shared entropy
// stage for the wavelet bitplane symbols. This is the correct scalar reference; the
// 8-way interleaved SIMD/GPU variant (the perf target) will match this bitstream-for-
// -semantics. ryg-rANS layout: 32-bit state, byte renorm, native-endian state prefix.
#pragma once

#include "core/types.hpp"

#include <array>
#include <cstring>
#include <span>
#include <vector>

namespace fenix::codec {

inline constexpr u32 rans_byte_l = 1u << 23;  // lower renorm bound
inline constexpr u32 rans_scale_bits = 12;    // frequency total = 4096
inline constexpr u32 rans_scale = 1u << rans_scale_bits;

// Normalized frequency model over the 256-symbol byte alphabet.
struct RansModel {
    std::array<u16, 256> freq{};
    std::array<u16, 256> cum{};        // exclusive prefix sums
    std::array<u8, rans_scale> slot{}; // slot -> symbol (decode lookup)

    // Rebuild the cum + slot tables from an already-normalized freq table (sum==rans_scale),
    // e.g. one deserialized from a block header.
    static RansModel from_freqs(std::span<const u16, 256> freq) {
        RansModel m;
        u32 c = 0;
        for (int s = 0; s < 256; ++s) {
            m.freq[static_cast<usize>(s)] = freq[static_cast<usize>(s)];
            m.cum[static_cast<usize>(s)] = static_cast<u16>(c);
            for (u32 i = 0; i < freq[static_cast<usize>(s)]; ++i) m.slot[c + i] = static_cast<u8>(s);
            c += freq[static_cast<usize>(s)];
        }
        return m;
    }

    // Build from raw symbol counts. Normalizes to sum == rans_scale, never zeroing a
    // symbol that actually occurs (so every input is representable).
    static RansModel from_counts(std::span<const u64> counts) {
        RansModel m;
        u64 total = 0;
        for (u64 c : counts) total += c;
        if (total == 0) {  // degenerate: uniform
            for (auto& f : m.freq) f = static_cast<u16>(rans_scale / 256);
        } else {
            u32 assigned = 0;
            int last = -1;
            for (int s = 0; s < 256; ++s) {
                if (counts[static_cast<usize>(s)] == 0) continue;
                u32 f = static_cast<u32>((counts[static_cast<usize>(s)] * rans_scale) / total);
                if (f == 0) f = 1;  // keep occurring symbols representable
                m.freq[static_cast<usize>(s)] = static_cast<u16>(f);
                assigned += f;
                last = s;
            }
            // Fix rounding so the table sums to exactly rans_scale (adjust the largest).
            if (assigned != rans_scale && last >= 0) {
                int big = 0;
                for (int s = 1; s < 256; ++s)
                    if (m.freq[static_cast<usize>(s)] > m.freq[static_cast<usize>(big)]) big = s;
                const s32 fix = static_cast<s32>(rans_scale) - static_cast<s32>(assigned);
                m.freq[static_cast<usize>(big)] =
                    static_cast<u16>(static_cast<s32>(m.freq[static_cast<usize>(big)]) + fix);
            }
        }
        u32 c = 0;
        for (int s = 0; s < 256; ++s) {
            m.cum[static_cast<usize>(s)] = static_cast<u16>(c);
            for (u32 i = 0; i < m.freq[static_cast<usize>(s)]; ++i) m.slot[c + i] = static_cast<u8>(s);
            c += m.freq[static_cast<usize>(s)];
        }
        return m;
    }
};

// Encode bytes -> compressed stream (state prefix + renorm bytes).
inline std::vector<u8> rans_encode(std::span<const u8> data, const RansModel& m) {
    std::vector<u8> buf(data.size() * 2 + 64);
    usize ptr = buf.size();
    u32 x = rans_byte_l;
    for (usize i = data.size(); i-- > 0;) {
        const u8 s = data[i];
        const u32 f = m.freq[s];
        const u32 start = m.cum[s];
        const u32 x_max = ((rans_byte_l >> rans_scale_bits) << 8) * f;
        while (x >= x_max) {
            buf[--ptr] = static_cast<u8>(x & 0xffu);
            x >>= 8;
        }
        x = ((x / f) << rans_scale_bits) + (x % f) + start;
    }
    ptr -= 4;
    std::memcpy(buf.data() + ptr, &x, 4);  // native-endian state prefix
    return std::vector<u8>(buf.begin() + static_cast<isize>(ptr), buf.end());
}

// Decode `n` bytes from a stream produced by rans_encode with the same model.
inline std::vector<u8> rans_decode(std::span<const u8> stream, usize n, const RansModel& m) {
    std::vector<u8> out(n);
    usize p = 0;
    u32 x = 0;
    std::memcpy(&x, stream.data() + p, 4);
    p += 4;
    const u32 mask = rans_scale - 1;
    for (usize i = 0; i < n; ++i) {
        const u32 s = m.slot[x & mask];
        out[i] = static_cast<u8>(s);
        x = m.freq[s] * (x >> rans_scale_bits) + (x & mask) - m.cum[s];
        while (x < rans_byte_l && p < stream.size()) x = (x << 8) | stream[p++];
    }
    return out;
}

}  // namespace fenix::codec
