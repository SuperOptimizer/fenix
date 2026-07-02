// codec/rans.hpp — static range-ANS entropy coder (byte alphabet). The shared entropy stage for both
// transform codecs (wavelet + DCT). INTERLEAVED: K independent rANS states round-robined by symbol
// index share one LIFO byte stream, which breaks the serial state→state dependency so the CPU pipelines
// the lanes (hiding the data-dependent slot-lookup latency on decode). Measured isolated: ~1.6x enc,
// ~1.75x dec on a large skewed buffer (test_rans_perf), ratio-identical. K is ADAPTIVE and DERIVED from
// the symbol count (no header byte): K=1 below a threshold so the extra state prefixes don't hurt the
// codec's many mid-size context streams; K=kRansK above it. NB the wavelet/DCT block codecs are
// transform-bound, so this barely moves them end-to-end — the win lands on rANS-dominated paths (the
// lossless codec, low-q large streams) and once the transforms are SIMD'd. ryg-rANS: 32-bit state, byte
// renorm, native-endian state prefixes. Robust on any bytes (prefix reads bounded; K derived not trusted).
#pragma once

#include <array>
#include <cstring>
#include <span>
#include <vector>

#include "core/types.hpp"

namespace fenix::codec {

inline constexpr u32 rans_byte_l = 1u << 23; // lower renorm bound
inline constexpr u32 rans_scale_bits = 12;   // frequency total = 4096
inline constexpr u32 rans_scale = 1u << rans_scale_bits;
inline constexpr int kRansK = 4;                  // interleaved lanes (ILP) for large streams
inline constexpr usize kRansInterleaveMin = 4096; // below this, single-state: 4 lanes cost 16 prefix
                                                  // bytes vs 4, so K=4 only where that's negligible
                                                  // (<0.3% at 4096) — keeps the codec's many mid-size
                                                  // context streams ratio-neutral. The win lands on
                                                  // large/rANS-dominated streams (lossless, low-q).

// Normalized frequency model over the 256-symbol byte alphabet.
struct RansModel {
    std::array<u16, 256> freq{};
    std::array<u16, 256> cum{};        // exclusive prefix sums
    std::array<u8, rans_scale> slot{}; // slot -> symbol (decode lookup)

    // Rebuild the cum + slot tables from an already-normalized freq table (sum==rans_scale),
    // e.g. one deserialized from a block header. UNTRUSTED INPUT: the caller may be deserializing
    // freqs straight from a corrupt/adversarial file, so the sum is validated here (not just
    // FENIX_ASSERT'd, which compiles out in release) before any slot[] write. A table that doesn't
    // sum to exactly rans_scale is rejected in favor of the uniform model — "wrong values OK, a
    // SEGV is a fail" (codec/CLAUDE.md); callers that need to signal corruption check freq against
    // the caller-owned table they deserialized, this function just refuses to overrun `slot`.
    static RansModel from_freqs(std::span<const u16, 256> freq) {
        u64 total = 0;
        for (u16 f : freq)
            total += f;
        if (total != rans_scale)
            return uniform_();
        RansModel m;
        u32 c = 0;
        for (int s = 0; s < 256; ++s) {
            m.freq[static_cast<usize>(s)] = freq[static_cast<usize>(s)];
            m.cum[static_cast<usize>(s)] = static_cast<u16>(c);
            for (u32 i = 0; i < freq[static_cast<usize>(s)]; ++i)
                m.slot[c + i] = static_cast<u8>(s);
            c += freq[static_cast<usize>(s)];
        }
        return m;
    }

    // True iff `freq` sums to exactly rans_scale — the precondition from_freqs relies on. Callers
    // that can propagate an Expected error (rather than silently falling back to the uniform model)
    // should check this first and reject the corrupt payload outright.
    static bool valid_freqs(std::span<const u16, 256> freq) {
        u64 total = 0;
        for (u16 f : freq)
            total += f;
        return total == rans_scale;
    }

    // Build from raw symbol counts. Normalizes to sum == rans_scale, never zeroing a
    // symbol that actually occurs (so every input is representable).
    //
    // The floor-share + "bump zero-floor occurring symbols to 1" pass can push `assigned` OVER
    // rans_scale by more than the single largest freq (e.g. 64 symbols at count 10000 + 192 at
    // count 1: floor gives 63 each for the common symbols, but the 192 rare-symbol bumps to 1
    // overshoot the deficit by more than any one freq can absorb). Redistributing the whole
    // overshoot into ONE symbol via unchecked subtraction can drive that freq negative and wrap
    // around u16, which then makes the slot-fill loop below write far past the 4096-entry `slot`
    // array (this exact histogram was the confirmed encode-side OOB write). Fix: repeatedly take
    // 1 from the CURRENT largest freq (>1, so occurring symbols stay representable) until the
    // overshoot is gone — always feasible since <=256 occurring symbols each need >=1 and
    // rans_scale=4096 >= 256. Deficits (assigned < rans_scale) still go to the single largest freq
    // in one step (cannot overflow u16: fix <= 4095).
    static RansModel from_counts(std::span<const u64> counts) {
        RansModel m;
        u64 total = 0;
        for (u64 c : counts)
            total += c;
        if (total == 0) { // degenerate: uniform
            for (auto& f : m.freq)
                f = static_cast<u16>(rans_scale / 256);
        } else {
            u32 assigned = 0;
            int last = -1;
            for (int s = 0; s < 256; ++s) {
                if (counts[static_cast<usize>(s)] == 0)
                    continue;
                u32 f = static_cast<u32>((counts[static_cast<usize>(s)] * rans_scale) / total);
                if (f == 0)
                    f = 1; // keep occurring symbols representable
                m.freq[static_cast<usize>(s)] = static_cast<u16>(f);
                assigned += f;
                last = s;
            }
            if (assigned > rans_scale && last >= 0) {
                u32 over = assigned - rans_scale;
                while (over > 0) {
                    int big = -1;
                    for (int s = 0; s < 256; ++s) {
                        if (m.freq[static_cast<usize>(s)] <= 1)
                            continue; // never zero an occurring symbol
                        if (big < 0 || m.freq[static_cast<usize>(s)] > m.freq[static_cast<usize>(big)])
                            big = s;
                    }
                    if (big < 0)
                        break; // every occurring symbol is already at the 1-floor (shouldn't happen: 4096>=256)
                    --m.freq[static_cast<usize>(big)];
                    --over;
                }
            } else if (assigned < rans_scale && last >= 0) {
                int big = 0;
                for (int s = 1; s < 256; ++s)
                    if (m.freq[static_cast<usize>(s)] > m.freq[static_cast<usize>(big)])
                        big = s;
                const u32 fix = rans_scale - assigned; // <= 4095: cannot overflow u16
                m.freq[static_cast<usize>(big)] = static_cast<u16>(m.freq[static_cast<usize>(big)] + fix);
            }
        }
        u32 c = 0;
        for (int s = 0; s < 256; ++s) {
            m.cum[static_cast<usize>(s)] = static_cast<u16>(c);
            for (u32 i = 0; i < m.freq[static_cast<usize>(s)]; ++i)
                m.slot[c + i] = static_cast<u8>(s);
            c += m.freq[static_cast<usize>(s)];
        }
        return m;
    }

    static RansModel uniform_() {
        RansModel m;
        for (auto& f : m.freq)
            f = static_cast<u16>(rans_scale / 256);
        u32 c = 0;
        for (int s = 0; s < 256; ++s) {
            m.cum[static_cast<usize>(s)] = static_cast<u16>(c);
            for (u32 i = 0; i < m.freq[static_cast<usize>(s)]; ++i)
                m.slot[c + i] = static_cast<u8>(s);
            c += m.freq[static_cast<usize>(s)];
        }
        return m;
    }
};

// Encode bytes -> compressed stream: 1-byte K + K native-endian state prefixes + interleaved renorm
// bytes. Symbol i is coded by lane i%K; the whole sequence is encoded in REVERSE (rANS is LIFO) into a
// single backward-written stream, so the decoder reads it forward. The kRansK lanes are SEPARATE NAMED
// state variables processed a full group at a time (constant lane index) so they stay in registers and
// their independent state updates pipeline — the array-indexed form does not. Streams below the
// threshold use a single state (K=1) to avoid the extra prefixes on tiny context streams.
inline std::vector<u8> rans_encode(std::span<const u8> data, const RansModel& m) {
    std::vector<u8> buf(data.size() * 2 + 128);
    usize ptr = buf.size();
    auto enc = [&](u32& x, u8 s) {
        const u32 f = m.freq[s], c = m.cum[s];
        const u32 x_max = ((rans_byte_l >> rans_scale_bits) << 8) * f;
        while (x >= x_max) {
            buf[--ptr] = static_cast<u8>(x & 0xffu);
            x >>= 8;
        }
        x = ((x / f) << rans_scale_bits) + (x % f) + c;
    };
    int K;
    if (data.size() < kRansInterleaveMin) {
        K = 1;
        u32 x0 = rans_byte_l;
        for (usize i = data.size(); i-- > 0;)
            enc(x0, data[i]);
        ptr -= 4;
        std::memcpy(buf.data() + ptr, &x0, 4);
    } else {
        K = kRansK; // == 4
        u32 x0 = rans_byte_l, x1 = rans_byte_l, x2 = rans_byte_l, x3 = rans_byte_l;
        const s64 n = static_cast<s64>(data.size()), ng = n / 4, base = ng * 4;
        for (s64 i = n - 1; i >= base; --i) // tail (highest indices), reverse, lane = i&3
            switch (i & 3) {
            case 0:
                enc(x0, data[static_cast<usize>(i)]);
                break;
            case 1:
                enc(x1, data[static_cast<usize>(i)]);
                break;
            case 2:
                enc(x2, data[static_cast<usize>(i)]);
                break;
            default:
                enc(x3, data[static_cast<usize>(i)]);
                break;
            }
        for (s64 g = ng - 1; g >= 0; --g) { // full groups, reverse; 4 independent lanes pipeline
            const usize b = static_cast<usize>(4 * g);
            enc(x3, data[b + 3]);
            enc(x2, data[b + 2]);
            enc(x1, data[b + 1]);
            enc(x0, data[b]);
        }
        ptr -= 4;
        std::memcpy(buf.data() + ptr, &x3, 4); // lane 0's prefix ends at the front
        ptr -= 4;
        std::memcpy(buf.data() + ptr, &x2, 4);
        ptr -= 4;
        std::memcpy(buf.data() + ptr, &x1, 4);
        ptr -= 4;
        std::memcpy(buf.data() + ptr, &x0, 4);
    }
    (void)K; // K is DERIVED from the symbol count on decode (same threshold) — no header byte, so the
             // extra lanes cost zero ratio on the codec's many tiny streams.
    return std::vector<u8>(buf.begin() + static_cast<isize>(ptr), buf.end());
}

// Decode `n` bytes from a stream produced by rans_encode with the same model. Robust on any bytes.
inline std::vector<u8> rans_decode(std::span<const u8> stream, usize n, const RansModel& m) {
    std::vector<u8> out(n);
    if (stream.empty())
        return out;
    usize p = 0;
    const int K = n >= kRansInterleaveMin ? kRansK : 1; // derived from the count, matching encode
    const u32 mask = rans_scale - 1;
    auto rd4 = [&]() {
        u32 v = rans_byte_l;
        if (p + 4 <= stream.size()) {
            std::memcpy(&v, stream.data() + p, 4);
            p += 4;
        }
        return v;
    };
    auto dec = [&](u32& x) -> u8 {
        const u32 s = m.slot[x & mask];
        x = m.freq[s] * (x >> rans_scale_bits) + (x & mask) - m.cum[s];
        while (x < rans_byte_l && p < stream.size())
            x = (x << 8) | stream[p++];
        return static_cast<u8>(s);
    };
    if (K == 1) {
        u32 x0 = rd4();
        for (usize i = 0; i < n; ++i)
            out[i] = dec(x0);
    } else {
        u32 x0 = rd4(), x1 = rd4(), x2 = rd4(), x3 = rd4();
        const s64 sn = static_cast<s64>(n), ng = sn / 4, base = ng * 4;
        for (s64 g = 0; g < ng; ++g) { // full groups: 4 independent lanes pipeline
            const usize b = static_cast<usize>(4 * g);
            out[b] = dec(x0);
            out[b + 1] = dec(x1);
            out[b + 2] = dec(x2);
            out[b + 3] = dec(x3);
        }
        for (s64 i = base; i < sn; ++i) // tail, lane = i&3
            switch (i & 3) {
            case 0:
                out[static_cast<usize>(i)] = dec(x0);
                break;
            case 1:
                out[static_cast<usize>(i)] = dec(x1);
                break;
            case 2:
                out[static_cast<usize>(i)] = dec(x2);
                break;
            default:
                out[static_cast<usize>(i)] = dec(x3);
                break;
            }
    }
    return out;
}

} // namespace fenix::codec
