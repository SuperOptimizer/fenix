// codec/entropy.hpp — shared entropy-stage serialization helpers for the block/tile codecs: fixed +
// LEB128 varint integer fields, rANS-coded byte planes with a COMPACT sparse frequency table (only used
// symbols), and an LSB-first bit packer for the raw mantissa/sign bits. Extracted from the original
// wavelet block codec when the DWT was retired (ADR 0005) so the DCT tile codec keeps them. See
// codec/CLAUDE.md.
#pragma once

#include "codec/rans.hpp"
#include "core/types.hpp"

#include <array>
#include <atomic>
#include <cstring>
#include <span>
#include <vector>

namespace fenix::codec::detail {

// Lightweight, always-on byte accounting for the rANS byte-plane coder (negligible: a couple of relaxed
// atomic adds per non-empty plane). Lets a bench size the freq-table-header overhead vs the rANS payload
// — the input to the table-signalling work (plan 1.2). Read + reset from a single-threaded point.
inline std::atomic<u64> g_plane_hdr_bytes{0};  // freq-table header bytes (sym/freq pairs + size fields)
inline std::atomic<u64> g_plane_enc_bytes{0};  // rANS-coded payload bytes
inline std::atomic<u64> g_plane_count{0};      // number of non-empty planes coded
inline std::atomic<u64> g_dc_bytes{0};         // tile DC-level varint section
inline std::atomic<u64> g_nsig_bytes{0};       // tile per-block nsig varint section
inline std::atomic<u64> g_map_bytes{0};        // tile K-byte + context-map
inline std::atomic<u64> g_bits_bytes{0};       // raw mantissa+sign bit stream (+ its size varint)

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
// LEB128 varint — size fields are mostly tiny (0/empty for sparse high-q context streams), so a varint
// (1 byte for <128) beats a fixed u32 and keeps per-stream overhead ~1 byte.
inline void put_var(std::vector<u8>& b, u32 v) {
    while (v >= 0x80) { b.push_back(static_cast<u8>(v) | 0x80u); v >>= 7; }
    b.push_back(static_cast<u8>(v));
}
inline u32 get_var(std::span<const u8> b, usize& p) {
    u32 v = 0;
    int sh = 0;
    u8 byte;
    do { byte = b[p++]; v |= static_cast<u32>(byte & 0x7fu) << sh; sh += 7; } while (byte & 0x80u);
    return v;
}

// rANS-encode a byte plane with a COMPACT sparse freq table (only used symbols) + varint size fields —
// the difference between a fixed 512-byte table and a couple of bytes when the plane is sparse/uniform.
// Layout: var n | var enc_size | var nused | nused×(u8 sym, var freq) | enc bytes.  n==0 -> just n.
inline void encode_plane(std::vector<u8>& out, std::span<const u8> plane) {
    put_var(out, static_cast<u32>(plane.size()));
    if (plane.empty()) return;
    const usize hdr_start = out.size();
    std::array<u64, 256> counts{};
    for (u8 b : plane) counts[b]++;
    RansModel m = RansModel::from_counts(counts);
    auto enc = rans_encode(plane, m);
    put_var(out, static_cast<u32>(enc.size()));
    u32 nused = 0;
    for (u16 f : m.freq) nused += (f != 0);
    put_var(out, nused);
    for (int s = 0; s < 256; ++s) {
        const u16 f = m.freq[static_cast<usize>(s)];
        if (!f) continue;
        out.push_back(static_cast<u8>(s));
        put_var(out, f);
    }
    g_plane_hdr_bytes.fetch_add(out.size() - hdr_start, std::memory_order_relaxed);
    g_plane_enc_bytes.fetch_add(enc.size(), std::memory_order_relaxed);
    g_plane_count.fetch_add(1, std::memory_order_relaxed);
    out.insert(out.end(), enc.begin(), enc.end());
}

inline std::vector<u8> decode_plane(std::span<const u8> in, usize& p) {
    const u32 n = get_var(in, p);
    if (n == 0) return {};
    const u32 enc_size = get_var(in, p);
    const u32 nused = get_var(in, p);
    std::array<u16, 256> freq{};
    for (u32 k = 0; k < nused; ++k) {
        const u8 s = in[p++];
        freq[s] = static_cast<u16>(get_var(in, p));
    }
    RansModel m = RansModel::from_freqs(freq);
    auto dec = rans_decode(in.subspan(p, enc_size), n, m);
    p += enc_size;
    return dec;
}

// LSB-first bit writer/reader for the raw mantissa+sign bits (which are ~incompressible, so we pack
// them rather than waste entropy-coder effort on them).
struct BitWriter {
    std::vector<u8> bytes;
    u64 acc = 0;
    int nbits = 0;
    void put(u32 v, int bits) {
        if (bits <= 0) return;
        acc |= static_cast<u64>(v & (bits >= 32 ? 0xffffffffu : ((1u << bits) - 1u))) << nbits;
        nbits += bits;
        while (nbits >= 8) { bytes.push_back(static_cast<u8>(acc & 0xffu)); acc >>= 8; nbits -= 8; }
    }
    void flush() { if (nbits > 0) { bytes.push_back(static_cast<u8>(acc & 0xffu)); acc = 0; nbits = 0; } }
};
struct BitReader {
    const u8* p;
    usize n, pos = 0;
    u64 acc = 0;
    int nbits = 0;
    u32 get(int bits) {
        if (bits <= 0) return 0;
        while (nbits < bits) { acc |= static_cast<u64>(pos < n ? p[pos++] : 0) << nbits; nbits += 8; }
        const u32 v = static_cast<u32>(acc & (bits >= 32 ? 0xffffffffu : ((1u << bits) - 1u)));
        acc >>= bits; nbits -= bits;
        return v;
    }
};

}  // namespace fenix::codec::detail
