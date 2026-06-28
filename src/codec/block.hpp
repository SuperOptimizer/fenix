// codec/block.hpp — the end-to-end lossy block codec for one side^3 f32 block:
//   forward DWT -> per-subband dead-zone quantize -> magnitude-category coding (rANS the
//   bit-length with a causal 3D significance context; raw mantissa+sign bits).
// Round-trips within ~step/2 max error per coefficient (the near-lossless knob). This is the
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

// Per-subband quantization step: CDF 9/7 is biorthogonal, so quantizing every subband with the
// same step is NOT rate-distortion optimal — low-frequency subbands carry more reconstruction
// energy and deserve a finer step. We shrink the step toward the LLL corner (higher scale).
inline f32 scale_step(f32 q, int scale) {
    return q * std::pow(0.82f, static_cast<f32>(scale - 1));
}

// One orientation subband of the deinterleaved separable DWT: a contiguous box in the block,
// tagged with its DWT scale. Scanning each box in 3D raster order lets us condition a
// coefficient's category on its already-coded 3D neighbours (a causal significance context),
// which the prior raster-order context could not do (raster predecessors cross subbands).
struct Subband {
    int scale;
    s64 z0, z1, y0, y1, x0, x1;
};
inline std::vector<Subband> subband_boxes(s64 side, int levels) {
    std::vector<Subband> b;
    s64 e = side;
    for (int l = 1; l <= levels; ++l) {
        const s64 h = (e + 1) / 2;  // low-band size at this level
        for (int o = 1; o < 8; ++o) {  // 7 detail octants (exclude all-low)
            const int oz = (o >> 2) & 1, oy = (o >> 1) & 1, ox = o & 1;
            b.push_back({l, oz ? h : 0, oz ? e : h, oy ? h : 0, oy ? e : h, ox ? h : 0, ox ? e : h});
        }
        e = h;
    }
    b.push_back({levels + 1, 0, e, 0, e, 0, e});  // LLL corner (lowest frequency)
    return b;
}

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
// LEB128 varint — size fields are mostly tiny (0/empty for sparse high-q context streams), so a
// varint (1 byte for <128) beats a fixed u32 and removes the per-stream overhead that fragmenting
// into many context streams would otherwise reintroduce.
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

// rANS-encode a byte plane with a COMPACT sparse freq table (only used symbols) and varint size
// fields — the difference between a fixed 512-byte table and a couple of bytes when the plane is
// sparse/uniform (the common case for wavelet detail bands and empty context streams).
// Layout: var n | var enc_size | var nused | nused×(u8 sym, var freq) | enc bytes.  n==0 -> just n.
inline void encode_plane(std::vector<u8>& out, std::span<const u8> plane) {
    put_var(out, static_cast<u32>(plane.size()));
    if (plane.empty()) return;
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

// LSB-first bit writer/reader for the raw mantissa+sign bits (which are ~incompressible, so we
// pack them rather than waste entropy-coder effort on them).
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

}  // namespace detail

// Number of causal-neighbour significance contexts (count of significant z-1/y-1/x-1 neighbours,
// 0..3) per scale, used to model the magnitude-category stream.
inline constexpr int kCtx = 4;

// Encode a contiguous side^3 f32 block: per-subband-quantized DWT coefficients. Each orientation
// subband is scanned in 3D raster order; the magnitude CATEGORY (bit-length) is rANS-coded with a
// model selected by a causal 3D significance context (how many of the z-1/y-1/x-1 neighbours are
// nonzero), per scale; the near-random mantissa+sign bits are packed raw. Returns the payload.
inline std::vector<u8> encode_block(std::span<const f32> block, s64 side, BlockParams params) {
    std::vector<f32> coef(block.begin(), block.end());
    dwt3_forward(coef, side, params.levels);

    const int nscale = params.levels + 1;
    std::vector<f32> istep(static_cast<usize>(nscale + 1));
    for (int s = 1; s <= nscale; ++s) istep[static_cast<usize>(s)] = 1.0f / detail::scale_step(params.q, s);

    std::vector<std::array<std::vector<u8>, kCtx>> cats(static_cast<usize>(nscale));  // [scale][ctx]
    std::vector<detail::BitWriter> bits(static_cast<usize>(nscale));                  // mantissa+sign, raw
    std::vector<u8> sig(static_cast<usize>(side * side * side), 0);                    // significance map
    const s64 sz = side * side;

    for (const auto& sb : detail::subband_boxes(side, params.levels)) {
        const usize s = static_cast<usize>(sb.scale - 1);
        const f32 iq = istep[static_cast<usize>(sb.scale)];
        for (s64 z = sb.z0; z < sb.z1; ++z)
            for (s64 y = sb.y0; y < sb.y1; ++y)
                for (s64 x = sb.x0; x < sb.x1; ++x) {
                    const usize i = static_cast<usize>(z * sz + y * side + x);
                    const int ctx = (z > sb.z0 ? sig[i - static_cast<usize>(sz)] : 0) +
                                    (y > sb.y0 ? sig[i - static_cast<usize>(side)] : 0) +
                                    (x > sb.x0 ? sig[i - 1] : 0);
                    const s32 level = std::clamp(static_cast<s32>(coef[i] * iq), -32768, 32767);
                    const u32 a = static_cast<u32>(level < 0 ? -level : level);
                    const int cat = a ? std::bit_width(a) : 0;  // 0 for zero, else 1..15
                    cats[s][static_cast<usize>(ctx)].push_back(static_cast<u8>(cat));
                    sig[i] = static_cast<u8>(cat > 0);
                    if (cat > 0) {
                        bits[s].put(a - (1u << (cat - 1)), cat - 1);  // mantissa
                        bits[s].put(level < 0 ? 1u : 0u, 1);          // sign
                    }
                }
    }

    std::vector<u8> out;
    detail::put_u32(out, static_cast<u32>(side));
    detail::put_u32(out, static_cast<u32>(params.levels));
    detail::put_u32(out, std::bit_cast<u32>(params.q));
    for (int s = 0; s < nscale; ++s) {
        for (int c = 0; c < kCtx; ++c) detail::encode_plane(out, cats[static_cast<usize>(s)][static_cast<usize>(c)]);
        bits[static_cast<usize>(s)].flush();
        detail::put_var(out, static_cast<u32>(bits[static_cast<usize>(s)].bytes.size()));
        out.insert(out.end(), bits[static_cast<usize>(s)].bytes.begin(), bits[static_cast<usize>(s)].bytes.end());
    }
    return out;
}

// Decode a block payload back to a side^3 f32 block (lossy, within ~step/2 per coefficient).
inline std::vector<f32> decode_block(std::span<const u8> payload) {
    usize p = 0;
    const s64 side = static_cast<s64>(detail::get_u32(payload, p));
    const int levels = static_cast<int>(detail::get_u32(payload, p));
    const f32 q = std::bit_cast<f32>(detail::get_u32(payload, p));
    const int nscale = levels + 1;

    std::vector<std::array<std::vector<u8>, kCtx>> cats(static_cast<usize>(nscale));
    std::vector<std::array<usize, kCtx>> cur(static_cast<usize>(nscale), std::array<usize, kCtx>{});
    std::vector<detail::BitReader> bits(static_cast<usize>(nscale));
    for (int s = 0; s < nscale; ++s) {
        for (int c = 0; c < kCtx; ++c) cats[static_cast<usize>(s)][static_cast<usize>(c)] = detail::decode_plane(payload, p);
        const u32 nb = detail::get_var(payload, p);
        bits[static_cast<usize>(s)] = detail::BitReader{payload.data() + p, nb};
        p += nb;
    }

    std::vector<f32> step(static_cast<usize>(nscale + 1));
    for (int s = 1; s <= nscale; ++s) step[static_cast<usize>(s)] = detail::scale_step(q, s);
    std::vector<f32> coef(static_cast<usize>(side * side * side), 0.0f);
    std::vector<u8> sig(static_cast<usize>(side * side * side), 0);
    const s64 sz = side * side;

    for (const auto& sb : detail::subband_boxes(side, levels)) {
        const usize s = static_cast<usize>(sb.scale - 1);
        for (s64 z = sb.z0; z < sb.z1; ++z)
            for (s64 y = sb.y0; y < sb.y1; ++y)
                for (s64 x = sb.x0; x < sb.x1; ++x) {
                    const usize i = static_cast<usize>(z * sz + y * side + x);
                    const int ctx = (z > sb.z0 ? sig[i - static_cast<usize>(sz)] : 0) +
                                    (y > sb.y0 ? sig[i - static_cast<usize>(side)] : 0) +
                                    (x > sb.x0 ? sig[i - 1] : 0);
                    const int cat = cats[s][static_cast<usize>(ctx)][cur[s][static_cast<usize>(ctx)]++];
                    sig[i] = static_cast<u8>(cat > 0);
                    if (cat > 0) {
                        const u32 a = (1u << (cat - 1)) + bits[s].get(cat - 1);
                        const s32 level = bits[s].get(1) ? -static_cast<s32>(a) : static_cast<s32>(a);
                        coef[i] = (static_cast<f32>(level) + (level > 0 ? 0.5f : -0.5f)) * step[static_cast<usize>(sb.scale)];
                    }
                }
    }
    dwt3_inverse(coef, side, levels);
    return coef;
}

}  // namespace fenix::codec
