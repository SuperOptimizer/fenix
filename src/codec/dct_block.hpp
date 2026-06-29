// codec/dct_block.hpp — the DCT-16 lossy block codec (the second transform codec). Pipeline for one
// 16³ block of any dtype T:
//   widen→f32 → subtract block DC (mean) → separable float DCT-16 → frequency-weighted dead-zone quant
//   → magnitude-category rANS with a causal 3D significance context (REUSED from block.hpp) + raw
//   mantissa/sign bits.
// Decode reverses it and narrows back to T. A rewrite of matter-compressor's mc_codec_float: the
// frequency-weighted step `q·(1+cz+cy+cx)^hf_exp`, the 0.8 dead-zone, and the centroid dequant are
// theirs; the entropy stage is fenix's rANS (not mc's binary range coder — rANS is SIMD/GPU-amenable,
// the codec invariant). All-float compute. See codec/CLAUDE.md.
#pragma once

#include "codec/block.hpp"  // detail::{encode_plane,decode_plane,BitWriter,BitReader,put_u32,get_u32,put_var,get_var}, kCtx
#include "codec/dct.hpp"
#include "codec/dtype.hpp"
#include "core/types.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <span>
#include <vector>

namespace fenix::codec {

struct DctParams {
    f32 q = 8.0f;         // base quantization step (smaller = higher fidelity / larger)
    f32 hf_exp = 0.65f;   // frequency weighting: step(cz,cy,cx) = q·(1+cz+cy+cx)^hf_exp (coarser HF)
    f32 dz_frac = 0.80f;  // dead-zone width as a fraction of step (truncate-to-zero below dz)
};

namespace detail {
inline f32 dct_step(const DctParams& p, int cz, int cy, int cx) {
    return p.q * std::pow(1.0f + static_cast<f32>(cz + cy + cx), p.hf_exp);
}
}  // namespace detail

// Encode a kDctN³ block of dtype T -> payload (carries the dtype + DC so decode reproduces them).
template <class T>
inline std::vector<u8> encode_block_dct(std::span<const T> block, DctParams params = {}) {
    constexpr s64 N = kDctN, sz = N * N, V = N * N * N;
    std::vector<f32> c = to_f32<T>(block);  // widen
    f64 msum = 0;
    for (f32 v : c) msum += static_cast<f64>(v);
    const f32 dc = static_cast<f32>(msum / static_cast<f64>(V));  // block DC
    for (f32& v : c) v -= dc;
    dct16_3d_fwd(c);

    std::array<std::vector<u8>, kCtx> cats;
    detail::BitWriter bits;
    std::vector<u8> sig(static_cast<usize>(V), 0);
    for (s64 z = 0; z < N; ++z)
        for (s64 y = 0; y < N; ++y)
            for (s64 x = 0; x < N; ++x) {
                const usize i = static_cast<usize>((z * N + y) * N + x);
                const int ctx = (z > 0 ? sig[i - static_cast<usize>(sz)] : 0) +
                                (y > 0 ? sig[i - static_cast<usize>(N)] : 0) + (x > 0 ? sig[i - 1] : 0);
                const f32 step = detail::dct_step(params, static_cast<int>(z), static_cast<int>(y), static_cast<int>(x));
                const f32 a0 = std::abs(c[i]);
                s32 level = 0;
                if (a0 >= params.dz_frac * step)
                    level = (c[i] < 0 ? -1 : 1) * static_cast<s32>(std::floor(a0 / step + (1.0f - params.dz_frac)));
                level = std::clamp(level, -32768, 32767);
                const u32 a = static_cast<u32>(level < 0 ? -level : level);
                const int cat = a ? std::bit_width(a) : 0;
                cats[static_cast<usize>(ctx)].push_back(static_cast<u8>(cat));
                sig[i] = static_cast<u8>(cat > 0);
                if (cat > 0) {
                    bits.put(a - (1u << (cat - 1)), cat - 1);
                    bits.put(level < 0 ? 1u : 0u, 1);
                }
            }

    std::vector<u8> out;
    detail::put_u32(out, static_cast<u32>(N));
    out.push_back(static_cast<u8>(dtype_of<T>()));
    detail::put_u32(out, std::bit_cast<u32>(dc));
    detail::put_u32(out, std::bit_cast<u32>(params.q));
    detail::put_u32(out, std::bit_cast<u32>(params.hf_exp));
    detail::put_u32(out, std::bit_cast<u32>(params.dz_frac));
    for (int ctx = 0; ctx < kCtx; ++ctx) detail::encode_plane(out, cats[static_cast<usize>(ctx)]);
    bits.flush();
    detail::put_var(out, static_cast<u32>(bits.bytes.size()));
    out.insert(out.end(), bits.bytes.begin(), bits.bytes.end());
    return out;
}

struct DctDecoded {
    DType dtype;
    std::vector<f32> data;  // reconstructed kDctN³ block in f32 (transform domain)
};

// Decode a payload to the reconstructed f32 block + its source dtype (use to_dtype to narrow back).
inline DctDecoded decode_block_dct(std::span<const u8> payload) {
    usize p = 0;
    const s64 N = static_cast<s64>(detail::get_u32(payload, p));
    const DType dt = static_cast<DType>(payload[p++]);
    const f32 dc = std::bit_cast<f32>(detail::get_u32(payload, p));
    DctParams params;
    params.q = std::bit_cast<f32>(detail::get_u32(payload, p));
    params.hf_exp = std::bit_cast<f32>(detail::get_u32(payload, p));
    params.dz_frac = std::bit_cast<f32>(detail::get_u32(payload, p));
    const s64 sz = N * N, V = N * N * N;

    std::array<std::vector<u8>, kCtx> cats;
    std::array<usize, kCtx> cur{};
    for (int ctx = 0; ctx < kCtx; ++ctx) cats[static_cast<usize>(ctx)] = detail::decode_plane(payload, p);
    const u32 nb = detail::get_var(payload, p);
    detail::BitReader bits{payload.data() + p, nb};
    p += nb;

    std::vector<f32> c(static_cast<usize>(V), 0.0f);
    std::vector<u8> sig(static_cast<usize>(V), 0);
    for (s64 z = 0; z < N; ++z)
        for (s64 y = 0; y < N; ++y)
            for (s64 x = 0; x < N; ++x) {
                const usize i = static_cast<usize>((z * N + y) * N + x);
                const int ctx = (z > 0 ? sig[i - static_cast<usize>(sz)] : 0) +
                                (y > 0 ? sig[i - static_cast<usize>(N)] : 0) + (x > 0 ? sig[i - 1] : 0);
                const int cat = cats[static_cast<usize>(ctx)][cur[static_cast<usize>(ctx)]++];
                sig[i] = static_cast<u8>(cat > 0);
                if (cat > 0) {
                    const u32 a = (1u << (cat - 1)) + bits.get(cat - 1);
                    const bool neg = bits.get(1) != 0;
                    const f32 step = detail::dct_step(params, static_cast<int>(z), static_cast<int>(y), static_cast<int>(x));
                    // centroid dequant: |recon| = step·(|level| − 1 + dz_frac + 0.40)
                    const f32 mag = step * (static_cast<f32>(a) - 1.0f + params.dz_frac + 0.40f);
                    c[i] = neg ? -mag : mag;
                }
            }
    dct16_3d_inv(c);
    for (f32& v : c) v += dc;
    return {dt, std::move(c)};
}

// Convenience: decode + narrow back to T (round/clamp per dtype).
template <class T>
inline std::vector<T> decode_block_dct_to(std::span<const u8> payload) {
    const DctDecoded d = decode_block_dct(payload);
    return from_f32<T>(d.data);
}

}  // namespace fenix::codec
