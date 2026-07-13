// io/dct3d.hpp — decoder for the standalone `dct3d` 16^3 lossy block codec
// (SuperOptimizer/3ddct, MIT — the matter-compressor lineage codec used by the
// dl.ash2txt.org community zarr exports). Fresh C++ port of the DECODE path only:
// fenix reads dct3d-compressed zarr v3 shards as a source format (like TIFF/PNG);
// it never writes them (.fxvol stays the native container).
//
// Blob layout: [u8 0xD3][u8 dtype] then ONE range-coded stream:
//   fval(vmin) fval(vspan) fval(qstep) seg(dc*16) | corr-flag | levels | corrections
// Pipeline: range-decode levels -> dead-zone dequant (step = qstep*(1+L1freq)^0.65,
// dz 0.80) -> inverse 16^3 DCT (separable orthonormal DCT-III) -> +dc -> denormalize
// from the 0..255 working domain -> round/clamp to dtype -> apply corrections.
// The range coder is bit-exact integer arithmetic (load-bearing); the value math is
// float and tolerance-grade, matching the codec's own not-bit-reproducible contract.
// Robust to corrupt input: never reads/writes out of bounds, returns false instead.
#pragma once

#include "core/core.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <numbers>
#include <span>

namespace fenix::io::dct3d {

inline constexpr s64 kN = 16;
inline constexpr s64 kN3 = kN * kN * kN;

namespace detail {

inline constexpr f32 kDzFrac = 0.80f;
inline constexpr f32 kHfExp = 0.65f;
inline constexpr int kDcQ = 16;

struct Tables {
    std::array<std::array<f32, kN>, kN> cm{};  // orthonormal DCT-II matrix
    std::array<f32, 3 * (kN - 1) + 1> hfw{};   // (1+L1freq)^kHfExp
    std::array<u16, kN3> scan{};               // ascending-L1-frequency order
    Tables() {
        for (int k = 0; k < kN; ++k) {
            const f64 ck = (k == 0) ? std::sqrt(1.0 / kN) : std::sqrt(2.0 / kN);
            for (int n = 0; n < kN; ++n)
                cm[static_cast<usize>(k)][static_cast<usize>(n)] =
                    static_cast<f32>(ck * std::cos(std::numbers::pi * (2.0 * n + 1.0) * k / (2.0 * kN)));
        }
        for (int f = 0; f <= 3 * (kN - 1); ++f)
            hfw[static_cast<usize>(f)] = std::pow(1.0f + static_cast<f32>(f), kHfExp);
        // counting sort by L1 frequency, raster order within a frequency
        std::array<int, 3 * (kN - 1) + 2> hist{};
        for (int idx = 0; idx < kN3; ++idx) {
            const int cz = idx / (kN * kN), cy = (idx / kN) % kN, cx = idx % kN;
            hist[static_cast<usize>(cz + cy + cx + 1)]++;
        }
        for (usize i = 1; i < hist.size(); ++i) hist[i] += hist[i - 1];
        for (int idx = 0; idx < kN3; ++idx) {
            const int cz = idx / (kN * kN), cy = (idx / kN) % kN, cx = idx % kN;
            scan[static_cast<usize>(hist[static_cast<usize>(cz + cy + cx)]++)] = static_cast<u16>(idx);
        }
    }
};
inline const Tables& tables() {
    static const Tables t;  // magic-static: built on first decode, no global ctor
    return t;
}

// --- binary range decoder (bit-exact integer; mirrors the reference coder) ---
struct Ctx {
    u16 p0 = 1u << 11;  // adaptive P(bit==0) in 1/4096 units
};
struct Dec {
    const u8* buf;
    usize len, pos = 0;
    u32 code = 0, range = 0xFFFFFFFFu;
    static constexpr u32 kTop = 1u << 24;
    Dec(const u8* b, usize n) : buf(b), len(n) {
        for (int i = 0; i < 5; ++i) code = (code << 8) | next();
    }
    u8 next() { return pos < len ? buf[pos++] : 0; }
    void norm() {
        while (range < kTop) {
            code = (code << 8) | next();
            range <<= 8;
        }
    }
    int bit(Ctx& c) {
        const u32 r0 = (range >> 12) * c.p0;
        int b;
        if (code < r0) {
            range = r0;
            b = 0;
            c.p0 = static_cast<u16>(c.p0 + ((4096 - c.p0) >> 4));
        } else {
            code -= r0;
            range -= r0;
            b = 1;
            c.p0 = static_cast<u16>(c.p0 - (c.p0 >> 4));
        }
        norm();
        return b;
    }
    int bypass() {
        range >>= 1;
        const int b = code >= range;
        if (b) code -= range;
        norm();
        return b;
    }
    u32 bypass_n(int k) {
        u32 v = 0;
        while (k > 16) {
            v = (v << 16) | bypass_n(16);
            k -= 16;
        }
        if (!k) return v;
        range >>= k;
        u32 q = code / range;
        const u32 m = (1u << k) - 1;
        if (q > m) q = m;
        code -= q * range;
        norm();
        return (v << k) | q;
    }
    u32 eg() {  // Exp-Golomb order 0
        u32 nb = 0;
        while (bypass()) {
            if (++nb > 31) return 0;
        }
        if (!nb) return 0;
        return ((1u << nb) | bypass_n(static_cast<int>(nb))) - 1;
    }
    s64 seg() {  // signed (zig-zag)
        const u32 z = eg();
        return (z & 1) ? -static_cast<s64>((z + 1) >> 1) : static_cast<s64>(z >> 1);
    }
    f32 fval() {  // integer (seg) or raw-f32 header field
        if (!bypass()) return static_cast<f32>(seg());
        const u32 b = bypass_n(32);
        f32 f;
        std::memcpy(&f, &b, 4);
        return f;
    }
};

// --- coefficient context coder (static priors trained on scroll CT) ---
inline constexpr int kBands = 8, kMagCtx = 12, kEobCtx = 14;
inline constexpr u16 kPriSig[kBands * 4] = {
    1980, 575,  846,  743,  3916, 3707, 3515, 1463, 3969, 3753, 3527, 2313, 4002, 3826, 3635, 2819,
    4039, 3851, 3613, 2974, 3985, 3840, 3703, 3475, 4021, 3991, 3951, 3969, 4065, 4024, 2048, 2048,
};
inline constexpr u16 kPriMag[kMagCtx] = {1893, 1217, 908, 728, 610, 527, 465, 416, 377, 346, 320, 297};
inline constexpr u16 kPriEob[kEobCtx] = {74, 1, 1, 1, 1, 5, 109, 504, 772, 1063, 1458, 2326, 4095, 2048};

struct CoefCtx {
    std::array<Ctx, kBands * 4> sig;
    std::array<Ctx, kMagCtx> mag;
    std::array<Ctx, kEobCtx> eob;
    CoefCtx() {
        for (usize i = 0; i < sig.size(); ++i) sig[i].p0 = kPriSig[i] ? kPriSig[i] : (1u << 11);
        for (usize i = 0; i < mag.size(); ++i) mag[i].p0 = kPriMag[i];
        for (usize i = 0; i < eob.size(); ++i) eob[i].p0 = kPriEob[i] ? kPriEob[i] : (1u << 11);
    }
};

inline int band_of(u32 idx) {
    const u32 cz = idx / (kN * kN), cy = (idx / kN) % kN, cx = idx % kN;
    int b = static_cast<int>((cz + cy + cx) * kBands / (3u * kN));
    return b >= kBands ? kBands - 1 : b;
}

inline u32 dec_magnitude(Dec& d, CoefCtx& ac) {
    u32 v = 0, k = 0;
    while (k < static_cast<u32>(kMagCtx - 1)) {
        if (d.bit(ac.mag[k])) {
            v += 1;
            k++;
        } else
            return v + 1;
    }
    if (!d.bit(ac.mag[k])) return v + 1;
    u32 nbits = 0;
    while (d.bypass()) {
        if (++nbits > 31) break;
    }
    const u32 x = nbits ? ((1u << nbits) | d.bypass_n(static_cast<int>(nbits))) - 1 : 0;
    return v + x + 1;
}

inline u32 dec_eob(Dec& d, CoefCtx& c) {
    int kmax = 0;
    while ((1u << kmax) <= static_cast<u32>(kN3)) kmax++;
    int k = 0;
    while (k < kmax && d.bit(c.eob[static_cast<usize>(k)])) k++;
    if (k == 0) return 0;
    if (k == 1) return 1;
    return (1u << (k - 1)) | d.bypass_n(k - 1);
}

inline void dec_coefs(Dec& d, std::span<f32, kN3> lvl) {
    CoefCtx ac;
    for (auto& v : lvl) v = 0.0f;
    u32 eob = dec_eob(d, ac);
    if (eob > static_cast<u32>(kN3)) eob = kN3;
    u32 hist = 0;
    const auto& scan = tables().scan;
    for (u32 p = 0; p < eob; ++p) {
        const u32 idx = scan[p];
        const int b = band_of(idx);
        int dens = std::popcount(hist & 0xFFFFu);
        dens = dens < 3 ? dens : 3;
        const int sig = (p == eob - 1) ? 1 : d.bit(ac.sig[static_cast<usize>(b * 4 + dens)]);
        hist = (hist << 1) | static_cast<u32>(sig);
        if (!sig) continue;
        const u32 m = dec_magnitude(d, ac);
        const int neg = d.bypass();
        lvl[idx] = neg ? -static_cast<f32>(m) : static_cast<f32>(m);
    }
}

// --- inverse transform ---
inline void dct1d_inv(const f32* in, f32* out) {
    constexpr int S = kN, H = S / 2;
    f32 e[H], o[H];
    for (int n = 0; n < H; ++n) e[n] = o[n] = 0.0f;
    const auto& cm = tables().cm;
    for (int k = 0; k < S; k += 2) {
        const f32 v = in[k];
        if (v != 0.0f)
            for (int n = 0; n < H; ++n) e[n] += cm[static_cast<usize>(k)][static_cast<usize>(n)] * v;
    }
    for (int k = 1; k < S; k += 2) {
        const f32 v = in[k];
        if (v != 0.0f)
            for (int n = 0; n < H; ++n) o[n] += cm[static_cast<usize>(k)][static_cast<usize>(n)] * v;
    }
    for (int n = 0; n < H; ++n) {
        out[n] = e[n] + o[n];
        out[S - 1 - n] = e[n] - o[n];
    }
}

inline void lines_inv(const f32* src, f32* dst) {
    constexpr int S = kN;
    f32 ol[S];
    for (int off = 0; off < S * S; ++off) {
        const f32* v = src + static_cast<usize>(off) * S;
        f32* o = dst + static_cast<usize>(off) * S;
        bool nz = false;
        for (int i = 0; i < S; ++i)
            if (v[i] != 0.0f) {
                nz = true;
                break;
            }
        if (!nz) {
            for (int i = 0; i < S; ++i) o[i] = 0.0f;
            continue;
        }
        dct1d_inv(v, ol);
        for (int i = 0; i < S; ++i) o[i] = ol[i];
    }
}

inline void rot(const f32* src, f32* dst) {  // (z,y,x) -> (x,z,y), tiled
    constexpr int S = kN, T = 8;
    for (int zt = 0; zt < S; zt += T)
        for (int xt = 0; xt < S; xt += T)
            for (int z = zt; z < zt + T; ++z)
                for (int x = xt; x < xt + T; ++x) {
                    const f32* sp = src + static_cast<usize>(z) * S * S + x;
                    f32* dp = dst + (static_cast<usize>(x) * S + static_cast<usize>(z)) * S;
                    for (int y = 0; y < S; ++y) dp[y] = sp[static_cast<usize>(y) * S];
                }
}

inline void dct3_inv(const f32* coef, f32* blk, f32* a, f32* b) {
    lines_inv(coef, a);
    rot(a, b);
    lines_inv(b, a);
    rot(a, b);
    lines_inv(b, a);
    rot(a, blk);
}

inline f32 deq_one(f32 lv, f32 step) {
    if (lv == 0.0f) return 0.0f;
    const f32 a = std::fabs(lv);
    const f32 r = (a - 1.0f) * step + kDzFrac * step + 0.40f * step;
    return lv < 0.0f ? -r : r;
}

// clamp with NaN -> lo (a corrupt blob can make voxels non-finite; casting a
// non-finite float to an integer type is UB — this makes the store defined).
inline f32 clamp_nan(f32 v, f32 lo, f32 hi) {
    if (!(v >= lo)) return lo;
    return v > hi ? hi : v;
}

enum : u8 { kDtU8 = 1, kDtU16 = 2, kDtU32 = 3, kDtS8 = 4, kDtS16 = 5, kDtS32 = 6, kDtF32 = 7 };
inline constexpr u8 kMagic = 0xD3;

inline bool decode_float(std::span<const u8> blob, u8 want_dtype, std::span<f32, kN3> out) {
    if (blob.size() < 2) return false;
    if (blob[0] != kMagic || blob[1] != want_dtype) return false;
    Dec d(blob.data() + 2, blob.size() - 2);
    const f32 vmin = d.fval();
    const f32 vspan = d.fval();
    const f32 qstep = d.fval();
    const f32 dc = static_cast<f32>(d.seg()) / static_cast<f32>(kDcQ);
    if (!(vspan > 0.0f) || !(qstep > 0.0f) || !std::isfinite(vmin) || !std::isfinite(dc)) return false;

    Ctx cflag;
    const int has_corr = d.bit(cflag);

    f32 lvl[kN3];
    dec_coefs(d, std::span<f32, kN3>(lvl));

    f32 coef[kN3], a[kN3], b[kN3], rb[kN3];
    const auto& hfw = tables().hfw;
    for (int cz = 0; cz < kN; ++cz)
        for (int cy = 0; cy < kN; ++cy)
            for (int cx = 0; cx < kN; ++cx) {
                const int i = (cz * kN + cy) * kN + cx;
                coef[i] = deq_one(lvl[i], qstep * hfw[static_cast<usize>(cz + cy + cx)]);
            }
    dct3_inv(coef, rb, a, b);

    for (int i = 0; i < kN3; ++i) out[static_cast<usize>(i)] = rb[i] + dc;

    if (has_corr) {
        u32 ncorr = d.eg() + 1, pos = 0;
        if (ncorr > static_cast<u32>(kN3)) ncorr = kN3;
        f32 cq = d.fval();
        if (!(cq > 0.0f) || !std::isfinite(cq)) cq = 1.0f;
        for (u32 c = 0; c < ncorr; ++c) {
            pos += d.eg();
            const int neg = d.bypass();
            const u32 mag = d.eg() + 1;
            if (pos >= static_cast<u32>(kN3)) break;
            const f32 delta = static_cast<f32>(mag) * cq;
            out[pos] += neg ? -delta : delta;
        }
    }

    const f32 inv = vspan / 255.0f;
    for (auto& v : out) v = v * inv + vmin;
    return true;
}

}  // namespace detail

// Decode one dct3d blob into 4096 voxels of T (z-major). Returns false on a
// malformed blob or dtype mismatch — callers must treat that as a hard decode
// error, never as fill/air (the io absent-vs-failed invariant).
template <typename T>
[[nodiscard]] inline bool decode(std::span<const u8> blob, std::span<T> out) {
    if (out.size() != static_cast<usize>(kN3)) return false;
    using detail::clamp_nan;
    f32 fb[kN3];
    u8 dt;
    f32 lo, hi;
    if constexpr (std::is_same_v<T, u8>) {
        dt = detail::kDtU8;
        lo = 0;
        hi = 255;
    } else if constexpr (std::is_same_v<T, u16>) {
        dt = detail::kDtU16;
        lo = 0;
        hi = 65535;
    } else if constexpr (std::is_same_v<T, f32>) {
        dt = detail::kDtF32;
        lo = 0;
        hi = 0;  // unused
    } else {
        static_assert(std::is_same_v<T, u8> || std::is_same_v<T, u16> || std::is_same_v<T, f32>,
                      "dct3d::decode supports u8/u16/f32 (the scroll-CT source dtypes)");
    }
    if (!detail::decode_float(blob, dt, std::span<f32, kN3>(fb))) return false;
    for (int i = 0; i < kN3; ++i) {
        if constexpr (std::is_same_v<T, f32>)
            out[static_cast<usize>(i)] = fb[i];
        else
            out[static_cast<usize>(i)] = static_cast<T>(clamp_nan(std::round(fb[i]), lo, hi));
    }
    return true;
}

}  // namespace fenix::io::dct3d
