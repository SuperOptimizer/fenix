// io/jpeg.hpp — first-party baseline (sequential) JPEG encoder. No libjpeg (io/CLAUDE.md).
// Grayscale (1 comp) or RGB (3 comp, YCbCr 4:4:4, no chroma subsampling — simpler, higher
// quality). Standard Annex-K quantization + Huffman tables, orthonormal 8x8 FDCT, bit writer
// with 0xFF byte stuffing. Good enough for "look at a slice"; not tuned for size.
#pragma once

#include "core/core.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace fenix::io {

// Interleaved 8-bit image: comps == 1 (gray) or 3 (RGB).
struct Image {
    int w = 0, h = 0, comps = 1;
    std::vector<u8> px;  // size w*h*comps, row-major, interleaved
    u8& at(int y, int x, int c) { return px[(static_cast<usize>(y) * w + x) * comps + c]; }
};

namespace detail {

inline constexpr std::array<int, 64> kZigZag = {
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15,
    23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

inline constexpr std::array<int, 64> kLumQuant = {
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57,
    69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64,
    81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
inline constexpr std::array<int, 64> kChromQuant = {
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99,
    99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

// Standard Huffman spec tables (JPEG Annex K.3): per-length code counts + symbol values.
inline constexpr std::array<u8, 16> kDcLumBits = {0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
inline constexpr std::array<u8, 12> kDcLumVal = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
inline constexpr std::array<u8, 16> kDcChrBits = {0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
inline constexpr std::array<u8, 12> kDcChrVal = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
inline constexpr std::array<u8, 16> kAcLumBits = {0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
inline constexpr std::array<u8, 162> kAcLumVal = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA};
inline constexpr std::array<u8, 16> kAcChrBits = {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
inline constexpr std::array<u8, 162> kAcChrVal = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA};

// (code,size) per symbol value, built from the spec bits/vals.
struct Huff {
    std::array<u16, 256> code{};
    std::array<u8, 256> size{};
};
inline Huff build_huff(const u8* bits, const u8* vals, int nval) {
    Huff h{};
    int k = 0;
    u16 code = 0;
    for (int len = 1; len <= 16; ++len) {
        for (int i = 0; i < bits[len - 1]; ++i) {
            h.code[vals[k]] = code;
            h.size[vals[k]] = static_cast<u8>(len);
            ++code;
            ++k;
        }
        code = static_cast<u16>(code << 1);
    }
    (void)nval;
    return h;
}

struct BitWriter {
    std::vector<u8>& out;
    u32 acc = 0;
    int nbits = 0;
    explicit BitWriter(std::vector<u8>& o) : out(o) {}
    void put(u16 code, int size) {
        acc |= static_cast<u32>(code) << (24 - nbits - size);
        nbits += size;
        while (nbits >= 8) {
            const u8 b = static_cast<u8>((acc >> 16) & 0xFF);
            out.push_back(b);
            if (b == 0xFF) out.push_back(0x00);  // byte stuffing
            acc <<= 8;
            nbits -= 8;
        }
    }
};

// magnitude category + the size-bit representation of a (possibly negative) value.
inline void encode_val(int v, int& cat, u16& bits) {
    int t = v < 0 ? -v : v;
    cat = 0;
    while (t) { ++cat; t >>= 1; }
    bits = static_cast<u16>(v < 0 ? (v + (1 << cat) - 1) : v) & static_cast<u16>((1 << cat) - 1);
}

// Orthonormal 8x8 forward DCT-II on a level-shifted block (in place, double).
inline void fdct8x8(double b[64]) {
    static double M[8][8];
    static bool init = false;
    if (!init) {
        for (int u = 0; u < 8; ++u)
            for (int x = 0; x < 8; ++x)
                M[u][x] = (u == 0 ? std::sqrt(1.0 / 8.0) : std::sqrt(2.0 / 8.0)) *
                          std::cos((2 * x + 1) * u * std::numbers::pi / 16.0);
        init = true;
    }
    double t[64];
    for (int u = 0; u < 8; ++u)        // rows: t = M * b
        for (int y = 0; y < 8; ++y) {
            double s = 0;
            for (int x = 0; x < 8; ++x) s += M[u][x] * b[y * 8 + x];
            t[y * 8 + u] = s;
        }
    for (int v = 0; v < 8; ++v)        // cols: out = t * M^T  => b[v][u] = sum_y M[v][y] t[y][u]
        for (int u = 0; u < 8; ++u) {
            double s = 0;
            for (int y = 0; y < 8; ++y) s += M[v][y] * t[y * 8 + u];
            b[v * 8 + u] = s;
        }
}

// Encode one 8x8 block: FDCT, quantize, Huffman DC(diff)+AC(run/size). Returns new DC predictor.
inline int encode_block(BitWriter& bw, const double in[64], const int* quant, int prev_dc,
                        const Huff& dc, const Huff& ac) {
    double b[64];
    for (int i = 0; i < 64; ++i) b[i] = in[i];
    fdct8x8(b);
    int q[64];
    for (int i = 0; i < 64; ++i) q[i] = static_cast<int>(std::lround(b[i] / quant[i]));

    const int dcv = q[0];
    int cat;
    u16 bits;
    encode_val(dcv - prev_dc, cat, bits);
    bw.put(dc.code[static_cast<usize>(cat)], dc.size[static_cast<usize>(cat)]);
    if (cat) bw.put(bits, cat);

    int run = 0;
    for (int k = 1; k < 64; ++k) {
        const int v = q[kZigZag[static_cast<usize>(k)]];
        if (v == 0) { ++run; continue; }
        while (run > 15) { bw.put(ac.code[0xF0], ac.size[0xF0]); run -= 16; }  // ZRL
        encode_val(v, cat, bits);
        const int sym = (run << 4) | cat;
        bw.put(ac.code[static_cast<usize>(sym)], ac.size[static_cast<usize>(sym)]);
        bw.put(bits, cat);
        run = 0;
    }
    if (run > 0) bw.put(ac.code[0x00], ac.size[0x00]);  // EOB
    return dcv;
}

inline void put16(std::vector<u8>& o, int v) { o.push_back(static_cast<u8>(v >> 8)); o.push_back(static_cast<u8>(v & 0xFF)); }
inline void scaled_quant(const std::array<int, 64>& base, int quality, int out[64]) {
    const int q = quality < 1 ? 1 : (quality > 100 ? 100 : quality);
    const int s = q < 50 ? 5000 / q : 200 - 2 * q;
    for (int i = 0; i < 64; ++i) {
        int v = (base[static_cast<usize>(i)] * s + 50) / 100;
        out[i] = v < 1 ? 1 : (v > 255 ? 255 : v);
    }
}

}  // namespace detail

// Write `img` (1 or 3 comps) as baseline JPEG at `quality` (1..100).
inline Expected<void> write_jpeg(const std::string& path, const Image& img, int quality = 90) {
    using namespace detail;
    if (img.comps != 1 && img.comps != 3) return err(Errc::invalid_argument, "jpeg: comps must be 1 or 3");
    const int W = img.w, H = img.h, NC = img.comps;

    int ql[64], qc[64];
    scaled_quant(kLumQuant, quality, ql);
    scaled_quant(kChromQuant, quality, qc);
    const Huff dcL = build_huff(kDcLumBits.data(), kDcLumVal.data(), 12);
    const Huff acL = build_huff(kAcLumBits.data(), kAcLumVal.data(), 162);
    const Huff dcC = build_huff(kDcChrBits.data(), kDcChrVal.data(), 12);
    const Huff acC = build_huff(kAcChrBits.data(), kAcChrVal.data(), 162);

    std::vector<u8> o;
    o.reserve(static_cast<usize>(W) * H * NC / 4 + 1024);
    auto marker = [&](int m) { o.push_back(0xFF); o.push_back(static_cast<u8>(m)); };

    marker(0xD8);                                   // SOI
    marker(0xE0); put16(o, 16);                     // APP0/JFIF
    for (char c : std::string("JFIF")) o.push_back(static_cast<u8>(c));
    o.push_back(0); o.push_back(1); o.push_back(1); o.push_back(0);
    put16(o, 1); put16(o, 1); o.push_back(0); o.push_back(0);

    // DQT (lum; + chrom if color)
    marker(0xDB); put16(o, 2 + 65); o.push_back(0x00); for (int i = 0; i < 64; ++i) o.push_back(static_cast<u8>(ql[kZigZag[static_cast<usize>(i)]]));
    if (NC == 3) { marker(0xDB); put16(o, 2 + 65); o.push_back(0x01); for (int i = 0; i < 64; ++i) o.push_back(static_cast<u8>(qc[kZigZag[static_cast<usize>(i)]])); }

    // SOF0
    marker(0xC0); put16(o, 8 + 3 * NC); o.push_back(8); put16(o, H); put16(o, W); o.push_back(static_cast<u8>(NC));
    for (int c = 0; c < NC; ++c) { o.push_back(static_cast<u8>(c + 1)); o.push_back(0x11); o.push_back(c == 0 ? 0 : 1); }

    // DHT (4 tables for color, 2 for gray)
    auto emit_dht = [&](int cls_id, const u8* bits, const u8* vals, int nval) {
        marker(0xC4); put16(o, 2 + 1 + 16 + nval); o.push_back(static_cast<u8>(cls_id));
        for (int i = 0; i < 16; ++i) o.push_back(bits[i]);
        for (int i = 0; i < nval; ++i) o.push_back(vals[i]);
    };
    emit_dht(0x00, kDcLumBits.data(), kDcLumVal.data(), 12);
    emit_dht(0x10, kAcLumBits.data(), kAcLumVal.data(), 162);
    if (NC == 3) {
        emit_dht(0x01, kDcChrBits.data(), kDcChrVal.data(), 12);
        emit_dht(0x11, kAcChrBits.data(), kAcChrVal.data(), 162);
    }

    // SOS
    marker(0xDA); put16(o, 6 + 2 * NC); o.push_back(static_cast<u8>(NC));
    for (int c = 0; c < NC; ++c) { o.push_back(static_cast<u8>(c + 1)); o.push_back(c == 0 ? 0x00 : 0x11); }
    o.push_back(0); o.push_back(63); o.push_back(0);

    // Entropy-coded scan, 4:4:4 (one 8x8 per component per MCU).
    BitWriter bw(o);
    int pdc[3] = {0, 0, 0};
    auto sample = [&](int yy, int xx, int c) -> double {
        const int y = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
        const int x = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
        const usize idx = (static_cast<usize>(y) * W + x) * NC;
        if (NC == 1) return img.px[idx];
        const double r = img.px[idx], g = img.px[idx + 1], b = img.px[idx + 2];
        if (c == 0) return 0.299 * r + 0.587 * g + 0.114 * b;
        if (c == 1) return -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
        return 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
    };
    double blk[64];
    for (int my = 0; my < H; my += 8) {
        for (int mx = 0; mx < W; mx += 8) {
            for (int c = 0; c < NC; ++c) {
                for (int y = 0; y < 8; ++y)
                    for (int x = 0; x < 8; ++x) blk[y * 8 + x] = sample(my + y, mx + x, c) - 128.0;
                pdc[c] = encode_block(bw, blk, c == 0 ? ql : qc, pdc[c],
                                      c == 0 ? dcL : dcC, c == 0 ? acL : acC);
            }
        }
    }
    // Flush remaining bits padded with 1s.
    if (bw.nbits > 0) bw.put(static_cast<u16>((1 << (8 - bw.nbits % 8)) - 1), 8 - bw.nbits % 8);
    marker(0xD9);  // EOI

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return err(Errc::io_error, "jpeg: cannot write " + path);
    std::fwrite(o.data(), 1, o.size(), f);
    std::fclose(f);
    return {};
}

}  // namespace fenix::io
