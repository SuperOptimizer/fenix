// io/tiff.hpp — minimal first-party TIFF reader for the formats we actually ingest:
// little-endian classic TIFF AND BigTIFF, single-sample grayscale, strips or tiles,
// uncompressed OR LZW (with horizontal predictor 2 / floating-point predictor 3 —
// TechNote3), f32 / u8 / u16 samples. This covers every VC tifxyz variant seen in the
// wild (classic uncompressed single-strip f32 + BigTIFF LZW+fp-predictor 1024² tiles).
// Everything else (big-endian, other codecs, multi-sample) is rejected with a typed
// error — UNTRUSTED INPUT: every offset/count from the file is bounds-checked.
#pragma once

#include "core/core.hpp"

#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace fenix::io {

struct TiffImage {
    s64 width = 0, height = 0;
    std::vector<f32> pix;  // height*width row-major, converted to f32 regardless of source dtype
};

namespace detail {

struct TiffTag {
    u16 tag = 0, type = 0;
    u64 count = 0, value = 0;  // inline value (raw bytes in `inl`) or offset
    u8 inl[8] = {};            // the raw value field (4 B classic / 8 B BigTIFF)
    bool big = false;
};

inline u16 tiff_u16(const u8* p) {
    u16 v;
    std::memcpy(&v, p, 2);
    return v;
}
inline u32 tiff_u32(const u8* p) {
    u32 v;
    std::memcpy(&v, p, 4);
    return v;
}
inline u64 tiff_u64(const u8* p) {
    u64 v;
    std::memcpy(&v, p, 8);
    return v;
}

// Read a tag's value array (SHORT=3, LONG=4, or LONG8=16), inline or via offset. Bounds-checked.
inline Expected<std::vector<u64>> tiff_values(std::span<const u8> d, const TiffTag& t) {
    const usize esz = t.type == 3 ? 2u : t.type == 4 ? 4u : t.type == 16 ? 8u : 0u;
    if (esz == 0) return err(Errc::decode_error, "tiff: unsupported tag type");
    const usize cap = t.big ? 8u : 4u;  // inline-value capacity
    if (t.count > (u64{1} << 24)) return err(Errc::decode_error, "tiff: tag count implausible");
    const usize total = esz * static_cast<usize>(t.count);
    const u8* src;
    if (total <= cap) {
        src = t.inl;
    } else {
        const u64 off = t.value;
        if (off > d.size() || total > d.size() - off) return err(Errc::decode_error, "tiff: tag array out of range");
        src = d.data() + off;
    }
    std::vector<u64> out(static_cast<usize>(t.count));
    for (usize i = 0; i < out.size(); ++i)
        out[i] = esz == 2 ? tiff_u16(src + 2 * i) : esz == 4 ? tiff_u32(src + 4 * i) : tiff_u64(src + 8 * i);
    return out;
}

// TIFF LZW (MSB-first codes, ClearCode=256, EOI=257, 9→12-bit with EARLY CHANGE).
// UNTRUSTED: output capped at `max_out`; malformed streams return an error.
inline Expected<std::vector<u8>> tiff_lzw_decode(std::span<const u8> in, usize max_out) {
    constexpr u32 kClear = 256, kEoi = 257, kFirst = 258, kMaxCode = 4096;
    std::vector<u32> prefix(kMaxCode);
    std::vector<u8> suffix(kMaxCode), first(kMaxCode);
    std::vector<u8> out;
    out.reserve(std::min(max_out, usize{1} << 22));
    u32 next = kFirst, bits = 9;
    u64 acc = 0;
    int nacc = 0;
    usize pos = 0;
    u32 prev = ~0u;
    // NOTE: not named `emit` — this header now reaches the Qt GUI TU, where emit is a macro.
    auto emit_code = [&](u32 code) -> bool {  // append code's string; false on overflow
        u8 stack[kMaxCode];
        int sp = 0;
        while (code >= kFirst) {
            if (sp >= static_cast<int>(kMaxCode)) return false;
            stack[sp++] = suffix[code];
            code = prefix[code];
        }
        if (out.size() + static_cast<usize>(sp) + 1 > max_out) return false;
        out.push_back(static_cast<u8>(code));
        while (sp > 0) out.push_back(stack[--sp]);
        return true;
    };
    auto first_of = [&](u32 code) -> u8 { return code < 256 ? static_cast<u8>(code) : first[code]; };
    while (true) {
        while (nacc < static_cast<int>(bits)) {
            if (pos >= in.size()) return out;  // stream ends without EOI: accept what we have
            acc = (acc << 8) | in[pos++];
            nacc += 8;
        }
        const u32 code = static_cast<u32>((acc >> (nacc - static_cast<int>(bits))) & ((1u << bits) - 1));
        nacc -= static_cast<int>(bits);
        if (code == kEoi) return out;
        if (code == kClear) {
            next = kFirst;
            bits = 9;
            prev = ~0u;
            continue;
        }
        if (prev == ~0u) {
            if (code >= kFirst) return err(Errc::decode_error, "tiff: lzw bad first code");
            if (!emit_code(code)) return err(Errc::decode_error, "tiff: lzw output overflow");
            prev = code;
        } else {
            if (code > next) return err(Errc::decode_error, "tiff: lzw code out of range");
            if (next < kMaxCode) {
                prefix[next] = prev;
                suffix[next] = code == next ? first_of(prev) : first_of(code);
                first[next] = first_of(prev);
                ++next;
            }
            if (!emit_code(code)) return err(Errc::decode_error, "tiff: lzw output overflow");
            prev = code;
        }
        // TIFF early change: widen one code EARLIER than the table actually fills.
        if (next == (1u << bits) - 1 && bits < 12) ++bits;
    }
}

// Undo TIFF predictors on a decompressed block of `rows` rows × `w` samples.
// pred 2: horizontal integer differencing per sample. pred 3 (TechNote3): each row's f32
// bytes are split MSB-plane-first and byte-delta'd; undo the delta then reassemble LE floats.
inline void tiff_undo_predictor(std::vector<u8>& b, u64 rows, u64 w, usize bpp, u64 pred) {
    if (pred == 2) {
        for (u64 r = 0; r < rows; ++r) {
            u8* p = b.data() + r * w * bpp;
            if (bpp == 1)
                for (u64 i = 1; i < w; ++i) p[i] = static_cast<u8>(p[i] + p[i - 1]);
            else if (bpp == 2)
                for (u64 i = 1; i < w; ++i) {
                    u16 a, c;
                    std::memcpy(&a, p + 2 * (i - 1), 2);
                    std::memcpy(&c, p + 2 * i, 2);
                    c = static_cast<u16>(c + a);
                    std::memcpy(p + 2 * i, &c, 2);
                }
        }
    } else if (pred == 3 && bpp == 4) {
        std::vector<u8> tmp(static_cast<usize>(w) * 4);
        for (u64 r = 0; r < rows; ++r) {
            u8* p = b.data() + r * w * 4;
            for (u64 i = 1; i < w * 4; ++i) p[i] = static_cast<u8>(p[i] + p[i - 1]);  // undo byte delta
            std::memcpy(tmp.data(), p, w * 4);
            for (u64 x = 0; x < w; ++x)  // plane k holds byte of significance (3-k); LE output
                for (int k = 0; k < 4; ++k) p[4 * x + static_cast<u64>(k)] = tmp[(3 - static_cast<u64>(k)) * w + x];
        }
    }
}

}  // namespace detail

// Decode a classic LE grayscale TIFF from memory into f32.
inline Expected<TiffImage> decode_tiff(std::span<const u8> d) {
    using namespace detail;
    if (d.size() < 8 || d[0] != 'I' || d[1] != 'I') return err(Errc::decode_error, "tiff: not little-endian TIFF");
    const u16 magic = tiff_u16(d.data() + 2);
    if (magic != 42 && magic != 43) return err(Errc::decode_error, "tiff: bad magic");
    const bool big = magic == 43;  // BigTIFF: 8-byte offsets, 20-byte IFD entries
    u64 ifd;
    if (big) {
        if (d.size() < 16) return err(Errc::decode_error, "tiff: truncated BigTIFF header");
        if (tiff_u16(d.data() + 4) != 8 || tiff_u16(d.data() + 6) != 0)
            return err(Errc::decode_error, "tiff: bad BigTIFF offset size");
        ifd = tiff_u64(d.data() + 8);
    } else {
        ifd = tiff_u32(d.data() + 4);
    }
    const usize cnt_sz = big ? 8u : 2u, ent_sz = big ? 20u : 12u;
    if (ifd > d.size() || cnt_sz > d.size() - ifd) return err(Errc::decode_error, "tiff: IFD out of range");
    const u64 nent = big ? tiff_u64(d.data() + ifd) : tiff_u16(d.data() + ifd);
    if (nent > 4096) return err(Errc::decode_error, "tiff: IFD entry count implausible");
    if (ent_sz * nent > d.size() - ifd - cnt_sz) return err(Errc::decode_error, "tiff: IFD entries out of range");

    u64 width = 0, height = 0, bits = 0, comp = 1, spp = 1, sfmt = 1, rps = ~u64{0};
    u64 tile_w = 0, tile_h = 0, pred = 1;
    std::vector<u64> strip_off, strip_cnt, tile_off, tile_cnt;
    for (u64 i = 0; i < nent; ++i) {
        TiffTag t;
        const u8* e = d.data() + ifd + cnt_sz + ent_sz * i;
        t.tag = tiff_u16(e);
        t.type = tiff_u16(e + 2);
        t.big = big;
        if (big) {
            t.count = tiff_u64(e + 4);
            t.value = tiff_u64(e + 12);
            std::memcpy(t.inl, e + 12, 8);
        } else {
            t.count = tiff_u32(e + 4);
            t.value = tiff_u32(e + 8);
            std::memcpy(t.inl, e + 8, 4);
        }
        auto one = [&](u64& dst) -> Expected<void> {
            auto v = tiff_values(d, t);
            if (!v) return std::unexpected(v.error());
            if (v->empty()) return err(Errc::decode_error, "tiff: empty tag");
            dst = (*v)[0];
            return {};
        };
        auto many = [&](std::vector<u64>& dst) -> Expected<void> {
            auto v = tiff_values(d, t);
            if (!v) return std::unexpected(v.error());
            dst = std::move(*v);
            return {};
        };
        Expected<void> r{};
        switch (t.tag) {
            case 256: r = one(width); break;
            case 257: r = one(height); break;
            case 258: r = one(bits); break;
            case 259: r = one(comp); break;
            case 273: r = many(strip_off); break;
            case 277: r = one(spp); break;
            case 278: r = one(rps); break;
            case 279: r = many(strip_cnt); break;
            case 317: r = one(pred); break;
            case 322: r = one(tile_w); break;
            case 323: r = one(tile_h); break;
            case 324: r = many(tile_off); break;
            case 325: r = many(tile_cnt); break;
            case 339: r = one(sfmt); break;
            default: break;  // ignore everything else
        }
        if (!r) return std::unexpected(r.error());
    }

    if (comp != 1 && comp != 5)
        return err(Errc::unsupported, "tiff: unsupported compression=" + std::to_string(comp) + " (raw/LZW only)");
    if (pred != 1 && pred != 2 && pred != 3)
        return err(Errc::unsupported, "tiff: unsupported predictor=" + std::to_string(pred));
    if (spp != 1) return err(Errc::unsupported, "tiff: multi-sample TIFF not supported");
    if (width == 0 || height == 0 || width > (1u << 20) || height > (1u << 20))
        return err(Errc::decode_error, "tiff: bad dimensions");
    const usize bpp = bits / 8;
    const bool is_f32 = (sfmt == 3 && bits == 32);
    const bool is_u8 = (sfmt == 1 && bits == 8);
    const bool is_u16 = (sfmt == 1 && bits == 16);
    if (!is_f32 && !is_u8 && !is_u16) return err(Errc::unsupported, "tiff: only f32/u8/u16 grayscale supported");

    TiffImage img;
    img.width = static_cast<s64>(width);
    img.height = static_cast<s64>(height);
    const u64 npix = width * height;
    if (npix > (u64{1} << 34) / bpp) return err(Errc::decode_error, "tiff: image implausibly large");
    img.pix.resize(static_cast<usize>(npix));

    auto scatter = [&](const u8* src, u64 n, usize dst0) {
        for (u64 i = 0; i < n; ++i) {
            f32 v;
            if (is_f32)
                std::memcpy(&v, src + 4 * i, 4);
            else if (is_u16)
                v = static_cast<f32>(tiff_u16(src + 2 * i));
            else
                v = static_cast<f32>(src[i]);
            img.pix[dst0 + static_cast<usize>(i)] = v;
        }
    };

    // Materialize one strip/tile: bounds-check the source range, LZW-decompress if needed,
    // undo the predictor. Returns the block bytes (row-major, `w_smp` samples per row).
    std::vector<u8> block;
    auto load_block = [&](u64 off, u64 cnt, u64 rows, u64 w_smp) -> Expected<std::span<const u8>> {
        const u64 need = rows * w_smp * bpp;
        if (off > d.size() || cnt > d.size() - off) return err(Errc::decode_error, "tiff: block out of range");
        if (comp == 1) {
            if (cnt < need) return err(Errc::decode_error, "tiff: short block");
            if (pred == 1) return d.subspan(static_cast<usize>(off), static_cast<usize>(need));
            block.assign(d.data() + off, d.data() + off + need);
        } else {
            auto dec = tiff_lzw_decode(d.subspan(static_cast<usize>(off), static_cast<usize>(cnt)),
                                       static_cast<usize>(need));
            if (!dec) return std::unexpected(dec.error());
            if (dec->size() < need) return err(Errc::decode_error, "tiff: lzw block short");
            block = std::move(*dec);
        }
        tiff_undo_predictor(block, rows, w_smp, bpp, pred);
        return std::span<const u8>(block);
    };

    if (!tile_off.empty()) {  // tiled layout
        if (tile_w == 0 || tile_h == 0 || tile_off.size() != tile_cnt.size())
            return err(Errc::decode_error, "tiff: bad tile layout");
        if (tile_w * tile_h > (u64{1} << 26)) return err(Errc::decode_error, "tiff: tile implausibly large");
        const u64 tx = (width + tile_w - 1) / tile_w, ty = (height + tile_h - 1) / tile_h;
        if (tile_off.size() != tx * ty) return err(Errc::decode_error, "tiff: tile count mismatch");
        for (u64 t = 0; t < tile_off.size(); ++t) {
            auto blk = load_block(tile_off[t], tile_cnt[t], tile_h, tile_w);
            if (!blk) return std::unexpected(blk.error());
            const u64 ty0 = (t / tx) * tile_h, tx0 = (t % tx) * tile_w;
            for (u64 row = 0; row < tile_h && ty0 + row < height; ++row) {
                const u64 cols = std::min(tile_w, width - tx0);
                scatter(blk->data() + row * tile_w * bpp, cols, static_cast<usize>((ty0 + row) * width + tx0));
            }
        }
    } else {  // strips
        if (strip_off.empty() || strip_off.size() != strip_cnt.size())
            return err(Errc::decode_error, "tiff: bad strip layout");
        if (rps == ~u64{0}) rps = height;
        u64 row = 0;
        for (usize s = 0; s < strip_off.size(); ++s) {
            const u64 rows = std::min(rps, height - row);
            auto blk = load_block(strip_off[s], strip_cnt[s], rows, width);
            if (!blk) return std::unexpected(blk.error());
            scatter(blk->data(), rows * width, static_cast<usize>(row * width));
            row += rows;
        }
        if (row < height) return err(Errc::decode_error, "tiff: strips cover fewer rows than height");
    }
    return img;
}

inline Expected<TiffImage> read_tiff(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return err(Errc::not_found, "tiff: cannot open " + path);
    const auto sz = static_cast<usize>(f.tellg());
    f.seekg(0);
    std::vector<u8> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
    if (!f) return err(Errc::io_error, "tiff: short read " + path);
    return decode_tiff(buf);
}

}  // namespace fenix::io
