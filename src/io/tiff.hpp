// io/tiff.hpp — minimal first-party TIFF reader for the formats we actually ingest:
// classic little-endian TIFF, single-sample grayscale, UNCOMPRESSED strips or tiles,
// f32 / u8 / u16 samples (VC tifxyz coordinate images are single-strip f32). Everything
// else (big-endian, BigTIFF, compression, multi-sample) is rejected with a typed error —
// UNTRUSTED INPUT: every offset/count from the file is bounds-checked before use.
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
    u32 count = 0, value = 0;  // inline value or offset (type-dependent)
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

// Read a tag's value array (SHORT=3 or LONG=4), inline or via offset. Bounds-checked.
inline Expected<std::vector<u64>> tiff_values(std::span<const u8> d, const TiffTag& t) {
    const usize esz = t.type == 3 ? 2u : t.type == 4 ? 4u : 0u;
    if (esz == 0) return err(Errc::decode_error, "tiff: unsupported tag type");
    const usize total = esz * t.count;
    usize off;
    u8 inl[4];
    std::memcpy(inl, &t.value, 4);
    const u8* src;
    if (total <= 4) {
        src = inl;
        off = 0;
    } else {
        off = t.value;
        if (off > d.size() || total > d.size() - off) return err(Errc::decode_error, "tiff: tag array out of range");
        src = d.data() + off;
    }
    std::vector<u64> out(t.count);
    for (u32 i = 0; i < t.count; ++i) out[i] = esz == 2 ? tiff_u16(src + 2 * i) : tiff_u32(src + 4 * i);
    return out;
}

}  // namespace detail

// Decode a classic LE grayscale TIFF from memory into f32.
inline Expected<TiffImage> decode_tiff(std::span<const u8> d) {
    using namespace detail;
    if (d.size() < 8 || d[0] != 'I' || d[1] != 'I') return err(Errc::decode_error, "tiff: not little-endian TIFF");
    const u16 magic = tiff_u16(d.data() + 2);
    if (magic == 43) return err(Errc::unsupported, "tiff: BigTIFF not supported");
    if (magic != 42) return err(Errc::decode_error, "tiff: bad magic");
    const u32 ifd = tiff_u32(d.data() + 4);
    if (ifd + 2 > d.size()) return err(Errc::decode_error, "tiff: IFD out of range");
    const u16 nent = tiff_u16(d.data() + ifd);
    if (ifd + 2 + usize{12} * nent > d.size()) return err(Errc::decode_error, "tiff: IFD entries out of range");

    u64 width = 0, height = 0, bits = 0, comp = 1, spp = 1, sfmt = 1, rps = ~u64{0};
    u64 tile_w = 0, tile_h = 0;
    std::vector<u64> strip_off, strip_cnt, tile_off, tile_cnt;
    for (u16 i = 0; i < nent; ++i) {
        TiffTag t;
        const u8* e = d.data() + ifd + 2 + usize{12} * i;
        t.tag = tiff_u16(e);
        t.type = tiff_u16(e + 2);
        t.count = tiff_u32(e + 4);
        t.value = tiff_u32(e + 8);
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
            case 322: r = one(tile_w); break;
            case 323: r = one(tile_h); break;
            case 324: r = many(tile_off); break;
            case 325: r = many(tile_cnt); break;
            case 339: r = one(sfmt); break;
            default: break;  // ignore everything else
        }
        if (!r) return std::unexpected(r.error());
    }

    if (comp != 1)
        return err(Errc::unsupported, "tiff: compressed TIFF not supported (compression=" + std::to_string(comp) + ")");
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

    if (!tile_off.empty()) {  // tiled layout
        if (tile_w == 0 || tile_h == 0 || tile_off.size() != tile_cnt.size())
            return err(Errc::decode_error, "tiff: bad tile layout");
        const u64 tx = (width + tile_w - 1) / tile_w, ty = (height + tile_h - 1) / tile_h;
        if (tile_off.size() != tx * ty) return err(Errc::decode_error, "tiff: tile count mismatch");
        for (u64 t = 0; t < tile_off.size(); ++t) {
            const u64 need = tile_w * tile_h * bpp;
            if (tile_cnt[t] < need) return err(Errc::decode_error, "tiff: short tile");
            if (tile_off[t] > d.size() || need > d.size() - tile_off[t])
                return err(Errc::decode_error, "tiff: tile out of range");
            const u64 ty0 = (t / tx) * tile_h, tx0 = (t % tx) * tile_w;
            for (u64 row = 0; row < tile_h && ty0 + row < height; ++row) {
                const u64 cols = std::min(tile_w, width - tx0);
                scatter(
                    d.data() + tile_off[t] + row * tile_w * bpp, cols, static_cast<usize>((ty0 + row) * width + tx0));
            }
        }
    } else {  // strips
        if (strip_off.empty() || strip_off.size() != strip_cnt.size())
            return err(Errc::decode_error, "tiff: bad strip layout");
        if (rps == ~u64{0}) rps = height;
        u64 row = 0;
        for (usize s = 0; s < strip_off.size(); ++s) {
            const u64 rows = std::min(rps, height - row);
            const u64 need = rows * width * bpp;
            if (strip_cnt[s] < need) return err(Errc::decode_error, "tiff: short strip");
            if (strip_off[s] > d.size() || need > d.size() - strip_off[s])
                return err(Errc::decode_error, "tiff: strip out of range");
            scatter(d.data() + strip_off[s], rows * width, static_cast<usize>(row * width));
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
