// io/png.hpp — first-party minimal PNG READER (no libpng; zlib inflate is the only dep).
// Scope: what the corpus actually ships — segment masks (<id>_mask.png) and ink labels
// (inklabel.png): 8-bit gray / RGB / palette / gray+alpha / RGBA, plus 1/2/4-bit gray and
// 16-bit gray (downshifted). Non-interlaced only; Adam7 and 16-bit color are TYPED
// rejections, never garbage. Alpha is dropped (composited on black), palette expanded.
// Untrusted-input discipline: every length/offset bounds-checked; CRCs verified.
#pragma once

#include "core/core.hpp"
#include "io/jpeg.hpp"  // io::Image

#include <zlib.h>

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace fenix::io {

namespace detail {

inline u32 png_u32(const u8* p) {
    return (static_cast<u32>(p[0]) << 24) | (static_cast<u32>(p[1]) << 16) | (static_cast<u32>(p[2]) << 8) |
           static_cast<u32>(p[3]);
}

inline Expected<std::vector<u8>> png_inflate(std::span<const u8> in, usize max_out) {
    std::vector<u8> out;
    out.resize(max_out);
    z_stream zs{};
    if (inflateInit(&zs) != Z_OK) return err(Errc::internal, "png: inflateInit failed");
    zs.next_in = const_cast<Bytef*>(in.data());
    zs.avail_in = static_cast<uInt>(in.size());
    zs.next_out = out.data();
    zs.avail_out = static_cast<uInt>(out.size());
    const int rc = inflate(&zs, Z_FINISH);
    const usize produced = out.size() - zs.avail_out;
    inflateEnd(&zs);
    if (rc != Z_STREAM_END || produced != max_out)
        return err(Errc::decode_error, "png: bad or truncated IDAT stream");
    return out;
}

inline int png_paeth(int a, int b, int c) {
    const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    return pb <= pc ? b : c;
}

}  // namespace detail

inline Expected<Image> decode_png(std::span<const u8> d) {
    static constexpr u8 kSig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    if (d.size() < 8 + 25 || std::memcmp(d.data(), kSig, 8) != 0) return err(Errc::decode_error, "png: bad signature");

    usize pos = 8;
    u32 w = 0, h = 0;
    int bit_depth = 0, color_type = -1;
    std::vector<u8> idat;
    std::vector<std::array<u8, 3>> palette;
    bool done = false;
    while (!done) {
        if (pos + 12 > d.size()) return err(Errc::decode_error, "png: truncated chunk header");
        const u32 len = detail::png_u32(d.data() + pos);
        if (pos + 12 + len > d.size() || len > (u32{1} << 30)) return err(Errc::decode_error, "png: bad chunk length");
        const char* type = reinterpret_cast<const char*>(d.data() + pos + 4);
        const u8* body = d.data() + pos + 8;
        const u32 crc_want = detail::png_u32(d.data() + pos + 8 + len);
        const u32 crc_have = static_cast<u32>(
            ::crc32(::crc32(0, d.data() + pos + 4, 4), body, static_cast<uInt>(len)));
        if (crc_want != crc_have) return err(Errc::decode_error, std::string("png: chunk CRC mismatch in ") + std::string(type, 4));
        if (std::memcmp(type, "IHDR", 4) == 0) {
            if (len != 13) return err(Errc::decode_error, "png: bad IHDR");
            w = detail::png_u32(body);
            h = detail::png_u32(body + 4);
            bit_depth = body[8];
            color_type = body[9];
            if (body[10] != 0 || body[11] != 0) return err(Errc::decode_error, "png: nonstandard compression/filter");
            if (body[12] != 0) return err(Errc::decode_error, "png: interlaced (Adam7) unsupported");
            if (w == 0 || h == 0 || w > (1u << 20) || h > (1u << 20)) return err(Errc::decode_error, "png: bad dims");
        } else if (std::memcmp(type, "PLTE", 4) == 0) {
            if (len % 3 != 0 || len > 768) return err(Errc::decode_error, "png: bad PLTE");
            palette.resize(len / 3);
            for (usize i = 0; i < palette.size(); ++i) palette[i] = {body[3 * i], body[3 * i + 1], body[3 * i + 2]};
        } else if (std::memcmp(type, "IDAT", 4) == 0) {
            idat.insert(idat.end(), body, body + len);
        } else if (std::memcmp(type, "IEND", 4) == 0) {
            done = true;
        }
        pos += 12 + len;
    }
    if (color_type < 0 || idat.empty()) return err(Errc::decode_error, "png: missing IHDR/IDAT");

    // channels + supported (bit_depth, color_type) matrix
    int channels = 0;
    switch (color_type) {
        case 0: channels = 1; break;  // gray: 1/2/4/8/16
        case 2: channels = 3; break;  // rgb: 8 only (16 rejected)
        case 3: channels = 1; break;  // palette: 1/2/4/8
        case 4: channels = 2; break;  // gray+alpha: 8 only
        case 6: channels = 4; break;  // rgba: 8 only
        default: return err(Errc::decode_error, "png: unknown color type");
    }
    if ((color_type == 2 || color_type == 4 || color_type == 6) && bit_depth != 8)
        return err(Errc::decode_error, "png: only 8-bit color/alpha supported");
    if (color_type == 0 && bit_depth != 1 && bit_depth != 2 && bit_depth != 4 && bit_depth != 8 && bit_depth != 16)
        return err(Errc::decode_error, "png: bad gray bit depth");
    if (color_type == 3 && (bit_depth > 8 || palette.empty()))
        return err(Errc::decode_error, "png: bad palette depth or missing PLTE");

    const usize bits_pp = static_cast<usize>(channels) * static_cast<usize>(bit_depth);
    const usize row_bytes = (static_cast<usize>(w) * bits_pp + 7) / 8;
    const usize bpp = std::max<usize>(1, bits_pp / 8);  // filter step
    auto raw = detail::png_inflate(idat, (row_bytes + 1) * h);
    if (!raw) return std::unexpected(raw.error());

    // unfilter in place (scanline filters 0..4)
    std::vector<u8> prev(row_bytes, 0);
    std::vector<u8> cur(row_bytes);
    Image img;
    img.w = static_cast<int>(w);
    img.h = static_cast<int>(h);
    img.comps = (color_type == 2 || color_type == 6 || (color_type == 3 && true)) ? 3 : 1;
    if (color_type == 0 || color_type == 4) img.comps = 1;
    img.px.assign(static_cast<usize>(img.w) * static_cast<usize>(img.h) * static_cast<usize>(img.comps), 0);

    for (u32 y = 0; y < h; ++y) {
        const u8* src = raw->data() + static_cast<usize>(y) * (row_bytes + 1);
        const u8 filter = src[0];
        std::memcpy(cur.data(), src + 1, row_bytes);
        switch (filter) {
            case 0: break;
            case 1:
                for (usize i = bpp; i < row_bytes; ++i) cur[i] = static_cast<u8>(cur[i] + cur[i - bpp]);
                break;
            case 2:
                for (usize i = 0; i < row_bytes; ++i) cur[i] = static_cast<u8>(cur[i] + prev[i]);
                break;
            case 3:
                for (usize i = 0; i < row_bytes; ++i) {
                    const int left = i >= bpp ? cur[i - bpp] : 0;
                    cur[i] = static_cast<u8>(cur[i] + (left + prev[i]) / 2);
                }
                break;
            case 4:
                for (usize i = 0; i < row_bytes; ++i) {
                    const int left = i >= bpp ? cur[i - bpp] : 0;
                    const int ul = i >= bpp ? prev[i - bpp] : 0;
                    cur[i] = static_cast<u8>(cur[i] + detail::png_paeth(left, prev[i], ul));
                }
                break;
            default: return err(Errc::decode_error, "png: bad filter type");
        }
        // expand this scanline into img
        auto put = [&](u32 x, u8 r, u8 g, u8 b) {
            const usize o = (static_cast<usize>(y) * w + x) * static_cast<usize>(img.comps);
            if (img.comps == 1) img.px[o] = r;
            else {
                img.px[o] = r;
                img.px[o + 1] = g;
                img.px[o + 2] = b;
            }
        };
        if (color_type == 0 && bit_depth == 8) {
            for (u32 x = 0; x < w; ++x) put(x, cur[x], 0, 0);
        } else if (color_type == 0 && bit_depth == 16) {
            for (u32 x = 0; x < w; ++x) put(x, cur[2 * x], 0, 0);  // high byte
        } else if (color_type == 0) {  // 1/2/4-bit gray, scaled to 8-bit
            const int per = 8 / bit_depth, maxv = (1 << bit_depth) - 1;
            for (u32 x = 0; x < w; ++x) {
                const u8 byte = cur[x / static_cast<u32>(per)];
                const int shift = (per - 1 - static_cast<int>(x % static_cast<u32>(per))) * bit_depth;
                const int v = (byte >> shift) & maxv;
                put(x, static_cast<u8>(v * 255 / maxv), 0, 0);
            }
        } else if (color_type == 2) {
            for (u32 x = 0; x < w; ++x) put(x, cur[3 * x], cur[3 * x + 1], cur[3 * x + 2]);
        } else if (color_type == 3) {
            const int per = 8 / bit_depth, maxv = (1 << bit_depth) - 1;
            for (u32 x = 0; x < w; ++x) {
                int idx;
                if (bit_depth == 8) idx = cur[x];
                else {
                    const u8 byte = cur[x / static_cast<u32>(per)];
                    const int shift = (per - 1 - static_cast<int>(x % static_cast<u32>(per))) * bit_depth;
                    idx = (byte >> shift) & maxv;
                }
                if (static_cast<usize>(idx) >= palette.size()) return err(Errc::decode_error, "png: palette index oob");
                put(x, palette[static_cast<usize>(idx)][0], palette[static_cast<usize>(idx)][1],
                    palette[static_cast<usize>(idx)][2]);
            }
        } else if (color_type == 4) {
            for (u32 x = 0; x < w; ++x) put(x, cur[2 * x], 0, 0);  // drop alpha
        } else {  // 6: rgba -> rgb (drop alpha)
            for (u32 x = 0; x < w; ++x) put(x, cur[4 * x], cur[4 * x + 1], cur[4 * x + 2]);
        }
        std::swap(prev, cur);
    }
    return img;
}

inline Expected<Image> read_png(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return err(Errc::not_found, "png: cannot open " + path);
    std::vector<u8> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return decode_png(d);
}

}  // namespace fenix::io
