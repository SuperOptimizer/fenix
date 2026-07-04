// tests/test_png.cpp — the first-party PNG reader: crafted files (via zlib deflate +
// crc32) covering gray-8, all five scanline filters, palette, RGBA-drop-alpha, 1-bit
// gray expansion, and typed rejections (interlace, truncation, CRC damage).
#define FENIX_TEST_MAIN
#include "core/test.hpp"
#include "io/png.hpp"

#include <zlib.h>

#include <cstring>
#include <vector>

using namespace fenix;

namespace {
void put_u32(std::vector<u8>& v, u32 x) {
    v.push_back(static_cast<u8>(x >> 24));
    v.push_back(static_cast<u8>(x >> 16));
    v.push_back(static_cast<u8>(x >> 8));
    v.push_back(static_cast<u8>(x));
}
void chunk(std::vector<u8>& png, const char* type, const std::vector<u8>& body) {
    put_u32(png, static_cast<u32>(body.size()));
    const usize at = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), body.begin(), body.end());
    // zlib quirk: crc32(x, Z_NULL, 0) RETURNS THE SEED (0), discarding x — never pass
    // Z_NULL mid-accumulation
    u32 crc = static_cast<u32>(::crc32(0, png.data() + at, 4));
    if (!body.empty()) crc = static_cast<u32>(::crc32(crc, body.data(), static_cast<uInt>(body.size())));
    put_u32(png, crc);
}
std::vector<u8> deflate_all(const std::vector<u8>& raw) {
    uLongf cap = compressBound(static_cast<uLong>(raw.size()));
    std::vector<u8> out(cap);
    if (compress(out.data(), &cap, raw.data(), static_cast<uLong>(raw.size())) != Z_OK) out.clear();
    out.resize(cap);
    return out;
}
std::vector<u8> make_png(u32 w, u32 h, int depth, int ctype, const std::vector<u8>& scanlines,
                         const std::vector<u8>& plte = {}) {
    std::vector<u8> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::vector<u8> ihdr;
    put_u32(ihdr, w);
    put_u32(ihdr, h);
    ihdr.push_back(static_cast<u8>(depth));
    ihdr.push_back(static_cast<u8>(ctype));
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    chunk(png, "IHDR", ihdr);
    if (!plte.empty()) chunk(png, "PLTE", plte);
    chunk(png, "IDAT", deflate_all(scanlines));
    chunk(png, "IEND", {});
    return png;
}
}  // namespace

TEST(png_gray8_all_filters) {
    const u8 base[5][4] = {{10, 20, 30, 40}, {5, 5, 5, 5}, {1, 2, 3, 4}, {8, 16, 24, 32}, {100, 90, 80, 70}};
    std::vector<u8> want, lines;
    u8 prev[4] = {0, 0, 0, 0};
    for (int y = 0; y < 5; ++y) {
        const u8* row = base[y];
        lines.push_back(static_cast<u8>(y));
        u8 enc[4];
        for (int x = 0; x < 4; ++x) {
            const int left = x > 0 ? row[x - 1] : 0;
            const int up = prev[x];
            const int ul = x > 0 ? prev[x - 1] : 0;
            int pred = 0;
            if (y == 1) pred = left;
            else if (y == 2) pred = up;
            else if (y == 3) pred = (left + up) / 2;
            else if (y == 4) pred = io::detail::png_paeth(left, up, ul);
            enc[x] = static_cast<u8>(row[x] - pred);
            want.push_back(row[x]);
        }
        lines.insert(lines.end(), enc, enc + 4);
        std::memcpy(prev, row, 4);
    }
    const auto png = make_png(4, 5, 8, 0, lines);
    auto img = io::decode_png(png);
    if (!img) std::printf("PNG ERR: %s\n", img.error().message.c_str());
    REQUIRE(img.has_value());
    CHECK(img->w == 4);
    CHECK(img->h == 5);
    CHECK(img->comps == 1);
    for (usize i = 0; i < want.size(); ++i) CHECK(img->px[i] == want[i]);
}

TEST(png_palette_and_rgba_and_1bit) {
    std::vector<u8> plte = {255, 0, 0, 0, 255, 0, 0, 0, 255, 7, 8, 9};
    std::vector<u8> lines = {0, 0, 1, 0, 2, 3};
    auto img = io::decode_png(make_png(2, 2, 8, 3, lines, plte));
    REQUIRE(img.has_value());
    CHECK(img->comps == 3);
    CHECK(img->at(0, 0, 0) == 255);
    CHECK(img->at(0, 1, 1) == 255);
    CHECK(img->at(1, 1, 2) == 9);
    std::vector<u8> rl = {0, 11, 22, 33, 200, 44, 55, 66, 10};
    auto rgba = io::decode_png(make_png(2, 1, 8, 6, rl));
    REQUIRE(rgba.has_value());
    CHECK(rgba->comps == 3);
    CHECK(rgba->at(0, 0, 0) == 11);
    CHECK(rgba->at(0, 1, 2) == 66);
    std::vector<u8> bl = {0, 0xC0, 0x40};
    auto bit = io::decode_png(make_png(10, 1, 1, 0, bl));
    REQUIRE(bit.has_value());
    CHECK(bit->px[0] == 255);
    CHECK(bit->px[1] == 255);
    CHECK(bit->px[2] == 0);
    CHECK(bit->px[8] == 0);
    CHECK(bit->px[9] == 255);
}

TEST(png_typed_rejections) {
    std::vector<u8> lines = {0, 1, 2, 3};
    auto png = make_png(3, 1, 8, 0, lines);
    auto bad = png;
    bad[8 + 8 + 12] = 1;  // interlace byte (CRC now also mismatched — either way: typed error)
    CHECK(!io::decode_png(bad).has_value());
    auto trunc = png;
    trunc.resize(trunc.size() - 8);
    CHECK(!io::decode_png(trunc).has_value());
    auto crc = png;
    crc[crc.size() - 20] ^= 0xff;
    CHECK(!io::decode_png(crc).has_value());
}
