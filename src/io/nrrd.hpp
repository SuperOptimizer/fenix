// io/nrrd.hpp — minimal NRRD reader/writer (raw encoding, little-endian) for ground-truth
// cube interop. ASCII header (key: value lines) + blank line + binary payload. NRRD lists
// sizes fastest-first (x,y,z); our Volume is ZYX (x-fastest), so payload maps 1:1.
// gzip encoding needs zlib -> deferred (we write/read raw). See io/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace fenix::io {

namespace detail {
// Parsed NRRD header; the stream is left positioned at the start of the binary payload.
struct NrrdHeader {
    std::string type = "float";
    s64 nx = 0, ny = 0, nz = 0;  // fastest-first (x,y,z)
};
inline Expected<NrrdHeader> parse_nrrd_header(std::ifstream& f) {
    std::string magic;
    std::getline(f, magic);
    if (magic.rfind("NRRD", 0) != 0) return err(Errc::decode_error, "not a NRRD file");
    std::string encoding = "raw";
    std::vector<s64> sizes;
    NrrdHeader h;
    for (std::string line; std::getline(f, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) break;  // end of header
        if (line[0] == '#') continue;
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string keyk = line.substr(0, colon);
        std::string val = line.substr(colon + 1);
        while (!val.empty() && val.front() == ' ') val.erase(val.begin());
        if (keyk == "type") h.type = val;
        else if (keyk == "encoding") encoding = val;
        else if (keyk == "sizes") {
            std::istringstream ss(val);
            for (s64 v; ss >> v;) sizes.push_back(v);
        }
    }
    if (encoding != "raw") return err(Errc::unsupported, "NRRD encoding '" + encoding + "' (raw only)");
    if (sizes.size() != 3) return err(Errc::unsupported, "NRRD must be 3D");
    h.nx = sizes[0]; h.ny = sizes[1]; h.nz = sizes[2];
    return h;
}
// Byte width of an NRRD scalar type (0 = unsupported).
inline s64 nrrd_type_bytes(const std::string& t) {
    if (t == "unsigned char" || t == "uint8" || t == "uchar" || t == "signed char" || t == "int8") return 1;
    if (t == "unsigned short" || t == "uint16" || t == "ushort" || t == "short" || t == "int16") return 2;
    if (t == "float" || t == "float32") return 4;
    return 0;
}
}  // namespace detail

// Read a 3D NRRD (raw, little-endian; u8/u16/s8/s16/f32) into an f32 Volume (ZYX).
inline Expected<Volume<f32>> read_nrrd(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return err(Errc::not_found, "cannot open " + path);
    auto hh = detail::parse_nrrd_header(f);
    if (!hh) return std::unexpected(hh.error());
    const std::string type = hh->type;
    const s64 nx = hh->nx, ny = hh->ny, nz = hh->nz;  // fastest-first
    Volume<f32> vol(Extent3{nz, ny, nx});
    const s64 n = nz * ny * nx;

    auto load = [&]<class T>() -> Expected<void> {
        // f32 input: read straight into the volume — no temp buffer (avoids doubling RAM on big
        // volumes; the f32->f32 copy was pure waste). Other dtypes need a typed staging buffer.
        if constexpr (std::is_same_v<T, f32>) {
            f.read(reinterpret_cast<char*>(vol.data()), static_cast<std::streamsize>(n * static_cast<s64>(sizeof(f32))));
            if (f.gcount() != n * static_cast<s64>(sizeof(f32))) return err(Errc::decode_error, "short read");
            return {};
        }
        std::vector<T> buf(static_cast<usize>(n));
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n * static_cast<s64>(sizeof(T))));
        if (f.gcount() != n * static_cast<s64>(sizeof(T))) return err(Errc::decode_error, "short read");
        for (s64 i = 0; i < n; ++i) vol.flat()[static_cast<usize>(i)] = static_cast<f32>(buf[static_cast<usize>(i)]);
        return {};
    };
    Expected<void> r;
    if (type == "unsigned char" || type == "uint8" || type == "uchar") r = load.operator()<u8>();
    else if (type == "unsigned short" || type == "uint16" || type == "ushort") r = load.operator()<u16>();
    else if (type == "signed char" || type == "int8") r = load.operator()<s8>();
    else if (type == "short" || type == "int16") r = load.operator()<s16>();
    else if (type == "float" || type == "float32") r = load.operator()<f32>();
    else return err(Errc::unsupported, "NRRD type '" + type + "'");
    if (!r) return std::unexpected(r.error());
    return vol;
}

namespace detail {
// Stream a raw NRRD payload through `sink(chunk_ptr, count, first_index)` in f32, CHUNK voxels at a
// time, converting from the on-disk dtype. Never materializes the whole source — peak extra RAM is
// one chunk. Used by the u8 reader / max scan so a big f32 NRRD doesn't cost 4x its u8 footprint.
template <class Sink>
inline Expected<void> stream_nrrd_f32(std::ifstream& f, const NrrdHeader& h, Sink&& sink) {
    const s64 n = h.nx * h.ny * h.nz;
    const s64 tb = nrrd_type_bytes(h.type);
    if (tb == 0) return err(Errc::unsupported, "NRRD type '" + h.type + "'");
    constexpr s64 kChunk = 1 << 20;  // 1 Mvox -> 4 MB f32 scratch
    std::vector<f32> out(static_cast<usize>(std::min(n, kChunk)));
    std::vector<char> raw(static_cast<usize>(std::min(n, kChunk) * tb));
    for (s64 base = 0; base < n; base += kChunk) {
        const s64 m = std::min(kChunk, n - base);
        f.read(raw.data(), static_cast<std::streamsize>(m * tb));
        if (f.gcount() != m * tb) return err(Errc::decode_error, "short read");
        const char* p = raw.data();
        if (h.type == "float" || h.type == "float32") {
            std::memcpy(out.data(), p, static_cast<usize>(m * 4));
        } else if (tb == 1) {
            const bool sgn = (h.type == "signed char" || h.type == "int8");
            for (s64 i = 0; i < m; ++i) out[static_cast<usize>(i)] = sgn ? static_cast<f32>(reinterpret_cast<const s8*>(p)[i]) : static_cast<f32>(reinterpret_cast<const u8*>(p)[i]);
        } else {  // 2-byte
            const bool sgn = (h.type == "short" || h.type == "int16");
            for (s64 i = 0; i < m; ++i) out[static_cast<usize>(i)] = sgn ? static_cast<f32>(reinterpret_cast<const s16*>(p)[i]) : static_cast<f32>(reinterpret_cast<const u16*>(p)[i]);
        }
        sink(out.data(), m, base);
    }
    return {};
}
}  // namespace detail

// Streaming raw->u8 read: u8[i] = clamp(round(raw[i]*scale), 0, 255). Never holds the full source as
// f32, so a 1024^3 f32 NRRD costs ~1 GB (the u8 volume) instead of ~5 GB (f32 + u8). For the tracer's
// resident prediction/CT volumes (which only need u8 precision) this is the load path of choice.
inline Expected<Volume<u8>> read_nrrd_u8(const std::string& path, f32 scale) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return err(Errc::not_found, "cannot open " + path);
    auto hh = detail::parse_nrrd_header(f);
    if (!hh) return std::unexpected(hh.error());
    Volume<u8> vol(Extent3{hh->nz, hh->ny, hh->nx});
    u8* d = vol.data();
    auto r = detail::stream_nrrd_f32(f, *hh, [&](const f32* c, s64 m, s64 base) {
        for (s64 i = 0; i < m; ++i) {
            const f32 v = c[static_cast<usize>(i)] * scale;
            d[static_cast<usize>(base + i)] = static_cast<u8>(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v) + 0.5f);
        }
    });
    if (!r) return std::unexpected(r.error());
    return vol;
}

// Streaming max of a numeric NRRD without materializing it (one chunk resident at a time).
inline Expected<f32> nrrd_max(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return err(Errc::not_found, "cannot open " + path);
    auto hh = detail::parse_nrrd_header(f);
    if (!hh) return std::unexpected(hh.error());
    f32 mx = -1e30f;
    auto r = detail::stream_nrrd_f32(f, *hh, [&](const f32* c, s64 m, s64) {
        for (s64 i = 0; i < m; ++i) mx = std::max(mx, c[static_cast<usize>(i)]);
    });
    if (!r) return std::unexpected(r.error());
    return mx;
}

// Write an f32 Volume as a raw little-endian NRRD.
inline Expected<void> write_nrrd(const std::string& path, VolumeView<const f32> vol) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return err(Errc::io_error, "cannot create " + path);
    const Extent3 d = vol.dims();
    f << "NRRD0004\n";
    f << "type: float\n";
    f << "dimension: 3\n";
    f << "sizes: " << d.x << " " << d.y << " " << d.z << "\n";  // fastest-first
    f << "encoding: raw\n";
    f << "endian: little\n";
    f << "\n";
    if (vol.is_contiguous()) {
        f.write(reinterpret_cast<const char*>(vol.data()),
                static_cast<std::streamsize>(d.count() * static_cast<s64>(sizeof(f32))));
    } else {
        for (s64 z = 0; z < d.z; ++z)
            for (s64 y = 0; y < d.y; ++y)
                for (s64 x = 0; x < d.x; ++x) {
                    f32 v = vol(z, y, x);
                    f.write(reinterpret_cast<const char*>(&v), sizeof(f32));
                }
    }
    return f ? Expected<void>{} : err(Errc::io_error, "write failed");
}

}  // namespace fenix::io
