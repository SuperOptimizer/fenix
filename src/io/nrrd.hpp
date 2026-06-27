// io/nrrd.hpp — minimal NRRD reader/writer (raw encoding, little-endian) for ground-truth
// cube interop. ASCII header (key: value lines) + blank line + binary payload. NRRD lists
// sizes fastest-first (x,y,z); our Volume is ZYX (x-fastest), so payload maps 1:1.
// gzip encoding needs zlib -> deferred (we write/read raw). See io/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fenix::io {

// Read a 3D NRRD (raw, little-endian; u8/u16/s8/s16/f32) into an f32 Volume (ZYX).
inline Expected<Volume<f32>> read_nrrd(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return err(Errc::not_found, "cannot open " + path);
    std::string magic;
    std::getline(f, magic);
    if (magic.rfind("NRRD", 0) != 0) return err(Errc::decode_error, "not a NRRD file");

    std::string type = "float", encoding = "raw";
    std::vector<s64> sizes;
    for (std::string line; std::getline(f, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) break;  // end of header
        if (line[0] == '#') continue;
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string keyk = line.substr(0, colon);
        std::string val = line.substr(colon + 1);
        while (!val.empty() && val.front() == ' ') val.erase(val.begin());
        if (keyk == "type") type = val;
        else if (keyk == "encoding") encoding = val;
        else if (keyk == "sizes") {
            std::istringstream ss(val);
            for (s64 v; ss >> v;) sizes.push_back(v);
        }
    }
    if (encoding != "raw") return err(Errc::unsupported, "NRRD encoding '" + encoding + "' (raw only)");
    if (sizes.size() != 3) return err(Errc::unsupported, "NRRD must be 3D");

    const s64 nx = sizes[0], ny = sizes[1], nz = sizes[2];  // fastest-first
    Volume<f32> vol(Extent3{nz, ny, nx});
    const s64 n = nz * ny * nx;

    auto load = [&]<class T>() -> Expected<void> {
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
