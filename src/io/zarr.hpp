// io/zarr.hpp — OME-Zarr v2 reader (RAW encoding). The canonical scroll storage format.
// Parses .zarray (minimal hand-rolled JSON, like fysics), reads chunk files at
// <z><sep><y><sep><x>, treats a MISSING chunk as fill_value (zarr omits empty chunks).
// Blosc/zstd/lz4 decompression (needs blosc2) + v3/sharded drop in later; this is the
// dependency-free raw path. ZYX, C-order. See io/CLAUDE.md, docs/research/villa-data.md.
#pragma once

#include "core/core.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fenix::io {

struct ZarrMeta {
    Extent3 shape{};   // z,y,x
    Extent3 chunks{};  // chunk dims
    std::string dtype = "|u1";
    f32 fill = 0.0f;
    char sep = '.';  // dimension_separator
};

namespace detail {
// Extract the N integers of a JSON array following "key": [ ... ].
inline std::vector<s64> json_int_array(const std::string& s, const std::string& key) {
    std::vector<s64> out;
    auto k = s.find("\"" + key + "\"");
    if (k == std::string::npos) return out;
    auto lb = s.find('[', k);
    auto rb = s.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return out;
    std::string inner = s.substr(lb + 1, rb - lb - 1);
    for (char& ch : inner)
        if (ch == ',') ch = ' ';
    std::istringstream ss(inner);
    for (s64 v; ss >> v;) out.push_back(v);
    return out;
}
inline std::string json_string(const std::string& s, const std::string& key) {
    auto k = s.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    auto colon = s.find(':', k);
    auto q1 = s.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    auto q2 = s.find('"', q1 + 1);
    return s.substr(q1 + 1, q2 - q1 - 1);
}
inline usize dtype_size(const std::string& dt) {
    if (dt.size() < 3) return 1;
    return static_cast<usize>(dt[2] - '0');  // |u1 ->1, <u2->2, <f4->4
}
inline f32 cast_dtype(const u8* p, const std::string& dt) {
    const char kind = dt.size() > 1 ? dt[1] : 'u';
    const usize sz = dtype_size(dt);
    if (kind == 'f' && sz == 4) {
        f32 v;
        std::memcpy(&v, p, 4);
        return v;
    }
    if (kind == 'u') {
        if (sz == 1) return static_cast<f32>(*p);
        if (sz == 2) { u16 v; std::memcpy(&v, p, 2); return static_cast<f32>(v); }
        if (sz == 4) { u32 v; std::memcpy(&v, p, 4); return static_cast<f32>(v); }
    }
    if (kind == 'i') {
        if (sz == 1) return static_cast<f32>(static_cast<s8>(*p));
        if (sz == 2) { s16 v; std::memcpy(&v, p, 2); return static_cast<f32>(v); }
        if (sz == 4) { s32 v; std::memcpy(&v, p, 4); return static_cast<f32>(v); }
    }
    return static_cast<f32>(*p);
}
}  // namespace detail

inline Expected<ZarrMeta> read_zarray(const std::string& root) {
    std::ifstream f(root + "/.zarray", std::ios::binary);
    if (!f) return err(Errc::not_found, "no .zarray in " + root);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string js = ss.str();
    ZarrMeta m;
    auto sh = detail::json_int_array(js, "shape");
    auto ch = detail::json_int_array(js, "chunks");
    if (sh.size() != 3 || ch.size() != 3) return err(Errc::unsupported, "zarr must be 3D");
    m.shape = {sh[0], sh[1], sh[2]};
    m.chunks = {ch[0], ch[1], ch[2]};
    m.dtype = detail::json_string(js, "dtype");
    const std::string compu = detail::json_string(js, "compressor");  // "" if null
    if (js.find("\"compressor\": null") == std::string::npos &&
        js.find("\"compressor\":null") == std::string::npos && !compu.empty())
        return err(Errc::unsupported, "zarr compressor not supported yet (raw only)");
    const std::string seps = detail::json_string(js, "dimension_separator");
    if (!seps.empty()) m.sep = seps[0];
    return m;
}

// Read an axis-aligned region [origin, origin+extent) into an f32 Volume. Missing chunks
// (omitted by zarr) read as fill_value. Raw encoding only.
inline Expected<Volume<f32>> read_zarr_region(const std::string& root, Index3 origin, Extent3 extent) {
    auto mm = read_zarray(root);
    if (!mm) return std::unexpected(mm.error());
    const ZarrMeta m = *mm;
    const usize esz = detail::dtype_size(m.dtype);
    const s64 ccount = m.chunks.count();
    Volume<f32> out = Volume<f32>::zeros(extent);
    VolumeView<f32> ov = out.view();

    const s64 cz0 = origin.z / m.chunks.z, cz1 = (origin.z + extent.z - 1) / m.chunks.z;
    const s64 cy0 = origin.y / m.chunks.y, cy1 = (origin.y + extent.y - 1) / m.chunks.y;
    const s64 cx0 = origin.x / m.chunks.x, cx1 = (origin.x + extent.x - 1) / m.chunks.x;
    std::vector<u8> buf;

    for (s64 cz = cz0; cz <= cz1; ++cz)
        for (s64 cy = cy0; cy <= cy1; ++cy)
            for (s64 cx = cx0; cx <= cx1; ++cx) {
                std::ostringstream path;
                path << root << "/" << cz << m.sep << cy << m.sep << cx;
                bool present = false;
                {
                    std::ifstream cf(path.str(), std::ios::binary);
                    if (cf) {
                        buf.assign(static_cast<usize>(ccount * static_cast<s64>(esz)), 0);
                        cf.read(reinterpret_cast<char*>(buf.data()),
                                static_cast<std::streamsize>(buf.size()));
                        present = cf.gcount() == static_cast<std::streamoff>(buf.size());
                    }
                }
                // Copy the overlap of this chunk into the output.
                for (s64 lz = 0; lz < m.chunks.z; ++lz) {
                    const s64 gz = cz * m.chunks.z + lz;
                    if (gz < origin.z || gz >= origin.z + extent.z || gz >= m.shape.z) continue;
                    for (s64 ly = 0; ly < m.chunks.y; ++ly) {
                        const s64 gy = cy * m.chunks.y + ly;
                        if (gy < origin.y || gy >= origin.y + extent.y || gy >= m.shape.y) continue;
                        for (s64 lx = 0; lx < m.chunks.x; ++lx) {
                            const s64 gx = cx * m.chunks.x + lx;
                            if (gx < origin.x || gx >= origin.x + extent.x || gx >= m.shape.x) continue;
                            f32 v = m.fill;
                            if (present) {
                                const s64 off = (lz * m.chunks.y + ly) * m.chunks.x + lx;
                                v = detail::cast_dtype(buf.data() + static_cast<usize>(off) * esz, m.dtype);
                            }
                            ov(gz - origin.z, gy - origin.y, gx - origin.x) = v;
                        }
                    }
                }
            }
    return out;
}

}  // namespace fenix::io
