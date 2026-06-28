// io/zarr.hpp — OME-Zarr v2 reader (RAW encoding). The canonical scroll storage format.
// Parses .zarray (minimal hand-rolled JSON, like fysics), reads chunk files at
// <z><sep><y><sep><x>, treats a MISSING chunk as fill_value (zarr omits empty chunks).
// Blosc/zstd/lz4 decompression (needs blosc2) + v3/sharded drop in later; this is the
// dependency-free raw path. ZYX, C-order. See io/CLAUDE.md, docs/research/villa-data.md.
#pragma once

#include "core/core.hpp"
#include "core/parallel.hpp"
#include "io/s3.hpp"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fenix::io {

// Fetch one object under a zarr root (a local dir or an http(s):///s3:// URL). Returns the
// bytes, or std::nullopt when the object is absent (404 / missing file = air), or a hard Error
// on a real fetch failure. One code path for local and remote sources.
inline Expected<std::optional<std::vector<u8>>> fetch_object(const std::string& root,
                                                             const std::string& sub) {
    if (is_remote(root)) return http_get(root + "/" + sub);
    std::ifstream f(root + "/" + sub, std::ios::binary);
    if (!f) return std::optional<std::vector<u8>>(std::nullopt);  // missing chunk = air
    std::vector<u8> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return std::optional<std::vector<u8>>(std::move(b));
}

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
    auto bytes = fetch_object(root, ".zarray");
    if (!bytes) return std::unexpected(bytes.error());
    if (!*bytes) return err(Errc::not_found, "no .zarray in " + root);
    const std::string js(reinterpret_cast<const char*>((*bytes)->data()), (*bytes)->size());
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

    // Enumerate the chunks covering the region; fetch + scatter them in parallel (each writes a
    // disjoint output region, so no locking on the volume). Remote fetches reuse a per-thread
    // libcurl connection (s3.hpp), so a slab of hundreds of chunks streams concurrently.
    struct ChunkId { s64 cz, cy, cx; };
    std::vector<ChunkId> chunks;
    for (s64 cz = cz0; cz <= cz1; ++cz)
        for (s64 cy = cy0; cy <= cy1; ++cy)
            for (s64 cx = cx0; cx <= cx1; ++cx) chunks.push_back({cz, cy, cx});

    std::atomic<bool> failed{false};
    std::string fail_msg;
    std::atomic<s64> done{0};

    parallel_for(0, static_cast<s64>(chunks.size()), [&](s64 i) {
        if (failed.load(std::memory_order_relaxed)) return;
        const ChunkId c = chunks[static_cast<usize>(i)];
        std::ostringstream sub;
        sub << c.cz << m.sep << c.cy << m.sep << c.cx;
        auto got = fetch_object(root, sub.str());
        if (!got) {  // hard fetch failure — record, do NOT treat as air
            bool expect = false;
            if (failed.compare_exchange_strong(expect, true)) fail_msg = got.error().message;
            return;
        }
        const std::optional<std::vector<u8>>& blob = *got;
        const bool present = blob && blob->size() >= static_cast<usize>(ccount) * esz;
        const u8* data = present ? blob->data() : nullptr;
        for (s64 lz = 0; lz < m.chunks.z; ++lz) {
            const s64 gz = c.cz * m.chunks.z + lz;
            if (gz < origin.z || gz >= origin.z + extent.z || gz >= m.shape.z) continue;
            for (s64 ly = 0; ly < m.chunks.y; ++ly) {
                const s64 gy = c.cy * m.chunks.y + ly;
                if (gy < origin.y || gy >= origin.y + extent.y || gy >= m.shape.y) continue;
                for (s64 lx = 0; lx < m.chunks.x; ++lx) {
                    const s64 gx = c.cx * m.chunks.x + lx;
                    if (gx < origin.x || gx >= origin.x + extent.x || gx >= m.shape.x) continue;
                    f32 v = m.fill;
                    if (present) {
                        const s64 off = (lz * m.chunks.y + ly) * m.chunks.x + lx;
                        v = detail::cast_dtype(data + static_cast<usize>(off) * esz, m.dtype);
                    }
                    ov(gz - origin.z, gy - origin.y, gx - origin.x) = v;
                }
            }
        }
        const s64 d = done.fetch_add(1) + 1;
        if (d % 64 == 0 || d == static_cast<s64>(chunks.size()))
            log(LogLevel::info, "zarr: fetched {}/{} chunks", d, chunks.size());
    });

    if (failed.load()) return err(Errc::fetch_failed, "zarr region fetch failed: " + fail_msg);
    return out;
}

}  // namespace fenix::io
