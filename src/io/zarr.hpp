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
template <class T>
inline T cast_dtype(const u8* p, const std::string& dt) {
    const char kind = dt.size() > 1 ? dt[1] : 'u';
    const usize sz = dtype_size(dt);
    if (kind == 'f' && sz == 4) { f32 v; std::memcpy(&v, p, 4); return static_cast<T>(v); }
    if (kind == 'u') {
        if (sz == 1) return static_cast<T>(*p);
        if (sz == 2) { u16 v; std::memcpy(&v, p, 2); return static_cast<T>(v); }
        if (sz == 4) { u32 v; std::memcpy(&v, p, 4); return static_cast<T>(v); }
    }
    if (kind == 'i') {
        if (sz == 1) return static_cast<T>(static_cast<s8>(*p));
        if (sz == 2) { s16 v; std::memcpy(&v, p, 2); return static_cast<T>(v); }
        if (sz == 4) { s32 v; std::memcpy(&v, p, 4); return static_cast<T>(v); }
    }
    return static_cast<T>(*p);
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

// Read an axis-aligned region [origin, origin+extent) into a Volume<T> in the SOURCE dtype. T defaults to
// f32 for back-compat, but a u8 source should be read with T=u8 so it is NOT widened 4x in RAM — the DCT
// codec widens to f32 itself, per 16³ block. Missing chunks (omitted by zarr) read as fill_value. Raw only.
template <class T = f32>
inline Expected<Volume<T>> read_zarr_region(const std::string& root, Index3 origin, Extent3 extent) {
    auto mm = read_zarray(root);
    if (!mm) return std::unexpected(mm.error());
    const ZarrMeta m = *mm;
    const usize esz = detail::dtype_size(m.dtype);
    const s64 ccount = m.chunks.count();
    Volume<T> out = Volume<T>::zeros(extent);
    VolumeView<T> ov = out.view();
    const T fillv = static_cast<T>(m.fill);

    const s64 cz0 = origin.z / m.chunks.z, cz1 = (origin.z + extent.z - 1) / m.chunks.z;
    const s64 cy0 = origin.y / m.chunks.y, cy1 = (origin.y + extent.y - 1) / m.chunks.y;
    const s64 cx0 = origin.x / m.chunks.x, cx1 = (origin.x + extent.x - 1) / m.chunks.x;
    FENIX_DEBUG("io", "zarr region origin({},{},{}) extent {}x{}x{}: {} chunks", origin.z, origin.y, origin.x,
                extent.z, extent.y, extent.x, (cz1 - cz0 + 1) * (cy1 - cy0 + 1) * (cx1 - cx0 + 1));

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
                    T v = fillv;
                    if (present) {
                        const s64 off = (lz * m.chunks.y + ly) * m.chunks.x + lx;
                        v = detail::cast_dtype<T>(data + static_cast<usize>(off) * esz, m.dtype);
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

// Pull a CHUNK-ALIGNED region of a raw OME-Zarr level into a self-contained LOCAL OME-Zarr v2
// group at `out_root` (one pyramid level "0"). Chunk bytes are copied VERBATIM — no decode/
// re-encode — so the source dtype and chunking are preserved (a u8 source stays u8, NOT widened
// to f32 the way read_zarr_region does). Chunk coords are re-indexed to start at 0 and a fresh
// .zarray (shape = extent) + .zgroup/.zattrs skeleton are written, so the slab is a valid
// standalone zarr. `origin` must be a multiple of the source chunk size on every axis (a raw
// copy can't shift sub-chunk). Missing source chunks are omitted (zarr air), as in the source.
inline Expected<void> copy_zarr_region_local(const std::string& src_level_root, const std::string& out_root,
                                             Index3 origin, Extent3 extent) {
    auto mm = read_zarray(src_level_root);
    if (!mm) return std::unexpected(mm.error());
    const ZarrMeta m = *mm;
    if (origin.z % m.chunks.z != 0 || origin.y % m.chunks.y != 0 || origin.x % m.chunks.x != 0)
        return err(Errc::invalid_argument, "copy_zarr_region_local: origin must be chunk-aligned");

    namespace fs = std::filesystem;
    const std::string lvl = out_root + "/0";
    std::error_code ec;
    fs::create_directories(lvl, ec);
    if (ec) return err(Errc::io_error, "mkdir " + lvl + ": " + ec.message());

    auto write_text = [](const std::string& path, const std::string& body) -> Expected<void> {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return err(Errc::io_error, "open " + path);
        f.write(body.data(), static_cast<std::streamsize>(body.size()));
        if (!f) return err(Errc::io_error, "write " + path);
        return {};
    };
    const std::string sep(1, m.sep);
    if (auto r = write_text(out_root + "/.zgroup", "{\"zarr_format\": 2}\n"); !r) return r;
    if (auto r = write_text(out_root + "/.zattrs",
                            "{\"multiscales\": [{\"version\": \"0.4\", \"axes\": ["
                            "{\"name\": \"z\", \"type\": \"space\"}, {\"name\": \"y\", \"type\": \"space\"}, "
                            "{\"name\": \"x\", \"type\": \"space\"}], \"datasets\": [{\"path\": \"0\", "
                            "\"coordinateTransformations\": [{\"type\": \"scale\", \"scale\": [1.0, 1.0, 1.0]}]}]}]}\n");
        !r)
        return r;
    {
        std::ostringstream za;
        za << "{\n  \"shape\": [" << extent.z << ", " << extent.y << ", " << extent.x << "],\n"
           << "  \"chunks\": [" << m.chunks.z << ", " << m.chunks.y << ", " << m.chunks.x << "],\n"
           << "  \"dtype\": \"" << m.dtype << "\",\n  \"fill_value\": 0,\n  \"order\": \"C\",\n"
           << "  \"filters\": null,\n  \"dimension_separator\": \"" << sep << "\",\n"
           << "  \"compressor\": null,\n  \"zarr_format\": 2\n}\n";
        if (auto r = write_text(lvl + "/.zarray", za.str()); !r) return r;
    }

    const s64 cz0 = origin.z / m.chunks.z, cz1 = (origin.z + extent.z - 1) / m.chunks.z;
    const s64 cy0 = origin.y / m.chunks.y, cy1 = (origin.y + extent.y - 1) / m.chunks.y;
    const s64 cx0 = origin.x / m.chunks.x, cx1 = (origin.x + extent.x - 1) / m.chunks.x;
    // Pre-create the nested chunk dirs serially (avoids racing create_directories in the loop).
    if (m.sep == '/')
        for (s64 z = 0; z <= cz1 - cz0; ++z)
            for (s64 y = 0; y <= cy1 - cy0; ++y) {
                fs::create_directories(lvl + "/" + std::to_string(z) + "/" + std::to_string(y), ec);
                if (ec) return err(Errc::io_error, "mkdir chunk dir: " + ec.message());
            }

    struct ChunkId { s64 cz, cy, cx; };
    std::vector<ChunkId> chunks;
    for (s64 cz = cz0; cz <= cz1; ++cz)
        for (s64 cy = cy0; cy <= cy1; ++cy)
            for (s64 cx = cx0; cx <= cx1; ++cx) chunks.push_back({cz, cy, cx});

    std::atomic<bool> failed{false};
    std::string fail_msg;  // single writer: the one thread whose CAS on `failed` wins
    std::atomic<s64> done{0}, copied{0};
    parallel_for(0, static_cast<s64>(chunks.size()), [&](s64 i) {
        if (failed.load(std::memory_order_relaxed)) return;
        const ChunkId c = chunks[static_cast<usize>(i)];
        std::ostringstream ssrc;
        ssrc << c.cz << m.sep << c.cy << m.sep << c.cx;
        auto got = fetch_object(src_level_root, ssrc.str());
        if (!got) {
            bool e = false;
            if (failed.compare_exchange_strong(e, true)) fail_msg = got.error().message;
            return;
        }
        if (!*got) return;  // absent source chunk = air; omit it in the copy too
        std::ostringstream sdst;
        sdst << (c.cz - cz0) << m.sep << (c.cy - cy0) << m.sep << (c.cx - cx0);
        const std::string dpath = lvl + "/" + sdst.str();
        std::ofstream of(dpath, std::ios::binary | std::ios::trunc);
        if (!of) {
            bool e = false;
            if (failed.compare_exchange_strong(e, true)) fail_msg = "open " + dpath;
            return;
        }
        of.write(reinterpret_cast<const char*>((*got)->data()), static_cast<std::streamsize>((*got)->size()));
        copied.fetch_add(1, std::memory_order_relaxed);
        const s64 d = done.fetch_add(1) + 1;
        if (d % 64 == 0 || d == static_cast<s64>(chunks.size()))
            log(LogLevel::info, "zarr-copy: {}/{} chunks", d, chunks.size());
    });
    if (failed.load()) return err(Errc::fetch_failed, "zarr copy failed: " + fail_msg);
    log(LogLevel::info, "zarr-copy: {} chunks ({} non-empty) -> {}", chunks.size(), copied.load(), out_root);
    return {};
}

}  // namespace fenix::io
