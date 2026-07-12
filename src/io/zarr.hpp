// io/zarr.hpp — OME-Zarr v2 reader (RAW encoding). The canonical scroll storage format.
// Parses .zarray (minimal hand-rolled JSON, like fysics), reads chunk files at
// <z><sep><y><sep><x>, treats a MISSING chunk as fill_value (zarr omits empty chunks).
// Blosc/zstd/lz4 decompression (needs blosc2) + v3/sharded drop in later; this is the
// dependency-free raw path. ZYX, C-order. See io/CLAUDE.md, docs/research/villa-data.md.
#pragma once

#include "core/core.hpp"
#include "core/parallel.hpp"
#include "io/dct3d.hpp"
#include "io/s3.hpp"

#ifdef FENIX_HAVE_BLOSC2
#include <blosc2.h>
#endif
#include <zlib.h>
#ifdef FENIX_HAVE_BLOSC2
#endif

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
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
//
// Local path uses ::fopen (not std::ifstream) so errno is authoritative: ENOENT is the only
// "absent" case (matches the remote 404), every other errno (EACCES/EMFILE/EIO/ENOMEM/...) is a
// hard fetch failure per the "absent != fetch-failed, never silent air" invariant (io/CLAUDE.md).
// The read loop checks ferror() so a mid-read EIO also hard-fails instead of yielding a silent
// short buffer.
inline Expected<std::optional<std::vector<u8>>> fetch_object(const std::string& root, const std::string& sub) {
    if (is_remote(root)) return http_get(root + "/" + sub);
    const std::string path = root + "/" + sub;
    errno = 0;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (errno == ENOENT || errno == ENOTDIR) return std::optional<std::vector<u8>>(std::nullopt);  // missing = air
        return err(Errc::fetch_failed, "open " + path + ": " + std::strerror(errno));
    }
    std::vector<u8> b;
    u8 chunk[65536];
    for (;;) {
        const usize n = std::fread(chunk, 1, sizeof chunk, f);
        if (n > 0) b.insert(b.end(), chunk, chunk + n);
        if (n < sizeof chunk) {
            if (std::ferror(f)) {
                const int e = errno;
                std::fclose(f);
                return err(Errc::fetch_failed, "read " + path + ": " + std::strerror(e));
            }
            break;  // eof
        }
    }
    std::fclose(f);
    return std::optional<std::vector<u8>>(std::move(b));
}

struct ZarrMeta {
    int version = 2;     // 2 (.zarray) or 3 (zarr.json)
    bool blosc = false;  // chunk payloads are blosc-framed (decompressed via blosc2)
    bool gzip = false;   // v3 "gzip" codec (zlib)
    bool dct3d = false;  // v3 "dct3d" codec (16^3 lossy blocks — io/dct3d.hpp decoder)
    bool sharded = false;   // v3 sharding_indexed: fetch unit = SHARD, inner chunks inside
    Extent3 inner{};        // sharded: inner chunk dims (the codec-level chunk)
    bool shard_index_crc = true;  // index_codecs include crc32c (4 trailing bytes)
    Extent3 shape{};     // z,y,x
    Extent3 chunks{};    // chunk dims (for sharded: the SHARD dims)
    std::string dtype = "|u1";
    f32 fill = 0.0f;
    char sep = '.';  // v2 dimension_separator | v3 chunk_key_encoding separator
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
template <class T> inline T cast_dtype(const u8* p, const std::string& dt) {
    const char kind = dt.size() > 1 ? dt[1] : 'u';
    const usize sz = dtype_size(dt);
    if (kind == 'f' && sz == 4) {
        f32 v;
        std::memcpy(&v, p, 4);
        return static_cast<T>(v);
    }
    if (kind == 'u') {
        if (sz == 1) return static_cast<T>(*p);
        if (sz == 2) {
            u16 v;
            std::memcpy(&v, p, 2);
            return static_cast<T>(v);
        }
        if (sz == 4) {
            u32 v;
            std::memcpy(&v, p, 4);
            return static_cast<T>(v);
        }
    }
    if (kind == 'i') {
        if (sz == 1) return static_cast<T>(static_cast<s8>(*p));
        if (sz == 2) {
            s16 v;
            std::memcpy(&v, p, 2);
            return static_cast<T>(v);
        }
        if (sz == 4) {
            s32 v;
            std::memcpy(&v, p, 4);
            return static_cast<T>(v);
        }
    }
    return static_cast<T>(*p);
}
}  // namespace detail

namespace detail {

inline Expected<std::vector<u8>> gzip_inflate(std::span<const u8> in, usize expected) {
    std::vector<u8> out(expected);
    z_stream zs{};
    if (inflateInit2(&zs, 15 + 32) != Z_OK) return err(Errc::internal, "gzip: inflateInit failed");  // auto gz/zlib
    zs.next_in = const_cast<Bytef*>(in.data());
    zs.avail_in = static_cast<uInt>(in.size());
    zs.next_out = out.data();
    zs.avail_out = static_cast<uInt>(out.size());
    const int rc = inflate(&zs, Z_FINISH);
    const usize produced = out.size() - zs.avail_out;
    inflateEnd(&zs);
    if (rc != Z_STREAM_END || produced != expected) return err(Errc::decode_error, "gzip: bad chunk stream");
    return out;
}

// v3 sharding_indexed with index_location=end: shard = concat(inner chunk payloads) ||
// index (u64le offset,nbytes per inner chunk, 0xffff.. = missing) [|| crc32c(4B)].
// Assembles the FULL shard-shaped block (missing inner chunks = zero fill).
inline Expected<std::vector<u8>> decode_shard(std::span<const u8> shard, const struct ZarrMeta& m, usize esz);

}  // namespace detail

// zarr v3 (zarr.json). Codecs supported: bytes(le) [+ blosc | gzip], and
// sharding_indexed wrapping the same — anything else is a typed rejection.
inline Expected<ZarrMeta> read_zarr_v3(const std::string& js) {
    ZarrMeta m;
    m.version = 3;
    auto sh = detail::json_int_array(js, "shape");
    if (sh.size() != 3) return err(Errc::unsupported, "zarr v3 must be 3D");
    m.shape = {sh[0], sh[1], sh[2]};
    auto ch = detail::json_int_array(js, "chunk_shape");  // first occurrence = chunk_grid
    if (ch.size() < 3) return err(Errc::decode_error, "zarr v3: missing chunk_shape");
    m.chunks = {ch[0], ch[1], ch[2]};
    const std::string dt = detail::json_string(js, "data_type");
    if (dt == "uint8") m.dtype = "|u1";
    else if (dt == "uint16") m.dtype = "<u2";
    else if (dt == "uint32") m.dtype = "<u4";
    else if (dt == "float32") m.dtype = "<f4";
    else return err(Errc::unsupported, "zarr v3 data_type '" + dt + "' unsupported");
    m.sep = '/';
    {   // chunk_key_encoding.configuration.separator (default "/")
        const auto k = js.find("chunk_key_encoding");
        if (k != std::string::npos) {
            const auto seg = js.substr(k, 200);
            const auto sp = detail::json_string(seg, "separator");
            if (!sp.empty()) m.sep = sp[0];
        }
    }
    const bool has_shard = js.find("sharding_indexed") != std::string::npos;
    if (has_shard) {
        m.sharded = true;
        // the sharding config carries the INNER chunk_shape (second chunk_shape array)
        const auto k = js.find("sharding_indexed");
        const auto seg = js.substr(k);
        auto ich = detail::json_int_array(seg, "chunk_shape");
        if (ich.size() < 3) return err(Errc::decode_error, "zarr v3: sharding without inner chunk_shape");
        m.inner = {ich[0], ich[1], ich[2]};
        if ((m.chunks.z % m.inner.z) | (m.chunks.y % m.inner.y) | (m.chunks.x % m.inner.x))
            return err(Errc::decode_error, "zarr v3: shard dims not a multiple of inner chunk dims");
        if (seg.find("index_location") != std::string::npos && seg.find("\"start\"") != std::string::npos)
            return err(Errc::unsupported, "zarr v3: index_location=start unsupported (end only)");
        m.shard_index_crc = seg.find("crc32c") != std::string::npos;
        if (seg.find("\"blosc\"") != std::string::npos) m.blosc = true;
        if (seg.find("\"gzip\"") != std::string::npos) m.gzip = true;
        if (seg.find("\"dct3d\"") != std::string::npos) m.dct3d = true;
    } else {
        if (js.find("\"blosc\"") != std::string::npos) m.blosc = true;
        if (js.find("\"gzip\"") != std::string::npos) m.gzip = true;
        if (js.find("\"dct3d\"") != std::string::npos) m.dct3d = true;
    }
    if (m.dct3d) {
        // dct3d is geometry-fixed: one blob = one 16^3 inner chunk (community
        // export convention: shard 1024^3, inner 16^3). Anything else is a
        // format we don't know how to slice — typed rejection, never garbage.
        const Extent3 g = m.sharded ? m.inner : m.chunks;
        if (g.z != io::dct3d::kN || g.y != io::dct3d::kN || g.x != io::dct3d::kN)
            return err(Errc::unsupported, "zarr v3 dct3d requires 16^3 inner chunks");
        if (m.blosc || m.gzip) return err(Errc::unsupported, "zarr v3 dct3d cannot stack with blosc/gzip");
    }
#ifndef FENIX_HAVE_BLOSC2
    if (m.blosc) return err(Errc::unsupported, "zarr blosc chunks need a blosc2 build (FENIX_DEP_BLOSC2)");
#endif
    if (js.find("\"zstd\"") != std::string::npos && !m.blosc)
        return err(Errc::unsupported, "zarr v3 raw-zstd codec unsupported (blosc/gzip/raw only)");
    return m;
}

namespace detail {

inline Expected<std::vector<u8>> decode_shard(std::span<const u8> shard, const ZarrMeta& m, usize esz) {
    const s64 nz = m.chunks.z / m.inner.z, ny = m.chunks.y / m.inner.y, nx = m.chunks.x / m.inner.x;
    const usize n_inner = static_cast<usize>(nz * ny * nx);
    const usize idx_bytes = n_inner * 16 + (m.shard_index_crc ? 4 : 0);
    if (shard.size() < idx_bytes) return err(Errc::decode_error, "shard smaller than its index");
    const u8* idx = shard.data() + shard.size() - idx_bytes;
    auto rd64 = [&](usize i) {
        u64 v = 0;
        std::memcpy(&v, idx + i * 8, 8);
        return v;  // little-endian host assumed (LE-only per io invariants)
    };
    const usize inner_count = static_cast<usize>(m.inner.z * m.inner.y * m.inner.x);
    const usize inner_bytes = inner_count * esz;
    std::vector<u8> block(static_cast<usize>(m.chunks.z * m.chunks.y * m.chunks.x) * esz, 0);
    // Inner chunks are independent and write disjoint block regions — decode in
    // parallel. Matters most for dct3d shards (a 1024^3 shard = 262144 blocks of
    // range-decode + inverse DCT; serial decode would dwarf the fetch).
    std::atomic<int> bad{0};
    parallel_for(0, static_cast<s64>(nz * ny * nx), [&](s64 ci) {
        if (bad.load(std::memory_order_relaxed)) return;
        const s64 cz = ci / (ny * nx), cy = (ci / nx) % ny, cx = ci % nx;
        const u64 off = rd64(static_cast<usize>(ci) * 2), nb = rd64(static_cast<usize>(ci) * 2 + 1);
        if (off == ~u64{0} && nb == ~u64{0}) return;  // missing inner chunk = fill
        if (off + nb > shard.size() - idx_bytes) {
            bad.store(1, std::memory_order_relaxed);
            return;
        }
        std::span<const u8> payload(shard.data() + off, static_cast<usize>(nb));
        std::vector<u8> raw;
        if (m.dct3d) {
            raw.resize(inner_bytes);
            bool ok = false;
            if (esz == 1)
                ok = io::dct3d::decode<u8>(payload, std::span<u8>(raw.data(), inner_count));
            else if (esz == 2)
                ok = io::dct3d::decode<u16>(payload,
                                            std::span<u16>(reinterpret_cast<u16*>(raw.data()), inner_count));
            else if (esz == 4)
                ok = io::dct3d::decode<f32>(payload,
                                            std::span<f32>(reinterpret_cast<f32*>(raw.data()), inner_count));
            if (!ok) {
                bad.store(2, std::memory_order_relaxed);
                return;
            }
        } else if (m.gzip) {
            auto r = gzip_inflate(payload, inner_bytes);
            if (!r) {
                bad.store(3, std::memory_order_relaxed);
                return;
            }
            raw = std::move(*r);
        }
#ifdef FENIX_HAVE_BLOSC2
        else if (m.blosc) {
            raw.resize(inner_bytes);
            const int n = blosc2_decompress(payload.data(), static_cast<int32_t>(payload.size()), raw.data(),
                                            static_cast<int32_t>(raw.size()));
            if (n < 0 || static_cast<usize>(n) != inner_bytes) {
                bad.store(4, std::memory_order_relaxed);
                return;
            }
        }
#endif
        else {
            if (payload.size() != inner_bytes) {
                bad.store(5, std::memory_order_relaxed);
                return;
            }
            raw.assign(payload.begin(), payload.end());
        }
        // scatter the inner chunk into the block (both row-major ZYX)
        for (s64 lz = 0; lz < m.inner.z; ++lz)
            for (s64 ly = 0; ly < m.inner.y; ++ly) {
                const usize src = (static_cast<usize>(lz * m.inner.y + ly) * static_cast<usize>(m.inner.x)) * esz;
                const usize dst = ((static_cast<usize>(cz * m.inner.z + lz) * static_cast<usize>(m.chunks.y) +
                                    static_cast<usize>(cy * m.inner.y + ly)) *
                                       static_cast<usize>(m.chunks.x) +
                                   static_cast<usize>(cx * m.inner.x)) *
                                  esz;
                std::memcpy(block.data() + dst, raw.data() + src, static_cast<usize>(m.inner.x) * esz);
            }
    });
    switch (bad.load()) {
        case 0: break;
        case 1: return err(Errc::decode_error, "shard index oob");
        case 2: return err(Errc::decode_error, "shard inner dct3d decode failed");
        case 3: return err(Errc::decode_error, "shard inner gzip decode failed");
        case 4: return err(Errc::decode_error, "shard inner blosc decompress failed");
        default: return err(Errc::decode_error, "shard inner size mismatch");
    }
    return block;
}

}  // namespace detail

inline Expected<ZarrMeta> read_zarray(const std::string& root) {
    auto bytes = fetch_object(root, ".zarray");
    if (!bytes) return std::unexpected(bytes.error());
    if (!*bytes) {
        auto v3 = fetch_object(root, "zarr.json");
        if (!v3) return std::unexpected(v3.error());
        if (*v3)
            return read_zarr_v3(std::string(reinterpret_cast<const char*>((*v3)->data()), (*v3)->size()));
        return err(Errc::not_found, "no .zarray or zarr.json in " + root);
    }
    const std::string js(reinterpret_cast<const char*>((*bytes)->data()), (*bytes)->size());
    ZarrMeta m;
    auto sh = detail::json_int_array(js, "shape");
    auto ch = detail::json_int_array(js, "chunks");
    if (sh.size() != 3 || ch.size() != 3) return err(Errc::unsupported, "zarr must be 3D");
    m.shape = {sh[0], sh[1], sh[2]};
    m.chunks = {ch[0], ch[1], ch[2]};
    m.dtype = detail::json_string(js, "dtype");
    if (js.find("\"compressor\": null") == std::string::npos && js.find("\"compressor\":null") == std::string::npos) {
        // {"id":"blosc", ...} (any cname/shuffle — blosc self-describes its frame) is supported
        // when built with blosc2; anything else is a typed rejection.
        if (detail::json_string(js, "id") == "blosc") {
#ifdef FENIX_HAVE_BLOSC2
            m.blosc = true;
#else
            return err(Errc::unsupported, "zarr blosc chunks need a blosc2 build (FENIX_DEP_BLOSC2)");
#endif
        } else {
            return err(Errc::unsupported, "zarr compressor not supported (raw + blosc only)");
        }
    }
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
    FENIX_DEBUG("io",
                "zarr region origin({},{},{}) extent {}x{}x{}: {} chunks",
                origin.z,
                origin.y,
                origin.x,
                extent.z,
                extent.y,
                extent.x,
                (cz1 - cz0 + 1) * (cy1 - cy0 + 1) * (cx1 - cx0 + 1));

    // Enumerate the chunks covering the region; fetch + scatter them in parallel (each writes a
    // disjoint output region, so no locking on the volume). Remote fetches reuse a per-thread
    // libcurl connection (s3.hpp), so a slab of hundreds of chunks streams concurrently.
    struct ChunkId {
        s64 cz, cy, cx;
    };
    std::vector<ChunkId> chunks;
    for (s64 cz = cz0; cz <= cz1; ++cz)
        for (s64 cy = cy0; cy <= cy1; ++cy)
            for (s64 cx = cx0; cx <= cx1; ++cx) chunks.push_back({cz, cy, cx});

    std::atomic<bool> failed{false};
    std::string fail_msg;
    std::atomic<s64> done{0};

    // Bound the number of concurrent remote GETs. Fanning one connection per core at a single
    // S3 endpoint self-congests: on a CPU-quota-limited box (e.g. a 27-CPU cgroup reporting
    // nproc=256) hundreds of transfers share a fraction of that bandwidth, drop below the
    // low-speed floor, and trip the per-transfer stall watchdog — a spurious hard-fail. Cap it
    // to a modest pool (override with FENIX_ZARR_FETCH_THREADS). Local chunks are mmap reads with
    // no endpoint contention, so they stay unbounded.
    int fetch_threads = 0;
    if (is_remote(root)) {
        fetch_threads = 24;
        if (const char* e = std::getenv("FENIX_ZARR_FETCH_THREADS")) {
            const int v = std::atoi(e);
            if (v > 0) fetch_threads = v;
        }
    }

    // I/O-bound fan-out: the workers block on the network, not the CPU, so use parallel_for_io — it sizes
    // the team to `fetch_threads` DIRECTLY and does NOT clamp to cpu_budget(). On a low-CPU-quota container
    // (e.g. 13 CPUs) clamping to the budget throttled S3 to ~13 concurrent transfers and throughput
    // collapsed (~8 MB/s); the endpoint can feed many more parallel streams than the box has cores.
    parallel_for_io(0, static_cast<s64>(chunks.size()), fetch_threads, [&](s64 i) {
        if (failed.load(std::memory_order_relaxed)) return;
        const ChunkId c = chunks[static_cast<usize>(i)];
        std::ostringstream sub;
        if (m.version == 3) sub << 'c' << m.sep;
        sub << c.cz << m.sep << c.cy << m.sep << c.cx;
        auto got = fetch_object(root, sub.str());
        if (!got) {  // hard fetch failure — record, do NOT treat as air
            bool expect = false;
            if (failed.compare_exchange_strong(expect, true)) fail_msg = got.error().message;
            return;
        }
        std::optional<std::vector<u8>>& blob = *got;
        if (!blob) return;  // absent (missing chunk file / 404) = legitimate air, fill stays
        const usize expected = static_cast<usize>(ccount) * esz;
        if (m.sharded) {  // v3 sharding_indexed: blob is a SHARD; assemble the full block
            auto raw = detail::decode_shard(*blob, m, esz);
            if (!raw) {
                bool expect = false;
                if (failed.compare_exchange_strong(expect, true))
                    fail_msg = "shard " + sub.str() + ": " + raw.error().message;
                return;
            }
            *blob = std::move(*raw);
        }
        if (m.dct3d && !m.sharded) {  // v3 dct3d codec on a plain (16^3) chunk
            std::vector<u8> raw(expected);
            bool ok = false;
            if (esz == 1) ok = io::dct3d::decode<u8>(*blob, std::span<u8>(raw.data(), ccount));
            else if (esz == 2)
                ok = io::dct3d::decode<u16>(*blob, std::span<u16>(reinterpret_cast<u16*>(raw.data()), ccount));
            else if (esz == 4)
                ok = io::dct3d::decode<f32>(*blob, std::span<f32>(reinterpret_cast<f32*>(raw.data()), ccount));
            if (!ok) {
                bool expect = false;
                if (failed.compare_exchange_strong(expect, true))
                    fail_msg = "chunk " + sub.str() + ": dct3d decode failed";
                return;
            }
            *blob = std::move(raw);
        }
        if (m.gzip && !m.sharded) {  // v3 gzip codec on a plain chunk
            auto raw = detail::gzip_inflate(*blob, expected);
            if (!raw) {
                bool expect = false;
                if (failed.compare_exchange_strong(expect, true))
                    fail_msg = "chunk " + sub.str() + ": " + raw.error().message;
                return;
            }
            *blob = std::move(*raw);
        }
#ifdef FENIX_HAVE_BLOSC2
        if (m.blosc && !m.sharded) {  // blosc frame -> raw chunk bytes (size mismatch below = corruption)
            std::vector<u8> raw(expected);
            const int n = blosc2_decompress(
                blob->data(), static_cast<int32_t>(blob->size()), raw.data(), static_cast<int32_t>(raw.size()));
            if (n < 0 || static_cast<usize>(n) != expected) {
                bool expect = false;
                if (failed.compare_exchange_strong(expect, true))
                    fail_msg = "chunk " + sub.str() + ": blosc decompress failed (rc=" + std::to_string(n) + ")";
                return;
            }
            *blob = std::move(raw);
        }
#endif
        if (blob->size() != expected) {  // present but wrong-sized = corruption, NEVER silent air
            bool expect = false;
            if (failed.compare_exchange_strong(expect, true))
                fail_msg = "chunk " + sub.str() + ": got " + std::to_string(blob->size()) + " bytes, expected " +
                           std::to_string(expected);
            return;
        }
        const bool present = true;
        const u8* data = blob->data();
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
inline Expected<void>
copy_zarr_region_local(const std::string& src_level_root, const std::string& out_root, Index3 origin, Extent3 extent) {
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
    if (auto r =
            write_text(out_root + "/.zattrs",
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

    struct ChunkId {
        s64 cz, cy, cx;
    };
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
        const std::string tmp = dpath + ".tmp" + std::to_string(i);  // per-task tmp: parallel_for tasks share no path
        {
            std::ofstream of(tmp, std::ios::binary | std::ios::trunc);
            if (!of) {
                bool e = false;
                if (failed.compare_exchange_strong(e, true)) fail_msg = "open " + tmp;
                return;
            }
            of.write(reinterpret_cast<const char*>((*got)->data()), static_cast<std::streamsize>((*got)->size()));
            of.close();
            if (!of) {  // ENOSPC/EIO on write or close-flush — never leave a truncated chunk file
                bool e = false;
                if (failed.compare_exchange_strong(e, true)) fail_msg = "write " + tmp + " (disk full?)";
                std::error_code rm_ec;
                fs::remove(tmp, rm_ec);
                return;
            }
        }
        std::error_code ren_ec;
        fs::rename(tmp, dpath, ren_ec);
        if (ren_ec) {
            bool e = false;
            if (failed.compare_exchange_strong(e, true))
                fail_msg = "rename " + tmp + " -> " + dpath + ": " + ren_ec.message();
            return;
        }
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
