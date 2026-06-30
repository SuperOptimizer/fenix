// codec/archive.hpp — the .fxvol container (v4). One file per volume: a sparse grid of 64³ chunks, each
// encoded by the DCT tile codec (a 64³ chunk = one tile of 4³=64 DCT blocks sharing rANS tables), indexed
// by a 3-level (12+12+12) compressed-Morton radix PAGE TABLE that lives IN the file and is mmap'd — so the
// resident RAM is the working set, never O(chunks) (a flat in-RAM index would be ~3.4 TB at the 2¹⁸/axis
// envelope). One file holds EVERY LOD octave (each its own page-table, reached via per-LOD roots in the
// superblock). This is the volume store + the out-of-core IO substrate. See ADR 0006 + docs/design/fxvol-v4-layout.md.
//
// PHASES 1-5 (this file): single-file LIVE form, crash-safe, with the explicit LOD pyramid, a decoded-16³-
// chunk cache, and a SEALED coarse-first repack. P1 = Morton key + 3-level radix table over a MAP_NORESERVE
// reservation + fallocate growth + bump allocator + sentinel-as-coverage. P2 = double-buffered crc32c
// superblock + per-blob crc + data-before-pointer commit. P3 = sharded-SIEVE decoded-16³-chunk cache
// (block16/voxel). P4 = explicit octave pyramid (per-LOD roots; global 2× box downsample → retile). P5 =
// finalize() → SEALED coarse-first repack (verbatim blob copy). S3 is READ-ONLY (anonymous range-GET via
// io/s3.hpp; no S3 writes / multi-writer CAS needed — the archive is written locally then uploaded).
// Optional later refinement: full copy-on-write page-table versioning. The public API is stable.
#pragma once

#include "codec/block_cache.hpp"
#include "codec/dct_block.hpp"
#include "core/core.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace fenix::codec {

enum class Coverage : u8 { Absent = 0, Zero = 1, Real = 2 };  // NOT_SURE / air / has-data

inline constexpr u32 fxvol_magic = 0x4c565846u;  // "FXVL"
inline constexpr u32 fxvol_version = 4;          // v4: mmap'd 3-level Morton radix page-table (ADR 0006)
inline constexpr s64 fxvol_chunk_side = 64;

namespace detail {
inline u32 crc32c(const u8* p, usize n, u32 crc = 0) {  // Castagnoli, software table; torn-write detection
    static const std::array<u32, 256> table = [] {
        std::array<u32, 256> t{};
        for (u32 i = 0; i < 256; ++i) {
            u32 c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (c >> 1) ^ 0x82f63b78u : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (usize i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xffu] ^ (crc >> 8);
    return ~crc;
}

inline u64 morton_part1by2(u64 x) {  // spread low 21 bits to every 3rd bit
    x &= 0x1fffffull;
    x = (x | (x << 32)) & 0x1f00000000ffffull;
    x = (x | (x << 16)) & 0x1f0000ff0000ffull;
    x = (x | (x << 8)) & 0x100f00f00f00f00full;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
    x = (x | (x << 2)) & 0x1249249249249249ull;
    return x;
}
inline u64 morton3(u64 z, u64 y, u64 x) {  // compressed-Morton ZYX key (halo locality)
    return (morton_part1by2(z) << 2) | (morton_part1by2(y) << 1) | morton_part1by2(x);
}

inline constexpr u64 kFxAbsent = ~0ull;  // tri-state: off==kFxAbsent ABSENT; else len==0 ZERO; else REAL
struct FxSlot {
    u64 off;
    u64 len;
};
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(memory_sanitizer) || __has_feature(thread_sanitizer)
#define FENIX_FXVOL_SMALL_RESERVE 1
#endif
#endif
#ifdef FENIX_FXVOL_SMALL_RESERVE
inline constexpr u64 kFxReserve = 1ull << 32;  // 4 GiB under sanitizers
#else
inline constexpr u64 kFxReserve = 1ull << 40;  // 1 TiB (caps a single archive; Phase-1 limitation)
#endif
inline constexpr u64 kFxGrow = 1ull << 26;            // grow the file 64 MiB at a time
inline constexpr u64 kFxNodeEnt = 4096;               // 2¹² entries per radix node / leaf
inline constexpr u64 kFxSuper = 4096;                 // one superblock slot = one page
inline constexpr u64 kFxDataStart = 2 * kFxSuper;     // append region after the two superblock slots
inline constexpr u64 kFxMaxLod = 20;                  // 2¹⁸→2⁶ needs 13; 20 is generous headroom
inline constexpr u64 kFxLodRootOff = 68;              // superblock: per-LOD root-offset array starts here
inline constexpr u64 kFxSbCrcLen = kFxLodRootOff + 8 * kFxMaxLod;  // crc32c covers [0, kFxSbCrcLen); stored at +kFxSbCrcLen
inline constexpr u64 kFxDefaultCache = 256ull << 20;  // default decoded-16³-chunk cache budget (256 MiB)
}  // namespace detail

// The .fxvol archive (v4). Owns one mmap'd file; move-only (RAII over the fd + mapping).
class VolumeArchive {
public:
    static Expected<VolumeArchive> create(const std::string& path, Extent3 dims, DctParams bp) {
        VolumeArchive a;
        a.path_ = path;
        a.dims_ = dims;
        a.params_ = bp;
        a.nlods_ = a.plan_nlods_(dims);
        a.fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (a.fd_ < 0) return err(Errc::io_error, "cannot create " + path);
        if (auto e = a.map_(); !e) return std::unexpected(e.error());
        if (auto e = a.ensure_(detail::kFxDataStart); !e) return std::unexpected(e.error());
        a.cursor_ = detail::kFxDataStart;
        a.committed_eof_ = detail::kFxDataStart;
        a.commit_seq_ = 1;
        a.write_superblock_(a.commit_seq_);  // a valid (empty) archive exists even if we crash right away
        a.writable_ = true;
        return a;
    }

    // open(path) — read-only. open(path, true) — read/write (append + commit; COW preserves the prior
    // committed snapshot, so a crash mid-append recovers the last committed state).
    static Expected<VolumeArchive> open(const std::string& path, bool writable = false) {
        VolumeArchive a;
        a.path_ = path;
        a.fd_ = ::open(path.c_str(), O_RDWR, 0644);
        if (a.fd_ < 0) return err(Errc::not_found, "cannot open " + path);
        const off_t sz = ::lseek(a.fd_, 0, SEEK_END);
        if (sz < static_cast<off_t>(detail::kFxDataStart)) return err(Errc::decode_error, "truncated .fxvol");
        a.file_size_ = static_cast<u64>(sz);
        if (auto e = a.map_(); !e) return std::unexpected(e.error());
        int best = -1;
        u64 best_seq = 0;
        for (int s = 0; s < 2; ++s) {  // adopt the highest-commit_seq superblock slot that passes crc
            const u64 base = static_cast<u64>(s) * detail::kFxSuper;
            u32 magic, version, crc_stored;
            std::memcpy(&magic, a.base_ + base + 0, 4);
            std::memcpy(&version, a.base_ + base + 4, 4);
            std::memcpy(&crc_stored, a.base_ + base + detail::kFxSbCrcLen, 4);
            if (magic != fxvol_magic || version != fxvol_version) continue;
            if (detail::crc32c(a.base_ + base, detail::kFxSbCrcLen) != crc_stored) continue;
            u64 seq;
            std::memcpy(&seq, a.base_ + base + 8, 8);
            if (best < 0 || seq > best_seq) { best = s; best_seq = seq; }
        }
        if (best < 0) return err(Errc::decode_error, "no valid superblock");
        const u64 b = static_cast<u64>(best) * detail::kFxSuper;
        std::memcpy(&a.commit_seq_, a.base_ + b + 8, 8);
        std::memcpy(&a.committed_eof_, a.base_ + b + 16, 8);
        std::memcpy(&a.nlods_, a.base_ + b + 24, 4);
        std::memcpy(&a.dims_.z, a.base_ + b + 32, 8);
        std::memcpy(&a.dims_.y, a.base_ + b + 40, 8);
        std::memcpy(&a.dims_.x, a.base_ + b + 48, 8);
        std::memcpy(&a.params_.q, a.base_ + b + 56, 4);
        std::memcpy(&a.params_.hf_exp, a.base_ + b + 60, 4);
        std::memcpy(&a.params_.dz_frac, a.base_ + b + 64, 4);
        if (a.nlods_ < 1 || a.nlods_ > detail::kFxMaxLod) return err(Errc::decode_error, "bad nlods");
        for (u32 l = 0; l < detail::kFxMaxLod; ++l) std::memcpy(&a.lod_root_[l], a.base_ + b + detail::kFxLodRootOff + 8 * l, 8);
        if (a.committed_eof_ > a.file_size_ || a.committed_eof_ < detail::kFxDataStart) return err(Errc::decode_error, "corrupt eof");
        for (u32 l = 0; l < a.nlods_; ++l)
            if (a.lod_root_[l] != 0 && a.lod_root_[l] + node_bytes_() > a.committed_eof_) return err(Errc::decode_error, "corrupt root");
        a.cursor_ = a.committed_eof_;
        a.writable_ = writable;
        return a;
    }

    VolumeArchive() = default;
    VolumeArchive(const VolumeArchive&) = delete;
    VolumeArchive& operator=(const VolumeArchive&) = delete;
    VolumeArchive(VolumeArchive&& o) noexcept { steal_(o); }
    VolumeArchive& operator=(VolumeArchive&& o) noexcept {
        if (this != &o) { release_(); steal_(o); }
        return *this;
    }
    ~VolumeArchive() { release_(); }

    [[nodiscard]] Extent3 dims() const { return dims_; }            // LOD 0 dims
    [[nodiscard]] DctParams params() const { return params_; }
    [[nodiscard]] u32 nlods() const { return nlods_; }
    [[nodiscard]] Extent3 dims_at(s64 lod) const {                  // dims of LOD level `lod` (2× per octave)
        Extent3 d = dims_;
        for (s64 i = 0; i < lod; ++i) d = {(d.z + 1) / 2, (d.y + 1) / 2, (d.x + 1) / 2};
        return d;
    }
    [[nodiscard]] ChunkCoord chunk_extent(s64 lod = 0) const {
        const Extent3 d = dims_at(lod);
        return {(d.z + fxvol_chunk_side - 1) / fxvol_chunk_side, (d.y + fxvol_chunk_side - 1) / fxvol_chunk_side,
                (d.x + fxvol_chunk_side - 1) / fxvol_chunk_side};
    }

    [[nodiscard]] Coverage coverage(s64 lod, ChunkCoord c) const {
        const detail::FxSlot s = slot_read_(lod, c);
        if (s.off == detail::kFxAbsent) return Coverage::Absent;
        return s.len == 0 ? Coverage::Zero : Coverage::Real;
    }
    [[nodiscard]] Coverage coverage(ChunkCoord c) const { return coverage(0, c); }

    // Write one chunk_side³ block into LOD `lod`. An all-`fill` block is recorded as ZERO (no blob).
    Expected<void> write_chunk(s64 lod, ChunkCoord c, std::span<const f32> block, f32 fill = 0.0f) {
        if (!writable_) return err(Errc::io_error, "archive opened read-only");
        if (lod < 0 || lod >= static_cast<s64>(detail::kFxMaxLod)) return err(Errc::invalid_argument, "bad lod");
        const s64 n = fxvol_chunk_side * fxvol_chunk_side * fxvol_chunk_side;
        if (static_cast<s64>(block.size()) != n) return err(Errc::invalid_argument, "block must be chunk_side³");
        bool all_fill = true;
        for (f32 v : block)
            if (v != fill) { all_fill = false; break; }
        auto sp = slot_write_(lod, c);
        if (!sp) return std::unexpected(sp.error());
        if (all_fill) {
            **sp = {0, 0};
            return {};
        }
        auto payload = encode_tile_dct<f32>(block, fxvol_chunk_side / kDctN, params_);
        const u32 crc = detail::crc32c(payload.data(), payload.size());
        auto off = alloc_(payload.size() + 4, false);  // blob = [payload][u32 crc32c]
        if (!off) return std::unexpected(off.error());
        std::memcpy(base_ + *off, payload.data(), payload.size());
        std::memcpy(base_ + *off + payload.size(), &crc, 4);
        **sp = {*off, payload.size()};  // base_ is fixed → the slot pointer is still valid after alloc_
        return {};
    }
    Expected<void> write_chunk(ChunkCoord c, std::span<const f32> block, f32 fill = 0.0f) { return write_chunk(0, c, block, fill); }

    // Read one chunk_side³ block from LOD `lod`. ABSENT/ZERO -> filled with `fill`.
    [[nodiscard]] Expected<std::vector<f32>> read_chunk(s64 lod, ChunkCoord c, f32 fill = 0.0f) const {
        const usize n = static_cast<usize>(fxvol_chunk_side * fxvol_chunk_side * fxvol_chunk_side);
        const detail::FxSlot s = slot_read_(lod, c);
        if (s.off == detail::kFxAbsent || s.len == 0) return std::vector<f32>(n, fill);
        const u64 hz = read_horizon_();
        if (s.off < detail::kFxDataStart || s.len + 4 > hz || s.off > hz - s.len - 4)
            return err(Errc::decode_error, "corrupt chunk offset");
        u32 crc_stored;
        std::memcpy(&crc_stored, base_ + s.off + s.len, 4);
        if (detail::crc32c(base_ + s.off, s.len) != crc_stored) return err(Errc::decode_error, "chunk crc mismatch");
        return decode_tile_dct<f32>(std::span<const u8>(base_ + s.off, s.len), fxvol_chunk_side / kDctN, params_);
    }
    [[nodiscard]] Expected<std::vector<f32>> read_chunk(ChunkCoord c, f32 fill = 0.0f) const { return read_chunk(0, c, fill); }

    // Tile a whole volume into LOD 0 AND build the full LOD pyramid (global 2× box downsample → retile per
    // octave, down to a single 64³ chunk). Edge-replicates partial chunks so the block-DCT doesn't ring.
    Expected<void> write_volume(VolumeView<const f32> vol) {
        if (!(vol.dims() == dims_)) return err(Errc::invalid_argument, "dims mismatch");
        if (auto e = write_level_(0, vol); !e) return e;
        Volume<f32> cur;
        VolumeView<const f32> cv = vol;
        s64 lod = 0;
        while (!fits_one_chunk_(cv.dims()) && lod + 1 < static_cast<s64>(detail::kFxMaxLod)) {
            Volume<f32> next = downsample2_(cv);
            ++lod;
            if (auto e = write_level_(lod, next.view()); !e) return e;
            cur = std::move(next);
            cv = cur.view();
        }
        nlods_ = static_cast<u32>(lod + 1);
        return {};
    }

    // Reassemble LOD level `lod` (cropping edge padding).
    [[nodiscard]] Expected<Volume<f32>> read_volume(s64 lod = 0) const {
        const Extent3 vd = dims_at(lod);
        Volume<f32> out = Volume<f32>::zeros(vd);
        VolumeView<f32> ov = out.view();
        const ChunkCoord ce = chunk_extent(lod);
        const s64 cs = fxvol_chunk_side;
        for (s64 cz = 0; cz < ce.z; ++cz)
            for (s64 cy = 0; cy < ce.y; ++cy)
                for (s64 cx = 0; cx < ce.x; ++cx) {
                    auto blk = read_chunk(lod, {cz, cy, cx});
                    if (!blk) return std::unexpected(blk.error());
                    for (s64 z = 0; z < cs; ++z)
                        for (s64 y = 0; y < cs; ++y)
                            for (s64 x = 0; x < cs; ++x) {
                                const s64 vz = cz * cs + z, vy = cy * cs + y, vx = cx * cs + x;
                                if (vz < vd.z && vy < vd.y && vx < vd.x)
                                    ov(vz, vy, vx) = (*blk)[static_cast<usize>((z * cs + y) * cs + x)];
                            }
                }
        return out;
    }

    [[nodiscard]] u64 lod_root_offset(s64 lod) const {  // byte offset of a LOD's radix root (0 = none); for tests/inspection
        return (lod >= 0 && lod < static_cast<s64>(detail::kFxMaxLod)) ? lod_root_[lod] : 0;
    }

    // Repack this (LIVE) archive into a fresh SEALED file at `dst`, ordered COARSE-FIRST (coarsest LOD's
    // data at the front, full-res last) so a truncated range-GET yields a coarse preview. Compressed blobs
    // are copied VERBATIM (no decode/re-encode → no extra loss); ZERO/ABSENT coverage is preserved. Within
    // a LOD, chunks are written in Morton order (on-disk halo locality). Call on a read-only/closed archive.
    Expected<void> finalize(const std::string& dst) const {
        auto outr = create(dst, dims_, params_);
        if (!outr) return std::unexpected(outr.error());
        VolumeArchive out = std::move(*outr);
        out.nlods_ = nlods_;
        for (s64 lod = static_cast<s64>(nlods_) - 1; lod >= 0; --lod) {  // coarse-first across octaves
            const ChunkCoord ce = chunk_extent(lod);
            std::vector<std::pair<u64, ChunkCoord>> items;
            for (s64 cz = 0; cz < ce.z; ++cz)
                for (s64 cy = 0; cy < ce.y; ++cy)
                    for (s64 cx = 0; cx < ce.x; ++cx) {
                        const ChunkCoord c{cz, cy, cx};
                        if (slot_read_(lod, c).off == detail::kFxAbsent) continue;
                        items.emplace_back(detail::morton3(static_cast<u64>(cz), static_cast<u64>(cy), static_cast<u64>(cx)), c);
                    }
            std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& [m, c] : items) {
                const detail::FxSlot s = slot_read_(lod, c);
                auto sp = out.slot_write_(lod, c);
                if (!sp) return std::unexpected(sp.error());
                if (s.len == 0) { **sp = {0, 0}; continue; }  // ZERO
                if (s.off < detail::kFxDataStart || s.len + 4 > committed_eof_ || s.off > committed_eof_ - s.len - 4)
                    return err(Errc::decode_error, "corrupt source blob");
                auto off = out.alloc_(s.len + 4, false);  // copy [payload][crc] verbatim
                if (!off) return std::unexpected(off.error());
                std::memcpy(out.base_ + *off, base_ + s.off, s.len + 4);
                **sp = {*off, s.len};
            }
        }
        out.nlods_ = nlods_;
        return out.close();
    }

    // ---- decoded-16³-chunk view (mmap-the-archive-as-an-array) ----
    void reserve_cache(u64 budget_bytes, int shards = 16) { block_cache_ = std::make_unique<BlockCache>(budget_bytes, shards); }
    [[nodiscard]] u64 cache_hits() const { return block_cache_ ? block_cache_->hits() : 0; }
    [[nodiscard]] u64 cache_misses() const { return block_cache_ ? block_cache_->misses() : 0; }
    [[nodiscard]] u64 cache_bytes() const { return block_cache_ ? block_cache_->bytes() : 0; }

    // Fetch a decoded 16³ chunk at LOD `lod` by its block-grid coord (= voxel/16). On a miss the containing
    // 64³ tile is decoded once (atomic decode unit) and ALL 64 of its 16³ chunks are cached.
    [[nodiscard]] Expected<BlockCache::Ref> block16(s64 lod, ChunkCoord bc) const {
        if (!block_cache_) block_cache_ = std::make_unique<BlockCache>(detail::kFxDefaultCache);
        const u64 key = block_key_(lod, bc.z, bc.y, bc.x);
        if (auto r = block_cache_->get(key)) return r;
        constexpr s64 BS = kDctN, NB = fxvol_chunk_side / kDctN;
        const ChunkCoord tc{bc.z / NB, bc.y / NB, bc.x / NB};
        auto tile = read_chunk(lod, tc);  // 64³ decoded f32 (ZYX contiguous); ABSENT/ZERO -> zeros
        if (!tile) return std::unexpected(tile.error());
        const std::vector<f32>& t = *tile;
        BlockCache::Ref want;
        for (s64 sz = 0; sz < NB; ++sz)
            for (s64 sy = 0; sy < NB; ++sy)
                for (s64 sx = 0; sx < NB; ++sx) {
                    auto blk = std::make_shared<BlockCache::Block>(static_cast<usize>(BS * BS * BS));
                    for (s64 z = 0; z < BS; ++z)
                        for (s64 y = 0; y < BS; ++y)
                            for (s64 x = 0; x < BS; ++x)
                                (*blk)[static_cast<usize>((z * BS + y) * BS + x)] =
                                    t[static_cast<usize>(((sz * BS + z) * fxvol_chunk_side + (sy * BS + y)) * fxvol_chunk_side + (sx * BS + x))];
                    const u64 k = block_key_(lod, tc.z * NB + sz, tc.y * NB + sy, tc.x * NB + sx);
                    BlockCache::Ref cr = blk;
                    block_cache_->put(k, cr);
                    if (k == key) want = cr;
                }
        if (!want) return err(Errc::decode_error, "block16 key not produced");
        return want;
    }
    [[nodiscard]] Expected<BlockCache::Ref> block16(ChunkCoord bc) const { return block16(0, bc); }

    [[nodiscard]] Expected<f32> voxel(s64 lod, s64 z, s64 y, s64 x) const {
        auto r = block16(lod, {z / kDctN, y / kDctN, x / kDctN});
        if (!r) return std::unexpected(r.error());
        return (**r)[static_cast<usize>(((z % kDctN) * kDctN + (y % kDctN)) * kDctN + (x % kDctN))];
    }
    [[nodiscard]] Expected<f32> voxel(s64 z, s64 y, s64 x) const { return voxel(0, z, y, x); }

    // Durable checkpoint: msync the data region, THEN write+msync the next superblock slot (data-before-
    // pointer). Safe to call repeatedly mid-session; a crash recovers the last committed state.
    Expected<void> commit() {
        if (!writable_) return {};
        committed_eof_ = cursor_;
        if (base_ && committed_eof_ > detail::kFxDataStart)
            ::msync(base_ + detail::kFxDataStart, committed_eof_ - detail::kFxDataStart, MS_SYNC);
        ++commit_seq_;
        write_superblock_(commit_seq_);
        return {};
    }
    Expected<void> close() {
        if (!writable_) return {};
        auto e = commit();
        writable_ = false;
        return e;
    }

private:
    static u64 node_bytes_() { return detail::kFxNodeEnt * sizeof(u64); }
    static u64 leaf_bytes_() { return detail::kFxNodeEnt * sizeof(detail::FxSlot); }
    static bool fits_one_chunk_(Extent3 d) { return d.z <= fxvol_chunk_side && d.y <= fxvol_chunk_side && d.x <= fxvol_chunk_side; }
    static u32 plan_nlods_(Extent3 d) {
        u32 n = 1;
        while (!fits_one_chunk_(d) && n < detail::kFxMaxLod) { d = {(d.z + 1) / 2, (d.y + 1) / 2, (d.x + 1) / 2}; ++n; }
        return n;
    }
    static u64 block_key_(s64 lod, s64 z, s64 y, s64 x) {  // cache key: LOD in the high byte + 16³-block Morton
        return (static_cast<u64>(lod) << 56) | detail::morton3(static_cast<u64>(z), static_cast<u64>(y), static_cast<u64>(x));
    }
    static Volume<f32> downsample2_(VolumeView<const f32> v) {  // global 2³ box prefilter + decimate (seam-free)
        const Extent3 d = v.dims();
        const Extent3 o{(d.z + 1) / 2, (d.y + 1) / 2, (d.x + 1) / 2};
        Volume<f32> out = Volume<f32>::zeros(o);
        VolumeView<f32> ov = out.view();
        for (s64 z = 0; z < o.z; ++z)
            for (s64 y = 0; y < o.y; ++y)
                for (s64 x = 0; x < o.x; ++x) {
                    f32 sum = 0;
                    int n = 0;
                    for (s64 dz = 0; dz < 2; ++dz)
                        for (s64 dy = 0; dy < 2; ++dy)
                            for (s64 dx = 0; dx < 2; ++dx) {
                                const s64 sz = 2 * z + dz, sy = 2 * y + dy, sx = 2 * x + dx;
                                if (sz < d.z && sy < d.y && sx < d.x) { sum += v(sz, sy, sx); ++n; }
                            }
                    ov(z, y, x) = n ? sum / static_cast<f32>(n) : 0.0f;
                }
        return out;
    }

    Expected<void> write_level_(s64 lod, VolumeView<const f32> v) {
        const Extent3 vd = v.dims();
        const ChunkCoord ce = chunk_extent(lod);
        const s64 cs = fxvol_chunk_side;
        std::vector<f32> block(static_cast<usize>(cs * cs * cs));
        for (s64 cz = 0; cz < ce.z; ++cz)
            for (s64 cy = 0; cy < ce.y; ++cy)
                for (s64 cx = 0; cx < ce.x; ++cx) {
                    for (s64 z = 0; z < cs; ++z)
                        for (s64 y = 0; y < cs; ++y)
                            for (s64 x = 0; x < cs; ++x) {
                                const s64 vz = std::min(cz * cs + z, vd.z - 1);
                                const s64 vy = std::min(cy * cs + y, vd.y - 1);
                                const s64 vx = std::min(cx * cs + x, vd.x - 1);
                                block[static_cast<usize>((z * cs + y) * cs + x)] = v(vz, vy, vx);
                            }
                    auto w = write_chunk(lod, {cz, cy, cx}, block);
                    if (!w) return w;
                }
        return {};
    }

    Expected<void> map_() {
        void* p = ::mmap(nullptr, detail::kFxReserve, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_, 0);
        if (p == MAP_FAILED) return err(Errc::io_error, "mmap failed");
        base_ = static_cast<u8*>(p);
        return {};
    }
    Expected<void> ensure_(u64 end) {  // back the file (fallocate, never ftruncate) up to `end`
        if (end <= file_size_) return {};
        const u64 want = (end + detail::kFxGrow - 1) / detail::kFxGrow * detail::kFxGrow;
        if (want > detail::kFxReserve) return err(Errc::io_error, "archive exceeds reservation");
        const int rc = ::posix_fallocate(fd_, static_cast<off_t>(file_size_), static_cast<off_t>(want - file_size_));
        if (rc != 0) return err(Errc::io_error, "fallocate failed");
        file_size_ = want;
        return {};
    }
    Expected<u64> alloc_(u64 nbytes, bool node_zero_init) {  // bump-allocate at EOF (8-aligned)
        const u64 off = (cursor_ + 7) & ~7ull;
        if (auto e = ensure_(off + nbytes); !e) return std::unexpected(e.error());
        std::memset(base_ + off, node_zero_init ? 0 : 0xFF, nbytes);
        cursor_ = off + nbytes;
        return off;
    }

    u64* node_at_(u64 off) const { return reinterpret_cast<u64*>(base_ + off); }
    detail::FxSlot* leaf_at_(u64 off) const { return reinterpret_cast<detail::FxSlot*>(base_ + off); }

    // The writer sees its own uncommitted writes (horizon = cursor_); readers + crash recovery see only the
    // committed snapshot (horizon = committed_eof_). With COW, the in-progress tree lives in [committed_eof_,
    // cursor_), so a read-only reopen (horizon=committed_eof_) never follows a pointer into uncommitted data.
    u64 read_horizon_() const { return writable_ ? cursor_ : committed_eof_; }
    detail::FxSlot slot_read_(s64 lod, ChunkCoord c) const {  // walk L0→L1→leaf; out-of-bounds ⇒ ABSENT
        const detail::FxSlot absent{detail::kFxAbsent, 0};
        if (lod < 0 || lod >= static_cast<s64>(detail::kFxMaxLod)) return absent;
        const u64 hz = read_horizon_();
        const u64 root = lod_root_[lod];
        if (root == 0 || root + node_bytes_() > hz) return absent;
        const u64 m = detail::morton3(static_cast<u64>(c.z), static_cast<u64>(c.y), static_cast<u64>(c.x));
        const u32 i0 = (m >> 24) & 0xfffu, i1 = (m >> 12) & 0xfffu, i2 = m & 0xfffu;
        const u64 c1 = node_at_(root)[i0];
        if (c1 == 0 || c1 + node_bytes_() > hz) return absent;
        const u64 c2 = node_at_(c1)[i1];
        if (c2 == 0 || c2 + leaf_bytes_() > hz) return absent;
        return leaf_at_(c2)[i2];
    }

    // Ensure parent[idx] points at a CURRENT-transaction node (offset >= committed_eof_): allocate fresh if
    // absent, or COPY-ON-WRITE if it's a committed node (offset < committed_eof_), so committed snapshots are
    // never mutated. Returns the (current-transaction) child offset. `is_leaf` selects node vs leaf layout.
    Expected<u64> cow_child_(u64 parent_off, u32 idx, bool is_leaf) {
        const u64 ch = node_at_(parent_off)[idx];
        const u64 bytes = is_leaf ? leaf_bytes_() : node_bytes_();
        if (ch != 0 && ch >= committed_eof_) return ch;  // already this transaction → mutate in place
        auto o = alloc_(bytes, !is_leaf);
        if (!o) return std::unexpected(o.error());
        if (ch != 0) std::memcpy(base_ + *o, base_ + ch, bytes);  // COW the committed node
        node_at_(parent_off)[idx] = *o;  // base_ never moves, so parent_off stays valid across alloc_
        return *o;
    }
    Expected<detail::FxSlot*> slot_write_(s64 lod, ChunkCoord c) {  // COW the root→leaf path, return the slot
        const u64 m = detail::morton3(static_cast<u64>(c.z), static_cast<u64>(c.y), static_cast<u64>(c.x));
        const u32 i0 = (m >> 24) & 0xfffu, i1 = (m >> 12) & 0xfffu, i2 = m & 0xfffu;
        if (lod_root_[lod] == 0) {  // root: alloc fresh / COW (it has no parent slot to update)
            auto o = alloc_(node_bytes_(), true);
            if (!o) return std::unexpected(o.error());
            lod_root_[lod] = *o;
        } else if (lod_root_[lod] < committed_eof_) {
            auto o = alloc_(node_bytes_(), true);
            if (!o) return std::unexpected(o.error());
            std::memcpy(base_ + *o, base_ + lod_root_[lod], node_bytes_());
            lod_root_[lod] = *o;
        }
        auto l1 = cow_child_(lod_root_[lod], i0, false);
        if (!l1) return std::unexpected(l1.error());
        auto leaf = cow_child_(*l1, i1, true);
        if (!leaf) return std::unexpected(leaf.error());
        return &leaf_at_(*leaf)[i2];
    }

    void write_superblock_(u64 seq) {  // write the slot for `seq` (alternating A/B), then msync it
        const u64 b = ((seq - 1) & 1ull) * detail::kFxSuper;
        std::memcpy(base_ + b + 0, &fxvol_magic, 4);
        std::memcpy(base_ + b + 4, &fxvol_version, 4);
        std::memcpy(base_ + b + 8, &seq, 8);
        std::memcpy(base_ + b + 16, &committed_eof_, 8);
        std::memcpy(base_ + b + 24, &nlods_, 4);
        const u32 pad = 0;
        std::memcpy(base_ + b + 28, &pad, 4);
        std::memcpy(base_ + b + 32, &dims_.z, 8);
        std::memcpy(base_ + b + 40, &dims_.y, 8);
        std::memcpy(base_ + b + 48, &dims_.x, 8);
        std::memcpy(base_ + b + 56, &params_.q, 4);
        std::memcpy(base_ + b + 60, &params_.hf_exp, 4);
        std::memcpy(base_ + b + 64, &params_.dz_frac, 4);
        for (u32 l = 0; l < detail::kFxMaxLod; ++l) std::memcpy(base_ + b + detail::kFxLodRootOff + 8 * l, &lod_root_[l], 8);
        const u32 crc = detail::crc32c(base_ + b, detail::kFxSbCrcLen);
        std::memcpy(base_ + b + detail::kFxSbCrcLen, &crc, 4);
        if (base_) ::msync(base_ + b, detail::kFxSuper, MS_SYNC);
    }

    void steal_(VolumeArchive& o) {
        fd_ = o.fd_;
        base_ = o.base_;
        file_size_ = o.file_size_;
        cursor_ = o.cursor_;
        committed_eof_ = o.committed_eof_;
        commit_seq_ = o.commit_seq_;
        nlods_ = o.nlods_;
        std::memcpy(lod_root_, o.lod_root_, sizeof lod_root_);
        dims_ = o.dims_;
        params_ = o.params_;
        writable_ = o.writable_;
        path_ = std::move(o.path_);
        block_cache_ = std::move(o.block_cache_);
        o.fd_ = -1;
        o.base_ = nullptr;
        o.writable_ = false;
    }
    void release_() {
        // No auto-commit: persistence is explicit via commit()/close(). Dropping a writable archive without
        // close() = a crash — uncommitted writes are lost (and never corrupt the last committed snapshot).
        if (base_) {
            ::munmap(base_, detail::kFxReserve);
            base_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    std::string path_;
    Extent3 dims_{};
    DctParams params_{};
    int fd_ = -1;
    u8* base_ = nullptr;
    u64 file_size_ = 0;
    u64 cursor_ = 0;
    u64 committed_eof_ = 0;
    u64 commit_seq_ = 0;
    u32 nlods_ = 1;
    u64 lod_root_[detail::kFxMaxLod] = {};  // per-LOD radix-table root offset (0 = none)
    bool writable_ = false;
    mutable std::unique_ptr<BlockCache> block_cache_;
};

}  // namespace fenix::codec
