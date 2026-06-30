// codec/archive.hpp — the .fxvol container (v4). One file per volume: a sparse grid of 64³ chunks, each
// encoded by the DCT tile codec (a 64³ chunk = one tile of 4³=64 DCT blocks sharing rANS tables), indexed
// by a 3-level (12+12+12) compressed-Morton radix PAGE TABLE that lives IN the file and is mmap'd — so the
// resident RAM is the working set, never O(chunks) (a flat in-RAM index would be ~3.4 TB at the 2¹⁸/axis
// envelope). This is the volume store + the out-of-core IO substrate. See ADR 0006 + docs/design/fxvol-v4-layout.md.
//
// PHASES 1-2 (this file): single-LOD, LIVE (append-mmap) form, CRASH-SAFE. Phase 1 = Morton key + 3-level
// radix table over a huge MAP_NORESERVE reservation with fallocate growth + a bump allocator, sentinel-as-
// coverage slots. Phase 2 = a DOUBLE-BUFFERED crc32c superblock (two alternating slots, monotonic seq) +
// per-blob crc32c + data-before-pointer commit (msync the data, THEN the superblock — HDF5-SWMR flush
// ordering). On open we adopt the highest-seq slot that passes crc; everything past its committed_eof is a
// crashed half-append, read as ABSENT via the committed_eof bounds checks (so an uncommitted append never
// corrupts the committed state). STILL TODO (design note §9): decoded-tile cache (3), explicit LOD pyramid
// (4), SEALED coarse-first repack + S3 If-Match CAS (5-6), and full copy-on-write page-table versioning
// (Phase 2 versions {committed_eof, root}; the table is mutated in place, so point-in-time recovery to an
// OLDER commit may read a since-overwritten chunk as ABSENT — never corrupt). The public API is stable.
#pragma once

#include "codec/block_cache.hpp"
#include "codec/dct_block.hpp"
#include "core/core.hpp"
#include "core/types.hpp"

#include <array>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace fenix::codec {

enum class Coverage : u8 { Absent = 0, Zero = 1, Real = 2 };  // NOT_SURE / air / has-data

inline constexpr u32 fxvol_magic = 0x4c565846u;  // "FXVL"
inline constexpr u32 fxvol_version = 4;          // v4: mmap'd 3-level Morton radix page-table (ADR 0006)
inline constexpr s64 fxvol_chunk_side = 64;

namespace detail {
// crc32c (Castagnoli, poly 0x82F63B78), software table — torn-write detection for blobs + the superblock.
// Blob-granularity (not per-voxel), so a software table is plenty; a hardware path can replace it later.
inline u32 crc32c(const u8* p, usize n, u32 crc = 0) {
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

inline u64 morton_part1by2(u64 x) {  // spread low 21 bits to every 3rd bit (12 used at 2¹²/axis)
    x &= 0x1fffffull;
    x = (x | (x << 32)) & 0x1f00000000ffffull;
    x = (x | (x << 16)) & 0x1f0000ff0000ffull;
    x = (x | (x << 8)) & 0x100f00f00f00f00full;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
    x = (x | (x << 2)) & 0x1249249249249249ull;
    return x;
}
inline u64 morton3(u64 z, u64 y, u64 x) {  // compressed-Morton ZYX chunk key (halo locality)
    return (morton_part1by2(z) << 2) | (morton_part1by2(y) << 1) | morton_part1by2(x);
}

// Tri-state coverage IN the slot: off == kFxAbsent ⇒ ABSENT; else len == 0 ⇒ ZERO; else REAL (a blob at
// [off, off+len) followed by a u32 crc32c trailer at off+len).
inline constexpr u64 kFxAbsent = ~0ull;
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
inline constexpr u64 kFxGrow = 1ull << 26;       // grow the file 64 MiB at a time
inline constexpr u64 kFxNodeEnt = 4096;          // 2¹² entries per radix node / leaf
inline constexpr u64 kFxSuper = 4096;            // one superblock slot = one page
inline constexpr u64 kFxDataStart = 2 * kFxSuper;  // append region after the two superblock slots
inline constexpr u64 kFxSbCrcLen = 68;           // superblock fields covered by crc32c (crc stored at +68)
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
        a.fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (a.fd_ < 0) return err(Errc::io_error, "cannot create " + path);
        if (auto e = a.map_(); !e) return std::unexpected(e.error());
        if (auto e = a.ensure_(detail::kFxDataStart); !e) return std::unexpected(e.error());  // back both superblocks
        a.cursor_ = detail::kFxDataStart;  // append region starts after the two superblock slots
        a.committed_eof_ = detail::kFxDataStart;
        a.root_off_ = 0;  // no L0 node until the first write
        a.commit_seq_ = 1;
        a.write_superblock_(a.commit_seq_);  // a valid (empty) archive exists even if we crash right away
        a.writable_ = true;
        return a;
    }

    static Expected<VolumeArchive> open(const std::string& path) {
        VolumeArchive a;
        a.path_ = path;
        a.fd_ = ::open(path.c_str(), O_RDWR, 0644);
        if (a.fd_ < 0) return err(Errc::not_found, "cannot open " + path);
        const off_t sz = ::lseek(a.fd_, 0, SEEK_END);
        if (sz < static_cast<off_t>(detail::kFxDataStart)) return err(Errc::decode_error, "truncated .fxvol");
        a.file_size_ = static_cast<u64>(sz);
        if (auto e = a.map_(); !e) return std::unexpected(e.error());
        // Adopt the highest-commit_seq superblock slot that passes magic/version/crc (double-buffer recovery).
        int best = -1;
        u64 best_seq = 0;
        for (int s = 0; s < 2; ++s) {
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
        std::memcpy(&a.root_off_, a.base_ + b + 24, 8);
        std::memcpy(&a.dims_.z, a.base_ + b + 32, 8);
        std::memcpy(&a.dims_.y, a.base_ + b + 40, 8);
        std::memcpy(&a.dims_.x, a.base_ + b + 48, 8);
        std::memcpy(&a.params_.q, a.base_ + b + 56, 4);
        std::memcpy(&a.params_.hf_exp, a.base_ + b + 60, 4);
        std::memcpy(&a.params_.dz_frac, a.base_ + b + 64, 4);
        if (a.committed_eof_ > a.file_size_ || a.committed_eof_ < detail::kFxDataStart ||
            (a.root_off_ != 0 && a.root_off_ + node_bytes_() > a.committed_eof_))
            return err(Errc::decode_error, "corrupt .fxvol header");
        a.cursor_ = a.committed_eof_;
        a.writable_ = false;
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

    [[nodiscard]] Extent3 dims() const { return dims_; }
    [[nodiscard]] DctParams params() const { return params_; }
    [[nodiscard]] ChunkCoord chunk_extent() const {
        return {(dims_.z + fxvol_chunk_side - 1) / fxvol_chunk_side,
                (dims_.y + fxvol_chunk_side - 1) / fxvol_chunk_side,
                (dims_.x + fxvol_chunk_side - 1) / fxvol_chunk_side};
    }

    [[nodiscard]] Coverage coverage(ChunkCoord c) const {
        const detail::FxSlot s = slot_read_(c);
        if (s.off == detail::kFxAbsent) return Coverage::Absent;
        return s.len == 0 ? Coverage::Zero : Coverage::Real;
    }

    // Write one chunk_side³ block. An all-`fill` block is recorded as ZERO (no blob).
    Expected<void> write_chunk(ChunkCoord c, std::span<const f32> block, f32 fill = 0.0f) {
        if (!writable_) return err(Errc::io_error, "archive opened read-only");
        const s64 n = fxvol_chunk_side * fxvol_chunk_side * fxvol_chunk_side;
        if (static_cast<s64>(block.size()) != n) return err(Errc::invalid_argument, "block must be chunk_side³");
        bool all_fill = true;
        for (f32 v : block)
            if (v != fill) { all_fill = false; break; }
        auto sp = slot_write_(c);
        if (!sp) return std::unexpected(sp.error());
        if (all_fill) {
            **sp = {0, 0};  // ZERO (off != kFxAbsent, len == 0)
            return {};
        }
        auto payload = encode_tile_dct<f32>(block, fxvol_chunk_side / kDctN, params_);
        const u32 crc = detail::crc32c(payload.data(), payload.size());
        auto off = alloc_(payload.size() + 4, false);  // blob = [payload][u32 crc32c]
        if (!off) return std::unexpected(off.error());
        std::memcpy(base_ + *off, payload.data(), payload.size());
        std::memcpy(base_ + *off + payload.size(), &crc, 4);
        **sp = {*off, payload.size()};  // base_ is fixed, so the slot pointer is still valid after alloc_
        return {};
    }

    // Read one chunk_side³ block. ABSENT/ZERO -> filled with `fill`.
    [[nodiscard]] Expected<std::vector<f32>> read_chunk(ChunkCoord c, f32 fill = 0.0f) const {
        const usize n = static_cast<usize>(fxvol_chunk_side * fxvol_chunk_side * fxvol_chunk_side);
        const detail::FxSlot s = slot_read_(c);
        if (s.off == detail::kFxAbsent || s.len == 0) return std::vector<f32>(n, fill);
        if (s.off < detail::kFxDataStart || s.len + 4 > committed_eof_ || s.off > committed_eof_ - s.len - 4)
            return err(Errc::decode_error, "corrupt chunk offset");  // bound before deref (no UB)
        u32 crc_stored;
        std::memcpy(&crc_stored, base_ + s.off + s.len, 4);
        if (detail::crc32c(base_ + s.off, s.len) != crc_stored) return err(Errc::decode_error, "chunk crc mismatch");
        return decode_tile_dct<f32>(std::span<const u8>(base_ + s.off, s.len), fxvol_chunk_side / kDctN, params_);
    }

    // Convenience: tile a whole volume into chunks (edge-replicating partial chunks) and write it.
    Expected<void> write_volume(VolumeView<const f32> vol) {
        if (!(vol.dims() == dims_)) return err(Errc::invalid_argument, "dims mismatch");
        const ChunkCoord ce = chunk_extent();
        const s64 cs = fxvol_chunk_side;
        std::vector<f32> block(static_cast<usize>(cs * cs * cs));
        for (s64 cz = 0; cz < ce.z; ++cz)
            for (s64 cy = 0; cy < ce.y; ++cy)
                for (s64 cx = 0; cx < ce.x; ++cx) {
                    // Pad past the volume edge by REPLICATING the boundary voxel, not zero-filling: a zero
                    // step at the edge makes the block-DCT ring; a flat extension stays clean. read_volume
                    // crops the padding, so this only affects edge codec quality, never reconstructed values.
                    for (s64 z = 0; z < cs; ++z)
                        for (s64 y = 0; y < cs; ++y)
                            for (s64 x = 0; x < cs; ++x) {
                                const s64 vz = std::min(cz * cs + z, dims_.z - 1);
                                const s64 vy = std::min(cy * cs + y, dims_.y - 1);
                                const s64 vx = std::min(cx * cs + x, dims_.x - 1);
                                block[static_cast<usize>((z * cs + y) * cs + x)] = vol(vz, vy, vx);
                            }
                    auto w = write_chunk({cz, cy, cx}, block);
                    if (!w) return w;
                }
        return {};
    }

    // Convenience: reassemble the whole volume (cropping edge padding).
    [[nodiscard]] Expected<Volume<f32>> read_volume() const {
        Volume<f32> out = Volume<f32>::zeros(dims_);
        VolumeView<f32> ov = out.view();
        const ChunkCoord ce = chunk_extent();
        const s64 cs = fxvol_chunk_side;
        for (s64 cz = 0; cz < ce.z; ++cz)
            for (s64 cy = 0; cy < ce.y; ++cy)
                for (s64 cx = 0; cx < ce.x; ++cx) {
                    auto blk = read_chunk({cz, cy, cx});
                    if (!blk) return std::unexpected(blk.error());
                    for (s64 z = 0; z < cs; ++z)
                        for (s64 y = 0; y < cs; ++y)
                            for (s64 x = 0; x < cs; ++x) {
                                const s64 vz = cz * cs + z, vy = cy * cs + y, vx = cx * cs + x;
                                if (vz < dims_.z && vy < dims_.y && vx < dims_.x)
                                    ov(vz, vy, vx) = (*blk)[static_cast<usize>((z * cs + y) * cs + x)];
                            }
                }
        return out;
    }

    // ---- decoded-16³-chunk view (mmap-the-archive-as-an-array) ----
    // Set the decoded-chunk cache byte budget (default 256 MiB, created lazily on first access).
    void reserve_cache(u64 budget_bytes, int shards = 16) { block_cache_ = std::make_unique<BlockCache>(budget_bytes, shards); }
    [[nodiscard]] u64 cache_hits() const { return block_cache_ ? block_cache_->hits() : 0; }
    [[nodiscard]] u64 cache_misses() const { return block_cache_ ? block_cache_->misses() : 0; }
    [[nodiscard]] u64 cache_bytes() const { return block_cache_ ? block_cache_->bytes() : 0; }

    // Fetch a decoded 16³ chunk by its block-grid coord (= voxel/16). On a miss, the containing 64³ tile
    // is decoded once (the atomic decode unit) and ALL 64 of its 16³ chunks are cached, so the decode
    // amortizes across the tile while eviction stays 16³-granular. Returns a pinned shared_ptr.
    [[nodiscard]] Expected<BlockCache::Ref> block16(ChunkCoord bc) const {
        if (!block_cache_) block_cache_ = std::make_unique<BlockCache>(detail::kFxDefaultCache);
        const u64 key = detail::morton3(static_cast<u64>(bc.z), static_cast<u64>(bc.y), static_cast<u64>(bc.x));
        if (auto r = block_cache_->get(key)) return r;
        constexpr s64 BS = kDctN, NB = fxvol_chunk_side / kDctN;  // 16³ chunks, 4³ per 64³ tile
        const ChunkCoord tc{bc.z / NB, bc.y / NB, bc.x / NB};
        auto tile = read_chunk(tc);  // 64³ decoded f32 (ZYX contiguous); ABSENT/ZERO -> zeros
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
                    const u64 k = detail::morton3(static_cast<u64>(tc.z * NB + sz), static_cast<u64>(tc.y * NB + sy), static_cast<u64>(tc.x * NB + sx));
                    BlockCache::Ref cr = blk;
                    block_cache_->put(k, cr);
                    if (k == key) want = cr;
                }
        if (!want) return err(Errc::decode_error, "block16 key not produced");
        return want;
    }

    // Read a single voxel through the decoded-chunk cache.
    [[nodiscard]] Expected<f32> voxel(s64 z, s64 y, s64 x) const {
        auto r = block16({z / kDctN, y / kDctN, x / kDctN});
        if (!r) return std::unexpected(r.error());
        return (**r)[static_cast<usize>(((z % kDctN) * kDctN + (y % kDctN)) * kDctN + (x % kDctN))];
    }

    // Durable checkpoint: msync the data region, THEN write+msync the next superblock slot (data-before-
    // pointer). Safe to call repeatedly mid-session; a crash recovers the last committed state.
    Expected<void> commit() {
        if (!writable_) return {};
        committed_eof_ = cursor_;
        if (base_ && committed_eof_ > detail::kFxDataStart)
            ::msync(base_ + detail::kFxDataStart, committed_eof_ - detail::kFxDataStart, MS_SYNC);
        ++commit_seq_;
        write_superblock_(commit_seq_);  // writes the inactive slot + msyncs it
        return {};
    }

    // Commit and seal read-only.
    Expected<void> close() {
        if (!writable_) return {};
        auto e = commit();
        writable_ = false;
        return e;
    }

private:
    static u64 node_bytes_() { return detail::kFxNodeEnt * sizeof(u64); }                  // radix node = 4096 × u64
    static u64 leaf_bytes_() { return detail::kFxNodeEnt * sizeof(detail::FxSlot); }        // leaf = 4096 × slot

    Expected<void> map_() {
        void* p = ::mmap(nullptr, detail::kFxReserve, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_, 0);
        if (p == MAP_FAILED) return err(Errc::io_error, "mmap failed");
        base_ = static_cast<u8*>(p);
        return {};
    }
    // Ensure the file is backed (real blocks via fallocate, never ftruncate) up to `end` bytes.
    Expected<void> ensure_(u64 end) {
        if (end <= file_size_) return {};
        const u64 want = (end + detail::kFxGrow - 1) / detail::kFxGrow * detail::kFxGrow;
        if (want > detail::kFxReserve) return err(Errc::io_error, "archive exceeds reservation");
        const int rc = ::posix_fallocate(fd_, static_cast<off_t>(file_size_), static_cast<off_t>(want - file_size_));
        if (rc != 0) return err(Errc::io_error, "fallocate failed");
        file_size_ = want;
        return {};
    }
    // Bump-allocate `nbytes` (8-aligned) at EOF; node=zero-init (children absent), leaf=0xFF-init (slots absent).
    Expected<u64> alloc_(u64 nbytes, bool node_zero_init) {
        const u64 off = (cursor_ + 7) & ~7ull;
        if (auto e = ensure_(off + nbytes); !e) return std::unexpected(e.error());
        std::memset(base_ + off, node_zero_init ? 0 : 0xFF, nbytes);
        cursor_ = off + nbytes;
        return off;
    }

    u64* node_at_(u64 off) const { return reinterpret_cast<u64*>(base_ + off); }
    detail::FxSlot* leaf_at_(u64 off) const { return reinterpret_cast<detail::FxSlot*>(base_ + off); }

    // Read path: walk L0→L1→leaf; any absent/out-of-bounds child ⇒ ABSENT (robust on corrupt/uncommitted bytes).
    detail::FxSlot slot_read_(ChunkCoord c) const {
        const detail::FxSlot absent{detail::kFxAbsent, 0};
        if (root_off_ == 0) return absent;
        const u64 m = detail::morton3(static_cast<u64>(c.z), static_cast<u64>(c.y), static_cast<u64>(c.x));
        const u32 i0 = (m >> 24) & 0xfffu, i1 = (m >> 12) & 0xfffu, i2 = m & 0xfffu;
        if (root_off_ + node_bytes_() > committed_eof_) return absent;
        const u64 c1 = node_at_(root_off_)[i0];
        if (c1 == 0 || c1 + node_bytes_() > committed_eof_) return absent;
        const u64 c2 = node_at_(c1)[i1];
        if (c2 == 0 || c2 + leaf_bytes_() > committed_eof_) return absent;
        return leaf_at_(c2)[i2];
    }

    // Write path: walk L0→L1→leaf, allocating missing nodes/leaf on the way down. Returns the slot ptr.
    Expected<detail::FxSlot*> slot_write_(ChunkCoord c) {
        const u64 m = detail::morton3(static_cast<u64>(c.z), static_cast<u64>(c.y), static_cast<u64>(c.x));
        const u32 i0 = (m >> 24) & 0xfffu, i1 = (m >> 12) & 0xfffu, i2 = m & 0xfffu;
        if (root_off_ == 0) {
            auto o = alloc_(node_bytes_(), true);
            if (!o) return std::unexpected(o.error());
            root_off_ = *o;
        }
        if (node_at_(root_off_)[i0] == 0) {
            auto o = alloc_(node_bytes_(), true);
            if (!o) return std::unexpected(o.error());
            node_at_(root_off_)[i0] = *o;
        }
        const u64 l1 = node_at_(root_off_)[i0];
        if (node_at_(l1)[i1] == 0) {
            auto o = alloc_(leaf_bytes_(), false);
            if (!o) return std::unexpected(o.error());
            node_at_(l1)[i1] = *o;
        }
        return &leaf_at_(node_at_(l1)[i1])[i2];
    }

    // Write a superblock into the slot for `seq` (alternating A/B), then msync it (the commit-pointer flush).
    void write_superblock_(u64 seq) {
        const u64 b = ((seq - 1) & 1ull) * detail::kFxSuper;
        std::memcpy(base_ + b + 0, &fxvol_magic, 4);
        std::memcpy(base_ + b + 4, &fxvol_version, 4);
        std::memcpy(base_ + b + 8, &seq, 8);
        std::memcpy(base_ + b + 16, &committed_eof_, 8);
        std::memcpy(base_ + b + 24, &root_off_, 8);
        std::memcpy(base_ + b + 32, &dims_.z, 8);
        std::memcpy(base_ + b + 40, &dims_.y, 8);
        std::memcpy(base_ + b + 48, &dims_.x, 8);
        std::memcpy(base_ + b + 56, &params_.q, 4);
        std::memcpy(base_ + b + 60, &params_.hf_exp, 4);
        std::memcpy(base_ + b + 64, &params_.dz_frac, 4);
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
        root_off_ = o.root_off_;
        commit_seq_ = o.commit_seq_;
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
        if (writable_) (void)commit();  // best-effort durable flush on graceful teardown
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
    u64 root_off_ = 0;
    u64 commit_seq_ = 0;
    bool writable_ = false;
    mutable std::unique_ptr<BlockCache> block_cache_;  // decoded-16³-chunk cache (lazily created)
};

}  // namespace fenix::codec
