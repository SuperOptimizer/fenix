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
#include <atomic>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <fenix_win_archive.h>  // Win32 backing store for the out-of-core mapping (see map_/release_)
#endif

namespace fenix::codec {

enum class Coverage : u8 { Absent = 0, Zero = 1, Real = 2 };  // NOT_SURE / air / has-data

inline constexpr u32 fxvol_magic = 0x4c565846u;  // "FXVL"
inline constexpr u32 fxvol_version = 5;          // v5: + source-dtype tag (native-dtype read path; no f32 widen)
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
// 32 TiB: a single MAP_NORESERVE VA reservation, not a RAM commitment (pages are backed only as the file
// grows via fallocate) — VA is cheap, so size for headroom rather than measured need. 1 TiB (the original
// Phase-1 constant) hard-capped export-scroll: PHerc Paris 3 (2^18/axis, ~1.12e14 voxels u8) plausibly
// exceeds 1 TiB even compressed once the LOD pyramid + COW page-table orphan bloat are counted, so a
// multi-day whole-scroll export could fail mid-run with no way to grow (mmap offsets must stay fixed).
// 32 TiB comfortably clears that worst case while staying well inside the 47-bit (128 TiB) x86-64/arm64
// user VA envelope with room for the rest of the process's address space; raise again (or move to a
// dims-derived reservation computed at create()/open() time) if a real archive ever approaches it.
inline constexpr u64 kFxReserve = 1ull << 45;  // 32 TiB
#endif

inline constexpr u64 kFxGrow = 1ull << 26;  // grow the file 64 MiB at a time

// Dims+q-derived VA reservation: a hard ceiling on the FILE (compressed) size computed from a
// conservative LOWER bound on the measured DCT compression curve (bench 2026-06/07 on real CT:
// q2 ~10-24x, q8 ~24-52x, q32 ~88-141x) so the estimate can only overshoot, never strand a run:
// ratio_lb(q) = clamp(1.5 + 1.6q, 2, 48). raw(u8-native — f32-on-disk is banned project-wide)
// / ratio_lb x1.25 (LOD pyramid + COW page-table orphans) + a page-table budget + slack. VA is
// free (MAP_NORESERVE), so overshooting by ~2-4x costs nothing, while a 2048^3 crop reserves a
// few GiB instead of the 32 TiB blanket cap (which remains the clamp for whole-scroll archives).
inline u64 reserve_for(Extent3 dims, f32 q) {
    const f64 raw = static_cast<f64>(dims.count());
    const f64 ratio = std::clamp(1.5 + 1.6 * static_cast<f64>(q > 0.0f ? q : 1.0f), 2.0, 48.0);
    const u64 nchunks = static_cast<u64>((dims.z + 63) / 64) * static_cast<u64>((dims.y + 63) / 64) *
                        static_cast<u64>((dims.x + 63) / 64);
    const u64 nodes = (64ull << 20) + nchunks * 32;
    u64 r = nodes + static_cast<u64>(raw / ratio * 1.25) + (128ull << 20);
    r = (r + kFxGrow - 1) / kFxGrow * kFxGrow;
    return std::clamp<u64>(r, kFxGrow, kFxReserve);
}
inline constexpr u64 kFxNodeEnt = 4096;            // 2¹² entries per radix node / leaf
inline constexpr u64 kFxSuper = 4096;              // one superblock slot = one page
inline constexpr u64 kFxDataStart = 2 * kFxSuper;  // append region after the two superblock slots
inline constexpr u64 kFxMaxLod = 20;               // 2¹⁸→2⁶ needs 13; 20 is generous headroom
inline constexpr u64 kFxLodRootOff = 68;           // superblock: per-LOD root-offset array starts here
inline constexpr u64 kFxDataOffField = kFxLodRootOff + 8 * kFxMaxLod;  // u64: SEALED blob-region start (0 = LIVE)
inline constexpr u64 kFxSbCrcLen = kFxDataOffField + 8;  // crc32c covers [0, kFxSbCrcLen); stored at +kFxSbCrcLen
inline constexpr u64 kFxDefaultCache = 256ull << 20;     // default decoded-16³-chunk cache budget (256 MiB)
}  // namespace detail

// The .fxvol archive (v4). Owns one mmap'd file; move-only (RAII over the fd + mapping).
class VolumeArchive {
  public:
    static Expected<VolumeArchive>
    create(const std::string& path, Extent3 dims, DctParams bp, DType src_dtype = DType::u8) {
        VolumeArchive a;
        a.path_ = path;
        a.dims_ = dims;
        a.params_ = bp;
        a.src_dtype_ = src_dtype;
        a.nlods_ = a.plan_nlods_(dims);
        a.reserve_ = detail::reserve_for(dims, bp.q);
#ifdef _WIN32
        a.fd_ = fenix_win_open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);  // share-delete (unlink-while-open)
#else
        a.fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
#endif
        if (a.fd_ < 0) return err(Errc::io_error, "cannot create " + path);
        if (auto e = a.map_(); !e) return std::unexpected(e.error());
        if (auto e = a.ensure_(detail::kFxDataStart); !e) return std::unexpected(e.error());
        a.cursor_ = detail::kFxDataStart;
        a.committed_eof_ = detail::kFxDataStart;
        a.commit_seq_ = 1;
        a.write_superblock_(a.commit_seq_);  // a valid (empty) archive exists even if we crash right away
        a.writable_ = true;
        a.write_opened_ = true;
        a.block_cache_ = std::make_unique<BlockCache>(detail::kFxDefaultCache);  // eager: see open()'s comment
        return a;
    }

    // open(path) — read-only. open(path, true) — read/write (append + commit; COW preserves the prior
    // committed snapshot, so a crash mid-append recovers the last committed state).
    static Expected<VolumeArchive> open(const std::string& path, bool writable = false) {
        VolumeArchive a;
        a.path_ = path;
#ifdef _WIN32
        a.fd_ = fenix_win_open(path.c_str(), O_RDWR, 0644);
#else
        a.fd_ = ::open(path.c_str(), O_RDWR, 0644);
#endif
        if (a.fd_ < 0) return err(Errc::not_found, "cannot open " + path);
        const off_t sz = ::lseek(a.fd_, 0, SEEK_END);
        if (sz < static_cast<off_t>(detail::kFxDataStart)) return err(Errc::decode_error, "truncated .fxvol");
        a.file_size_ = static_cast<u64>(sz);
        {
            // Size the VA reservation from the superblock's dims+q BEFORE mapping (pread — the map
            // doesn't exist yet). Take the best-crc slot's values; a file with no valid slot fails
            // below anyway, so the blanket cap is a fine reservation for that path. The reservation
            // must also cover the CURRENT file (readable even if larger than the estimate).
            u8 sb[2 * detail::kFxSuper];
            if (::pread(a.fd_, sb, sizeof sb, 0) == static_cast<ssize_t>(sizeof sb)) {
                for (int slot = 0; slot < 2; ++slot) {
                    const u8* p = sb + static_cast<usize>(slot) * detail::kFxSuper;
                    u32 magic, version, crc_stored;
                    std::memcpy(&magic, p + 0, 4);
                    std::memcpy(&version, p + 4, 4);
                    std::memcpy(&crc_stored, p + detail::kFxSbCrcLen, 4);
                    if (magic != fxvol_magic || version != fxvol_version) continue;
                    if (detail::crc32c(p, detail::kFxSbCrcLen) != crc_stored) continue;
                    Extent3 d;
                    f32 q;
                    std::memcpy(&d.z, p + 32, 8);
                    std::memcpy(&d.y, p + 40, 8);
                    std::memcpy(&d.x, p + 48, 8);
                    std::memcpy(&q, p + 56, 4);
                    constexpr s64 kMaxAxis = 1ll << 18;
                    if (d.z <= 0 || d.z > kMaxAxis || d.y <= 0 || d.y > kMaxAxis || d.x <= 0 || d.x > kMaxAxis)
                        continue;  // corrupt dims: keep the blanket cap; the full parse below rejects
                    a.reserve_ = std::max(detail::reserve_for(d, q),
                                          (a.file_size_ + detail::kFxGrow - 1) / detail::kFxGrow * detail::kFxGrow);
                    a.reserve_ = std::min<u64>(std::max(a.reserve_, a.file_size_), detail::kFxReserve);
                    break;
                }
            }
        }
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
            if (best < 0 || seq > best_seq) {
                best = s;
                best_seq = seq;
            }
        }
        if (best < 0) return err(Errc::decode_error, "no valid superblock");
        const u64 b = static_cast<u64>(best) * detail::kFxSuper;
        std::memcpy(&a.commit_seq_, a.base_ + b + 8, 8);
        std::memcpy(&a.committed_eof_, a.base_ + b + 16, 8);
        std::memcpy(&a.nlods_, a.base_ + b + 24, 4);
        u32 dt = 0;
        std::memcpy(&dt, a.base_ + b + 28, 4);
        if ((dt & 0xff) > static_cast<u32>(DType::f32)) return err(Errc::decode_error, "bad src dtype");
        a.src_dtype_ = static_cast<DType>(static_cast<u8>(dt & 0xff));
        std::memcpy(&a.dims_.z, a.base_ + b + 32, 8);
        std::memcpy(&a.dims_.y, a.base_ + b + 40, 8);
        std::memcpy(&a.dims_.x, a.base_ + b + 48, 8);
        std::memcpy(&a.params_.q, a.base_ + b + 56, 4);
        std::memcpy(&a.params_.hf_exp, a.base_ + b + 60, 4);
        std::memcpy(&a.params_.dz_frac, a.base_ + b + 64, 4);
        if (a.nlods_ < 1 || a.nlods_ > detail::kFxMaxLod) return err(Errc::decode_error, "bad nlods");
        // dims_/params_.q come straight from the superblock with no range check beyond the whole-block
        // CRC (which only catches random corruption, not a crafted file): an out-of-envelope dims_ later
        // drives Volume<T>::zeros(vd) in read_volume_as into an absurd allocation (or signed-arithmetic
        // trouble), and a non-finite/non-positive q corrupts every dequant downstream. Reject both here.
        constexpr s64 kMaxAxis = 1ll << 18;  // the project's documented envelope (root CLAUDE.md §2.4)
        if (a.dims_.z <= 0 || a.dims_.z > kMaxAxis || a.dims_.y <= 0 || a.dims_.y > kMaxAxis || a.dims_.x <= 0 ||
            a.dims_.x > kMaxAxis)
            return err(Errc::decode_error, "dims out of range");
        if (!(a.params_.q > 0.0f) || !std::isfinite(a.params_.q))  // !(x>0) also catches NaN
            return err(Errc::decode_error, "bad quant step");
        for (u32 l = 0; l < detail::kFxMaxLod; ++l)
            std::memcpy(&a.lod_root_[l], a.base_ + b + detail::kFxLodRootOff + 8 * l, 8);
        std::memcpy(&a.data_off_, a.base_ + b + detail::kFxDataOffField, 8);
        if (a.committed_eof_ > a.file_size_ || a.committed_eof_ < detail::kFxDataStart)
            return err(Errc::decode_error, "corrupt eof");
        for (u32 l = 0; l < a.nlods_; ++l)  // non-overflowing (see in_bounds_): a huge corrupt root must not wrap
            if (a.lod_root_[l] != 0 && !in_bounds_(a.lod_root_[l], node_bytes_(), a.committed_eof_))
                return err(Errc::decode_error, "corrupt root");
        a.cursor_ = a.committed_eof_;
        a.last_msync_ = a.committed_eof_;  // the recovered committed state is already durable
        a.writable_ = writable;
        a.write_opened_ = writable;
        // Eagerly construct the block cache here (cheap: empty shards, a few hundred bytes) rather than
        // lazily in block16() — block16() is const and gather_box_f32 is documented thread-safe, so a
        // lazy `if (!block_cache_) block_cache_ = make_unique<...>` on a `mutable unique_ptr` under
        // concurrent first calls is an unsynchronized test-and-assign: two racing constructions, one
        // destroying the BlockCache the other is still using (data race / UAF). Constructing it in
        // create()/open() means every archive has a valid cache before any thread can call block16(),
        // and reserve_cache() only replaces it before that concurrent use begins (documented below).
        a.block_cache_ = std::make_unique<BlockCache>(detail::kFxDefaultCache);
        return a;
    }

    VolumeArchive() = default;
    VolumeArchive(const VolumeArchive&) = delete;
    VolumeArchive& operator=(const VolumeArchive&) = delete;

    VolumeArchive(VolumeArchive&& o) noexcept { steal_(o); }

    VolumeArchive& operator=(VolumeArchive&& o) noexcept {
        if (this != &o) {
            release_();
            steal_(o);
        }
        return *this;
    }

    ~VolumeArchive() { release_(); }

    [[nodiscard]] Extent3 dims() const { return dims_; }  // LOD 0 dims

    [[nodiscard]] DctParams params() const { return params_; }

    [[nodiscard]] u32 nlods() const { return nlods_; }

    [[nodiscard]] Extent3 dims_at(s64 lod) const {  // dims of LOD level `lod` (2× per octave)
        Extent3 d = dims_;
        for (s64 i = 0; i < lod; ++i) d = {(d.z + 1) / 2, (d.y + 1) / 2, (d.x + 1) / 2};
        return d;
    }

    [[nodiscard]] ChunkCoord chunk_extent(s64 lod = 0) const {
        const Extent3 d = dims_at(lod);
        return {(d.z + fxvol_chunk_side - 1) / fxvol_chunk_side,
                (d.y + fxvol_chunk_side - 1) / fxvol_chunk_side,
                (d.x + fxvol_chunk_side - 1) / fxvol_chunk_side};
    }

    [[nodiscard]] Coverage coverage(s64 lod, ChunkCoord c) const {
        const detail::FxSlot s = slot_read_(lod, c);
        if (s.off == detail::kFxAbsent) return Coverage::Absent;
        return s.len == 0 ? Coverage::Zero : Coverage::Real;
    }

    [[nodiscard]] Coverage coverage(ChunkCoord c) const { return coverage(0, c); }

    // Write one chunk_side³ block into LOD `lod`. An all-`fill` block is recorded as ZERO (no blob). The
    // element type T flows straight into the DCT codec (encode_tile_dct<T> widens to f32 per 16³ block —
    // a u8 source is NEVER inflated to an f32 buffer here). The DCT bitstream is dtype-independent (T only
    // drives widen-in/clamp-out), so a u8 tile still reads back through read_chunk<f32> (finalize/export).
    template <class T> Expected<void> write_chunk_(s64 lod, ChunkCoord c, std::span<const T> block, T fill) {
        if (!writable_) return err(Errc::io_error, "archive opened read-only");
        if (lod < 0 || lod >= static_cast<s64>(detail::kFxMaxLod)) return err(Errc::invalid_argument, "bad lod");
        const s64 n = fxvol_chunk_side * fxvol_chunk_side * fxvol_chunk_side;
        if (static_cast<s64>(block.size()) != n) return err(Errc::invalid_argument, "block must be chunk_side³");
        bool all_fill = true;
        for (T v : block)
            if (v != fill) {
                all_fill = false;
                break;
            }
        auto sp = slot_write_(lod, c);
        if (!sp) return std::unexpected(sp.error());
        if (all_fill) {
            **sp = {0, 0};
            return {};
        }
        auto payload = encode_tile_dct<T>(block, fxvol_chunk_side / kDctN, params_);
        const u32 crc = detail::crc32c(payload.data(), payload.size());
        auto off = alloc_(payload.size() + 4, false);  // blob = [payload][u32 crc32c]
        if (!off) return std::unexpected(off.error());
        std::memcpy(base_ + *off, payload.data(), payload.size());
        std::memcpy(base_ + *off + payload.size(), &crc, 4);
        **sp = {*off, payload.size()};  // base_ is fixed → the slot pointer is still valid after alloc_
        return {};
    }

    Expected<void> write_chunk(s64 lod, ChunkCoord c, std::span<const f32> block, f32 fill = 0.0f) {
        return write_chunk_<f32>(lod, c, block, fill);
    }

    Expected<void> write_chunk(s64 lod, ChunkCoord c, std::span<const u8> block, u8 fill = 0) {
        return write_chunk_<u8>(lod, c, block, fill);
    }

    Expected<void> write_chunk(ChunkCoord c, std::span<const f32> block, f32 fill = 0.0f) {
        return write_chunk_<f32>(0, c, block, fill);
    }

    // Read one chunk_side³ block decoded to dtype T. ABSENT/ZERO -> filled with `fill`. The DCT bitstream is
    // dtype-independent (T only drives the codec's widen-in/clamp-out), so a u8 tile decodes DIRECTLY back to
    // u8 here — no f32 intermediate volume. This is the primitive; pick T = the archive's src_dtype to stay
    // native (scrolls: u8), or T = f32 for a widened read (finalize/export/LOD build).
    template <class T> [[nodiscard]] Expected<std::vector<T>> read_chunk_as(s64 lod, ChunkCoord c, T fill = T{}) const {
        const usize n = static_cast<usize>(fxvol_chunk_side * fxvol_chunk_side * fxvol_chunk_side);
        const detail::FxSlot s = slot_read_(lod, c);
        if (s.off == detail::kFxAbsent || s.len == 0) return std::vector<T>(n, fill);
        const u64 hz = read_horizon_();
        // Non-overflowing bounds check: page-table leaves carry no CRC (unlike blobs/superblock), so a
        // corrupt/adversarial s.off/s.len must be range-checked BEFORE any subtraction — `s.off <= hz`
        // first, THEN compare s.len against the remaining span (hz - s.off), so neither subtraction can
        // wrap. The old `s.len + 4 > hz || s.off > hz - s.len - 4` form wraps for a huge s.len (e.g.
        // 2^64-4), passing both checks and driving crc32c() ~2^64 bytes past the mapping (SIGSEGV).
        if (s.off < detail::kFxDataStart || s.off > hz || s.len > hz - s.off || hz - s.off - s.len < 4)
            return err(Errc::decode_error, "corrupt chunk offset");
        u32 crc_stored;
        std::memcpy(&crc_stored, base_ + s.off + s.len, 4);
        if (detail::crc32c(base_ + s.off, s.len) != crc_stored) return err(Errc::decode_error, "chunk crc mismatch");
        return decode_tile_dct<T>(std::span<const u8>(base_ + s.off, s.len),
                                  fxvol_chunk_side / kDctN,
                                  params_);  // Expected: propagates corrupt-payload errors
    }

    // f32 convenience (finalize/export/LOD build widen deliberately).
    [[nodiscard]] Expected<std::vector<f32>> read_chunk(s64 lod, ChunkCoord c, f32 fill = 0.0f) const {
        return read_chunk_as<f32>(lod, c, fill);
    }

    [[nodiscard]] Expected<std::vector<f32>> read_chunk(ChunkCoord c, f32 fill = 0.0f) const {
        return read_chunk(0, c, fill);
    }

    [[nodiscard]] DType src_dtype() const { return src_dtype_; }

    // Tile a whole volume into LOD 0 AND build the full LOD pyramid (global 2× box downsample → retile per
    // octave, down to a single 64³ chunk). Edge-replicates partial chunks so the block-DCT doesn't ring.
    //
    // Templated on the SOURCE dtype so a native u8/u16 volume is encoded WITHOUT first widening the whole
    // volume to f32: LOD 0 reads straight from the `T` view (each 64³ tile is widened per-chunk inside the
    // codec, which is unavoidable for a float DCT — but the transient f32 copy is one tile, not the whole
    // volume). The LOD pyramid is built in f32 from the (already tiny) decimated levels. This is the fix for
    // the u8→f32 blow-up on ingest (a 2048³ crop is 8 GiB u8, not 34 GiB f32).
    template <class T> Expected<void> write_volume(VolumeView<const T> vol) {
        if (!(vol.dims() == dims_)) return err(Errc::invalid_argument, "dims mismatch");
        src_dtype_ = dtype_traits<T>::id;  // reads reconstruct THIS dtype natively (no f32 widen)
        if (auto e = write_level_<T>(0, vol); !e) return e;
        // Downsample once from the source dtype into f32, then continue the pyramid in f32. `cur` is the
        // decimated level (≤ 1/8 the voxels of LOD 0), so holding it in f32 is cheap.
        if (fits_one_chunk_(vol.dims())) {
            nlods_ = 1;
            return {};
        }
        Volume<f32> cur = downsample2_<T>(vol);
        s64 lod = 1;
        if (auto e = write_level_<f32>(lod, cur.view()); !e) return e;
        while (!fits_one_chunk_(cur.dims()) && lod + 1 < static_cast<s64>(detail::kFxMaxLod)) {
            Volume<f32> next = downsample2_<f32>(cur.view());
            ++lod;
            if (auto e = write_level_<f32>(lod, next.view()); !e) return e;
            cur = std::move(next);
        }
        nlods_ = static_cast<u32>(lod + 1);
        return {};
    }

    // Convenience overload: f32 callers (pipeline stages) keep working unchanged.
    Expected<void> write_volume(VolumeView<const f32> vol) { return write_volume<f32>(vol); }

    // Out-of-core LOD pyramid build: populate LODs 1..N from an already-written LOD 0 by 2× box-downsampling
    // each octave IN PLACE — no whole-volume buffer (RAM is O(cores · tile)), so it works on the 70k³ scroll
    // exports that write_volume can't hold. A 64³ output chunk at LOD L+1 maps to its 8 aligned source chunks
    // at LOD L (a 128³ region); the 2³ box is disjoint so NO halo is read. Downsampling matches downsample2_
    // (mean of the IN-BOUNDS source voxels — beyond-dims positions are excluded, not the codec's edge-replica).
    // All-absent sources → the output chunk stays ABSENT (the sparse air structure is preserved); all-air →
    // ZERO; else encoded (f32, like write_volume's LOD1+). Cascades level-on-STORED-level (each LOD from the
    // decoded previous one, not a pristine f32 buffer as write_volume does) — a small extra loss, fine for the
    // preview/LOD levels (tolerance-only). Two-phase per batch (parallel decode+downsample+encode → serial
    // commit), commit=N crash-safe checkpoints. Resumable: a dst chunk already Real/Zero is skipped.
    template <class Prog>
    Expected<void> build_pyramid_ooc(s64 commit_every, Prog&& progress) {
        if (!writable_) return err(Errc::io_error, "archive opened read-only");
        data_off_ = 0;  // a LIVE archive being extended — correct a stale SEALED tag if one is present
        const s64 cs = fxvol_chunk_side, RS = 2 * cs;  // output tile side / source region side (128)
        s64 lod = 0;
        while (!fits_one_chunk_(dims_at(lod)) && lod + 1 < static_cast<s64>(detail::kFxMaxLod)) {
            const s64 src = lod, dst = lod + 1;
            const Extent3 sd = dims_at(src);
            const ChunkCoord sce = chunk_extent(src);
            const ChunkCoord ce = chunk_extent(dst);
            const s64 nout = ce.z * ce.y * ce.x;
            struct Enc { std::vector<u8> payload; int state = 0; };  // state: -1 skip, 0 absent, 1 zero, 2 real
            for (s64 batch = 0; batch < nout; batch += commit_every) {
                const s64 hi = std::min(batch + commit_every, nout);
                std::vector<Enc> enc(static_cast<usize>(hi - batch));
                std::atomic<bool> failed{false};
                std::mutex emu;
                Error ferr = err(Errc::decode_error, "build_pyramid_ooc").error();
                parallel_for(batch, hi, [&](s64 i) {
                    if (failed.load(std::memory_order_relaxed)) return;
                    const s64 cx = i % ce.x, cy = (i / ce.x) % ce.y, cz = i / (ce.x * ce.y);
                    Enc& e = enc[static_cast<usize>(i - batch)];
                    if (coverage(dst, {cz, cy, cx}) != Coverage::Absent) { e.state = -1; return; }  // resume-skip
                    // Phase A: probe the 8 source octants' coverage (cheap, no decode/alloc). Record the REAL
                    // ones; absent-everywhere bails before any 128³ allocation (most air chunks take this path).
                    struct Oct { ChunkCoord c; s64 ez, ey, ex; };
                    Oct real_oct[8];
                    int nreal = 0, present = 0;
                    for (s64 ez = 0; ez < 2; ++ez)
                        for (s64 ey = 0; ey < 2; ++ey)
                            for (s64 ex = 0; ex < 2; ++ex) {
                                const ChunkCoord scc{2 * cz + ez, 2 * cy + ey, 2 * cx + ex};
                                if (scc.z >= sce.z || scc.y >= sce.y || scc.x >= sce.x) continue;
                                const Coverage sv = coverage(src, scc);
                                if (sv == Coverage::Absent) continue;
                                ++present;
                                if (sv == Coverage::Real) real_oct[nreal++] = {scc, ez, ey, ex};
                            }
                    if (present == 0) { e.state = 0; return; }  // fully absent → keep dst ABSENT (sparse)
                    // Phase B: assemble the 128³ region from the REAL octants (air/zero octants stay 0).
                    std::vector<f32> region(static_cast<usize>(RS * RS * RS), 0.0f);
                    for (int k = 0; k < nreal; ++k) {
                        auto blk = read_chunk_as<f32>(src, real_oct[k].c, 0.0f);
                        if (!blk) { bool f = false; if (failed.compare_exchange_strong(f, true)) { std::lock_guard<std::mutex> lk(emu); ferr = blk.error(); } return; }
                        const s64 oz = real_oct[k].ez * cs, oy = real_oct[k].ey * cs, ox = real_oct[k].ex * cs;
                        for (s64 z = 0; z < cs; ++z)
                            for (s64 y = 0; y < cs; ++y) {
                                const usize drow = static_cast<usize>(((oz + z) * RS + (oy + y)) * RS + ox);
                                const usize srow = static_cast<usize>((z * cs + y) * cs);
                                for (s64 x = 0; x < cs; ++x) region[drow + static_cast<usize>(x)] = (*blk)[srow + static_cast<usize>(x)];
                            }
                    }
                    // Downsample 128³ → 64³, IN-BOUNDS mean (global source coord < sd), matching downsample2_.
                    std::vector<f32> out(static_cast<usize>(cs * cs * cs), 0.0f);
                    const s64 gbz = (2 * cz) * cs, gby = (2 * cy) * cs, gbx = (2 * cx) * cs;
                    bool all_air = true;
                    for (s64 z = 0; z < cs; ++z)
                        for (s64 y = 0; y < cs; ++y)
                            for (s64 x = 0; x < cs; ++x) {
                                f32 sum = 0; int n = 0;
                                for (s64 dz = 0; dz < 2; ++dz)
                                    for (s64 dy = 0; dy < 2; ++dy)
                                        for (s64 dx = 0; dx < 2; ++dx) {
                                            const s64 iz = 2 * z + dz, iy = 2 * y + dy, ix = 2 * x + dx;
                                            if (gbz + iz < sd.z && gby + iy < sd.y && gbx + ix < sd.x) {
                                                sum += region[static_cast<usize>((iz * RS + iy) * RS + ix)]; ++n;
                                            }
                                        }
                                const f32 v = n ? sum / static_cast<f32>(n) : 0.0f;
                                out[static_cast<usize>((z * cs + y) * cs + x)] = v;
                                if (v != 0.0f) all_air = false;
                            }
                    if (all_air) { e.state = 1; return; }  // in-bounds all zero → ZERO slot
                    e.payload = encode_tile_dct<f32>(std::span<const f32>(out), cs / kDctN, params_);
                    e.state = 2;
                });
                if (failed.load()) return std::unexpected(ferr);
                for (s64 i = batch; i < hi; ++i) {
                    const Enc& e = enc[static_cast<usize>(i - batch)];
                    if (e.state <= 0) continue;  // -1 skip / 0 absent → no slot written
                    const s64 cx = i % ce.x, cy = (i / ce.x) % ce.y, cz = i / (ce.x * ce.y);
                    if (auto w = commit_encoded_(dst, {cz, cy, cx}, e.payload, e.state == 1); !w) return w;
                }
                if (auto c = commit(); !c) return std::unexpected(c.error());
                progress(dst, hi, nout);
            }
            lod = dst;
        }
        nlods_ = static_cast<u32>(lod + 1);
        if (auto c = commit(); !c) return std::unexpected(c.error());
        return {};
    }

    // Reassemble LOD level `lod` into a dense Volume<T> (T = the native dtype to avoid widening: a u8 archive
    // → Volume<u8>, 8 GiB for a 2048³, NOT 34 GiB f32). PARALLEL decode: each 64³ tile decodes independently
    // (read_chunk_as<T>) and scatters into a disjoint output region (no locking). This one-time decode is far
    // cheaper than the streaming path re-decoding overlapping tiles per patch — use it whenever the volume
    // fits in RAM.
    template <class T> [[nodiscard]] Expected<Volume<T>> read_volume_as(s64 lod = 0) const {
        const Extent3 vd = dims_at(lod);
        Volume<T> out = Volume<T>::zeros(vd);
        VolumeView<T> ov = out.view();
        const ChunkCoord ce = chunk_extent(lod);
        const s64 cs = fxvol_chunk_side;
        const s64 ntiles = ce.z * ce.y * ce.x;
        std::atomic<bool> failed{false};
        std::mutex emu;
        Error first_err = err(Errc::decode_error, "read_volume").error();
        parallel_for(0, ntiles, [&](s64 i) {
            if (failed.load(std::memory_order_relaxed)) return;
            const s64 cx = i % ce.x, cy = (i / ce.x) % ce.y, cz = i / (ce.x * ce.y);
            auto blk = read_chunk_as<T>(lod, {cz, cy, cx});
            if (!blk) {
                bool e = false;
                if (failed.compare_exchange_strong(e, true)) {
                    std::lock_guard<std::mutex> lk(emu);
                    first_err = blk.error();
                }
                return;
            }
            for (s64 z = 0; z < cs; ++z)
                for (s64 y = 0; y < cs; ++y) {
                    const s64 vz = cz * cs + z, vy = cy * cs + y;
                    if (vz >= vd.z || vy >= vd.y) continue;
                    const usize srow = static_cast<usize>((z * cs + y) * cs);
                    for (s64 x = 0; x < cs; ++x) {
                        const s64 vx = cx * cs + x;
                        if (vx < vd.x) ov(vz, vy, vx) = (*blk)[srow + static_cast<usize>(x)];
                    }
                }
        });
        if (failed.load()) return std::unexpected(first_err);
        return out;
    }

    [[nodiscard]] Expected<Volume<f32>> read_volume(s64 lod = 0) const { return read_volume_as<f32>(lod); }

    [[nodiscard]] u64
    lod_root_offset(s64 lod) const {  // byte offset of a LOD's radix root (0 = none); for tests/inspection
        return (lod >= 0 && lod < static_cast<s64>(detail::kFxMaxLod)) ? lod_root_[lod] : 0;
    }

    [[nodiscard]] u64 data_offset() const {
        return data_off_;
    }  // SEALED: where blobs begin (all index nodes precede it)

    [[nodiscard]] u64 committed_size() const { return committed_eof_; }  // bytes of actual committed data (≤ file size)

    // Repack this (LIVE) archive into a fresh SEALED file at `dst`, ordered COARSE-FIRST (coarsest LOD's
    // data at the front, full-res last) so a truncated range-GET yields a coarse preview. Compressed blobs
    // are copied VERBATIM (no decode/re-encode → no extra loss); ZERO/ABSENT coverage is preserved. Within
    // a LOD, chunks are written in Morton order (on-disk halo locality). Call on a read-only/closed archive.
    Expected<void> finalize(const std::string& dst) const {
        auto outr = create(dst, dims_, params_, src_dtype_);
        if (!outr) return std::unexpected(outr.error());
        VolumeArchive out = std::move(*outr);
        out.nlods_ = nlods_;

        struct Pending {
            detail::FxSlot* slot;  // into out's mmap node region — stable (base_ never moves)
            u64 src_off;
            u64 src_len;
        };

        std::vector<Pending> pending;
        // Pass 1: build the radix tree FRONT-LOADED — allocate index nodes only (no blobs yet), so all nodes
        // are contiguous right after the superblocks (a remote reader pulls the structure up front). Coarse-
        // first across octaves, Morton order within. Set ZERO slots now; queue REAL chunks for pass 2.
        for (s64 lod = static_cast<s64>(nlods_) - 1; lod >= 0; --lod) {
            const ChunkCoord ce = chunk_extent(lod);
            std::vector<std::pair<u64, ChunkCoord>> items;
            for (s64 cz = 0; cz < ce.z; ++cz)
                for (s64 cy = 0; cy < ce.y; ++cy)
                    for (s64 cx = 0; cx < ce.x; ++cx) {
                        const ChunkCoord c{cz, cy, cx};
                        if (slot_read_(lod, c).off == detail::kFxAbsent) continue;
                        items.emplace_back(
                            detail::morton3(static_cast<u64>(cz), static_cast<u64>(cy), static_cast<u64>(cx)), c);
                    }
            std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& [m, c] : items) {
                const detail::FxSlot s = slot_read_(lod, c);
                auto sp = out.slot_write_(lod, c);
                if (!sp) return std::unexpected(sp.error());
                if (s.len == 0) {
                    **sp = {0, 0};
                    continue;
                }  // ZERO
                // Same non-overflowing rewrite as read_chunk_as: bound s.off first, then subtract.
                if (s.off < detail::kFxDataStart || s.off > committed_eof_ || s.len > committed_eof_ - s.off ||
                    committed_eof_ - s.off - s.len < 4)
                    return err(Errc::decode_error, "corrupt source blob");
                pending.push_back({*sp, s.off, s.len});
            }
        }
        out.data_off_ = out.cursor_;  // blobs begin here, after the front-loaded index node region
        // Pass 2: append the compressed blobs VERBATIM (coarse-first order preserved) + patch each slot.
        for (const Pending& p : pending) {
            auto off = out.alloc_(p.src_len + 4, false);
            if (!off) return std::unexpected(off.error());
            std::memcpy(out.base_ + *off, base_ + p.src_off, p.src_len + 4);
            *p.slot = {*off, p.src_len};
        }
        return out.close();
    }

    // ---- decoded-16³-chunk view (mmap-the-archive-as-an-array) ----
    // create()/open() already install a default-budget cache, so this only needs calling to pick a
    // non-default budget/shard count. Like the default cache, it unconditionally REPLACES the pointer —
    // call it before any concurrent block16()/gather_box_f32() use (e.g. before fuse_main spawns
    // threads), never while another thread may be mid-lookup, or the old BlockCache can be destroyed out
    // from under a concurrent reader (same hazard the eager-init fix closes for the lazy-init case).
    void reserve_cache(u64 budget_bytes, int shards = 16) {
        block_cache_ = std::make_unique<BlockCache>(budget_bytes, shards);
    }

    [[nodiscard]] u64 cache_hits() const { return block_cache_ ? block_cache_->hits() : 0; }

    [[nodiscard]] u64 cache_misses() const { return block_cache_ ? block_cache_->misses() : 0; }

    [[nodiscard]] u64 cache_bytes() const { return block_cache_ ? block_cache_->bytes() : 0; }

    // Fetch a decoded 16³ chunk at LOD `lod` by its block-grid coord (= voxel/16). On a miss the containing
    // 64³ tile is decoded once (atomic decode unit) and ALL 64 of its 16³ chunks are cached. The cache holds
    // NATIVE-DTYPE bytes: a u8 archive caches u8 (16³·1 B = 4 KiB/block), NEVER f32 — the whole-volume f32
    // slab is gone, and f32 exists only ephemerally when sample_f32() widens a voxel for compute. u8 is the
    // only native-cache path today (the scrolls); other dtypes decode to u8 too iff they fit (they don't in
    // general) — so non-u8 archives fall back to f32-in-bytes to stay correct.
    [[nodiscard]] Expected<BlockCache::Ref> block16(s64 lod, ChunkCoord bc) const {
        // No lazy init here (was a data race: two threads racing an unsynchronized test-and-assign of the
        // `mutable unique_ptr` could each construct a BlockCache, then one assignment destroys the other's
        // object while it's still in use). create()/open() always install a default-budget cache, so a
        // null cache here means this VolumeArchive was default-constructed and never opened — a caller bug,
        // not a state block16() should paper over concurrently.
        if (!block_cache_) return err(Errc::internal, "block16: archive has no cache (not created/opened)");
        const u64 key = block_key_(lod, bc.z, bc.y, bc.x);
        if (auto r = block_cache_->get(key)) return r;
        constexpr s64 BS = kDctN, NB = fxvol_chunk_side / kDctN;
        const ChunkCoord tc{bc.z / NB, bc.y / NB, bc.x / NB};
        const bool u8_native = (src_dtype_ == DType::u8);
        // Decode the tile ONCE, in its native dtype when u8, else f32; scatter its 64 sub-blocks into the
        // cache as raw bytes. esz = bytes per voxel in the cache.
        const usize esz = u8_native ? sizeof(u8) : sizeof(f32);
        std::vector<u8> tbytes;
        if (u8_native) {
            auto tile = read_chunk_as<u8>(lod, tc);
            if (!tile) return std::unexpected(tile.error());
            tbytes.resize(tile->size());
            std::memcpy(tbytes.data(), tile->data(), tile->size());
        } else {
            auto tile = read_chunk_as<f32>(lod, tc);
            if (!tile) return std::unexpected(tile.error());
            tbytes.resize(tile->size() * sizeof(f32));
            std::memcpy(tbytes.data(), tile->data(), tbytes.size());
        }
        const usize cs = static_cast<usize>(fxvol_chunk_side);
        BlockCache::Ref want;
        for (s64 sz = 0; sz < NB; ++sz)
            for (s64 sy = 0; sy < NB; ++sy)
                for (s64 sx = 0; sx < NB; ++sx) {
                    auto blk = std::make_shared<BlockCache::Block>(static_cast<usize>(BS * BS * BS) * esz);
                    for (s64 z = 0; z < BS; ++z)
                        for (s64 y = 0; y < BS; ++y) {
                            const usize dst = (static_cast<usize>(z) * BS + static_cast<usize>(y)) * BS * esz;
                            const usize src = ((static_cast<usize>(sz) * BS + static_cast<usize>(z)) * cs +
                                               (static_cast<usize>(sy) * BS + static_cast<usize>(y))) *
                                                  cs * esz +
                                              static_cast<usize>(sx) * BS * esz;
                            std::memcpy(blk->data() + dst, tbytes.data() + src, static_cast<usize>(BS) * esz);
                        }
                    const u64 k = block_key_(lod, tc.z * NB + sz, tc.y * NB + sy, tc.x * NB + sx);
                    BlockCache::Ref cr = blk;
                    block_cache_->put(k, cr);
                    if (k == key) want = cr;
                }
        if (!want) return err(Errc::decode_error, "block16 key not produced");
        return want;
    }

    [[nodiscard]] Expected<BlockCache::Ref> block16(ChunkCoord bc) const { return block16(0, bc); }

    // Read a single voxel widened to f32 EPHEMERALLY (the cache stays native-dtype). This is the sampling
    // primitive for consumers that compute in float (ML patch gather, resampling) — the f32 is a transient
    // return value, never a stored volume.
    [[nodiscard]] Expected<f32> sample_f32(s64 lod, s64 z, s64 y, s64 x) const {
        // Negative coordinates are not legal (ZYX conventions, docs/conventions.md), but C's truncating
        // `/`+`%` don't agree on sign: z=-1 gives z/16==0 (a valid-looking block) and z%16==-1, which then
        // makes `off` wrap to a huge usize -> OOB read of the 4 KiB cache block. Reject explicitly instead
        // of silently reading garbage or crashing on a misbehaving caller.
        if (z < 0 || y < 0 || x < 0) return err(Errc::invalid_argument, "sample_f32: negative coordinate");
        auto r = block16(lod, {z / kDctN, y / kDctN, x / kDctN});
        if (!r) return std::unexpected(r.error());
        const usize off = static_cast<usize>(((z % kDctN) * kDctN + (y % kDctN)) * kDctN + (x % kDctN));
        const BlockCache::Block& b = **r;
        if (src_dtype_ == DType::u8) return static_cast<f32>(b[off]);
        f32 v;
        std::memcpy(&v, b.data() + off * sizeof(f32), sizeof(f32));
        return v;
    }

    [[nodiscard]] Expected<f32> sample_f32(s64 z, s64 y, s64 x) const { return sample_f32(0, z, y, x); }

    // Gather a [D×H×W] box at (oz,oy,ox) into `out` (ZYX-contiguous f32, size D·H·W), edge-clamped to the
    // volume. BLOCK-MAJOR: iterate the covering 16³ blocks and fetch each EXACTLY ONCE (one cache-lock per
    // block), then scatter all its voxels into the output — vs the per-row version which re-locked the same
    // block ~16× per patch. For an interior 256³ patch that is 16³=4096 block16() calls instead of ~1M, so
    // the block cache stops being lock-bound and inference becomes GPU-bound. Thread-safe for disjoint boxes.
    // The common case (patch fully inside the volume) takes the fast no-clamp path with contiguous x-copies.
    // u8-NATIVE fast gather: rows memcpy'd straight from the decoded-block cache — no f32
    // widen + reconvert (the training feeder's per-patch 2M-voxel convert loop measured as
    // pure overhead). Requires src dtype u8 and the box fully inside the volume; callers
    // fall back to gather_box_f32 otherwise.
    Expected<void> gather_box_u8(s64 lod, s64 oz, s64 oy, s64 ox, s64 D, s64 H, s64 W, u8* out) const {
        if (src_dtype_ != DType::u8) return err(Errc::unsupported, "gather_box_u8: non-u8 archive");
        const Extent3 vd = dims_at(lod);
        if (!(oz >= 0 && oy >= 0 && ox >= 0 && oz + D <= vd.z && oy + H <= vd.y && ox + W <= vd.x))
            return err(Errc::invalid_argument, "gather_box_u8: box not inside volume");
        constexpr s64 N = kDctN;
        const s64 bz0 = oz / N, bz1 = (oz + D - 1) / N;
        const s64 by0 = oy / N, by1 = (oy + H - 1) / N;
        const s64 bx0 = ox / N, bx1 = (ox + W - 1) / N;
        for (s64 bz = bz0; bz <= bz1; ++bz)
            for (s64 by = by0; by <= by1; ++by)
                for (s64 bx = bx0; bx <= bx1; ++bx) {
                    auto r = block16(lod, {bz, by, bx});
                    if (!r) return std::unexpected(r.error());
                    const BlockCache::Block& b = **r;
                    const s64 lz0 = std::max<s64>(0, oz - bz * N), lz1 = std::min<s64>(N, oz + D - bz * N);
                    const s64 ly0 = std::max<s64>(0, oy - by * N), ly1 = std::min<s64>(N, oy + H - by * N);
                    const s64 lx0 = std::max<s64>(0, ox - bx * N), lx1 = std::min<s64>(N, ox + W - bx * N);
                    const s64 run = lx1 - lx0;
                    for (s64 lz = lz0; lz < lz1; ++lz)
                        for (s64 ly = ly0; ly < ly1; ++ly)
                            std::memcpy(out + (((bz * N + lz - oz) * H) + (by * N + ly - oy)) * W + (bx * N + lx0 - ox),
                                        b.data() + static_cast<usize>((lz * N + ly) * N + lx0),
                                        static_cast<usize>(run));
                }
        return {};
    }

    Expected<void> gather_box_f32(s64 lod, s64 oz, s64 oy, s64 ox, s64 D, s64 H, s64 W, f32* out) const {
        const Extent3 vd = dims_at(lod);
        const bool u8n = (src_dtype_ == DType::u8);
        const bool inside = oz >= 0 && oy >= 0 && ox >= 0 && oz + D <= vd.z && oy + H <= vd.y && ox + W <= vd.x;
        constexpr s64 N = kDctN;
        // Block-index range covering the box on each axis (clamped source coords).
        auto brange = [](s64 o, s64 n, s64 dim, s64& b0, s64& b1) {
            const s64 g0 = o < 0 ? 0 : (o >= dim ? dim - 1 : o);
            const s64 g1 = (o + n - 1) < 0 ? 0 : ((o + n - 1) >= dim ? dim - 1 : (o + n - 1));
            b0 = g0 / N;
            b1 = g1 / N;
        };
        s64 bz0, bz1, by0, by1, bx0, bx1;
        brange(oz, D, vd.z, bz0, bz1);
        brange(oy, H, vd.y, by0, by1);
        brange(ox, W, vd.x, bx0, bx1);
        for (s64 bz = bz0; bz <= bz1; ++bz)
            for (s64 by = by0; by <= by1; ++by)
                for (s64 bx = bx0; bx <= bx1; ++bx) {
                    auto r = block16(lod, {bz, by, bx});
                    if (!r) return std::unexpected(r.error());
                    const BlockCache::Block& b = **r;
                    if (inside) {
                        // Fast path: this block's voxels [bz*N..)(local) map to out at fixed offsets, full
                        // 16-wide contiguous x-runs, no clamping.
                        const s64 lz0 = std::max<s64>(0, oz - bz * N), lz1 = std::min<s64>(N, oz + D - bz * N);
                        const s64 ly0 = std::max<s64>(0, oy - by * N), ly1 = std::min<s64>(N, oy + H - by * N);
                        const s64 lx0 = std::max<s64>(0, ox - bx * N), lx1 = std::min<s64>(N, ox + W - bx * N);
                        for (s64 lz = lz0; lz < lz1; ++lz)
                            for (s64 ly = ly0; ly < ly1; ++ly) {
                                const usize base = static_cast<usize>((lz * N + ly) * N);
                                f32* orow =
                                    out + (((bz * N + lz - oz) * H) + (by * N + ly - oy)) * W + (bx * N + lx0 - ox);
                                if (u8n)
                                    for (s64 lx = lx0; lx < lx1; ++lx)
                                        *orow++ = static_cast<f32>(b[base + static_cast<usize>(lx)]);
                                else
                                    for (s64 lx = lx0; lx < lx1; ++lx) {
                                        std::memcpy(orow,
                                                    b.data() + (base + static_cast<usize>(lx)) * sizeof(f32),
                                                    sizeof(f32));
                                        ++orow;
                                    }
                            }
                    } else {
                        // Edge-clamp path: for each OUTPUT voxel whose clamped source lands in THIS block, copy.
                        for (s64 z = 0; z < D; ++z) {
                            const s64 gz = std::clamp<s64>(oz + z, 0, vd.z - 1);
                            if (gz / N != bz) continue;
                            for (s64 y = 0; y < H; ++y) {
                                const s64 gy = std::clamp<s64>(oy + y, 0, vd.y - 1);
                                if (gy / N != by) continue;
                                f32* orow = out + ((z * H) + y) * W;
                                for (s64 x = 0; x < W; ++x) {
                                    const s64 gx = std::clamp<s64>(ox + x, 0, vd.x - 1);
                                    if (gx / N != bx) continue;
                                    const usize off = static_cast<usize>(((gz % N) * N + (gy % N)) * N + (gx % N));
                                    if (u8n)
                                        orow[x] = static_cast<f32>(b[off]);
                                    else
                                        std::memcpy(&orow[x], b.data() + off * sizeof(f32), sizeof(f32));
                                }
                            }
                        }
                    }
                }
        return {};
    }

    // Durable checkpoint: msync the data region, THEN write+msync the next superblock slot (data-before-
    // pointer). Safe to call repeatedly mid-session; a crash recovers the last committed state.
    Expected<void> commit() {
        if (!writable_) return {};
        committed_eof_ = cursor_;
        // msync only the bytes written SINCE the last commit (page-aligned start) — re-syncing the whole
        // region every checkpoint would be O(N²) page scans over a long (thousands-of-commits) export.
        if (base_ && committed_eof_ > last_msync_) {
            const u64 from = last_msync_ & ~static_cast<u64>(4095);  // msync addr must be page-aligned
            ::msync(base_ + from, committed_eof_ - from, MS_SYNC);
        }
        last_msync_ = committed_eof_;
        ++commit_seq_;
        write_superblock_(commit_seq_);
        return {};
    }

    Expected<void> close() {
        if (!writable_) return {};
        auto e = commit();
        // Trim the fallocate'd tail to the actual data size (the file grows 64 MiB at a time; without this
        // a tiny archive is padded to 64 MiB). Shrinking is safe — we never read past committed_eof_, and a
        // reopen-rw re-grows via fallocate. The mmap reservation (VA) is unaffected.
#ifndef _WIN32
        // Windows compacts the sparse file in release_() instead: it cannot be shrunk while the
        // section is still mapped (the mapping pins the file size).
        if (committed_eof_ < file_size_ && ::ftruncate(fd_, static_cast<off_t>(committed_eof_)) == 0)
            file_size_ = committed_eof_;
#endif
        writable_ = false;
        return e;
    }

  private:
    static u64 node_bytes_() { return detail::kFxNodeEnt * sizeof(u64); }

    static u64 leaf_bytes_() { return detail::kFxNodeEnt * sizeof(detail::FxSlot); }

    static bool fits_one_chunk_(Extent3 d) {
        return d.z <= fxvol_chunk_side && d.y <= fxvol_chunk_side && d.x <= fxvol_chunk_side;
    }

    static u32 plan_nlods_(Extent3 d) {
        u32 n = 1;
        while (!fits_one_chunk_(d) && n < detail::kFxMaxLod) {
            d = {(d.z + 1) / 2, (d.y + 1) / 2, (d.x + 1) / 2};
            ++n;
        }
        return n;
    }

    static u64 block_key_(s64 lod, s64 z, s64 y, s64 x) {  // cache key: LOD in the high byte + 16³-block Morton
        return (static_cast<u64>(lod) << 56) |
               detail::morton3(static_cast<u64>(z), static_cast<u64>(y), static_cast<u64>(x));
    }

    template <class T>
    static Volume<f32> downsample2_(VolumeView<const T> v) {  // global 2³ box prefilter + decimate (seam-free)
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
                                if (sz < d.z && sy < d.y && sx < d.x) {
                                    sum += static_cast<f32>(v(sz, sy, sx));
                                    ++n;
                                }
                            }
                    ov(z, y, x) = n ? sum / static_cast<f32>(n) : 0.0f;
                }
        return out;
    }

    template <class T> Expected<void> write_level_(s64 lod, VolumeView<const T> v) {
        const Extent3 vd = v.dims();
        const ChunkCoord ce = chunk_extent(lod);
        const s64 cs = fxvol_chunk_side;
        const s64 ntiles = ce.z * ce.y * ce.x;

        // TWO-PHASE tiling so the expensive DCT+rANS encode runs on ALL cores while the mmap page-table /
        // bump-allocator (which is single-writer) stays serial:
        //   (1) PARALLEL: gather each 64³ tile in the source dtype (u8 never widened to a whole f32 block) and
        //       encode_tile_dct<T> it into its own byte buffer. all-fill tiles → empty payload (ZERO slot).
        //   (2) SERIAL: append the payloads to the archive in coord order (slot_write_ + alloc_ + memcpy).
        struct Enc {
            std::vector<u8> payload;
            bool zero;
        };

        std::vector<Enc> enc(static_cast<usize>(ntiles));
        std::atomic<bool> failed{false};
        parallel_for(0, ntiles, [&](s64 i) {
            const s64 cx = i % ce.x, cy = (i / ce.x) % ce.y, cz = i / (ce.x * ce.y);
            std::vector<T> block(static_cast<usize>(cs * cs * cs));
            bool all_fill = true;
            const T fill = T{};
            for (s64 z = 0; z < cs; ++z)
                for (s64 y = 0; y < cs; ++y)
                    for (s64 x = 0; x < cs; ++x) {
                        const s64 vz = std::min(cz * cs + z, vd.z - 1);
                        const s64 vy = std::min(cy * cs + y, vd.y - 1);
                        const s64 vx = std::min(cx * cs + x, vd.x - 1);
                        const T val = v(vz, vy, vx);
                        block[static_cast<usize>((z * cs + y) * cs + x)] = val;
                        if (val != fill) all_fill = false;
                    }
            if (all_fill) {
                enc[static_cast<usize>(i)].zero = true;
                return;
            }
            enc[static_cast<usize>(i)].payload = encode_tile_dct<T>(std::span<const T>(block), cs / kDctN, params_);
        });
        for (s64 i = 0; i < ntiles; ++i) {
            const s64 cx = i % ce.x, cy = (i / ce.x) % ce.y, cz = i / (ce.x * ce.y);
            if (auto w = commit_encoded_(
                    lod, {cz, cy, cx}, enc[static_cast<usize>(i)].payload, enc[static_cast<usize>(i)].zero);
                !w)
                return w;
        }
        (void)failed;
        return {};
    }

    // Serial commit of a pre-encoded tile payload (phase 2 of write_level_). Empty/zero → ZERO slot (no blob).
    Expected<void> commit_encoded_(s64 lod, ChunkCoord c, const std::vector<u8>& payload, bool zero) {
        if (!writable_) return err(Errc::io_error, "archive opened read-only");
        auto sp = slot_write_(lod, c);
        if (!sp) return std::unexpected(sp.error());
        if (zero || payload.empty()) {
            **sp = {0, 0};
            return {};
        }
        const u32 crc = detail::crc32c(payload.data(), payload.size());
        auto off = alloc_(payload.size() + 4, false);
        if (!off) return std::unexpected(off.error());
        std::memcpy(base_ + *off, payload.data(), payload.size());
        std::memcpy(base_ + *off + payload.size(), &crc, 4);
        **sp = {*off, payload.size()};
        return {};
    }

    Expected<void> map_() {
#ifdef _WIN32
        // Sparse-file-backed section of reserve_, mapped whole at a stable base (win_compat.cpp).
        void* p = fenix_win_archive_map(fd_, reserve_);
        if (!p) return err(Errc::io_error, "Win32 archive mapping failed");
#else
        void* p = ::mmap(nullptr, reserve_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd_, 0);
        if (p == MAP_FAILED) return err(Errc::io_error, "mmap failed");
#endif
        base_ = static_cast<u8*>(p);
        return {};
    }

    Expected<void> ensure_(u64 end) {  // back the file up to `end` (preallocate blocks, then grow logical size)
        if (end <= file_size_) return {};
        const u64 want = (end + detail::kFxGrow - 1) / detail::kFxGrow * detail::kFxGrow;
        if (want > reserve_)
            return err(Errc::io_error,
                       "archive exceeds its VA reservation (" + std::to_string(reserve_ >> 20) +
                           " MiB, dims+q-derived) — the compression-ratio lower bound was too optimistic");
#if defined(_WIN32)
        // The sparse section already spans reserve_ (see map_); mapped writes fault in pages on
        // demand, so there is nothing to preallocate here -- just advance the logical size.
        file_size_ = want;
        return {};
#elif defined(__APPLE__)
        // macOS has no posix_fallocate. Best-effort F_PREALLOCATE (reserve contiguous blocks) then
        // ftruncate to grow the logical size — ftruncate alone is correct (sparse), preallocation is
        // just to avoid fragmentation. crash-safety unaffected: the page-table commit ordering holds.
        fstore_t st{F_ALLOCATECONTIG | F_ALLOCATEALL, F_PEOFPOSMODE, 0, static_cast<off_t>(want - file_size_), 0};
        if (::fcntl(fd_, F_PREALLOCATE, &st) == -1) {
            st.fst_flags = F_ALLOCATEALL;  // retry without the contiguous requirement
            ::fcntl(fd_, F_PREALLOCATE, &st);
        }
        if (::ftruncate(fd_, static_cast<off_t>(want)) != 0) return err(Errc::io_error, "ftruncate failed");
#else
        const int rc = ::posix_fallocate(fd_, static_cast<off_t>(file_size_), static_cast<off_t>(want - file_size_));
        if (rc != 0) return err(Errc::io_error, "fallocate failed");
#endif
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

    // in_bounds_(off, need, hz): non-overflowing "[off, off+need) fits within [0, hz)" check — off/need
    // are untrusted (radix nodes/leaves carry no CRC, unlike blobs/superblock), so `off + need > hz` is
    // NOT safe: a corrupt off near u64 max makes the addition wrap small and pass. Rewritten so the only
    // arithmetic is a subtraction gated by `off <= hz` first.
    static bool in_bounds_(u64 off, u64 need, u64 hz) { return off <= hz && need <= hz - off; }

    detail::FxSlot slot_read_(s64 lod, ChunkCoord c) const {  // walk L0→L1→leaf; out-of-bounds ⇒ ABSENT
        const detail::FxSlot absent{detail::kFxAbsent, 0};
        if (lod < 0 || lod >= static_cast<s64>(detail::kFxMaxLod)) return absent;
        const u64 hz = read_horizon_();
        const u64 root = lod_root_[lod];
        if (root == 0 || !in_bounds_(root, node_bytes_(), hz)) return absent;
        const u64 m = detail::morton3(static_cast<u64>(c.z), static_cast<u64>(c.y), static_cast<u64>(c.x));
        const u32 i0 = (m >> 24) & 0xfffu, i1 = (m >> 12) & 0xfffu, i2 = m & 0xfffu;
        const u64 c1 = node_at_(root)[i0];
        if (c1 == 0 || !in_bounds_(c1, node_bytes_(), hz)) return absent;
        const u64 c2 = node_at_(c1)[i1];
        if (c2 == 0 || !in_bounds_(c2, leaf_bytes_(), hz)) return absent;
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
        // byte 28: source dtype (v5). The upper 3 bytes stay zero (reserved).
        const u32 dt = static_cast<u32>(static_cast<u8>(src_dtype_));
        std::memcpy(base_ + b + 28, &dt, 4);
        std::memcpy(base_ + b + 32, &dims_.z, 8);
        std::memcpy(base_ + b + 40, &dims_.y, 8);
        std::memcpy(base_ + b + 48, &dims_.x, 8);
        std::memcpy(base_ + b + 56, &params_.q, 4);
        std::memcpy(base_ + b + 60, &params_.hf_exp, 4);
        std::memcpy(base_ + b + 64, &params_.dz_frac, 4);
        for (u32 l = 0; l < detail::kFxMaxLod; ++l)
            std::memcpy(base_ + b + detail::kFxLodRootOff + 8 * l, &lod_root_[l], 8);
        std::memcpy(base_ + b + detail::kFxDataOffField, &data_off_, 8);
        const u32 crc = detail::crc32c(base_ + b, detail::kFxSbCrcLen);
        std::memcpy(base_ + b + detail::kFxSbCrcLen, &crc, 4);
        if (base_) ::msync(base_ + b, detail::kFxSuper, MS_SYNC);
    }

    void steal_(VolumeArchive& o) {
        fd_ = o.fd_;
        base_ = o.base_;
        reserve_ = o.reserve_;
        file_size_ = o.file_size_;
        cursor_ = o.cursor_;
        committed_eof_ = o.committed_eof_;
        commit_seq_ = o.commit_seq_;
        last_msync_ = o.last_msync_;
        nlods_ = o.nlods_;
        std::memcpy(lod_root_, o.lod_root_, sizeof lod_root_);
        data_off_ = o.data_off_;
        dims_ = o.dims_;
        params_ = o.params_;
        src_dtype_ = o.src_dtype_;
        writable_ = o.writable_;
        write_opened_ = o.write_opened_;
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
#ifdef _WIN32
            // Unmap + close the section, then compact the sparse file to the committed size —
            // but only if we opened it for writing (never resize a read-only file).
            fenix_win_archive_unmap(base_, fd_,
                                    write_opened_ ? static_cast<long long>(committed_eof_) : -1);
#else
            ::munmap(base_, reserve_);
#endif
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
    DType src_dtype_ = DType::u8;  // the volume's native dtype — reads reconstruct THIS, not f32 (scrolls are u8)
    int fd_ = -1;
    u8* base_ = nullptr;
    u64 reserve_ = detail::kFxReserve;
    u64 file_size_ = 0;
    u64 cursor_ = 0;
    u64 committed_eof_ = 0;
    u64 commit_seq_ = 0;
    u64 last_msync_ = detail::kFxDataStart;  // high-water of msync'd data (incremental commit)
    u32 nlods_ = 1;
    u64 lod_root_[detail::kFxMaxLod] = {};  // per-LOD radix-table root offset (0 = none)
    u64 data_off_ = 0;                      // SEALED: byte offset where blob data begins (0 = LIVE, no front index)
    bool writable_ = false;
    bool write_opened_ = false;  // opened for writing (create/open-rw) — gates compact-on-close (Win32)
    mutable std::unique_ptr<BlockCache> block_cache_;
};

}  // namespace fenix::codec
