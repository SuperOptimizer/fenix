// codec/archive.hpp — the .fxvol container (v4). One file per volume: a sparse grid of 64³ chunks, each
// encoded by the DCT tile codec (a 64³ chunk = one tile of 4³=64 DCT blocks sharing rANS tables), indexed
// by a 3-level (12+12+12) compressed-Morton radix PAGE TABLE that lives IN the file and is mmap'd — so the
// resident RAM is the working set, never O(chunks) (a flat in-RAM index would be ~3.4 TB at the 2¹⁸/axis
// envelope). This is the volume store + the out-of-core IO substrate. See ADR 0006 + docs/design/fxvol-v4-layout.md.
//
// PHASE 1 (this file): single-LOD, LIVE (append-mmap) form — Morton key, 3-level radix table over a huge
// MAP_NORESERVE reservation with fallocate growth + a bump allocator, sentinel-as-coverage slots. A single
// superblock is flushed on close/commit. STILL TODO (later phases per the design note §9): double-buffered
// crc32c superblock + data-before-pointer durability (Phase 2), decoded-tile cache (Phase 3), the explicit
// LOD pyramid (Phase 4), the SEALED coarse-first repack + S3 If-Match CAS (Phases 5-6). The public API is
// stable across all phases.
#pragma once

#include "codec/dct_block.hpp"
#include "core/core.hpp"
#include "core/types.hpp"

#include <cstring>
#include <fcntl.h>
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
// Spread the low 21 bits of x into every 3rd bit (for a 3D Morton interleave). At 2¹⁸/axis the chunk grid
// is 2¹²/axis so only 12 bits are used → a 36-bit Morton key.
inline u64 morton_part1by2(u64 x) {
    x &= 0x1fffffull;
    x = (x | (x << 32)) & 0x1f00000000ffffull;
    x = (x | (x << 16)) & 0x1f0000ff0000ffull;
    x = (x | (x << 8)) & 0x100f00f00f00f00full;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
    x = (x | (x << 2)) & 0x1249249249249249ull;
    return x;
}
// Compressed-Morton ZYX chunk key: spatially-adjacent chunks share radix nodes/leaves → halo locality.
inline u64 morton3(u64 z, u64 y, u64 x) {
    return (morton_part1by2(z) << 2) | (morton_part1by2(y) << 1) | morton_part1by2(x);
}

// A leaf slot encodes the tri-state coverage IN the slot (no side bitmap): off == kAbsent ⇒ ABSENT
// (unknown / not written), else len == 0 ⇒ ZERO (proven all-air), else ⇒ REAL (blob at [off, off+len)).
inline constexpr u64 kFxAbsent = ~0ull;
struct FxSlot {
    u64 off;
    u64 len;
};
// Address-space reservation: huge so the mmap base never moves as the file grows (offsets stay valid
// lock-free). Smaller under sanitizers (a multi-TiB shadow is hostile to ASan/MSan/TSan).
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
inline constexpr u64 kFxGrow = 1ull << 26;     // grow the file 64 MiB at a time
inline constexpr u64 kFxNodeEnt = 4096;        // 2¹² entries per radix node / leaf
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
        if (auto e = a.ensure_(256); !e) return std::unexpected(e.error());  // back the superblock
        a.cursor_ = 256;  // append region starts after the 256-byte superblock
        a.committed_eof_ = 256;
        a.root_off_ = 0;  // no L0 node until the first write
        a.write_superblock_();
        a.writable_ = true;
        return a;
    }

    static Expected<VolumeArchive> open(const std::string& path) {
        VolumeArchive a;
        a.path_ = path;
        a.fd_ = ::open(path.c_str(), O_RDWR, 0644);
        if (a.fd_ < 0) return err(Errc::not_found, "cannot open " + path);
        const off_t sz = ::lseek(a.fd_, 0, SEEK_END);
        if (sz < 256) return err(Errc::decode_error, "truncated .fxvol");
        a.file_size_ = static_cast<u64>(sz);
        if (auto e = a.map_(); !e) return std::unexpected(e.error());
        u32 magic, version;
        std::memcpy(&magic, a.base_ + 0, 4);
        std::memcpy(&version, a.base_ + 4, 4);
        if (magic != fxvol_magic) return err(Errc::decode_error, "bad magic");
        if (version != fxvol_version) return err(Errc::unsupported, "version mismatch (no migration)");
        std::memcpy(&a.dims_.z, a.base_ + 8, 8);
        std::memcpy(&a.dims_.y, a.base_ + 16, 8);
        std::memcpy(&a.dims_.x, a.base_ + 24, 8);
        std::memcpy(&a.params_.q, a.base_ + 32, 4);
        std::memcpy(&a.params_.hf_exp, a.base_ + 36, 4);
        std::memcpy(&a.params_.dz_frac, a.base_ + 40, 4);
        std::memcpy(&a.root_off_, a.base_ + 48, 8);
        std::memcpy(&a.committed_eof_, a.base_ + 56, 8);
        if (a.committed_eof_ > a.file_size_ || (a.root_off_ != 0 && a.root_off_ + node_bytes_() > a.committed_eof_))
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
            **sp = {0, 0};  // ZERO (off != kAbsent, len == 0)
            return {};
        }
        auto payload = encode_tile_dct<f32>(block, fxvol_chunk_side / kDctN, params_);
        auto off = alloc_(payload.size(), false);
        if (!off) return std::unexpected(off.error());
        std::memcpy(base_ + *off, payload.data(), payload.size());
        // re-resolve the slot: alloc_ may have created nodes, but base_ is fixed so the pointer is stable.
        **sp = {*off, payload.size()};
        return {};
    }

    // Read one chunk_side³ block. ABSENT/ZERO -> filled with `fill`.
    [[nodiscard]] Expected<std::vector<f32>> read_chunk(ChunkCoord c, f32 fill = 0.0f) const {
        const usize n = static_cast<usize>(fxvol_chunk_side * fxvol_chunk_side * fxvol_chunk_side);
        const detail::FxSlot s = slot_read_(c);
        if (s.off == detail::kFxAbsent || s.len == 0) return std::vector<f32>(n, fill);
        if (s.off < 256 || s.len > committed_eof_ || s.off > committed_eof_ - s.len)  // bound before deref
            return err(Errc::decode_error, "corrupt chunk offset");
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

    // Flush the superblock (root + committed EOF) and msync the used range. Call once after writing.
    Expected<void> close() {
        if (!writable_) return {};
        committed_eof_ = cursor_;
        write_superblock_();
        if (base_) ::msync(base_, committed_eof_, MS_SYNC);
        writable_ = false;
        return {};
    }

private:
    static u64 node_bytes_() { return detail::kFxNodeEnt * sizeof(u64); }   // radix node = 4096 × u64
    static u64 leaf_bytes_() { return detail::kFxNodeEnt * sizeof(detail::FxSlot); }  // leaf = 4096 × slot

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

    // Read path: walk L0→L1→leaf; any absent/out-of-bounds child ⇒ ABSENT (robust on corrupt bytes).
    detail::FxSlot slot_read_(ChunkCoord c) const {
        const detail::FxSlot absent{detail::kFxAbsent, 0};
        if (root_off_ == 0) return absent;
        const u64 m = detail::morton3(static_cast<u64>(c.z), static_cast<u64>(c.y), static_cast<u64>(c.x));
        const u32 i0 = (m >> 24) & 0xfffu, i1 = (m >> 12) & 0xfffu, i2 = m & 0xfffu;
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

    void write_superblock_() {
        std::memcpy(base_ + 0, &fxvol_magic, 4);
        std::memcpy(base_ + 4, &fxvol_version, 4);
        std::memcpy(base_ + 8, &dims_.z, 8);
        std::memcpy(base_ + 16, &dims_.y, 8);
        std::memcpy(base_ + 24, &dims_.x, 8);
        std::memcpy(base_ + 32, &params_.q, 4);
        std::memcpy(base_ + 36, &params_.hf_exp, 4);
        std::memcpy(base_ + 40, &params_.dz_frac, 4);
        std::memcpy(base_ + 48, &root_off_, 8);
        std::memcpy(base_ + 56, &committed_eof_, 8);
    }

    void steal_(VolumeArchive& o) {
        fd_ = o.fd_;
        base_ = o.base_;
        file_size_ = o.file_size_;
        cursor_ = o.cursor_;
        committed_eof_ = o.committed_eof_;
        root_off_ = o.root_off_;
        dims_ = o.dims_;
        params_ = o.params_;
        writable_ = o.writable_;
        path_ = std::move(o.path_);
        o.fd_ = -1;
        o.base_ = nullptr;
    }
    void release_() {
        if (base_) {
            if (writable_) { committed_eof_ = cursor_; write_superblock_(); ::msync(base_, committed_eof_, MS_SYNC); }
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
    bool writable_ = false;
};

}  // namespace fenix::codec
