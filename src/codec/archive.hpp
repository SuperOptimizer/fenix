// codec/archive.hpp — the .fxvol container. Stores a sparse grid of 64^3 chunks, each encoded by the
// DCT tile codec (a 64^3 chunk = one tile of 4^3=64 DCT blocks sharing rANS tables), addressed by chunk
// coord, with coverage tri-state. This is the volume store + the out-of-core IO substrate.
//
// NOTE: this first cut is file-backed via std::fstream with the index serialized at the
// tail. The target (per codec/CLAUDE.md + ADR 0002) is an mmap'd, append-at-EOF, dense
// in-place 2-level page table with release-store commit + fallocate growth for lock-free
// crash-safe concurrent appends. The on-disk format here is NOT yet that and will change
// (no back-compat); the public API is stable.
#pragma once

#include "codec/dct_block.hpp"
#include "core/core.hpp"
#include "core/types.hpp"

#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fenix::codec {

enum class Coverage : u8 { Absent = 0, Zero = 1, Real = 2 };  // NOT_SURE/air/has-data

inline constexpr u32 fxvol_magic = 0x4c565846u;  // "FXVL"
inline constexpr u32 fxvol_version = 3;  // v3: DCT tile codec w/ clustered context-map tables (wavelet retired, ADR 0005)
inline constexpr s64 fxvol_chunk_side = 64;

class VolumeArchive {
public:
    static Expected<VolumeArchive> create(const std::string& path, Extent3 dims, DctParams bp) {
        VolumeArchive a;
        a.path_ = path;
        a.dims_ = dims;
        a.params_ = bp;
        a.file_.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!a.file_) return err(Errc::io_error, "cannot create " + path);
        std::vector<u8> hdr(256, 0);
        a.file_.write(reinterpret_cast<const char*>(hdr.data()), 256);
        a.cursor_ = 256;
        a.writable_ = true;
        return a;
    }

    static Expected<VolumeArchive> open(const std::string& path) {
        VolumeArchive a;
        a.path_ = path;
        a.file_.open(path, std::ios::binary | std::ios::in);
        if (!a.file_) return err(Errc::not_found, "cannot open " + path);
        u8 hdr[256];
        a.file_.read(reinterpret_cast<char*>(hdr), 256);
        u32 magic, version;
        std::memcpy(&magic, hdr + 0, 4);
        std::memcpy(&version, hdr + 4, 4);
        if (magic != fxvol_magic) return err(Errc::decode_error, "bad magic");
        if (version != fxvol_version) return err(Errc::unsupported, "version mismatch (no migration)");
        std::memcpy(&a.dims_.z, hdr + 8, 8);
        std::memcpy(&a.dims_.y, hdr + 16, 8);
        std::memcpy(&a.dims_.x, hdr + 24, 8);
        std::memcpy(&a.params_.q, hdr + 32, 4);
        std::memcpy(&a.params_.hf_exp, hdr + 36, 4);
        std::memcpy(&a.params_.dz_frac, hdr + 40, 4);
        u64 index_off, index_count;
        std::memcpy(&index_off, hdr + 48, 8);
        std::memcpy(&index_count, hdr + 56, 8);
        a.file_.seekg(static_cast<std::streamoff>(index_off));
        for (u64 i = 0; i < index_count; ++i) {
            u64 key, off, len;
            a.file_.read(reinterpret_cast<char*>(&key), 8);
            a.file_.read(reinterpret_cast<char*>(&off), 8);
            a.file_.read(reinterpret_cast<char*>(&len), 8);
            a.index_[key] = {off, len};
        }
        a.writable_ = false;
        return a;
    }

    [[nodiscard]] Extent3 dims() const { return dims_; }
    [[nodiscard]] DctParams params() const { return params_; }
    [[nodiscard]] ChunkCoord chunk_extent() const {
        return {(dims_.z + fxvol_chunk_side - 1) / fxvol_chunk_side,
                (dims_.y + fxvol_chunk_side - 1) / fxvol_chunk_side,
                (dims_.x + fxvol_chunk_side - 1) / fxvol_chunk_side};
    }

    [[nodiscard]] Coverage coverage(ChunkCoord c) const {
        auto it = index_.find(key(c));
        if (it == index_.end()) return Coverage::Absent;
        return it->second.len == 0 ? Coverage::Zero : Coverage::Real;
    }

    // Write one chunk_side^3 block. An all-`fill` block is recorded as ZERO (no blob).
    Expected<void> write_chunk(ChunkCoord c, std::span<const f32> block, f32 fill = 0.0f) {
        if (!writable_) return err(Errc::io_error, "archive opened read-only");
        const s64 n = fxvol_chunk_side * fxvol_chunk_side * fxvol_chunk_side;
        if (static_cast<s64>(block.size()) != n)
            return err(Errc::invalid_argument, "block must be chunk_side^3");
        bool all_fill = true;
        for (f32 v : block)
            if (v != fill) {
                all_fill = false;
                break;
            }
        if (all_fill) {
            index_[key(c)] = {0, 0};  // ZERO
            return {};
        }
        auto payload = encode_tile_dct<f32>(block, fxvol_chunk_side / kDctN, params_);
        file_.seekp(static_cast<std::streamoff>(cursor_));
        file_.write(reinterpret_cast<const char*>(payload.data()),
                    static_cast<std::streamsize>(payload.size()));
        index_[key(c)] = {cursor_, payload.size()};
        cursor_ += payload.size();
        return {};
    }

    // Read one chunk_side^3 block. ABSENT/ZERO -> filled with `fill`.
    [[nodiscard]] Expected<std::vector<f32>> read_chunk(ChunkCoord c, f32 fill = 0.0f) const {
        const usize n = static_cast<usize>(fxvol_chunk_side * fxvol_chunk_side * fxvol_chunk_side);
        auto it = index_.find(key(c));
        if (it == index_.end() || it->second.len == 0) return std::vector<f32>(n, fill);
        std::vector<u8> payload(it->second.len);
        // const read: use a private mutable file handle copy via a fresh ifstream.
        std::ifstream f(path_, std::ios::binary);
        if (!f) return err(Errc::io_error, "reopen failed");
        f.seekg(static_cast<std::streamoff>(it->second.off));
        f.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(it->second.len));
        return decode_tile_dct<f32>(payload, fxvol_chunk_side / kDctN, params_);
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
                    // Pad past the volume edge by REPLICATING the boundary voxel, not zero-filling: a
                    // zero step at the edge makes the block-DCT ring (~60-unit outliers); a flat
                    // extension is smooth so the transform stays clean. read_volume crops the padding,
                    // so this only affects the codec's edge quality, never the reconstructed values.
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

    // Serialize the index + finalize the header. Call once after writing.
    Expected<void> close() {
        if (!writable_) return {};
        const u64 index_off = cursor_;
        file_.seekp(static_cast<std::streamoff>(index_off));
        for (auto& [k, e] : index_) {
            file_.write(reinterpret_cast<const char*>(&k), 8);
            file_.write(reinterpret_cast<const char*>(&e.off), 8);
            file_.write(reinterpret_cast<const char*>(&e.len), 8);
        }
        u8 hdr[256] = {};
        std::memcpy(hdr + 0, &fxvol_magic, 4);
        std::memcpy(hdr + 4, &fxvol_version, 4);
        std::memcpy(hdr + 8, &dims_.z, 8);
        std::memcpy(hdr + 16, &dims_.y, 8);
        std::memcpy(hdr + 24, &dims_.x, 8);
        std::memcpy(hdr + 32, &params_.q, 4);
        std::memcpy(hdr + 36, &params_.hf_exp, 4);
        std::memcpy(hdr + 40, &params_.dz_frac, 4);
        std::memcpy(hdr + 48, &index_off, 8);
        u64 count = index_.size();
        std::memcpy(hdr + 56, &count, 8);
        file_.seekp(0);
        file_.write(reinterpret_cast<const char*>(hdr), 256);
        file_.flush();
        file_.close();
        writable_ = false;
        return {};
    }

private:
    struct Entry {
        u64 off = 0, len = 0;
    };
    static u64 key(ChunkCoord c) {
        return (static_cast<u64>(c.z) << 40) | (static_cast<u64>(c.y) << 20) | static_cast<u64>(c.x);
    }

    std::string path_;
    Extent3 dims_{};
    DctParams params_{};
    std::fstream file_;
    u64 cursor_ = 0;
    bool writable_ = false;
    std::unordered_map<u64, Entry> index_;
};

}  // namespace fenix::codec
