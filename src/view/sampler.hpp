// view/sampler.hpp — a per-thread trilinear sampler over the .fxvol decoded-16³ block
// cache: memoizes the last two blocks touched (a trilinear tap cluster usually straddles
// ≤2 blocks along a scanline) so the sharded cache lock is off the per-voxel hot path.
// A decode failure latches `failed` and samples return 0 — the caller checks once per
// image, never per voxel (a viewer must not turn one bad chunk into a crash).
#pragma once

#include "codec/archive.hpp"
#include "codec/source.hpp"
#include "core/core.hpp"

#include <cmath>
#include <optional>
#include <unordered_set>
#include <vector>

namespace fenix::view {

class BlockSampler {
public:
    // best_effort: NEVER block on the network — sample the finest LOCAL level per voxel
    // (fallback ladder lod..nlods-1), schedule missing target-level chunks in the
    // background, and latch incomplete() so the caller redraws on the next arrival.
    BlockSampler(codec::VolumeSource& s, s64 lod, bool best_effort = false)
        : src_(&s), lod_(lod), best_(best_effort), dims_(s.dims_at(lod)),
          u8_(s.src_dtype() == codec::DType::u8) {
        if (best_)
            for (u32 l = 0; l < s.nlods(); ++l) ldims_.push_back(s.dims_at(l));
    }
    BlockSampler(codec::VolumeArchive& a, s64 lod)
        : own_(std::in_place, a), src_(&*own_), lod_(lod), dims_(a.dims_at(lod)),
          u8_(a.src_dtype() == codec::DType::u8) {}

    [[nodiscard]] bool failed() const { return failed_; }
    [[nodiscard]] bool incomplete() const { return incomplete_; }
    [[nodiscard]] Extent3 dims() const { return dims_; }

    // Voxel fetch, edge-clamped.
    [[nodiscard]] f32 at(s64 z, s64 y, s64 x) {
        z = std::clamp<s64>(z, 0, dims_.z - 1);
        y = std::clamp<s64>(y, 0, dims_.y - 1);
        x = std::clamp<s64>(x, 0, dims_.x - 1);
        constexpr s64 N = codec::kDctN;
        if (best_) return at_available_(z, y, x);
        const ChunkCoord bc{z / N, y / N, x / N};
        const codec::BlockCache::Block* const b = block_(lod_, bc);
        if (!b) return 0.0f;
        const usize off = static_cast<usize>(((z % N) * N + (y % N)) * N + (x % N));
        if (u8_) return static_cast<f32>((*b)[off]);
        f32 v;
        std::memcpy(&v, b->data() + off * sizeof(f32), sizeof(f32));
        return v;
    }

    // Trilinear at a continuous LOD-space coordinate (voxel-index space).
    [[nodiscard]] f32 trilinear(f32 z, f32 y, f32 x) {
        const f32 fz = std::floor(z), fy = std::floor(y), fx = std::floor(x);
        const s64 iz = static_cast<s64>(fz), iy = static_cast<s64>(fy), ix = static_cast<s64>(fx);
        const f32 tz = z - fz, ty = y - fy, tx = x - fx;
        const f32 c00 = at(iz, iy, ix) * (1 - tx) + at(iz, iy, ix + 1) * tx;
        const f32 c01 = at(iz, iy + 1, ix) * (1 - tx) + at(iz, iy + 1, ix + 1) * tx;
        const f32 c10 = at(iz + 1, iy, ix) * (1 - tx) + at(iz + 1, iy, ix + 1) * tx;
        const f32 c11 = at(iz + 1, iy + 1, ix) * (1 - tx) + at(iz + 1, iy + 1, ix + 1) * tx;
        return (c00 * (1 - ty) + c01 * ty) * (1 - tz) + (c10 * (1 - ty) + c11 * ty) * tz;
    }

private:
    // Best-effort voxel: walk the fallback ladder lod_..top and take the finest level
    // whose covering chunk is local. A missing TARGET-level chunk is scheduled once per
    // sampler (local dedup set keeps the pyramid's mutex off the per-voxel path).
    [[nodiscard]] f32 at_available_(s64 z, s64 y, s64 x) {
        constexpr s64 N = codec::kDctN, NB = codec::fxvol_chunk_side / codec::kDctN;
        for (s64 l = lod_; l < static_cast<s64>(ldims_.size()); ++l) {
            const s64 k = l - lod_;
            const Extent3 ld = ldims_[static_cast<usize>(l)];
            const s64 zz = std::min(z >> k, ld.z - 1), yy = std::min(y >> k, ld.y - 1),
                      xx = std::min(x >> k, ld.x - 1);
            const ChunkCoord bc{zz / N, yy / N, xx / N};
            if (src_->chunk_state(l, {bc.z / NB, bc.y / NB, bc.x / NB}) == codec::Coverage::Absent) {
                if (l == lod_) note_missing_({bc.z / NB, bc.y / NB, bc.x / NB});
                continue;
            }
            const codec::BlockCache::Block* const b = block_(l, bc);
            if (!b) continue;  // decode hiccup: fall to a coarser level, retry next frame
            const usize off = static_cast<usize>(((zz % N) * N + (yy % N)) * N + (xx % N));
            if (u8_) return static_cast<f32>((*b)[off]);
            f32 v;
            std::memcpy(&v, b->data() + off * sizeof(f32), sizeof(f32));
            return v;
        }
        incomplete_ = true;  // nothing local at any level yet
        return 0.0f;
    }

    void note_missing_(ChunkCoord chunk) {
        incomplete_ = true;
        const u64 key = (static_cast<u64>(chunk.z) << 36) | (static_cast<u64>(chunk.y) << 18) |
                        static_cast<u64>(chunk.x);
        if (scheduled_.insert(key).second) src_->schedule_chunk(lod_, chunk);
    }

    const codec::BlockCache::Block* block_(s64 lod, ChunkCoord bc) {
        if (lod == l0_ && bc.z == bc0_.z && bc.y == bc0_.y && bc.x == bc0_.x && b0_) return b0_.get();
        if (lod == l1_ && bc.z == bc1_.z && bc.y == bc1_.y && bc.x == bc1_.x && b1_) {
            std::swap(bc0_, bc1_);
            std::swap(b0_, b1_);
            std::swap(l0_, l1_);
            return b0_.get();
        }
        auto r = best_ ? src_->block16_local(lod, bc) : src_->block16(lod, bc);
        if (!r) {
            if (!best_) failed_ = true;
            return nullptr;
        }
        bc1_ = bc0_;
        b1_ = std::move(b0_);
        l1_ = l0_;
        bc0_ = bc;
        b0_ = std::move(*r);
        l0_ = lod;
        return b0_.get();
    }

    std::optional<codec::ArchiveSource> own_;  // set by the archive-convenience ctor only
    codec::VolumeSource* src_;
    s64 lod_;
    bool best_ = false;
    Extent3 dims_;
    std::vector<Extent3> ldims_;  // per-level dims (best-effort ladder only)
    bool u8_;
    bool failed_ = false;
    bool incomplete_ = false;
    std::unordered_set<u64> scheduled_;  // per-sampler dedup of schedule_chunk calls
    ChunkCoord bc0_{-1, -1, -1}, bc1_{-1, -1, -1};
    s64 l0_ = -1, l1_ = -1;
    codec::BlockCache::Ref b0_, b1_;
};

}  // namespace fenix::view
