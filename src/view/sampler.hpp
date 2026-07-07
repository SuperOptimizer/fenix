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

namespace fenix::view {

class BlockSampler {
public:
    BlockSampler(codec::VolumeSource& s, s64 lod)
        : src_(&s), lod_(lod), dims_(s.dims_at(lod)), u8_(s.src_dtype() == codec::DType::u8) {}
    BlockSampler(codec::VolumeArchive& a, s64 lod)
        : own_(std::in_place, a), src_(&*own_), lod_(lod), dims_(a.dims_at(lod)),
          u8_(a.src_dtype() == codec::DType::u8) {}

    [[nodiscard]] bool failed() const { return failed_; }
    [[nodiscard]] Extent3 dims() const { return dims_; }

    // Voxel fetch, edge-clamped.
    [[nodiscard]] f32 at(s64 z, s64 y, s64 x) {
        z = std::clamp<s64>(z, 0, dims_.z - 1);
        y = std::clamp<s64>(y, 0, dims_.y - 1);
        x = std::clamp<s64>(x, 0, dims_.x - 1);
        constexpr s64 N = codec::kDctN;
        const ChunkCoord bc{z / N, y / N, x / N};
        const codec::BlockCache::Block* const b = block_(bc);
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
    const codec::BlockCache::Block* block_(ChunkCoord bc) {
        if (bc.z == bc0_.z && bc.y == bc0_.y && bc.x == bc0_.x && b0_) return b0_.get();
        if (bc.z == bc1_.z && bc.y == bc1_.y && bc.x == bc1_.x && b1_) {
            std::swap(bc0_, bc1_);
            std::swap(b0_, b1_);
            return b0_.get();
        }
        auto r = src_->block16(lod_, bc);
        if (!r) {
            failed_ = true;
            return nullptr;
        }
        bc1_ = bc0_;
        b1_ = std::move(b0_);
        bc0_ = bc;
        b0_ = std::move(*r);
        return b0_.get();
    }

    std::optional<codec::ArchiveSource> own_;  // set by the archive-convenience ctor only
    codec::VolumeSource* src_;
    s64 lod_;
    Extent3 dims_;
    bool u8_;
    bool failed_ = false;
    ChunkCoord bc0_{-1, -1, -1}, bc1_{-1, -1, -1};
    codec::BlockCache::Ref b0_, b1_;
};

}  // namespace fenix::view
