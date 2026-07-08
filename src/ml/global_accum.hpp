// ml/global_accum.hpp — torch-free machinery for the GLOBAL (zero-waste) predict-scroll grid:
// one patch grid over the whole scroll bbox, every patch computed exactly once (no region halo,
// no discarded compute), Gaussian-blended into a sparse chunk-keyed accumulator that a z-sweep
// finalizes (normalize → u8 → sink → evict) behind the last row that can still touch it.
//
// Because the global tile set is the full Cartesian product zs×ys×xs, the accumulated Gaussian
// weight factorizes exactly: wacc(z,y,x) = Wz(z)·Wy(y)·Wx(x). Three 1-D profiles replace any
// per-voxel weight volume — the accumulator stores ONLY the f32 probability sum (4 B/voxel,
// sparse). Air-skipped / constant-CT patches leave their contribution at 0 while the profile
// still counts them — voxels they touch normalize low. This mirrors the region path's
// tile_filter semantics exactly (those voxels are air; attenuation there is accepted).
#pragma once

#include "core/core.hpp"
#include "ml/tiling.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace fenix::ml {

// Accumulated 1-D Gaussian weight profile along one axis of the GLOBAL grid: for each bbox-local
// coordinate, the sum of gaussian1d weights of every tile start covering it. Same math as the
// region path's weight_profile, over the full bbox span.
inline std::vector<float>
global_weight_profile(const std::vector<float>& g, const std::vector<s64>& starts, s64 dim, int P) {
    std::vector<float> w(static_cast<usize>(dim), 0.0f);
    for (s64 s0 : starts)
        for (int i = 0; i < P && s0 + i < dim; ++i) w[static_cast<usize>(s0 + i)] += g[static_cast<usize>(i)];
    return w;
}

// Sparse chunk-keyed (64³) f32 probability accumulator over a 64-aligned bbox [org, org+ext).
// scatter() is thread-safe (per-block mutex; block creation under a shard lock). finalize_below()
// must be called from ONE thread (the writer); it may run concurrently with scatters ABOVE the
// finalize limit — the sweep protocol (rows complete in order before the limit advances)
// guarantees no scatter ever lands below it.
class GlobalAccum {
  public:
    static constexpr s64 T = 64;

    GlobalAccum(Index3 org, Extent3 ext, std::vector<float> Wz, std::vector<float> Wy, std::vector<float> Wx)
        : org_(org),
          ext_(ext),
          Wz_(std::move(Wz)),
          Wy_(std::move(Wy)),
          Wx_(std::move(Wx)),
          cz0_(org.z / T),
          cy0_(org.y / T),
          cx0_(org.x / T),
          ncz_((ext.z + T - 1) / T),
          ncy_((ext.y + T - 1) / T),
          ncx_((ext.x + T - 1) / T) {
        FENIX_ASSERT(org.z % T == 0 && org.y % T == 0 && org.x % T == 0);
        floor_row_ = 0;
    }

    // Resume: chunk rows below `floor_row` (bbox-local chunk-z index) are already finalized in the
    // output archive — scatters below it are dropped, finalize skips it.
    void set_floor_row(s64 floor_row) { floor_row_ = std::clamp<s64>(floor_row, 0, ncz_); }
    [[nodiscard]] s64 floor_row() const { return floor_row_; }
    [[nodiscard]] s64 chunk_rows() const { return ncz_; }
    [[nodiscard]] usize live_blocks() const {
        usize n = 0;
        for (const auto& sh : shards_) {
            std::lock_guard lk(sh.m);
            n += sh.blocks.size();
        }
        return n;
    }

    // Gaussian-blend a P³ patch result `sp` at SCROLL coords (z0,y0,x0) into the accumulator.
    // gz/gy/gx are the per-axis gaussian1d(P) windows. Voxels outside the bbox or below the
    // finalized floor are skipped (the resume re-run rows overlap the floor by design).
    void scatter(s64 z0,
                 s64 y0,
                 s64 x0,
                 int P,
                 const float* sp,
                 const std::vector<float>& gz,
                 const std::vector<float>& gy,
                 const std::vector<float>& gx) {
        const s64 lz0 = z0 - org_.z, ly0 = y0 - org_.y, lx0 = x0 - org_.x;  // bbox-local patch org
        const s64 zlo = std::max<s64>(std::max<s64>(lz0, 0), floor_row_ * T);
        const s64 zhi = std::min<s64>(lz0 + P, ext_.z);
        const s64 ylo = std::max<s64>(ly0, 0), yhi = std::min<s64>(ly0 + P, ext_.y);
        const s64 xlo = std::max<s64>(lx0, 0), xhi = std::min<s64>(lx0 + P, ext_.x);
        if (zlo >= zhi || ylo >= yhi || xlo >= xhi) return;
        // Walk the overlapped 64³ chunks; accumulate this patch's intersection under the block lock.
        for (s64 cz = zlo / T; cz <= (zhi - 1) / T; ++cz)
            for (s64 cy = ylo / T; cy <= (yhi - 1) / T; ++cy)
                for (s64 cx = xlo / T; cx <= (xhi - 1) / T; ++cx) {
                    Block& b = block(cz, cy, cx);
                    const s64 bz0 = cz * T, by0 = cy * T, bx0 = cx * T;
                    const s64 iz0 = std::max(zlo, bz0), iz1 = std::min(zhi, bz0 + T);
                    const s64 iy0 = std::max(ylo, by0), iy1 = std::min(yhi, by0 + T);
                    const s64 ix0 = std::max(xlo, bx0), ix1 = std::min(xhi, bx0 + T);
                    std::lock_guard lk(b.m);
                    for (s64 z = iz0; z < iz1; ++z) {
                        const float wz = gz[static_cast<usize>(z - lz0)];
                        for (s64 y = iy0; y < iy1; ++y) {
                            const float wzy = wz * gy[static_cast<usize>(y - ly0)];
                            float* row = b.data.get() + ((z - bz0) * T + (y - by0)) * T;
                            const float* srow = sp + (static_cast<usize>(z - lz0) * static_cast<usize>(P) +
                                                      static_cast<usize>(y - ly0)) *
                                                         static_cast<usize>(P);
                            for (s64 x = ix0; x < ix1; ++x)
                                row[x - bx0] += wzy * gx[static_cast<usize>(x - lx0)] * srow[x - lx0];
                        }
                    }
                }
    }

    // Finalize every bbox chunk row STRICTLY below bbox-local chunk-z `row_lim` that isn't already
    // finalized: normalize by the factorized profiles, clamp to [0,1] → u8, and hand each chunk to
    // `sink(ChunkCoord /*archive coords*/, const u8* /*T³ ZYX*/)`. Chunks never scattered into are
    // emitted as zeros (assessed air — coverage Zero in the archive, distinct from Absent). Blocks
    // are freed as they finalize. Single-caller (writer thread).
    template <class Sink> void finalize_rows_below(s64 row_lim, Sink&& sink) {
        row_lim = std::min(row_lim, ncz_);
        std::vector<u8> blk(static_cast<usize>(T * T * T));
        for (s64 cz = floor_row_; cz < row_lim; ++cz) {
            for (s64 cy = 0; cy < ncy_; ++cy)
                for (s64 cx = 0; cx < ncx_; ++cx) {
                    std::unique_ptr<float[]> data = take_block(cz, cy, cx);
                    if (!data) {
                        std::fill(blk.begin(), blk.end(), u8{0});
                    } else {
                        const s64 bz0 = cz * T, by0 = cy * T, bx0 = cx * T;
                        for (s64 z = 0; z < T; ++z) {
                            const float wz = bz0 + z < ext_.z ? Wz_[static_cast<usize>(bz0 + z)] : 0.0f;
                            for (s64 y = 0; y < T; ++y) {
                                const float wzy = by0 + y < ext_.y ? wz * Wy_[static_cast<usize>(by0 + y)] : 0.0f;
                                const float* row = data.get() + (z * T + y) * T;
                                u8* orow = blk.data() + (z * T + y) * T;
                                for (s64 x = 0; x < T; ++x) {
                                    const float w = bx0 + x < ext_.x ? wzy * Wx_[static_cast<usize>(bx0 + x)] : 0.0f;
                                    const float v = w > 0.0f ? row[x] / w : 0.0f;
                                    orow[x] = static_cast<u8>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
                                }
                            }
                        }
                    }
                    sink(ChunkCoord{cz0_ + cz, cy0_ + cy, cx0_ + cx}, blk.data());
                }
        }
        floor_row_ = std::max(floor_row_, row_lim);
    }

  private:
    struct Block {
        std::unique_ptr<float[]> data;
        std::mutex m;
    };
    static constexpr usize kShards = 64;
    struct Shard {
        mutable std::mutex m;
        std::unordered_map<u64, std::unique_ptr<Block>> blocks;
    };

    [[nodiscard]] u64 key(s64 cz, s64 cy, s64 cx) const {
        return (static_cast<u64>(cz) * static_cast<u64>(ncy_) + static_cast<u64>(cy)) * static_cast<u64>(ncx_) +
               static_cast<u64>(cx);
    }
    Block& block(s64 cz, s64 cy, s64 cx) {
        const u64 k = key(cz, cy, cx);
        Shard& sh = shards_[static_cast<usize>(hash_value(k) % kShards)];
        std::lock_guard lk(sh.m);
        auto& up = sh.blocks[k];
        if (!up) {
            up = std::make_unique<Block>();
            up->data = std::make_unique<float[]>(static_cast<usize>(T * T * T));
            std::fill(up->data.get(), up->data.get() + T * T * T, 0.0f);
        }
        return *up;
    }
    std::unique_ptr<float[]> take_block(s64 cz, s64 cy, s64 cx) {
        const u64 k = key(cz, cy, cx);
        Shard& sh = shards_[static_cast<usize>(hash_value(k) % kShards)];
        std::lock_guard lk(sh.m);
        auto it = sh.blocks.find(k);
        if (it == sh.blocks.end()) return nullptr;
        auto data = std::move(it->second->data);
        sh.blocks.erase(it);
        return data;
    }

    Index3 org_;
    Extent3 ext_;
    std::vector<float> Wz_, Wy_, Wx_;
    s64 cz0_, cy0_, cx0_, ncz_, ncy_, ncx_;
    s64 floor_row_ = 0;
    Shard shards_[kShards];
};

}  // namespace fenix::ml
