// ml/sampler.hpp — the occupancy-guided training patch sampler (torch-free, always built).
// Guided by the MESH, not CT occupancy: the mesh is the supervision, so patches centered on
// valid surface cells are exactly the material-rich, label-bearing samples. Deterministic:
// draw(i) depends only on (seed, i) — reproducible runs, resumable mid-epoch, and the
// producer threads of `fenix train-feed` can partition i-space without coordination.
#pragma once

#include "core/core.hpp"
#include "core/hash.hpp"
#include "core/surface.hpp"
#include "core/vec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace fenix::ml {

struct PatchDraw {
    s32 mesh = -1;   // index into the sampler's mesh list
    Vec3f center{};  // surface point (LOD-0 voxels, same space as the paired CT volume)
};

// Samples (mesh, surface-point) pairs: mesh ∝ its valid-cell count, cell ~uniform over the
// mesh's valid cells (rejection over the grid — the corpus is ~85-90% valid so this
// terminates fast), plus a ±jitter so patches aren't always cell-centered.
class PatchSampler {
  public:
    // `meshes` must outlive the sampler. jitter = max |offset| per axis in voxels.
    PatchSampler(std::span<const Surface* const> meshes, u64 seed, f32 jitter = 64.0f)
        : meshes_(meshes.begin(), meshes.end()), seed_(seed), jitter_(jitter) {
        cum_.reserve(meshes_.size());
        s64 acc = 0;
        for (const Surface* s : meshes_) {
            acc += s ? s->valid_count() : 0;
            cum_.push_back(acc);
        }
        total_ = acc;
    }

    [[nodiscard]] s64 total_weight() const { return total_; }

    // Deterministic draw #i. Returns mesh<0 only if the corpus has zero valid cells.
    [[nodiscard]] PatchDraw draw(u64 i) const {
        PatchDraw d;
        if (total_ <= 0) return d;
        // one u64 hash stream per draw: h0 picks the mesh, h1.. drive the rejection + jitter
        const s64 pick = static_cast<s64>(hash_value(std::array<u64, 2>{seed_, i}) % static_cast<u64>(total_));
        usize m = 0;
        while (m + 1 < cum_.size() && cum_[m] <= pick) ++m;
        const Surface& s = *meshes_[m];
        d.mesh = static_cast<s32>(m);
        for (u64 attempt = 0;; ++attempt) {
            const u64 h = hash_value(std::array<u64, 3>{seed_, i, attempt});
            const s64 u = static_cast<s64>(h % static_cast<u64>(s.nu));
            const s64 v = static_cast<s64>((h >> 32) % static_cast<u64>(s.nv));
            if (!s.is_valid(u, v)) continue;
            const u64 hj = hash_value(std::array<u64, 3>{seed_ ^ 0x9e3779b97f4a7c15ull, i, attempt});
            auto jit = [&](int k) {
                // map 16 hash bits -> [-jitter, jitter]
                const f32 t = static_cast<f32>((hj >> (16 * k)) & 0xffff) / 65535.0f;
                return (t * 2.0f - 1.0f) * jitter_;
            };
            Vec3f c = s.at(u, v);
            d.center = Vec3f{c.z + jit(0), c.y + jit(1), c.x + jit(2)};
            return d;
        }
    }

    // Clamp a patch of `extent` around `center` into [0, dims): returns the patch origin.
    static Index3 patch_origin(Vec3f center, Extent3 extent, Extent3 dims) {
        auto cl = [](f32 c, s64 e, s64 dmax) {
            const s64 o = static_cast<s64>(std::lround(c)) - e / 2;
            return std::clamp<s64>(o, 0, std::max<s64>(0, dmax - e));
        };
        return Index3{cl(center.z, extent.z, dims.z), cl(center.y, extent.y, dims.y), cl(center.x, extent.x, dims.x)};
    }

  private:
    std::vector<const Surface*> meshes_;
    std::vector<s64> cum_;  // cumulative valid-cell weights
    s64 total_ = 0;
    u64 seed_;
    f32 jitter_;
};

}  // namespace fenix::ml
