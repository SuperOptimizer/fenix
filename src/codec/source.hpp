// codec/source.hpp — the abstract multi-LOD voxel source the viewer engine renders from.
// Exactly the read surface of VolumeArchive that view/ needs (pyramid dims, dense box
// gathers, 16³ decoded-block access, cache sizing), as a virtual interface so a source
// can also be a *streaming* stack (io::CachedPyramid: per-level .fxvol caches lazily
// filled from a remote zarr). Methods are non-const because streaming sources mutate
// (fetch + append) on read; implementations must be thread-safe like VolumeArchive.
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"

namespace fenix::codec {

class VolumeSource {
  public:
    virtual ~VolumeSource() = default;

    [[nodiscard]] virtual u32 nlods() const = 0;
    [[nodiscard]] virtual Extent3 dims_at(s64 lod) const = 0;
    [[nodiscard]] virtual DType src_dtype() const = 0;
    [[nodiscard]] virtual ChunkCoord chunk_extent(s64 lod) const = 0;
    // 16³ decoded block (see VolumeArchive::block16). Streaming sources fetch the
    // covering chunk first — a failure is a hard error, never silent air.
    [[nodiscard]] virtual Expected<BlockCache::Ref> block16(s64 lod, ChunkCoord bc) = 0;
    virtual Expected<void>
    gather_box_f32(s64 lod, s64 oz, s64 oy, s64 ox, s64 D, s64 H, s64 W, f32* out) = 0;
    virtual void reserve_cache(u64 bytes) = 0;

    [[nodiscard]] Extent3 dims() const { return dims_at(0); }
};

// The trivial adapter over a local archive; `arch` must outlive it.
class ArchiveSource final : public VolumeSource {
  public:
    explicit ArchiveSource(VolumeArchive& arch) : a_(&arch) {}

    [[nodiscard]] u32 nlods() const override { return a_->nlods(); }
    [[nodiscard]] Extent3 dims_at(s64 lod) const override { return a_->dims_at(lod); }
    [[nodiscard]] DType src_dtype() const override { return a_->src_dtype(); }
    [[nodiscard]] ChunkCoord chunk_extent(s64 lod) const override { return a_->chunk_extent(lod); }
    [[nodiscard]] Expected<BlockCache::Ref> block16(s64 lod, ChunkCoord bc) override {
        return a_->block16(lod, bc);
    }
    Expected<void>
    gather_box_f32(s64 lod, s64 oz, s64 oy, s64 ox, s64 D, s64 H, s64 W, f32* out) override {
        return a_->gather_box_f32(lod, oz, oy, ox, D, H, W, out);
    }
    void reserve_cache(u64 bytes) override { a_->reserve_cache(bytes); }

  private:
    VolumeArchive* a_;
};

}  // namespace fenix::codec
