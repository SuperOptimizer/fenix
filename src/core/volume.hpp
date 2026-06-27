// core/volume.hpp — the one 3D array/view type. ZYX (z-major, x-fastest), explicit
// 64-bit strides, owning + non-owning variants. THE only voxel accessor in fenix:
// never hand-roll an index macro (docs/conventions.md, the #1 taberna bug source).
#pragma once

#include "core/types.hpp"

#include <algorithm>
#include <memory>
#include <span>

namespace fenix {

// Non-owning strided view. Strides are in elements (not bytes). Default contiguous ZYX:
// stride_x = 1, stride_y = nx, stride_z = nx*ny. Sub-views reuse parent strides.
// T may be cv-qualified (e.g. VolumeView<const f32>) for read-only views.
template <class T>
    requires VoxelScalar<std::remove_cv_t<T>>
class VolumeView {
public:
    constexpr VolumeView() = default;

    constexpr VolumeView(T* data, Extent3 dims, Index3 strides)
        : data_(data), dims_(dims), strides_(strides) {}

    static constexpr Index3 contiguous_strides(Extent3 d) {
        return {d.y * d.x, d.x, 1};  // z, y, x strides for contiguous ZYX
    }

    constexpr VolumeView(T* data, Extent3 dims)
        : VolumeView(data, dims, contiguous_strides(dims)) {}

    // Non-const -> const view conversion (e.g. VolumeView<f32> -> VolumeView<const f32>).
    template <class U>
        requires std::convertible_to<U*, T*>
    constexpr VolumeView(VolumeView<U> o) : data_(o.data()), dims_(o.dims()), strides_(o.strides()) {}

    [[nodiscard]] constexpr Extent3 dims() const { return dims_; }
    [[nodiscard]] constexpr Index3 strides() const { return strides_; }
    [[nodiscard]] constexpr T* data() const { return data_; }
    [[nodiscard]] constexpr s64 size() const { return dims_.count(); }
    [[nodiscard]] constexpr bool empty() const { return data_ == nullptr || size() == 0; }

    [[nodiscard]] constexpr bool in_bounds(s64 z, s64 y, s64 x) const {
        return z >= 0 && z < dims_.z && y >= 0 && y < dims_.y && x >= 0 && x < dims_.x;
    }

    [[nodiscard]] constexpr s64 offset(s64 z, s64 y, s64 x) const {
        return z * strides_.z + y * strides_.y + x * strides_.x;
    }

    // Unchecked access (hot path). Debug builds still catch OOB via FENIX_ASSERT callers.
    constexpr T& operator()(s64 z, s64 y, s64 x) const { return data_[offset(z, y, x)]; }

    // Clamp-to-edge sampling (safe at borders).
    constexpr T at_clamped(s64 z, s64 y, s64 x) const {
        z = std::clamp<s64>(z, 0, dims_.z - 1);
        y = std::clamp<s64>(y, 0, dims_.y - 1);
        x = std::clamp<s64>(x, 0, dims_.x - 1);
        return data_[offset(z, y, x)];
    }

    [[nodiscard]] constexpr bool is_contiguous() const {
        return strides_ == contiguous_strides(dims_);
    }

    // Flat span — only valid when contiguous.
    [[nodiscard]] constexpr std::span<T> flat() const {
        return is_contiguous() ? std::span<T>(data_, static_cast<usize>(size())) : std::span<T>{};
    }

    // Axis-aligned sub-box [origin, origin+extent) reusing this view's strides.
    [[nodiscard]] constexpr VolumeView crop(Index3 origin, Extent3 extent) const {
        return VolumeView(data_ + offset(origin.z, origin.y, origin.x), extent, strides_);
    }

private:
    T* data_ = nullptr;
    Extent3 dims_{};
    Index3 strides_{};
};

// Owning contiguous ZYX volume.
template <VoxelScalar T>
class Volume {
public:
    Volume() = default;

    explicit Volume(Extent3 dims)
        : dims_(dims),
          storage_(std::make_unique_for_overwrite<T[]>(static_cast<usize>(dims.count()))) {}

    static Volume zeros(Extent3 dims) {
        Volume v;
        v.dims_ = dims;
        v.storage_ = std::make_unique<T[]>(static_cast<usize>(dims.count()));  // value-inits 0
        return v;
    }

    // Deep copy (models/volumes are copyable); moves stay cheap.
    Volume(const Volume& o) : dims_(o.dims_) {
        if (o.storage_) {
            storage_ = std::make_unique_for_overwrite<T[]>(static_cast<usize>(dims_.count()));
            std::copy_n(o.storage_.get(), static_cast<usize>(dims_.count()), storage_.get());
        }
    }
    Volume& operator=(const Volume& o) {
        if (this != &o) {
            Volume tmp(o);
            *this = std::move(tmp);
        }
        return *this;
    }
    Volume(Volume&&) noexcept = default;
    Volume& operator=(Volume&&) noexcept = default;
    ~Volume() = default;

    [[nodiscard]] Extent3 dims() const { return dims_; }
    [[nodiscard]] s64 size() const { return dims_.count(); }
    [[nodiscard]] bool empty() const { return size() == 0; }
    [[nodiscard]] T* data() { return storage_.get(); }
    [[nodiscard]] const T* data() const { return storage_.get(); }

    [[nodiscard]] VolumeView<T> view() { return VolumeView<T>(storage_.get(), dims_); }
    [[nodiscard]] VolumeView<const T> view() const {
        return VolumeView<const T>(storage_.get(), dims_);
    }

    T& operator()(s64 z, s64 y, s64 x) { return view()(z, y, x); }
    const T& operator()(s64 z, s64 y, s64 x) const { return view()(z, y, x); }

    [[nodiscard]] std::span<T> flat() { return {storage_.get(), static_cast<usize>(size())}; }
    [[nodiscard]] std::span<const T> flat() const {
        return {storage_.get(), static_cast<usize>(size())};
    }

private:
    Extent3 dims_{};
    std::unique_ptr<T[]> storage_;
};

}  // namespace fenix
