// core/types.hpp — fixed-width scalar aliases, strong units, and 3D extents/indices.
// Encodes the ZYX + unit foot-guns in the type system (see docs/conventions.md).
#pragma once

#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace fenix {

// ---- fixed-width scalars ---------------------------------------------------
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using s8 = std::int8_t;
using s16 = std::int16_t;
using s32 = std::int32_t;
using s64 = std::int64_t;
using f16 = _Float16;  // IEEE half (clang builtin); widens to/narrows from f32 for the codec dtype layer
using f32 = float;
using f64 = double;
using usize = std::size_t;
using isize = std::ptrdiff_t;

// Dense voxel dtypes the codec/Volume support (NO f64/u64/s64 dense — see fenix-codec).
template <class T>
concept VoxelScalar = std::same_as<T, u8> || std::same_as<T, u16> || std::same_as<T, u32> ||
                      std::same_as<T, s8> || std::same_as<T, s16> || std::same_as<T, s32> ||
                      std::same_as<T, f16> || std::same_as<T, f32>;

// ---- strong units ----------------------------------------------------------
// A minimal strong-typedef wrapper: explicit construction, comparable, with the few
// arithmetic ops that make physical sense. Prevents mixing microns with keV, etc.
template <class Tag, class Rep>
struct Quantity {
    Rep value{};
    constexpr Quantity() = default;
    constexpr explicit Quantity(Rep v) : value(v) {}
    friend constexpr auto operator<=>(Quantity, Quantity) = default;
    constexpr Quantity operator+(Quantity o) const { return Quantity{value + o.value}; }
    constexpr Quantity operator-(Quantity o) const { return Quantity{value - o.value}; }
    constexpr Quantity operator*(Rep s) const { return Quantity{value * s}; }
    constexpr Quantity operator/(Rep s) const { return Quantity{value / s}; }
};

struct MicronTag {};
struct KeVTag {};
struct RadianTag {};
struct WindingTag {};
using Micron = Quantity<MicronTag, f32>;        // physical length / voxel size
using KeV = Quantity<KeVTag, f32>;              // scan energy
using Radian = Quantity<RadianTag, f32>;        // angle
using WindingNumber = Quantity<WindingTag, f32>;// continuous winding (level sets = sheets)

// LOD: 0 = full resolution, higher = coarser.
enum class Lod : u32 {};
constexpr Lod lod(u32 l) { return static_cast<Lod>(l); }
constexpr u32 to_u32(Lod l) { return static_cast<u32>(l); }

// ---- 3D extents & indices (ZYX) -------------------------------------------
// Sizes are i64 so index math never overflows at scroll scale (2^18/axis, 2^54 dense).
struct Extent3 {
    s64 z = 0, y = 0, x = 0;
    [[nodiscard]] constexpr s64 count() const { return z * y * x; }
    friend constexpr bool operator==(Extent3, Extent3) = default;
};

// Integer voxel coordinate (z,y,x). Distinct from a spatial direction (see Vec3).
struct Index3 {
    s64 z = 0, y = 0, x = 0;
    friend constexpr bool operator==(Index3, Index3) = default;
};

// Chunk coordinate = voxel >> 6 (64^3 chunks). Tagged distinct from voxel Index3.
struct ChunkCoord {
    s64 z = 0, y = 0, x = 0;
    friend constexpr bool operator==(ChunkCoord, ChunkCoord) = default;
};

inline constexpr s64 chunk_side = 64;
constexpr ChunkCoord chunk_of(Index3 v) {
    return {v.z / chunk_side, v.y / chunk_side, v.x / chunk_side};
}

}  // namespace fenix
