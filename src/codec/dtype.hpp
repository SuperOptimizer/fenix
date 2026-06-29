// codec/dtype.hpp — the shared dtype I/O layer for both transform codecs. Every codec works in f32
// (the transform domain); this widens any input dtype to f32 and narrows the reconstruction back
// (round-to-nearest + clamp for integers, range-clamp for f16, pass-through for f32). The source dtype
// is stored in the block header so decode can reproduce it. Supports u8/u16/u32/s8/s16/s32/f16/f32.
// Caveat: u32/s32 magnitudes above 2^24 lose low bits in f32 — acceptable for a lossy codec; the
// integer-LABEL path uses the separate lossless codec, not this.
#pragma once

#include "core/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace fenix::codec {

enum class DType : u8 { u8 = 0, u16, u32, s8, s16, s32, f16, f32 };

template <class T> struct dtype_traits;  // maps a C++ scalar to its DType id
template <> struct dtype_traits<fenix::u8> { static constexpr DType id = DType::u8; };
template <> struct dtype_traits<fenix::u16> { static constexpr DType id = DType::u16; };
template <> struct dtype_traits<fenix::u32> { static constexpr DType id = DType::u32; };
template <> struct dtype_traits<fenix::s8> { static constexpr DType id = DType::s8; };
template <> struct dtype_traits<fenix::s16> { static constexpr DType id = DType::s16; };
template <> struct dtype_traits<fenix::s32> { static constexpr DType id = DType::s32; };
template <> struct dtype_traits<fenix::f16> { static constexpr DType id = DType::f16; };
template <> struct dtype_traits<fenix::f32> { static constexpr DType id = DType::f32; };

template <class T>
constexpr DType dtype_of() {
    return dtype_traits<T>::id;
}

// widen a typed span to f32.
template <class T>
inline std::vector<f32> to_f32(std::span<const T> in) {
    std::vector<f32> o(in.size());
    for (usize i = 0; i < in.size(); ++i) o[i] = static_cast<f32>(in[i]);
    return o;
}

// narrow one f32 back to T: integers round-to-nearest + clamp to range; f16 clamps to its finite range;
// f32 passes through.
template <class T>
inline T from_f32_one(f32 v) {
    if constexpr (std::is_integral_v<T>) {
        const f64 r = std::nearbyint(static_cast<f64>(v));
        const f64 lo = static_cast<f64>(std::numeric_limits<T>::min());
        const f64 hi = static_cast<f64>(std::numeric_limits<T>::max());
        return static_cast<T>(std::clamp(r, lo, hi));
    } else if constexpr (std::same_as<T, f16>) {
        return static_cast<f16>(std::clamp(v, -65504.0f, 65504.0f));
    } else {
        return static_cast<T>(v);  // f32
    }
}

template <class T>
inline std::vector<T> from_f32(std::span<const f32> in) {
    std::vector<T> o(in.size());
    for (usize i = 0; i < in.size(); ++i) o[i] = from_f32_one<T>(in[i]);
    return o;
}

}  // namespace fenix::codec
