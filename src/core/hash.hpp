// core/hash.hpp — fast non-cryptographic 64-bit hash for integrity tags + keys.
// wyhash-style mixing (public-domain lineage); xxh3-class quality at trivial size.
// Deterministic within a build (we only need within-build consistency, never crypto).
#pragma once

#include "core/types.hpp"

#include <cstring>
#include <span>

namespace fenix {

namespace detail {
inline constexpr u64 wy_p0 = 0xa0761d6478bd642fULL;
inline constexpr u64 wy_p1 = 0xe7037ed1a0b428dbULL;

// 64x64 -> 128 multiply, folded (the wymum primitive).
inline u64 wymix(u64 a, u64 b) {
    __uint128_t r = static_cast<__uint128_t>(a) * b;
    return static_cast<u64>(r) ^ static_cast<u64>(r >> 64);
}

inline u64 wy_read64(const u8* p) {
    u64 v;
    std::memcpy(&v, p, 8);
    return v;
}
inline u64 wy_read32(const u8* p) {
    u32 v;
    std::memcpy(&v, p, 4);
    return v;
}
}  // namespace detail

// Hash an arbitrary byte span. Seed lets you derive independent hash families.
inline u64 hash64(std::span<const u8> data, u64 seed = 0) {
    using namespace detail;
    const u8* p = data.data();
    usize len = data.size();
    u64 a = 0, b = 0;
    seed ^= wymix(seed ^ wy_p0, wy_p1);
    if (len <= 16) {
        if (len >= 4) {
            a = (wy_read32(p) << 32) | wy_read32(p + ((len >> 3) << 2));
            b = (wy_read32(p + len - 4) << 32) | wy_read32(p + len - 4 - ((len >> 3) << 2));
        } else if (len > 0) {
            a = (static_cast<u64>(p[0]) << 16) | (static_cast<u64>(p[len >> 1]) << 8) | p[len - 1];
        }
    } else {
        usize i = len;
        while (i > 16) {
            seed = wymix(wy_read64(p) ^ wy_p1, wy_read64(p + 8) ^ seed);
            p += 16;
            i -= 16;
        }
        a = wy_read64(p + i - 16);
        b = wy_read64(p + i - 8);
    }
    return wymix(wy_p1 ^ len, wymix(a ^ wy_p1, b ^ seed));
}

// Convenience for trivially-copyable values (e.g. chunk coords, headers).
template <class T>
    requires std::is_trivially_copyable_v<T>
u64 hash_value(const T& v, u64 seed = 0) {
    return hash64({reinterpret_cast<const u8*>(&v), sizeof(T)}, seed);
}

}  // namespace fenix
