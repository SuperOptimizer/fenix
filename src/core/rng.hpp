// core/rng.hpp — seeded deterministic RNG (PCG). Used for sampling (e.g. material-rich
// box draws); same seed => same sequence regardless of threads (draw per-stream).
#pragma once

#include "core/types.hpp"

namespace fenix {

// PCG-XSH-RR 64/32 (O'Neill 2014). Compact, fast, well-distributed. 32-bit output.
class Pcg32 {
public:
    constexpr Pcg32() = default;
    constexpr explicit Pcg32(u64 seed, u64 stream = 0xda3e39cb94b95bdbULL) {
        inc_ = (stream << 1u) | 1u;
        state_ = 0;
        next_u32();
        state_ += seed;
        next_u32();
    }

    constexpr u32 next_u32() {
        u64 old = state_;
        state_ = old * 6364136223846793005ULL + inc_;
        u32 xorshifted = static_cast<u32>(((old >> 18u) ^ old) >> 27u);
        u32 rot = static_cast<u32>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    constexpr u64 next_u64() {
        u64 hi = next_u32();
        return (hi << 32) | next_u32();
    }

    // Uniform f32 in [0, 1).
    f32 next_f32() { return static_cast<f32>(next_u32() >> 8) * (1.0f / 16777216.0f); }

    // Uniform integer in [0, bound) without modulo bias.
    constexpr u32 bounded(u32 bound) {
        u32 threshold = (~bound + 1u) % bound;
        for (;;) {
            u32 r = next_u32();
            if (r >= threshold) return r % bound;
        }
    }

private:
    u64 state_ = 0x853c49e6748fea9bULL;
    u64 inc_ = 0xda3e39cb94b95bdbULL;
};

}  // namespace fenix
