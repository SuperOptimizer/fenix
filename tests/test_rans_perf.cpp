// test_rans_perf.cpp — isolated rANS microbenchmark: single-state (K=1) vs interleaved (K=4) on the
// SAME large skewed byte buffer, to measure the interleave's effect on the entropy coder ITSELF
// (the codec bench is transform-bound + its decode timer includes error-stats, so it can't see this).
// Standalone main (NOT a unit test).
#include "codec/rans.hpp"
#include "core/core.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace fenix;
using namespace fenix::codec;
using clk = std::chrono::steady_clock;
static double mbs(usize bytes, clk::time_point a, clk::time_point b) {
    return (static_cast<double>(bytes) / 1e6) / std::chrono::duration<double>(b - a).count();
}

namespace {
// Single-state (K=1) reference encode/decode, inlined here so we A/B in one run (no rebuild).
std::vector<u8> enc_k1(std::span<const u8> data, const RansModel& m) {
    std::vector<u8> buf(data.size() * 2 + 64);
    usize ptr = buf.size();
    u32 x = rans_byte_l;
    for (usize i = data.size(); i-- > 0;) {
        const u8 s = data[i];
        const u32 f = m.freq[s], c = m.cum[s];
        const u32 xm = ((rans_byte_l >> rans_scale_bits) << 8) * f;
        while (x >= xm) { buf[--ptr] = static_cast<u8>(x & 0xffu); x >>= 8; }
        x = ((x / f) << rans_scale_bits) + (x % f) + c;
    }
    ptr -= 4;
    std::memcpy(buf.data() + ptr, &x, 4);
    return std::vector<u8>(buf.begin() + static_cast<isize>(ptr), buf.end());
}
std::vector<u8> dec_k1(std::span<const u8> stream, usize n, const RansModel& m) {
    std::vector<u8> out(n);
    usize p = 0;
    u32 x;
    std::memcpy(&x, stream.data() + p, 4);
    p += 4;
    const u32 mask = rans_scale - 1;
    for (usize i = 0; i < n; ++i) {
        const u32 s = m.slot[x & mask];
        out[i] = static_cast<u8>(s);
        x = m.freq[s] * (x >> rans_scale_bits) + (x & mask) - m.cum[s];
        while (x < rans_byte_l && p < stream.size()) x = (x << 8) | stream[p++];
    }
    return out;
}
}  // namespace

int main() {
    const usize N = 64u << 20;  // 64 MB
    std::vector<u8> data(N);
    u64 seed = 0x9e3779b97f4a7c15ull;  // skewed (geometric-ish): mostly small bytes, like quantized coefs
    for (u8& b : data) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const u32 r = static_cast<u32>(seed >> 40) & 0xffffu;
        b = static_cast<u8>(__builtin_ctz(r | 0x10000u));  // P(0)=1/2, P(1)=1/4, ...
    }
    std::array<u64, 256> cnt{};
    for (u8 b : data) cnt[b]++;
    const RansModel m = RansModel::from_counts(std::span<const u64>(cnt));

    for (int rep = 0; rep < 3; ++rep) {
        auto a0 = clk::now();
        const std::vector<u8> e1 = enc_k1(data, m);
        auto a1 = clk::now();
        const std::vector<u8> d1 = dec_k1(e1, N, m);
        auto a2 = clk::now();
        const std::vector<u8> e4 = rans_encode(data, m);  // interleaved (K=4, N >> threshold)
        auto a3 = clk::now();
        const std::vector<u8> d4 = rans_decode(e4, N, m);
        auto a4 = clk::now();
        std::printf("rep%d  K1: enc %.0f dec %.0f MB/s (%.3fx) | K4: enc %.0f dec %.0f MB/s (%.3fx)  ok=%d/%d\n",
                    rep, mbs(N, a0, a1), mbs(N, a1, a2), static_cast<double>(N) / static_cast<double>(e1.size()),
                    mbs(N, a2, a3), mbs(N, a3, a4), static_cast<double>(N) / static_cast<double>(e4.size()),
                    static_cast<int>(d1 == data), static_cast<int>(d4 == data));
    }
    return 0;
}
