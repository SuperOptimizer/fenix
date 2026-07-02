// fuzz_lossless_decode.cpp — libFuzzer harness for codec/lossless.hpp's lossless_decode<T>,
// the general lossless (rANS byte-plane) decoder. Reachable from any bytes claiming to be an
// encoded label/mask/exact-prior blob inside a .fxvol; codec/CLAUDE.md's "robustness is a
// hard rule: no UB/crash on any bytes" makes this in-scope even though lossless_decode has
// no Expected<> return (it's a raw std::vector<T> decode, not yet a validated entry point).
#include <cstdint>
#include <span>

#include "codec/lossless.hpp"
#include "core/core.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::span<const fenix::u8> in{data, size};
    // Exercise a couple of element widths; the byte-plane count/splitting logic differs by
    // sizeof(T), and both are reachable call sites in the codebase (labels=s32, masks=u8).
    (void)fenix::codec::lossless_decode<fenix::u8>(in);
    (void)fenix::codec::lossless_decode<fenix::s32>(in);
    return 0;
}
