// fuzz_dct_tile_decode.cpp — libFuzzer harness for codec/dct_block.hpp's decode_tile_dct<T>,
// the DCT-16 tile decoder (the hot path for every .fxvol chunk read). codec/CLAUDE.md states
// robustness against arbitrary bytes is a hard rule for this decoder specifically ("no UB/
// crash on any bytes (fuzzed; bounds-checked); wrong values OK, a SEGV is a fail"). Returns
// Expected<std::vector<T>> (bounds-checked); this harness discards Ok/Err either way — a
// SEGV/ASan report, not a returned Error, is the failure this exists to catch.
#include <cstdint>
#include <span>

#include "codec/dct_block.hpp"
#include "core/core.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::span<const fenix::u8> in{data, size};
    fenix::codec::DctParams params{};
    // bpa=1 (single 16^3 block, decode_block_dct_to) and bpa=4 (a full 64^3 tile — the
    // shared-rANS-table "group" path, the actual on-disk chunk unit) are both reachable.
    (void)fenix::codec::decode_tile_dct<fenix::u8>(in, 1, params);
    (void)fenix::codec::decode_tile_dct<fenix::u8>(in, 4, params);
    (void)fenix::codec::decode_tile_dct<fenix::f32>(in, 4, params);
    return 0;
}
