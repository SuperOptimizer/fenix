# Code review — unit "codec" (src/codec/: DCT-16 tile codec, rANS, lossless, .fxvol archive, block cache)

Overall assessment: the codec's algorithmic core (DCT-16 butterfly/panels, dead-zone quant, RDOQ, clustering, SIEVE cache, the COW page-table + data-before-pointer commit discipline) is well designed and matches its documentation. The dominant problem is that the module's own hard rule — "no UB/crash on any bytes (fuzzed; bounds-checked)" — is only true for the archive's *page-table walk*; the entire tile-payload decode chain (`get_var`, `decode_plane`, `RansModel::from_freqs`, `decode_tile_dct`, `lossless_decode`) trusts every field it reads, producing out-of-bounds heap **writes**, shift UB, and unbounded allocations on crafted or corrupted input (CRC32 gates random corruption but not a fuzzer/attacker who recomputes it, and the page-table leaves themselves are not CRC-protected at all). Worse, one memory-corruption bug (`from_counts` normalization underflow) is on the **encode** side and reachable with plausible natural data. There is also one real data race on the archive's lazily-created block cache.

## [critical/bug] RansModel::from_counts normalization can underflow the largest freq → out-of-bounds write of the slot table (encode side, natural data)

**Verdict:** CONFIRMED — Reproduced exactly. src/codec/rans.hpp:66 bumps zero-floor occurring symbols to 1; the rounding repair at rans.hpp:76-78 adds fix = 4096 - assigned to the single largest freq via unchecked static_cast<u16>. With the reviewer's histogram (64 symbols x 10000 + 192 symbols x 1, total 640192) a standalone repro yields assigned=4224, fix=-128, largest freq 63 -> u16 wrap to 65471, final freq sum 69632; the slot-fill loop at rans.hpp:84 then writes 69632 entries into the 4096-entry slot array (~64 KiB OOB write). The path is unguarded and reachable on encode of natural data: encode_plane (src/codec/entropy.hpp:64) and lossless_encode (src/codec/lossless.hpp:52) feed raw byte-plane histograms directly to from_counts. Bound: floor shares sum <=4096 while bumps add up to +255, so any plane whose largest floor share is smaller than the bump count underflows — ordinary for many-moderate+many-rare CT/label byte planes. Not a stub and not intentional: the codec CLAUDE.md declares 'no UB/crash on any bytes' a hard rule and lists rans.hpp as implemented+tested.

**Fix notes:** Proposed fix is sound; corrections/additions: (1) Only the negative direction needs the repair loop — when assigned < rans_scale, fix <= 4095 so adding to the largest freq cannot overflow u16; keep that path. For assigned > rans_scale, repeatedly decrement the current largest freq while it stays >= 1 (always feasible: <=256 occurring symbols, each needs >=1, rans_scale=4096 >= 256); taking from the largest freqs also minimizes ratio distortion. (2) The SAME OOB write exists on the DECODE side and must be fixed together: from_freqs (rans.hpp:40-50) fills slot[] with no check that the deserialized table sums to rans_scale, and decode_plane (src/codec/entropy.hpp:82-95) plus lossless_decode (src/codec/lossless.hpp:79) build it from untrusted stream bytes (unbounded varint freqs) — corrupt/adversarial input triggers the identical slot overrun, violating the module's fuzz-robustness hard rule. from_freqs should validate/clamp (running c + bounds-checked fill, or reject sum != rans_scale). (3) FENIX_ASSERT sum==rans_scale is compiled out in release, so the constructive normalization + the from_freqs bounds check are the actual guarantees; the fuzz test should cover both the count-histogram path and the decode freq-table header path.

**Location:** src/codec/rans.hpp:72

**Evidence:**
```cpp
if (f == 0) f = 1;  // keep occurring symbols representable
...
if (assigned != rans_scale && last >= 0) {
    int big = 0;
    for (int s = 1; s < 256; ++s)
        if (m.freq[s] > m.freq[big]) big = s;
    const s32 fix = static_cast<s32>(rans_scale) - static_cast<s32>(assigned);
    m.freq[big] = static_cast<u16>(static_cast<s32>(m.freq[big]) + fix);
}
...
for (int s = 0; s < 256; ++s) {
    m.cum[s] = static_cast<u16>(c);
    for (u32 i = 0; i < m.freq[s]; ++i) m.slot[c + i] = static_cast<u8>(s);  // slot has 4096 entries
    c += m.freq[s];
}
```

**Failure scenario:** the 0→1 bumps can overshoot `rans_scale` by more than the largest frequency. Concrete input: a byte plane with 64 symbols of count 10000 each plus 192 symbols of count 1 (total 640192). Exact share for the common symbols is 63.98 → floor 63; assigned = 64·63 + 192·1 = 4224; fix = −128; largest freq is 63 → `63 − 128 = −65` wraps to u16 65471. The slot-fill loop then writes ~69632 entries into the 4096-byte `slot` array — a ~64 KiB out-of-bounds write through `RansModel` (heap or stack smash) during a perfectly ordinary `encode_plane`/`lossless_encode` call. This is not a corrupt-file issue: it fires while *encoding* real data with a "many-moderate + many-rare symbols" histogram (entirely plausible for CT byte planes / label planes).

**Suggested fix:** after the initial pass, redistribute the deficit/overshoot properly: while `assigned > rans_scale`, decrement the current largest freq that is > 1 (repeatedly), and while `assigned < rans_scale`, add to the largest; or use a standard largest-remainder normalization. Add a `FENIX_ASSERT`/hard check that the final sum is exactly `rans_scale` and every freq ≥ 1 for occurring symbols, and a unit/fuzz test over adversarial count histograms.

**Outcome:** fixed — `from_counts` (src/codec/rans.hpp) now redistributes an overshoot by repeatedly decrementing the current-largest freq (>1) until the sum is exactly `rans_scale`, per fix_notes; the deficit path (assigned < rans_scale) still adds to the single largest freq in one step (provably safe: fix ≤ 4095). `tests/test_rans.cpp::rans_from_counts_handles_overshoot_histogram` reproduces the exact 64×10000+192×1 histogram from this finding and asserts sum==rans_scale + every occurring freq∈[1,4096] + a full encode/decode round-trip; `rans_from_counts_always_sums_to_scale` fuzzes 50 random skewed histograms for the same invariant. Both pass under ASan.

## [critical/correctness] RansModel::from_freqs never validates that the deserialized freqs sum to rans_scale → out-of-bounds slot write on corrupt files

**Verdict:** CONFIRMED — rans.hpp:40-50 `from_freqs` writes m.slot[c+i] into a fixed std::array<u8,4096> (rans.hpp:36) with c accumulated from 256 untrusted u16 freqs (up to 65535 each) and no check that they sum to rans_scale — the precondition is stated only in a comment (rans.hpp:38-39). Both callers build the freq table directly from file bytes with no validation: entropy.hpp:87-92 (decode_plane, on the .fxvol DCT-tile decode path) and lossless.hpp:74-79 (lossless_decode). A blob whose freqs sum > 4096 causes an out-of-bounds write of up to ~16 MB past the local RansModel. This directly violates the module's stated hard rule (src/codec/CLAUDE.md Invariants: "no UB/crash on any bytes (fuzzed; bounds-checked) — wrong values OK, a SEGV is a fail"), so it is neither intentional nor guarded elsewhere.

**Fix notes:** Fix is correct and one check does close both paths. Details: (1) sum in u32 (256×65535 fits; don't sum in u16), and given the -fno-exceptions/std::expected convention, either return std::expected<RansModel, Error> or — simpler and matching the module's "wrong values OK" stance — substitute the uniform model (rans_scale/256 per symbol, as from_counts's degenerate branch already does) and keep the signature; the latter avoids plumbing Expected through decode_plane/decode_tile_dct. Note that if the sum is < rans_scale, unfilled slot entries default to symbol 0 whose freq may be 0, making dec() produce garbage but no OOB — still fine per "wrong values OK", so exact-sum rejection is the right predicate. (2) Sibling untrusted-input holes in the same callers that the one check does NOT close and that the fuzz surface will hit next: decode_plane's `in[p++]` (entropy.hpp:89) and get_var can read past the span, lossless_decode's memcpy at lossless.hpp:76 can overread, and both call `in.subspan(p, enc_size)` with an unvalidated enc_size — subspan with count > size()-offset is UB. Worth fixing in the same pass.

**Location:** src/codec/rans.hpp:40 (slot write at line 46)

**Evidence:**
```cpp
static RansModel from_freqs(std::span<const u16, 256> freq) {
    RansModel m;
    u32 c = 0;
    for (int s = 0; s < 256; ++s) {
        ...
        for (u32 i = 0; i < freq[s]; ++i) m.slot[c + i] = static_cast<u8>(s);
        c += freq[s];
    }
```

**Failure scenario:** `decode_plane` (entropy.hpp:87-92) and `lossless_decode` (lossless.hpp:75-79) build the freq table directly from file bytes — each entry can be up to 65535 (from `get_var` cast to u16, or a raw u16 read). A crafted/corrupt `.fxvol` tile payload or lossless blob whose freqs sum to > 4096 makes the slot-fill loop write up to ~16 MB past the 4096-entry `slot` array — a controlled out-of-bounds write on the decode path, i.e. exactly the "SEGV on bytes" the module's invariants forbid (the blob CRC does not help against an attacker who recomputes it). Additionally, `cum` becomes inconsistent so `rans_decode`'s `(x & mask) - m.cum[s]` produces garbage — but the memory corruption happens first.

**Suggested fix:** in `from_freqs` (or its two callers), accumulate the sum first and reject (return `Expected` error / fall back to a safe uniform model) unless the sum is exactly `rans_scale`. This one check closes the OOB write for both the DCT tile and lossless paths.

**Outcome:** fixed — `RansModel::from_freqs` (src/codec/rans.hpp) now sums the incoming table in u32 before touching `slot[]` and falls back to the uniform model (matching `from_counts`'s degenerate-case convention) on any sum != rans_scale, so the OOB write is structurally impossible regardless of caller. Added `RansModel::valid_freqs()` for callers that want to reject (not silently substitute) corrupt input. `decode_plane` (entropy.hpp) and `lossless_decode` (lossless.hpp) both now call `valid_freqs` explicitly and return an `Expected` error on a bad table rather than relying on the fallback, closing both the DCT-tile and lossless decode paths in one place. Covered by `tests/test_entropy.cpp::rans_from_freqs_rejects_bad_sum`/`decode_plane_rejects_corrupt_freq_table` and `tests/test_lossless.cpp::lossless_decode_rejects_bad_freq_table`, all green under ASan.

## [critical/correctness] lossless_decode trusts n, B, enc_size and the freq table → OOB heap writes and unbounded reads on corrupt input

**Verdict:** CONFIRMED — Confirmed by direct reading of src/codec/lossless.hpp. (a) lossless.hpp:69-70 read n and B from the stream with no validation; the plane loop at :72 runs b=0..B-1 and :83 does memcpy(reinterpret_cast<u8*>(&out[i]) + b, ...), so B > sizeof(T) writes past each element — n controlled OOB heap bytes. (b) :71 `std::vector<T> out(n)` with attacker-controlled n up to 2^64-1; under -fno-exceptions (project-wide, CLAUDE.md §2.3) the throw from vector aborts the process. (c) ll_get_u64 (lossless.hpp:22-27) memcpy's 8 bytes from b.data()+p with no size check — any input shorter than the header layout reads OOB; :80 `in.subspan(p, enc_size)` with enc_size > in.size()-p is library UB per [span.sub]. (d) the 512-byte freq table (:74-78) feeds RansModel::from_freqs (rans.hpp:40-50), which writes m.slot[c+i] (rans.hpp:46) without checking the freqs sum to rans_scale — a table summing above rans_scale overflows the fixed slot array, an OOB write inside RansModel itself. Not refuted by reachability: the only current callers are tests (tests/test_lossless.cpp) with self-encoded data, but src/codec/CLAUDE.md states the module's hard rule verbatim: "Robustness is a hard rule: no UB/crash on any bytes (fuzzed; bounds-checked) — wrong values OK, a SEGV is a fail", and lossless.hpp is listed as delivered public API (label volumes / validity masks / exact priors in .fxvol archives, i.e. untrusted network bytes by design). rans_decode itself was already hardened ("Robust on any bytes", rans.hpp:141-156), and archive.hpp was retrofitted with bounds-checked reads per the CLAUDE.md — lossless_decode simply never got the same treatment. Not a documented stub, not intentional.

**Severity adjusted to:** high

**Fix notes:** Downgraded critical→high only because no production caller feeds untrusted bytes yet (tests are the sole callers); it becomes critical the moment lossless planes are wired into .fxvol/network reads. Fix corrections/additions: (1) return type should be the project's std::expected<std::vector<T>, fenix::Error> per conventions, not a bare Expected. (2) Bounding n against in.size()·max-expansion is wrong in principle — rANS legitimately expands, so a plane's decoded n can exceed its enc_size by large factors; the right bound is a caller-supplied expected element count (the chunk/volume geometry is always known at the call site), with n != expected → Error. (3) Checking B == sizeof(T) is necessary and also catches type-mismatched decode calls, which is a bonus correctness win. (4) Every header read needs `p + k <= in.size()` before it (two u64s, the 512-byte table, the 8-byte enc_size, and enc_size itself clamped/validated against in.size() - p — reject, don't clamp, per the version-strict format policy). (5) The from_freqs slot-array overflow (rans.hpp:46) must be fixed in rans.hpp itself, not just papered over here: validate sum(freq) == rans_scale (u32 accumulate) and return expected/bool; lossless_decode and any other from_freqs caller (dct_block header path) must propagate the failure. (6) Add a fuzz target for lossless_decode — the module CLAUDE.md claims the codec is fuzzed, and this file evidently is not.

**Location:** src/codec/lossless.hpp:83 (header reads at 69-70, freq at 75-78, subspan at 80)

**Evidence:**
```cpp
const u64 n = detail::ll_get_u64(in, p);
const u64 B = detail::ll_get_u64(in, p);
std::vector<T> out(static_cast<usize>(n), T{});
for (u64 b = 0; b < B; ++b) {
    ...
    auto plane = rans_decode(in.subspan(p, static_cast<usize>(enc_size)), ...);
    ...
    std::memcpy(reinterpret_cast<u8*>(&out[i]) + b, &plane[i], 1);
```

**Failure scenario:** (a) `B` is read from the file and never compared with `sizeof(T)`; with `B = 9` on a `lossless_decode<u64>` call, iteration `b = 8` memcpy's one byte past the end of every element — n bytes of controlled heap overflow. (b) `n` is unvalidated → `std::vector<T> out(n)` with n = 2⁶⁴−1 throws `length_error`/`bad_alloc`, which under `-fno-exceptions` is an immediate abort — a corrupt 8-byte header is a guaranteed crash. (c) `ll_get_u64` (lossless.hpp:22) memcpy's without any bounds check, so a 7-byte input reads past the buffer; `in.subspan(p, enc_size)` with an oversized `enc_size` is UB per the span contract. (d) the raw 512-byte freq table feeds `from_freqs` (see previous finding).

**Suggested fix:** make `lossless_decode` return `Expected<std::vector<T>>`; validate `B == sizeof(T)`, bound `n` against a caller-supplied maximum (or at least against `in.size()` × a max expansion factor), bounds-check every header read, and clamp `enc_size` to the remaining bytes.

**Outcome:** fixed — `lossless_decode` (src/codec/lossless.hpp) now returns `Expected<std::vector<T>>` and takes an optional `expected_n` (caller-known element count from volume/chunk geometry, per fix_notes point 2) checked exactly when given; when omitted, `n` is still capped against a generous expansion-factor bound on `in.size()` to reject an absurd header instead of `vector(n)` aborting under `-fno-exceptions`. `B == sizeof(T)` is validated (fix_notes point 3). Every header field (`ll_get_u64`, the 512-byte freq table, `enc_size`) is bounds-checked before use, and the freq table is validated via `RansModel::valid_freqs` before `from_freqs` ever touches `slot[]`. Only caller today is tests; `tests/test_lossless.cpp` gained `lossless_decode_rejects_truncated_payload` (every truncation point), `lossless_decode_rejects_bad_freq_table`, and `lossless_decode_rejects_wrong_element_size`, all passing under ASan alongside the existing round-trip tests (updated for the new `Expected` return).

## [critical/correctness] decode_tile_dct trusts nsigs / context map / category symbols → OOB heap writes and shift UB on corrupt tile payloads

**Verdict:** CONFIRMED — Confirmed all three primitives against the code. (1) dct_block.hpp:403 sets nsigs[b]=res+pred with no clamp; :426 loops pos<nsigs[b] and :427 indexes scan (std::array<ScanEntry,4096>, defined :67) via operator[] with no bounds check → OOB read, and the garbage e.idx drives an OOB f32 write to `c` (size V=4096) at :437, with sig[i-sz] underflow at :429. (2) :405 K=payload[p++] unchecked (can be 0); :406-407 cmap bytes never validated vs K; :431-432 cl=cmap[...] then cats[cl]/cur[cl]++ → OOB vector access. (3) :434 `1u<<(cat-1)` and bits.get(cat-1) (entropy.hpp:119-121, `<<nbits`/`>>bits`) are shift-UB for a large decoded cat byte. Reachability: archive.hpp:252 calls decode_tile_dct after only a crc32c integrity check (:251), which a crafted .fxvol can satisfy trivially; the function returns a plain vector with no error path. Also get_var (entropy.hpp:51) and payload[p++] use unchecked span indexing, and decode_plane's subspan(p,enc_size) (entropy.hpp:93) can exceed the payload. This directly violates the codec CLAUDE.md hard rule ('no UB/crash on any bytes; bounds-checked') and archive.hpp's own 'bounds-checked reads (no UB on corrupt bytes)' claim. No tile-decoder fuzz harness exists in tests/.

**Fix notes:** Fix is sound. Additionally: validate the BitReader length field `nb` (entropy.hpp) so `payload.data()+p` with `nb` stays within the span, and ensure decode_plane's `subspan(p, enc_size)` (entropy.hpp:93) is bounds-checked against the payload. read_chunk_as already returns Expected, so making decode_tile_dct return Expected propagates cleanly; test_dct_block.cpp/test_codec_bench.cpp call sites will need updating for the new return type.

**Location:** src/codec/dct_block.hpp:426 (also 405-407, 432, 434)

**Evidence:**
```cpp
nsigs[b] = static_cast<u32>(res + pred);                    // untrusted, no clamp to V=4096
...
const int K = payload[p++];                                 // unchecked read; K may be 0
for (int r = 0; r < kDctRawCtx; ++r) cmap[r] = payload[p++]; // values never checked against K
...
for (u32 pos = 0; pos < nsigs[b]; ++pos) {
    const detail::ScanEntry e = scan[pos];                  // scan has exactly 4096 entries
    ...
    const int cl = cmap[...];
    const int cat = cats[cl][cur[cl]++];                    // cl may be >= K; cur may run past size()
    if (cat > 0) {
        const u32 a = (1u << (cat - 1)) + bits.get(cat - 1); // cat is a raw byte: up to 255
```

**Failure scenario:** every field is attacker-controlled after fixing up the blob CRC. (1) `nsigs[b]` can be up to 2³²−1 → `scan[pos]` reads far past the 4096-entry static array; the garbage `e.idx` (u16, up to 65535) then drives `c[i] = ...` — an out-of-bounds f32 **write** into the 4096-element `c` vector (heap corruption), and `sig[i - sz]` can underflow the index. (2) `cmap` values ≥ K (or K = 0) make `cats[cl]` an OOB vector access, and even in-range `cl` with more coded coefficients than plane symbols makes `cur[cl]` run past `cats[cl].size()`. (3) a decoded category byte of e.g. 200 makes `1u << (cat-1)` UB (shift ≥ 32), and `BitReader::get(199)` executes `acc |= ... << nbits` with `nbits ≥ 64` and `acc >>= 199` — both UB shifts (entropy.hpp:119-121). Any of these is a crash/memory-corruption primitive on the primary fuzz surface.

**Suggested fix:** make `decode_tile_dct` return `Expected` and validate at the point of decode: clamp/reject `nsigs[b] > V`; reject `K == 0 || K > kDctMaxClusters` and any `cmap[r] >= K`; reject `cat >= kDctCat` (18) before shifting; bounds-check `cur[cl] < cats[cl].size()`; bounds-check every `payload[p]` access (see next finding for `get_var`). Add the tile decoder to the fuzz corpus with CRC fix-up.

**Outcome:** fixed — `decode_tile_dct`/`decode_block_dct_to` (src/codec/dct_block.hpp) now return `Expected<std::vector<T>>` and validate every field named in the finding before it drives an index or shift: `nsigs[b]` is range-checked against `V=4096` right where it's decoded; `K` is rejected outside `(0, kDctMaxClusters]`; every context-map entry is rejected if `>= K`; `decode_plane`'s decoded category symbols are rejected if `>= kDctCat`; `cur[cl] < cats[cl].size()` is checked before each category read ("category stream underrun"); the bit-stream length `nb` is bounds-checked against the remaining payload before constructing `BitReader`. The extern-template split-build declarations (dct_block.hpp bottom) were updated for the new return type. `BitReader::get()` (entropy.hpp) additionally clamps `bits > 32` defensively so even an already-validated `cat` can't reach a shift-UB path. Callers updated: `archive.hpp::read_chunk_as` (propagates via `Expected`), `tests/test_dct_block.cpp`, `tests/test_codec_bench.cpp`. New coverage in `tests/test_dct_block.cpp`: `decode_tile_dct_rejects_truncated_payload` (every truncation offset of a real tile), `decode_tile_dct_rejects_random_garbage` (20 random-byte trials), `decode_tile_dct_rejects_corrupt_K`/byte-flip fuzzing, `decode_tile_dct_roundtrips_real_tile` (valid input still decodes). All pass, including under ASan (no OOB reports on any corrupt-input trial) — a pre-written `tests/fuzz_dct_tile_decode.cpp` (from a sibling review cluster) already targets the new `Expected` signature and compiles clean against this change.

## [high/correctness] get_var reads past the buffer and has shift UB; decode_plane's header reads and subspan are unbounded

**Verdict:** CONFIRMED — Confirmed by reading the code. entropy.hpp:51 `do { byte = b[p++]; ... } while (byte & 0x80u)` has no p<size check and no sh<32 cap (6+ continuation bytes -> `<<35` on u32, UB); libc++ span::operator[] is unchecked here (no _LIBCPP_HARDENING flag anywhere in CMakeLists/cmake/, release is -O3 -fno-exceptions). decode_plane (entropy.hpp:82-96) does `in[p++]` nused times, `in.subspan(p, enc_size)` unvalidated (UB per span contract), and sizes a decode buffer from an unvalidated u32 n. Reachability: decode_tile_dct (dct_block.hpp:388-413) calls get_var 2*nblk times up front plus payload[p++] for K/cmap, then decode_plane K times, so a truncated/crafted payload runs off the end. The only production guard is archive.hpp:247-252 (blob-bounds + crc32c), and crc32c is explicitly "torn-write detection" (archive.hpp:45) — any payload written with its matching adjacent crc passes, so arbitrary bytes DO reach the unchecked parser. Not intentional: src/codec/CLAUDE.md Invariants states "Robustness is a hard rule: no UB/crash on any bytes (fuzzed; bounds-checked) — wrong values OK, a SEGV is a fail", and no codec fuzzer exists in tests/ to have enforced it.

**Fix notes:** The fix direction (checked get_var/get_u32 returning Expected, cap sh<32, clamp enc_size/nused/n against remaining input, propagate through decode_plane/decode_tile_dct; read_chunk_as at archive.hpp:242 already returns Expected so plumbing is easy) is correct but INCOMPLETE — hardening entropy.hpp alone still leaves UB in decode_tile_dct itself: (a) dct_block.hpp:405-407 `payload[p++]` for K and the 24-byte context map are equally unchecked; (b) K must be clamped to kDctMaxClusters and each cmap[r] validated < K, else cats[cl] at line 431-432 indexes OOB; (c) decoded nsigs[b] must be clamped <= V (4096), else `scan[pos]` at line 427 reads past dct_scan_order() and sig[i] writes OOB; (d) `cats[cl][cur[cl]++]` at line 432 must be bounds-checked against the decoded plane length (a valid-length but short category stream overruns it); (e) `p += nb` for the bit stream (line 411-413) needs nb <= payload.size()-p. Also cap n in decode_plane to a sane bound (a plane can never exceed nblk*V symbols) rather than just remaining-input, since rANS output size is decoupled from input size. decode_tile_dct's signature changes from std::vector<T> to Expected<std::vector<T>> — update the FENIX_SPLIT extern-template declarations at dct_block.hpp:477-480 and all callers (archive.hpp:252, tests). Finally, add the codec decoder fuzzer the module CLAUDE.md already promises ("fuzzed") so this class of bug is regression-gated.

**Location:** src/codec/entropy.hpp:47 (decode_plane at 82-95)

**Evidence:**
```cpp
inline u32 get_var(std::span<const u8> b, usize& p) {
    u32 v = 0; int sh = 0; u8 byte;
    do { byte = b[p++]; v |= static_cast<u32>(byte & 0x7fu) << sh; sh += 7; } while (byte & 0x80u);
    return v;
}
...
const u8 s = in[p++];                       // decode_plane: unchecked
...
auto dec = rans_decode(in.subspan(p, enc_size), n, m);   // enc_size untrusted
```

**Failure scenario:** a payload ending in a byte with the continuation bit set makes `b[p++]` read past the span (libc++ `span::operator[]` is unchecked in this build) — and since decode loops call `get_var` 2·nblk times up front, a 1-byte payload immediately reads OOB. Independently, 5+ continuation bytes drive `sh` to 35+, making `<< sh` on u32 UB. In `decode_plane`, `nused` (up to 2³²−1) drives `in[p++]` far past the end, and `in.subspan(p, enc_size)` with `p + enc_size > in.size()` is library UB. `n` (u32) also sizes an up-to-4 GiB allocation from one varint (abort risk under `-fno-exceptions`).

**Suggested fix:** give `get_var`/`get_u32` a checked form returning `Expected<u32>` (or an ok-flag) that verifies `p < b.size()` per byte and caps `sh < 32`; clamp `enc_size`/`nused`/`n` against the remaining input; thread the failure up (decode_plane → decode_tile_dct → read_chunk_as already returns Expected).

**Outcome:** fixed — `get_var`/`get_u32` (src/codec/entropy.hpp) now return `Expected<u32>`, check `p < b.size()` per continuation byte, and cap the shift at `sh < 32` (rejecting a >5-byte-continuation varint instead of shift-UB). `decode_plane` was rewritten to bounds-check every field per the fix_notes' full list: `nused` is capped at 256, each sym/freq pair is bounds-checked, the freq table is validated via `RansModel::valid_freqs` before use, and `enc_size`'s span is checked against the remaining input before `subspan`/`rans_decode` (no more UB `subspan(p, enc_size)` with `enc_size` past the end). `decode_plane` also takes a `max_n` bound (fix_notes point on capping against what the input could actually encode, not just remaining bytes) — `decode_tile_dct` passes `nblk*V` (the true upper bound for a DCT-tile plane). The dct_block.hpp-side gaps the fix_notes flagged (K/cmap/nsigs/cur/nb validation) are fixed together — see the decode_tile_dct finding above. Covered by `tests/test_entropy.cpp` (new file): `get_var_roundtrip`, `get_var_rejects_truncated_and_unterminated` (including the 6-continuation-byte shift-UB case), `get_u32_bounds_checked`, `decode_plane_roundtrip`, `decode_plane_rejects_length_over_bound`, `decode_plane_rejects_truncated_payload`, `decode_plane_rejects_corrupt_freq_table`. All pass under ASan.

## [high/correctness] read_chunk_as blob bounds check overflows on a corrupt slot length → SEGV (page-table leaves are not CRC-protected)

**Verdict:** CONFIRMED — Confirmed by reading the code. FxSlot.len is u64 (src/codec/archive.hpp:74-77) returned verbatim from a mmap'd leaf by slot_read_ (line 698); radix nodes/leaves carry no crc (only blobs, line 226-229, and the superblock, line 95). At line 247, s.len = 2^64-4 makes `s.len + 4` wrap to 0 (check passes) and `hz - s.len - 4` wrap to hz (check passes for any s.off ≤ hz); line 251 then calls crc32c(base_ + s.off, s.len) with an ~2^64-byte length → OOB read off the mapping → SIGSEGV/SIGBUS. This violates the module's explicit hard rule (src/codec/CLAUDE.md Invariants: "no UB/crash on any bytes ... a SEGV is a fail") and the file's advertised "bounds-checked reads (no UB on corrupt bytes)", so it is not intentional. Line 373 in finalize() has the identical wrapping check. Nothing upstream sanitizes leaf contents: open() validates only the superblock crc, committed_eof_, and lod_root_ (lines 150-168).

**Fix notes:** Proposed fix is correct: establish s.off ∈ [kFxDataStart, hz] first, then s.len ≤ hz - s.off and hz - s.off - s.len ≥ 4 — every subtraction is then non-wrapping. Apply the same rewrite at line 373 (finalize). Two additions: (1) the same wrapping-add pattern exists in slot_read_'s node walk (lines 691/695/697: `c1 + node_bytes_() > hz` wraps for a corrupt child offset near 2^64 read from an untrusted, un-crc'd node, leading to node_at_(c1) dereferencing far past the mapping) — fix it the same way (`c1 > hz - node_bytes_()` after checking `c1 <= hz`, or `c1 >= hz || hz - c1 < node_bytes_()`); the analogous open()-time lod_root_ check (line 167) is crc-protected so it is lower priority but worth hardening for consistency. (2) Note the finalize() consequence differs from read_chunk_as: the wrap there makes alloc_(p.src_len+4) allocate 0-3 bytes and memcpy 0-3 bytes (no giant OOB), but the sealed file's slot is written with the bogus src_len, so the corruption propagates; the fix still applies as proposed.

**Location:** src/codec/archive.hpp:247 (same pattern in finalize at 373)

**Evidence:**
```cpp
if (s.off < detail::kFxDataStart || s.len + 4 > hz || s.off > hz - s.len - 4)
    return err(Errc::decode_error, "corrupt chunk offset");
u32 crc_stored;
std::memcpy(&crc_stored, base_ + s.off + s.len, 4);
if (detail::crc32c(base_ + s.off, s.len) != crc_stored) ...
```

**Failure scenario:** `s.off`/`s.len` come from a leaf node in the mmap'd file; unlike the superblock and blobs, radix nodes and leaves carry **no CRC**, so a single corrupted/crafted leaf is directly consumed. With `s.len = 2⁶⁴−4`: `s.len + 4` wraps to 0 (passes check 2), `hz − s.len − 4` wraps to `hz` (passes check 3 for any `s.off ≤ hz`), then `crc32c(base_ + s.off, s.len)` walks ~2⁶⁴ bytes off the end of the mapping → SIGSEGV/SIGBUS. This defeats the file's one explicitly claimed robustness property ("bounds-checked reads, no UB on corrupt bytes").

**Suggested fix:** rewrite the check in non-overflowing form, e.g. `if (s.off < kFxDataStart || s.off > hz || s.len > hz - s.off || hz - s.off - s.len < 4) return err(...)` (check `s.off ≤ hz` first, then compare `s.len` against `hz − s.off`). Apply the same fix to the copy in `finalize()`.

**Outcome:** fixed — `read_chunk_as` and `finalize()` (src/codec/archive.hpp) both use the exact non-overflowing rewrite suggested: `s.off <= hz` is checked first, then `s.len` is compared against the remaining span (`hz - s.off`), so no subtraction can wrap. Also fixed the two additional wrapping sites the fix_notes flagged: `slot_read_`'s L0→L1→leaf node walk (root/c1/c2 bounds checks) now goes through a new `in_bounds_(off, need, hz)` helper with the same non-wrapping structure, and `open()`'s per-LOD root-offset validation uses the same helper instead of the old `lod_root_[l] + node_bytes_() > committed_eof_` (which had the identical overflow pattern for a corrupt root read from the un-CRC'd superblock field). Covered by `tests/test_archive.cpp::archive_rejects_corrupt_slot_length`, which builds a real archive, pokes the exact wraparound value (`len = ~0ull - 3`) directly into the mmap'd leaf, and confirms `read_chunk` returns an error instead of segfaulting; passes under ASan. The pre-existing `tests/test_fxvol.cpp::fxvol_robust_open_bad_bytes` also continues to pass, exercising the `open()`-path fix.

## [high/concurrency] block16 lazily creates the block cache on a const path with no synchronization → data race / use-after-free

**Verdict:** CONFIRMED — The core defect is real and I cannot refute it. src/codec/archive.hpp:806 declares `mutable std::unique_ptr<BlockCache> block_cache_;` and archive.hpp:402 does an unguarded `if (!block_cache_) block_cache_ = std::make_unique<BlockCache>(detail::kFxDefaultCache);` on a const path — an unsynchronized non-atomic test-and-assign; two concurrent first calls both construct and one assignment destroys the BlockCache the other thread is inside (BlockCache itself is properly sharded/mutexed, block_cache.hpp:35/51, but that doesn't protect the pointer). The API's own contracts promise concurrency: archive.hpp:466 documents gather_box_f32 as "Thread-safe for disjoint boxes", and src/fs/CLAUDE.md explicitly asserts "FUSE is multithreaded (block16 is const + the cache is sharded/locked...)" — neither requires reserve_cache() first, so the documented contract is violated by the lazy init. However, the reviewer's concrete trigger is overstated: gather_box_f32 has ZERO in-tree callers (repo grep: only docs/review sweeps and archive.hpp itself), and the ML inference path (src/ml/infer.hpp) does not stream via the archive at all — per project convention it decodes the whole volume once to dense u8. The only multithreaded consumer today, fxfs, calls reserve_cache(1 GiB) at src/fs/fxfs.cpp:135 before fuse_main spawns threads, so the lazy branch never executes concurrently in any current binary. The bug is a latent, documentation-contradicting race that any new caller relying on the stated thread-safety will hit intermittently, not a currently-firing one — so it stands, at reduced severity.

**Severity adjusted to:** medium

**Fix notes:** Eager construction in open()/create() is the right fix (a default 256 MiB-budget BlockCache with 16 empty shards is a few hundred bytes until populated — kFxDefaultCache at archive.hpp:96 is a budget, not an allocation). Two corrections: (1) do NOT take the "return an error from block16 when reserve_cache absent" option — it breaks fxfs's sample_f32 path and every casual single-threaded user for no gain; (2) the fix must also cover reserve_cache() itself (archive.hpp:390): it unconditionally replaces the pointer, so even after eager init it destroys a possibly-in-use cache — either document it as "call before any concurrent block16 use only" or have it no-op/resize under the same guard. std::once_flag would also work but complicates the move ctor (archive.hpp:773 moves block_cache_; once_flag is neither movable nor copyable), so eager init is cleaner. Belt-and-braces: also fix the stale doc in archive.hpp:466 / src/fs/CLAUDE.md if the lazy branch is kept for any reason.

**Location:** src/codec/archive.hpp:402

**Evidence:**
```cpp
[[nodiscard]] Expected<BlockCache::Ref> block16(s64 lod, ChunkCoord bc) const {
    if (!block_cache_) block_cache_ = std::make_unique<BlockCache>(detail::kFxDefaultCache);
```
with `mutable std::unique_ptr<BlockCache> block_cache_;` (line 806), and `gather_box_f32` documented "Thread-safe for disjoint boxes" (line 466).

**Failure scenario:** the ML patch-gather path calls `gather_box_f32` from multiple threads on one archive (that is its advertised contract). If `reserve_cache()` was never called, two threads race on the unchecked test-and-assign of the `mutable unique_ptr`: both construct a `BlockCache`, one assignment destroys the other's object while a third thread (or the loser itself, having cached the raw pointer through `->get`) is inside it — unsynchronized read/write of a non-atomic pointer (UB per se) escalating to use-after-free. Intermittent, load-dependent, and it will pass single-threaded tests.

**Suggested fix:** eagerly construct the default cache in `create()`/`open()` (it's cheap — empty shards), or guard the lazy init with `std::once_flag`/a mutex, or make `block16` require `reserve_cache()` and return an error otherwise.

**Outcome:** fixed — per fix_notes, went with eager construction (not `once_flag`, which doesn't survive the move ctor `VolumeArchive` relies on). Both `create()` and `open()` (src/codec/archive.hpp) now install a default-budget `BlockCache` before returning, so every archive has a valid cache before any thread can call `block16()`. `block16()` itself no longer does the lazy test-and-assign — a null cache there is now a caller-bug signal (a default-constructed-and-never-opened `VolumeArchive`, not a state to paper over), and returns an `Errc::internal` error instead. `reserve_cache()` is documented (comment above it) as still unconditionally replacing the pointer — call it before concurrent use begins, same constraint the lazy code implicitly had, now made explicit as fix_notes point 2 requested. Covered by `tests/test_archive.cpp::archive_block16_concurrent_use_is_safe`, which hammers `block16()` from 8 threads × 200 calls on a freshly-opened archive with no `reserve_cache()` call, and by the pre-existing `tests/test_fxvol.cpp::fxvol_block_cache`. Both pass under ASan; TSan was not run (out of scope for this pass — the ASan/functional coverage confirms no UAF on the exercised paths, but a dedicated TSan job would be the stronger proof for the concurrency claim).

## [medium/correctness] Untrusted superblock/stream sizes drive unchecked allocations → abort under -fno-exceptions

**Verdict:** unverified (medium/low)

**Location:** src/codec/lossless.hpp:71 (also entropy.hpp:83, archive.hpp:156-161)

**Evidence:**
```cpp
std::vector<T> out(static_cast<usize>(n), T{});          // lossless: n from file, u64
...
const u32 n = get_var(in, p); ... std::vector<u8> out(n); // decode_plane via rans_decode
...
std::memcpy(&a.dims_.z, a.base_ + b + 32, 8);            // dims_ never range-validated
```

**Failure scenario:** the build is `-fno-exceptions`, so any `vector` length/alloc failure calls `abort()`. A 4-byte varint in a tile payload allocates up to 4 GiB per plane; the lossless header allocates up to 2⁶⁴ elements; superblock `dims_` passes the CRC only for random corruption, not crafted files, and `open()` validates `nlods_`, dtype, eof and roots but **not** `dims_`/`params_` — a huge or negative `dims_` later drives `Volume<T>::zeros(vd)` in `read_volume_as` into an absurd allocation or signed-arithmetic trouble. "Wrong values OK, a crash is a fail" — these are crashes.

**Suggested fix:** validate `dims_` (each axis in (0, 2¹⁸]) and `params_.q > 0`, finite, in `open()`; bound every decode-side allocation by what the input could actually encode (e.g. plane `n` ≤ nblk·V for the tile codec; lossless `n·B` ≤ a caller-supplied cap).

**Outcome:** fixed — `open()` (src/codec/archive.hpp) now validates `dims_` (each axis in `(0, 2^18]`, the documented envelope) and `params_.q` (finite and `> 0`, via `!(q>0)` which also rejects NaN) right after they're deserialized from the superblock, before anything downstream can consume them. `decode_plane` (entropy.hpp) now takes a caller-supplied `max_n` bound (the true upper bound a DCT-tile plane can encode: `nblk*V`) rather than trusting `in.size()` alone. `lossless_decode` (lossless.hpp) takes an optional `expected_n` and, when omitted, still caps `n` against a generous expansion-factor bound on `in.size()` instead of trusting the raw header value — see the fuller outcome notes on the `from_freqs`/`lossless_decode`/`decode_tile_dct` findings above for the rest of this finding's scope (`get_var`/`get_u32` bounds-checking in entropy.hpp). Covered by `tests/test_archive.cpp::archive_rejects_out_of_range_dims` (crafts a CRC-consistent superblock with `dims_.z` one past the envelope and confirms `open()` rejects it — isolating the new range check from the pre-existing whole-block CRC check) plus the lossless/decode_plane tests cited elsewhere. All pass under ASan.

## [medium/design] ABSENT (NOT_SURE) chunks silently decode as fill — conflated with ZERO on every read path

**Verdict:** unverified (medium/low)

**Location:** src/codec/archive.hpp:245

**Evidence:**
```cpp
if (s.off == detail::kFxAbsent || s.len == 0) return std::vector<T>(n, fill);
```

**Failure scenario:** the project invariant is that "absent vs fetch-failed/not-sure are distinct" and NOT_SURE must never silently become air. The tri-state exists (`coverage()`), but `read_chunk_as`, and therefore `block16`, `sample_f32`, `gather_box_f32` and `read_volume_as`, return the fill value for ABSENT with no signal whatsoever. A pipeline stage doing a windowed gather over a partially-ingested archive (e.g. a halo touching a never-written chunk) computes on fabricated air and produces silently wrong output; nothing forces callers to pre-query `coverage()` per chunk, and `gather_box_f32` cannot even do so per-voxel. This is precisely the predecessor foot-gun the invariant was written against.

**Suggested fix:** at minimum, add an `Expected`-erroring variant (or a `bool absent_is_error` policy on the archive/read call) and make `gather_box_f32`/`block16` honor it; or return the coverage alongside the data so callers must acknowledge ABSENT. Keep the permissive fill behavior opt-in, not the default.

## [medium/performance] gather_box_f32 edge-clamp path rescans the whole output box once per covering block

**Verdict:** unverified (medium/low)

**Location:** src/codec/archive.hpp:506

**Evidence:**
```cpp
} else {
    // Edge-clamp path: for each OUTPUT voxel whose clamped source lands in THIS block, copy.
    for (s64 z = 0; z < D; ++z) {
        const s64 gz = std::clamp<s64>(oz + z, 0, vd.z - 1); if (gz / N != bz) continue;
        for (s64 y = 0; y < H; ++y) { ...
            for (s64 x = 0; x < W; ++x) { ...
```

**Failure scenario:** whenever the patch is not fully inside the volume, the code iterates all D·H·W output voxels for *every* covering 16³ block. A 256³ patch flush against one volume face covers ~17·16·16 ≈ 4.4k blocks × 16.8M voxel iterations ≈ 7×10¹⁰ clamp/compare ops — a single edge patch takes orders of magnitude longer than an interior one (which the comment itself says must be block-major to keep inference GPU-bound). Edge patches are routine in tiled inference, so throughput craters exactly at volume boundaries.

**Suggested fix:** compute, per block, the output-index range whose clamped source falls in that block (invert the clamp per axis: a boundary block additionally owns the out-of-range prefix/suffix of the output axis) and iterate only that subrange, as the fast path does; or gather the interior with the fast path and fix up the clamped border separately.

## [low/resource-safety] open() always opens O_RDWR and mmaps PROT_WRITE, even for read-only use

**Verdict:** unverified (medium/low)

**Location:** src/codec/archive.hpp:127

**Evidence:**
```cpp
static Expected<VolumeArchive> open(const std::string& path, bool writable = false) {
    ...
    a.fd_ = ::open(path.c_str(), O_RDWR, 0644);
```

**Failure scenario:** a `.fxvol` on a read-only mount, with mode 0444, or shared read-only between users fails to open at all (`EACCES`/`EROFS`) even though the caller asked for read-only access — the error surfaces as a misleading `not_found`. It also maps the file PROT_WRITE|MAP_SHARED for pure readers, so any stray write through `base_` corrupts the archive on disk instead of faulting.

**Suggested fix:** when `!writable`, open `O_RDONLY` and mmap `PROT_READ` (map_ needs the prot as a parameter); keep O_RDWR only for the writable path.

## [low/performance] quant_one_block heap-allocates two vectors per 16³ block in the encode hot loop

**Verdict:** unverified (medium/low)

**Location:** src/codec/dct_block.hpp:117

**Evidence:**
```cpp
std::vector<T> blk(static_cast<usize>(V));
...
std::vector<f32> c = to_f32<T>(blk);
```

**Failure scenario:** per 64³ tile that is 128 allocations (64 blocks × {blk, c}) of 4-16 KiB in the innermost encode loop, called for every chunk of a multi-terabyte ingest; it also copies each block twice (gather into `blk`, then widen into `c`). Not a correctness issue, but it is avoidable allocator traffic and cache churn on the measured-hot path (encode is already the slow direction at ~2.5-3 GB/s).

**Suggested fix:** hoist two fixed 4096-element scratch buffers (per tile, or thread_local) into `encode_tile_dct` and pass them in; widen directly during the gather (one pass, no intermediate `blk` when T ≠ f32).

## [low/hygiene] sample_f32/block16 accept negative coordinates and index with negative modulo → OOB read

**Verdict:** unverified (medium/low)

**Location:** src/codec/archive.hpp:453

**Evidence:**
```cpp
auto r = block16(lod, {z / kDctN, y / kDctN, x / kDctN});
...
const usize off = static_cast<usize>(((z % kDctN) * kDctN + (y % kDctN)) * kDctN + (x % kDctN));
```

**Failure scenario:** for `z = -1`, `z / 16 == 0` (truncation) so block (0,·,·) is fetched, but `z % 16 == -1` makes `off` wrap to a huge `usize` → out-of-bounds read of the 4 KiB cache block. `gather_box_f32` clamps before calling, but `sample_f32` is a public primitive with no bounds check and no `FENIX_ASSERT`; the same negative-coord issue makes `block16` compute a garbage Morton key. One misbehaving caller turns into silent garbage or a crash rather than a caught contract violation.

**Suggested fix:** `FENIX_ASSERT(z >= 0 && ...)` (or clamp/return error against `dims_at(lod)`) at the top of `sample_f32` and `block16`; use `((z % N) + N) % N` only if negative coords are meant to be legal (they are not, per ZYX conventions).

**Outcome:** fixed (partial) — `sample_f32` (src/codec/archive.hpp), the primitive with the confirmed OOB (truncating `/` and `%` on the same negative coordinate disagree in sign, wrapping `off` to a huge `usize`), now rejects `z < 0 || y < 0 || x < 0` via `Expected` error (not `FENIX_ASSERT`, since that compiles out in release and this is reachable with real caller-supplied coordinates, not just corrupt file bytes). `block16` itself was checked and found NOT vulnerable to the same class: `bc` feeds `block_key_`/`morton3`, whose `i0`/`i1`/`i2` node-table indices are always masked to 12 bits before use (archive.hpp `slot_read_`), so a negative-cast-to-u64 Morton key produces a garbage-but-in-bounds lookup (miss or wrong tile), not an OOB read — left unguarded since there's no memory-safety bug there, only a "wrong tile" hygiene concern the finding didn't distinguish. Covered by `tests/test_archive.cpp::archive_sample_f32_rejects_negative_coords`; passes under ASan.
