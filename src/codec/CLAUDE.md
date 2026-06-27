# codec — CLAUDE.md

## Purpose
The compression codecs and the `.fxvol` archive container — fenix's volume store and the
network-IO substrate for out-of-core processing. Greenfield rewrite of the
matter-compressor (`.mca`) **container** ideas + the **c3d** wavelet codec. See
`docs/research/research-mc.md` and `docs/research/research-c3d.md`.

## Public API & key types
- **One dim-parameterized CDF 9/7 wavelet core** (lifting + bitplane + 8-way interleaved
  rANS), instantiated for **3D (64³, volumes/prediction fields)** and **2D (64², images/
  parametric surfaces/texture layers)**. Lossy, **bitplane-progressive in LOD + quality**
  (decode stops at any resolution/bitplane). Dtypes: u8/u16/u32/s8/s16/s32/f16/f32.
- **General lossless codec** (rANS + delta/RLE/bitpacking filters) for integer label
  volumes, validity masks, exact priors. Selected via the container's codec-version field.
- **The archive** (`.fxvol`): 64³ chunk = base IO unit; **2-level page table** (each node
  2⁶/axis = 64³ slots) over 2¹² chunks/axis → 2¹⁸ voxels/axis; **top level sparse, 2nd
  dense, chunk dense**; slot = u64 offset + tri-state coverage (NOT_SURE/ZERO/REAL);
  append-at-EOF + release-store commit + `fallocate` growth (crash-safe, append-while-
  readable); self-describing chunk headers (q, hash, material-fraction map, subband/LOD/
  bitplane offsets); adaptive whole/partial (sub-chunk: coarse-LOD/low-bitplane) fetch.
- Quality: named presets (archival/working/preview) + raw q + target-ratio, per-chunk q.
- Image interop: first-party **PNG + JPEG + TIFF read+write** to/from the 2D codec.

## Inputs / outputs & formats
In: `Volume<T>` regions (from io transcode or pipeline stages). Out: `.fxvol` archives;
PNG/JPEG/TIFF. Carries spatial metadata + provenance.

## Dependencies
Intra: `core` (Volume, simd, hash, arena, threadpool). Third-party: none required for the
codec itself (rANS/wavelet are ours); `blosc2/zlib` only at the io boundary, not here.

## Invariants & numerics
**Tolerance-only, non-deterministic** — fast-math/float math throughout; correctness =
max-error τ / PSNR, never bit-exact. **SIMD + GPU are first-class** (entropy coder must be
SIMD/GPU-amenable — that's why rANS, not serial CABAC). Robustness is a **hard rule**:
no UB/crash on any bytes (fuzzed; bounds-checked) — wrong values OK, a SEGV is a failure.

## Performance notes
Blocks/chunks are independent → embarrassingly parallel. Per-worker persistent scratch,
no TLS dynamic lookups, allocation-free decode hotpath. Target 10×–1000× (wavelet);
near-lossless at low ratio. Seam handling: symmetric extension + optional decode-time heal.

## Gotchas / pitfalls
- DCT was explicitly **dropped** — wavelet-only. Don't reintroduce a DCT path.
- 2D and 3D **share** the lifting/bitplane/rANS core (dim-parameterized) — don't fork them.
- Carry mc's crash-safety invariants (release-store commit, fallocate-not-ftruncate,
  8-aligned atomic node slots, absent-vs-failed). Don't let docs drift from code (mc's
  header described removed features — keep this CLAUDE.md true).

## Status & TODO
**Implemented + tested** (release + ASan, warning-free): `wavelet.hpp` (CDF 9/7 lifting,
1D fwd/inv + multi-level separable 3D `dwt3_forward`/`dwt3_inverse`, mirror boundary —
roundtrip exact to fp error, 0.998 energy compaction on smooth 64³); `rans.hpp` (static
byte rANS, exact roundtrip, compresses skewed data); `block.hpp` (end-to-end lossy block
codec: DWT→dead-zone-quant→zigzag→rANS; near-lossless at small q, ~900× on flat regions,
graceful at high q). Tests: test_wavelet/test_rans/test_block.

**TODO (next):** the `.fxvol` archive/container (page table, coverage tri-state, append);
**bitplane-progressive** coefficient coding (currently quantize-then-rANS, not yet embedded
LOD+quality scalable); per-subband perceptual step weighting + Laplacian-α + RD allocator;
the 2D codec instantiation; the lossless (label) codec; 8-way interleaved SIMD rANS; GPU.
Open ADRs: bitplane scan order; rANS table-overhead amortization at 64³; lossless algo.
