# codec — CLAUDE.md

## Purpose
The compression codecs and the `.fxvol` archive container — fenix's volume store and the
network-IO substrate for out-of-core processing. Greenfield rewrite of the
matter-compressor (`.mca`) **container** ideas + the **c3d** wavelet codec. See
`docs/research/research-mc.md` and `docs/research/research-c3d.md`.

## Public API & key types
- **Two selectable lossy transform codecs** (per-archive codec-version field) over a shared
  rANS + dead-zone-quant + dtype + container substrate:
  - **CDF 9/7 wavelet core** (lifting + bitplane + 8-way interleaved rANS), dim-parameterized
    for **3D (64³)** and **2D (64²)**, **bitplane-progressive in LOD + quality**.
  - **Separable all-float DCT-16** (DCT-II, 16³/16² blocks, band-weighted dead-zone quant) —
    rewrite of matter-compressor `mc_codec` (`mc_codec_float.h`); strong near-lossless/low-ratio.
- **Dtype layer:** all codecs accept/emit u8/u16/u32/s8/s16/s32/f16/f32 (convert to f32 for
  the transform, round+clamp back on decode; the dtype is stored in the block/chunk header).
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
- **Two transform codecs (codec still being settled): CDF 9/7 wavelet AND a separable
  all-float DCT-16**, selected per-archive via the codec-version field. They SHARE the rANS
  entropy core, dead-zone quant + magnitude-category coding, the `.fxvol` container, and the
  u8..f32 dtype layer — do NOT fork those. (The earlier "DCT dropped" rule is reversed; see
  ADR 0002 amended 2026-06-29. mc's DCT was integer — ours is float, per the fast-math rule.)
- 2D and 3D **share** the lifting/bitplane/rANS core (dim-parameterized) — don't fork them.
- Carry mc's crash-safety invariants (release-store commit, fallocate-not-ftruncate,
  8-aligned atomic node slots, absent-vs-failed). Don't let docs drift from code (mc's
  header described removed features — keep this CLAUDE.md true).

## Status & TODO
**Implemented + tested** (release + ASan, warning-free): `wavelet.hpp` (CDF 9/7 lifting,
1D fwd/inv + multi-level separable 3D `dwt3_forward`/`dwt3_inverse`, mirror boundary —
roundtrip exact to fp error, 0.998 energy compaction on smooth 64³); `rans.hpp` (static
byte rANS, exact roundtrip, compresses skewed data); `block.hpp` (end-to-end lossy block
codec: DWT→**per-subband** dead-zone-quant→zigzag→**per-scale grouped** rANS). Tests:
test_wavelet/test_rans/test_block + **test_codec_bench** (ratio/PSNR/SSIM/MAE/percentile +
enc/dec MB/s on a real CT volume; skips cleanly if the data file is absent).

**Benchmarked on PHerc Paris 4 CT (512³ crop, 64³ blocks).** Improvements landed (all in
block.hpp / wavelet.hpp), each measured:
1. **per-scale grouping** — coefficients grouped by DWT scale, each scale entropy-coded with its
   own model (subband stats differ sharply);
2. **compact sparse freq tables** — store only used symbols, not a fixed 512-byte table;
3. **magnitude-category coding** (JPEG-style) — rANS only the skewed bit-length category; pack
   the near-random mantissa+sign bits raw (better ratio AND less rANS work);
4. **per-orientation subband boxes + causal 3D significance context** — each subband is a
   contiguous box (`subband_boxes`); scan it in 3D and select the category model by how many of
   the z-1/y-1/x-1 neighbours are nonzero (`kCtx`=4) — exploits spatial clustering far better than
   a raster-order context;
5. **per-subband quantization** (`scale_step`, finer step for low-freq bands; 9/7 is biorthogonal
   so uniform-q is not RD-optimal);
6. **dead-zone quantizer** (truncate-to-zero, 2× zero bin) + centroid reconstruction — RD-better
   for the Laplacian-like coefficients;
7. **varint size fields + compact sparse freq tables** — keeps per-stream overhead ~1 byte so
   fragmenting into many context streams stays a net win even at high q;
8. **wavelet scratch reuse** — no per-1-D-line allocation.
Net vs the original uniform-q/single-order-0 codec: **~3–3.6× better compression at matched
quality** in the useful regime, and faster (~1.4 GB/s enc, ~1.5 GB/s dec). Ratios (vs 8-bit CT):
q1 3.5×@52dB, q2 7.4×@45dB, q4 21.3×@39.5dB, q8 63.3×@34.3dB, q16 173×@29.5dB, q32 467×@25dB.
Iso-PSNR vs baseline: @39.5dB 7.3×→21.3× (2.9×), @34dB 18×→~63× (~3.5×), @29.5dB 47.5×→173×
(~3.6×). Metrics via `test_codec_bench` (ratio, enc/dec MB/s, PSNR, block-SSIM, MAE, percentiles).

**DCT-16 codec (DONE + tested):** `dct.hpp` (orthonormal DCT-II 16×16, separable 3D-16³ + 2D-16²,
all-float; round-trip fp-exact, Parseval-preserving, 100% energy compaction on smooth — `test_dct`);
`dtype.hpp` (the shared **u8/u16/u32/s8/s16/s32/f16/f32** I/O layer — widen→f32, narrow back with
round+clamp; wired `f16=_Float16`); `dct_block.hpp` (DC-removal → DCT16 → frequency-weighted dead-zone
quant `q·(1+cz+cy+cx)^0.65` → block.hpp's magnitude-category rANS — rewrite of mc_codec_float, but
float + rANS not mc's range coder). `test_dct_block`: all 8 dtypes round-trip within the quant step +
compress (u8 45×, f32 147× on smooth). **Head-to-head on crop512 CT (`test_codec_bench`):** wavelet
wins high-ratio (q8 63.3×@34.3dB vs DCT 31.4×@34.4dB; q16 173×@29.5dB) + is LOD-progressive; DCT is
competitive at near-lossless (q1 8.9×@46dB vs wavelet ~7×@45dB) and decodes ~2.5× faster (~3.7–4.3
GB/s vs ~1.5–1.8). Settles toward: **wavelet = archival/high-ratio default, DCT = fast near-lossless**.
The DCT is a faithful v1 — the wavelet's 8 RD-tuning rounds (per-band context, etc.) haven't been
applied to it yet, so it has headroom.

**TODO (next):** the `.fxvol` archive/container (page table, coverage tri-state, append);
**bitplane-progressive** coefficient coding (currently quantize-then-rANS, not yet embedded
LOD+quality scalable); Laplacian-α + RD step allocator; the 2D codec instantiation; the
lossless (label) codec; 8-way interleaved SIMD rANS; GPU. Open ADRs: bitplane scan order;
lossless algo.
