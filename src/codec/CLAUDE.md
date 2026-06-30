# codec — CLAUDE.md

## Purpose
The compression codec + the `.fxvol` archive container — fenix's volume store and the network-IO
substrate for out-of-core processing. Greenfield rewrite of the matter-compressor (`.mca`) **container**
ideas + `mc_codec_float`. The CDF 9/7 wavelet (c3d lineage) was evaluated and **retired** (ADR 0005) —
the DCT-16 tile codec beat it on ratio@quality AND speed across the range. See `docs/research/research-mc.md`.

## Public API & key types
- **One lossy transform codec — the DCT-16 tile codec** (`dct.hpp`, `dct_block.hpp`), over a shared
  rANS + dead-zone-quant + dtype substrate:
  - `dct.hpp` — separable orthonormal float **DCT-II 16** (even/odd partial butterfly), 3D-16³ + 2D-16².
  - `dct_block.hpp` — the codec. Per 16³ block: widen→f32 → subtract DC (quantized q/16, zigzag-varint)
    → DCT-16 → frequency-weighted dead-zone quant `q·(1+cz+cy+cx)^hf_exp` → magnitude-category rANS with
    a causal **(frequency-band × neighbour-magnitude-sum)** context, **clustered** per tile into ≤8 rANS
    tables (`cluster_contexts`) + raw mantissa/sign bits + **end-of-block** (frequency-ascending scan,
    code to last nonzero). Blocks are coded in **tiles**:
    `encode_tile_dct`/`decode_tile_dct` code a `bpa³`-block tile (a 64³ tile = 4³=64 blocks) that
    **SHARE one set of rANS category tables** (the JPEG XL "group" model — the big ratio win).
    `encode_block_dct`/`decode_block_dct_to` are `bpa=1` wrappers (byte-compatible single block).
- **Shared entropy substrate** (`entropy.hpp`): fixed+LEB128 varint fields, rANS byte-plane coder with
  a compact sparse freq table, LSB-first bit packer. `rans.hpp`: static byte rANS, 4-way interleaved.
- **Dtype layer** (`dtype.hpp`): u8/u16/u32/s8/s16/s32/f16/f32 — widen→f32 for the transform, round+clamp
  back on decode.
- **General lossless codec** (`lossless.hpp`, rANS + delta/RLE/bitpack filters) for label volumes,
  validity masks, exact priors.
- **The archive** (`archive.hpp`, `.fxvol`): 64³ chunk = base IO unit = one DCT tile + atomic decode +
  network + cache unit (16³-block/voxel = a view via a decoded-tile cache). Slot = u64 offset + tri-state
  coverage (NOT_SURE/ZERO/REAL). **The v4 layout is specified in [ADR 0006](../../docs/adr/0006-fxvol-v4-container.md)
  + [docs/design/fxvol-v4-layout.md](../../docs/design/fxvol-v4-layout.md)**: single file/volume, mmap'd
  **3-level (12+12+12) compressed-Morton radix page-table** (sparse-file, RAM = working set, sized for
  2¹⁸/axis = 2³⁶ chunks), explicit LOD octave pyramid (LOD-only quality scaling), LIVE append-mmap ↔ SEALED
  coarse-first (`fxvol finalize`), double-buffered crc superblock + data-before-pointer commit, S3 If-Match
  CAS. **`archive.hpp` now implements v4 Phase 1**: the mmap'd 3-level Morton radix page-table over a
  `MAP_NORESERVE` reservation + `posix_fallocate` growth + a bump allocator; sentinel-as-coverage slots
  (off==~0 ABSENT / len==0 ZERO / else REAL); bounds-checked reads (no UB on corrupt bytes); single
  superblock flushed on close; move-only RAII over the fd+mapping. Tested release + ASan (`test_archive`,
  `test_fxvol`, `test_pipeline`). Remaining phases (design note §9): double-buffered crc superblock +
  data-before-pointer durability (2), decoded-tile cache (3), explicit LOD pyramid (4), SEALED coarse-first
  repack + S3 If-Match CAS (5-6). Edge chunks are **edge-replicated** (not zero-padded) so the DCT doesn't
  ring at the volume boundary.
- Image interop: first-party **PNG + JPEG + TIFF read+write** to/from the 2D codec.

## Inputs / outputs & formats
In: `Volume<T>` regions (from io transcode or pipeline stages). Out: `.fxvol` archives (format v3);
PNG/JPEG/TIFF. Carries spatial metadata + provenance. Readers reject mismatched versions (no migration).

## Dependencies
Intra: `core` (Volume, simd, hash, arena, threadpool). Third-party: none for the codec itself (rANS/DCT
are ours); `blosc2/zlib` only at the io boundary.

## Invariants & numerics
**Tolerance-only, non-deterministic** — fast-math/float throughout; correctness = max-error τ / PSNR,
never bit-exact. **SIMD + GPU first-class** (the entropy coder must be SIMD/GPU-amenable — that's why
rANS, NOT serial CABAC/arithmetic; this is load-bearing for the tile-table decisions below). Robustness
is a **hard rule**: no UB/crash on any bytes (fuzzed; bounds-checked) — wrong values OK, a SEGV is a fail.

## Performance notes
Tiles/chunks are independent → embarrassingly parallel. The 64³ chunk = the random-access unit, so
sharing tables within it costs nothing. Per-block step table (46 `pow`/block, not per-coefficient) +
baked scan metadata (no per-coefficient division). Encode ~4.5 GB/s, decode ~4.5 GB/s on real CT.

## Gotchas / pitfalls
- **One transform codec now (DCT-16 tile), wavelet retired (ADR 0005).** Don't reintroduce a second
  transform codec or a codec-version branch without a new ADR. The shared substrate (rANS, dead-zone
  quant, magnitude-category coding, `entropy.hpp` helpers, dtype layer, `.fxvol`) is shared — don't fork.
- **Per-block static rANS tables overfit the tiny 16³ histogram** — this one fact explains every codec
  negative we hit. So: tables are **tile-global** (shared across the 64 blocks in a chunk); never go back
  to per-16³-block tables, and never expand contexts/alphabets without the shared tables to pay for them.
- **Stay STATIC rANS** (two-pass, tile-shared). Adaptive/per-symbol ANS breaks the 4-way SIMD interleave.
- **Guardrails from the SOTA research (2026-06-29):** do NOT re-add a zero-tree (a DCT block has no
  subband pyramid; it's redundant with the neighbour-significance context per EBCOT; and it dilutes the
  model). Keep the FIXED DCT basis (a per-block Tucker/HOSVD pays ~19% factor overhead). 
- Carry mc's crash-safety invariants into the real container (release-store commit, fallocate-not-
  ftruncate, absent-vs-failed). Keep this doc true to the code.

## Status & TODO
**Implemented + tested** (release + ASan, warning-free): `dct.hpp` (orthonormal DCT-II 16, butterfly,
Parseval-preserving — `test_dct`); `dct_block.hpp` (the tile codec — `test_dct_block` round-trips all 8
dtypes); `entropy.hpp`/`rans.hpp` (`test_rans`, `test_rans_perf`); `dtype.hpp`; `archive.hpp` (`.fxvol`
v3, DCT tiles, clustered tables, edge-replicated padding — `test_archive`, `test_pipeline`); `test_codec_bench`
(ratio/PSNR/MAE + enc/dec MB/s on a real CT volume or local `.zarr`; `FENIX_DCT_HF`/`FENIX_DCT_DZ` env).

**DCT optimization campaign (measured: 512³ PHerc Paris 4 smooth crop + 1024³ PHerc0211 dense centre,
8-bit CT, all wins PSNR-bit-identical).** In landing order:
1. **constant-header hoist** — N/dtype/params are archive-level config, not per-16³-block (~17 redundant
   bytes/block, up to ~12% at high q).
2. **DC zigzag-varint** (DC quantized q/16; was a raw 4-byte f32).
3. **end-of-block** — frequency-ascending scan (`dct_scan_order`), code to last nonzero; +3-4% ratio AND
   big speed (less to code AND scan).
4. **per-block step table + baked scan metadata** — kills the per-coefficient `std::pow` (~1e9→12M on a
   1024³) and the per-coefficient z/y/x division; encode +50-85%, ratio-neutral.
5. **tile-global rANS tables (THE big win)** — a 64³ tile's 64 blocks share one table set; the per-block
   tables had been ~73% of the payload at high q. **crop512 +48%/+113%/+279%** (q2/q8/q32),
   **dense +24%/+52%/+141%**. This is what flipped the DCT past the wavelet.
6. **neighbour-magnitude-SUM context** (AV1/HEVC-style; affordable only on the shared tables) — +3-4% on
   dense; on smooth crop512 +3-4% low/mid but ~-1.5% at q32 (subsumed by #7).
7. **context-map clustering (`cluster_contexts`)** — the raw context is now (frequency-band × neighbour-
   magnitude-sum) = kDctRawCtx=24 contexts, greedily MERGED per tile into ≤ kDctMaxClusters=8 rANS tables
   with the table-signalling cost IN the merge objective (JPEG XL / Brotli). So richer contexts can't
   fragment — sparse ones auto-collapse. Two-pass encode (quant+histogram → cluster → emit); the 25-byte
   context map is in the tile header. **+7-9% across the board on BOTH datasets** (crop512 q2/q8/q32
   +8.7/+7.5/+8.9%, dense +6.9/+7.5/+7.2%), PSNR-identical. The research's predicted "unlock" — it
   delivered, and it also made RDOQ viable (#8).
8. **RDOQ resurrected (`params.rdoq`, λ=0.15·step²)** — re-decide each significant coefficient's level
   among {0,L-1,L} to minimize D+λ·R, D=(|coef|−dequant)² (exact voxel-MSE, Parseval), R = the CLUSTERED
   rANS bit-cost + mantissa+sign. It failed before ONLY because the rate model came from noisy per-block
   tables; on the clustered tile-shared tables it's **RD-positive across the range on both datasets**
   (+0.5-1.2% iso-PSNR, no near-lossless regression at λ=0.15; 0.20+ over-trims at low q). Encoder-only:
   two-pass (cluster from provisional histograms → RDOQ re-quant → emit), decoder/bitstream unchanged.
9. **quant-gated 3D deblocking (`params.deblock`, plan 1.1)** — decode-time, NON-NORMATIVE (no bitstream
   change): an HEVC-lineage weak filter smooths only the 16³ block-boundary seams left by independent-block
   quantization, gated by a 2nd-difference flatness test + a quant-derived clamp (beta=2.5·q, tc=0.5·q) so
   real edges/texture are untouched. Decode reconstructs a full f32 tile, deblocks the INTERIOR seams, then
   rounds to dtype (64³ tile faces need a neighbour halo → container-level concern, deferred). **PSNR
   +0.05/+0.17/+0.43/+0.51 dB at q1/q2/q8/q32 (crop512), +0.01/+0.03/+0.30/+0.33 (dense)**, MAE down, ratio
   bit-identical, no near-lossless regression; insensitive to beta/tc (→ generalizes).
10. **DC + nsig spatial prediction (`params.dc_predict`, NOT in the original plan — found via the payload
   breakdown)** — predict each block's DC level AND significant-coeff count from the mean of its causal
   neighbour blocks WITHIN the tile (smooth volume → both vary slowly), code the zigzag residual. Causal in
   raster block order so decode reproduces it; no cross-tile ref → random access preserved. **ratio
   +0.4/+2.6/+0.2% (crop512 q2/q8/q32), +0.9/+1.0/+1.8% (dense)**, PSNR bit-identical. Best at q8 (large DC
   values) and dense q32 (nsig); the varint 1-byte floor caps the gain on tiny high-q values.
11. **SIMD DCT-16 (plan 1.4, the transform-bound speed lever)** — decode is provably IDCT-bound (throughput
   flat across q as rANS work drops). The separable 3D transform now runs through a vectorized 16×16 PANEL
   (`dct16_fwd_panel`/`dct16_inv_panel`) with the contiguous x-axis as the 16-wide SIMD lane: y-pass in place
   (a z-slab IS a contiguous [y][x] panel), x-pass transposes, z-pass gathers strided rows. Built on a Clang
   `ext_vector_type(16)` (the project is Clang-only; libc++ has no `<experimental/simd>`), memcpy load/store
   so any alignment is safe. **Decode +25-32% (crop512 ~3.5→4.3, dense ~3.5→4.6 GB/s)**, PSNR/ratio bit-
   identical. NOTE: a first naive "auto-vectorized panel" attempt REGRESSED ~2× (clang didn't vectorize the
   scalar c-loops + added gather/scatter) — explicit vector types were required. Encode ~flat (RDOQ/cluster-
   bound, not transform-bound). Still open: SIMD the dead-zone quant; widen rANS 4→8; a SIMD 16×16 transpose
   for the x/z-pass gather.

Cumulative DCT vs the v1: **crop512 +84%/+178%/+430%, dense +47%/+90%/+222%** (at the clustering stage;
RDOQ adds +0.5-1.2% iso-PSNR on top). Speed: **decode ~2.4→~4 GB/s** throughout; encode ~1.7→~4.5 then
back to **~2.5-3.2 GB/s** once clustering's two-pass + RDOQ landed (a one-time compression cost; decode
is unaffected). **The tile-DCT beats the (retired) wavelet at iso-quality across the whole range**
on both datasets (dense, before clustering: ~40 dB 10.2×/6.3× +62%, ~33 dB 24.2×/12.9× +88%, ~26 dB
88.8×/50.6× +75% — clustering widens this further) + ~2× faster encode, ~1.5× faster decode → ADR 0005.

**Honest negatives (the entropy + quant side is now well-tuned; further cheap levers are exhausted):**
`hf_exp`/`dz_frac` tuning marginal + data-dependent (kept 0.65/0.80, env-overridable); zero-tree parent
context regressed 3-15% (a DCT block has no subband pyramid); **hybrid-uint** small-level tokens
neutral-to-negative (the dead-zone + neighbour-sum context already capture the small-level distribution;
the larger alphabet adds freq-table overhead at sparsity); the **reconstruction-centroid** (Laplacian
offset) is already optimal at 0.40 (swept: 0.30→37.20, 0.40→37.24, 0.50→37.18 dB at fixed ratio).
**More clusters (kDctMaxClusters 8→16): bit-identical no-op** — the greedy merge already stops at its
beneficial optimum (<8) before the cap; capacity isn't the limit. **Richer raw context (MagCap/Sum/Band
3/6/4 → 4/8/6, 24→48 ctx): a wash-to-negative** — it does cut cat-enc ~1% rel, but the RAW (uncompressed)
context map doubles 24→48 B/tile and eats it at high q (q32 crop512 218→214). A compressed context map
would be needed first, for a <1% prize.

**Payload breakdown (measured 2026-06-30, FENIX_DCT_STATS):** category rANS payload **55-61%** | raw
mantissa+sign **27-41%** (mantissa proven-uniform; only signs attackable, ~fraction-of-a-bit) | freq-table
headers **1-4%** | DC/nsig/ctxmap each **<6%**. This **overturns the SOTA-research plan's #1 ratio pick
(1.2 rANS table-signalling)**: our compact-sparse-table + clustering already keeps table headers to 1-4%,
NOT "rivals payload" — so the full reuse/predefined-family/log-count mode-machine is poor ROI here. The
real ratio headroom is the **transform** (cuts BOTH big buckets): plan Tier 2 — secondary low-freq
transform, transform-skip/DST, trained KLT — but those need a corpus/offline training and/or are uncertain
on smooth CT (vs screen content). Per forrest 2026-06-30 the **ink-survivability metric (plan 0.1) and the
inverted task-quality track (Tier 3) are DROPPED** — the goal is whole-volume RD (PSNR/SSIM/MAE/pct-err),
which the bench already measures, so nothing is metric-gated.

**TODO — landed: deblock (#9), DC/nsig prediction (#10). The cheap entropy/quant levers are now provably
mined out (negatives above).** Next real levers, in rough ROI order: **(a) SIMD** the DCT-16 butterfly
(we're transform-bound; no SIMD infra yet → greenfield std::simd) + the dead-zone quant + widen rANS 4→8
— a guaranteed speed win, no bitstream risk; **(b) Tier-2 transforms** for ratio (transform-skip/DST has
no training need but uncertain on CT — probe first; secondary-transform/KLT need an offline corpus);
**(c) CDEF-3D** dering to finish 1.1 (deblock done); **(d) LOD** explicit multiscale pyramid + free DC
thumbnail; the real out-of-core `.fxvol` page table + append; the 2D codec instantiation; GPU. **Diag:**
`entropy.hpp` carries always-on byte-accounting atomics (g_plane_*/g_dc/g_nsig/g_map/g_bits), printed by
the bench under `FENIX_DCT_STATS` — negligible (per-plane/tile, relaxed). **Validation:** DCT-wins verdict
still from 2 regions — pull more diverse scroll regions opportunistically. Open ADRs: lossless algo.
