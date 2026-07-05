# codec — CLAUDE.md

## Purpose
The compression codecs + the `.fxvol` archive container — fenix's volume store and the
network-IO substrate for out-of-core processing, plus the 2D surface-field codec underlying
`.fxsurf`. Greenfield rewrite of matter-compressor (`.mca`) **container** ideas +
`mc_codec_float`. The CDF 9/7 wavelet (c3d lineage) was evaluated and **retired** (ADR 0005)
— the DCT-16 tile codec beat it on ratio@quality AND speed across the range. See
`docs/research/research-mc.md`.

## Public API & key types
- **Volume transform codec — the DCT-16 tile codec** (`dct.hpp`, `dct_block.hpp`), over a
  shared rANS + dead-zone-quant + dtype substrate:
  - `dct.hpp` — separable orthonormal float **DCT-II 16** (even/odd partial butterfly,
    SIMD panel path), 3D-16³ + 2D-16²; `kDctN = 16`. `dct16_fwd_panel`/`dct16_inv_panel` run
    the transform through a vectorized 16-wide `ext_vector_type` panel (Clang-only; no
    libc++ `<experimental/simd>`).
  - `dct_block.hpp` — the codec, `DctParams{q, hf_exp, dz_frac, rdoq, rdoq_lambda, deblock,
    deblock_beta, deblock_tc, dc_predict}`. Per 16³ block: widen→f32 → subtract DC
    (predicted from causal neighbour blocks when `dc_predict`, zigzag-varint residual) →
    DCT-16 → frequency-weighted dead-zone quant `q·(1+cz+cy+cx)^hf_exp` (RDOQ-refined when
    `rdoq`) → magnitude-category rANS with a causal **(frequency-band × neighbour-
    magnitude-sum)** context, **clustered** per tile into ≤`kDctMaxClusters`=8 rANS tables
    (`cluster_contexts`, `kDctRawCtx`=24 raw contexts) + raw mantissa/sign bits +
    **end-of-block** (frequency-ascending scan, `dct_scan_order`, code to last nonzero).
    Blocks are coded in **tiles**: `encode_tile_dct<T>`/`decode_tile_dct<T>` code a `bpa³`-
    block tile (a 64³ tile = 4³=64 blocks, `bpa=4`) that **SHARE one set of rANS category
    tables** (the JPEG XL "group" model). Decode applies quant-gated 3D deblocking
    (non-normative, decode-only) when `deblock`.
- **Shared entropy substrate** (`entropy.hpp`): fixed+LEB128 varint fields, rANS byte-plane
  coder with a compact sparse freq table, LSB-first bit packer, always-on byte-accounting
  atomics (`g_plane_*`/`g_dc`/`g_nsig`/`g_map`/`g_bits`, read under `FENIX_DCT_STATS`).
  `rans.hpp`: static byte rANS, 4-way interleaved.
- **Dtype layer** (`dtype.hpp`): `DType{u8,u16,u32,s8,s16,s32,f16,f32}` + `dtype_traits<T>` +
  `to_f32`/`from_f32` — widen→f32 for the transform, round+clamp back on decode. u32/s32
  magnitudes above 2^24 lose low bits in f32 (accepted for a lossy codec; labels use
  `lossless.hpp` instead).
- **General lossless codec** (`lossless.hpp`, rANS + delta/RLE/bitpack filters,
  `lossless_encode<T>`/`lossless_decode<T>`) for label volumes, validity masks, exact priors.
  Exact roundtrip; shares `rans.hpp`.
- **2D surface codec** (`dct64.hpp` + `tile2d.hpp`, ADR 0010) — the `.fxsurf` backend
  (container lives in `src/io/surface.hpp`, not here): a SECOND transform codec (64×64
  orthonormal DCT, `dct64_fwd_tile`/`dct64_inv_tile`) + dead-zone quant + zigzag/nsig +
  the shared lossless rANS substrate. **Tolerance-only AND verified at encode time**: each
  tile is decoded back and checked against `tau`; violators halve `q` (up to
  `kT2MaxQShift`=4×) then fall back to a raw-quantized tile (error ≤ tau/4 by construction).
  Three front-ends: `encode_field2d`/`decode_field2d` (one scalar channel — texture, height,
  confidence); `encode_rgb2d`/`decode_rgb2d` (3-channel color, YCoCg-decorrelated, per-
  channel tau/2); `encode_coords2d`/`decode_coords2d` (ZYX volume-sampling coords — per-tile
  LSQ affine fit, quantized 1/16 vox, + deterministic tangent-frame projection turning 3
  correlated channels into 1 height field + 2 near-zero remainders, per-channel tau/√3 for
  an exact 3D max-error bound). All decode paths treat input as untrusted (every
  count/length validated). This is deliberately a second transform codec next to the
  volume DCT-16 — permitted only because ADR 0010 records it; don't add a third without one.
- **The archive** (`archive.hpp`, `.fxvol` v5, class `VolumeArchive`): 64³ chunk = base IO
  unit = one DCT tile + atomic decode + network + cache unit (16³-block/voxel = a view via
  `block16()`/the decoded-chunk cache). `Coverage{Absent,Zero,Real}` tri-state via sentinel
  slots (`FxSlot{off,len}`: `off==kFxAbsent` ABSENT / `len==0` ZERO / else REAL). Layout
  per [ADR 0006](../../docs/adr/0006-fxvol-v4-container.md) +
  [docs/design/fxvol-v4-layout.md](../../docs/design/fxvol-v4-layout.md): single
  mmap'd file/volume, **3-level (12+12+12) compressed-Morton radix page-table** over a
  `MAP_NORESERVE` reservation (dims+q-derived via `reserve_for`, clamped to 32 TiB) +
  `posix_fallocate` growth + bump allocator, explicit per-LOD octave pyramid (own radix
  root per LOD in the superblock, global 2× box downsample → retile, edge-replicated —
  not zero-padded — partial chunks), double-buffered crc32c superblock + per-blob crc +
  data-before-pointer commit, LIVE append-mmap ↔ SEALED coarse-first (`finalize()`,
  verbatim blob copy, Morton order within a LOD). **All 5 phases from the design note are
  implemented**: P1 page-table+alloc, P2 crash-safe commit, P3 sharded-SIEVE decoded-chunk
  cache (`block_cache.hpp`), P4 LOD pyramid, P5 SEALED repack. S3 is read-only (anonymous
  range-GET via `io/s3.hpp`; no multi-writer CAS needed). Key methods: `create`/`open`,
  `write_volume<T>`/`write_chunk`, `read_volume<T>`/`read_chunk`, `block16(lod, ChunkCoord)`
  (decoded-16³ view via the cache), `sample_f32`, `gather_box_f32` (patch gather via
  `block16`, ~16× fewer cache calls than per-voxel), `gather_box_u8` (u8-native archives
  only: row-memcpy straight from the cached block, no f32 widen/reconvert — the ML feeder's
  fast path at scale==1), `finalize`, `commit`/`close`, `reserve_cache`.
- **`block_cache.hpp`**: `BlockCache`, a byte-budgeted, **sharded SIEVE** (NSDI'24) cache of
  decoded 16³ chunks stored as **raw native-dtype bytes** (`std::vector<u8>` — a u8 archive
  caches u8, 4× less RAM than f32; f32 exists only ephemerally when a consumer widens a
  voxel). `Ref = shared_ptr<const Block>` gives natural pinning (an evicted chunk survives
  as long as a reader holds its `Ref`). Sharded by key to cut lock contention under
  `parallel_for`.
- Image interop: first-party **PNG + JPEG + TIFF** read+write live in `src/io/`, not here;
  this dir owns the transform/entropy codecs they build on.

## Inputs / outputs & formats
In: `Volume<T>`/`VolumeView<const T>` regions (native dtype, from io transcode or pipeline
stages — no whole-volume f32 widen on ingest). Out: `.fxvol` archives (**format v5**: v4
page-table/LOD/commit layout + a source-dtype tag enabling the native-dtype read path);
`.fxsurf` payloads (v3, via `tile2d.hpp`, container in `src/io/surface.hpp`). Carries
spatial metadata + provenance at the archive/container level. Readers reject mismatched
versions — **no migration, no back/forward compat** (regenerate from source).

## Dependencies
Intra: `core` (Volume, Extent3/ChunkCoord, simd, hash, arena, threadpool, error/Expected).
Third-party: none for the codec itself (rANS/DCT/entropy are ours); `blosc2`/`zlib` only at
the io boundary (zarr chunk transcode), not used inside `src/codec/`.

## Invariants & numerics
**Tolerance-only, non-deterministic** — fast-math/float throughout; volume codec
correctness = max-error τ / PSNR, never bit-exact; the 2D codec makes this an explicit
runtime contract (encode-time per-tile verification against `tau`, not just an offline
bench). **SIMD + GPU first-class** (rANS chosen over serial CABAC/arithmetic specifically
for SIMD/GPU-amenability — load-bearing for the tile-table design). **Native dtype
end-to-end**: u8 CT stays u8 through ingest/cache/gather; never widen a whole volume to f32
(only a transient per-tile widen inside the DCT, which is unavoidable for a float
transform). Robustness is a **hard rule**: no UB/crash on any bytes (fuzzed; bounds-checked
decode paths throughout archive/dct_block/lossless/tile2d) — wrong values OK, a SEGV is a
fail.

## Performance notes
Tiles/chunks are independent → embarrassingly parallel. The 64³ chunk = the random-access
unit, so sharing rANS tables within it costs nothing. Per-block step table (not per-
coefficient `pow`) + baked scan metadata (no per-coefficient division). SIMD DCT-16 panel:
decode ~3.5→4.3-4.6 GB/s (transform-bound); encode ~2.5-3.2 GB/s (RDOQ/clustering two-pass
bound, not transform-bound). `gather_box_f32`/`gather_box_u8` cut inference-time cache
lookups ~16× per patch vs per-voxel `sample_f32`; `gather_box_u8` additionally skips the
f32 widen/reconvert entirely for u8-native archives. `BlockCache` is sharded (default 16)
to keep lock contention off the hot gather path under `parallel_for`.

## Gotchas / pitfalls
- **Two transform codecs now, by design: DCT-16 (3D, volumes) + DCT-64 (2D, surfaces,
  ADR 0010).** Wavelet retired (ADR 0005). Don't add a THIRD without a new ADR, and don't
  fork the shared substrate (rANS, dead-zone quant, magnitude-category coding,
  `entropy.hpp` helpers, dtype layer) between them.
- **Per-block static rANS tables overfit the tiny 16³ histogram** — this one fact explains
  every codec negative hit historically. Tables are **tile-global** (shared across the 64
  blocks in a 64³ tile); never go back to per-block tables, and never expand
  contexts/alphabets without the shared+clustered tables to pay for them.
- **Stay STATIC rANS** (two-pass, tile-shared). Adaptive/per-symbol ANS breaks the 4-way
  SIMD interleave.
- **No zero-tree** (a DCT block has no subband pyramid; redundant with the neighbour-
  significance context; regressed 3-15% when tried). **Keep the FIXED DCT basis** (a
  per-block Tucker/HOSVD pays ~19% factor overhead).
- **`gather_box_u8` requires a u8-native archive AND the box fully inside the volume** —
  callers must fall back to `gather_box_f32` otherwise (it does not itself fall back).
- **`reserve_cache`/replacing `block_cache_` is not safe concurrently with in-flight
  `block16()`/`gather_box_*` calls** — call it before spawning reader threads, never mid-use.
- `core/vec.hpp cross()` had a broken x-component (silently wrong every tangent
  frame/normal/triangle-area in the codebase) until the tile2d roundtrip test caught it
  (ADR 0010) — a reminder that geometry helpers used by codecs need their own direct tests.
- Carry mc's crash-safety invariants into the container (release-store/data-before-pointer
  commit, fallocate-not-ftruncate, absent-vs-failed distinct) — already implemented in
  `.fxvol` v5's P2; don't regress it. Keep this doc true to the code.

## Status & TODO
**Implemented + tested** (release + ASan, warning-free): `dct.hpp` (orthonormal DCT-II 16,
butterfly + SIMD panel, Parseval-preserving — `test_dct`); `dct_block.hpp` (the tile codec,
all `DctParams` levers landed — `test_dct_block` round-trips all 8 dtypes);
`entropy.hpp`/`rans.hpp` (`test_rans`, `test_rans_perf`); `dtype.hpp`; `lossless.hpp`;
`archive.hpp` (`.fxvol` **v5**, all 5 design-note phases: page-table+alloc, crash-safe
commit, decoded-chunk cache, LOD pyramid, SEALED repack, native-dtype src tag —
`test_archive`, `test_fxvol`, `test_pipeline`); `block_cache.hpp` (sharded SIEVE —
covered via `test_archive`); `dct64.hpp` + `tile2d.hpp` (64×64 2D codec, encode-time
tolerance verification, scalar/RGB/coords front-ends — ADR 0010, roundtrip-tested);
`test_codec_bench` (ratio/PSNR/MAE + enc/dec MB/s on real CT or local `.zarr`;
`FENIX_DCT_HF`/`FENIX_DCT_DZ`/`FENIX_DCT_STATS` env).

**DCT-16 optimization campaign (measured on 512³ PHerc Paris 4 smooth crop + 1024³
PHerc0211 dense centre, 8-bit CT; all wins PSNR-bit-identical unless noted). Landed, in
order: (1) constant-header hoist, (2) DC zigzag-varint, (3) end-of-block scan, (4)
per-block step table + baked scan metadata, (5) tile-global rANS tables (the big win —
flipped DCT past the wavelet), (6) neighbour-magnitude-sum context, (7) context-map
clustering (`cluster_contexts`, +7-9% across the board, unlocked RDOQ), (8) RDOQ
(`params.rdoq`, +0.5-1.2% iso-PSNR), (9) quant-gated 3D deblock (`params.deblock`,
decode-only, +0.05 to +0.51 dB depending on q), (10) DC+nsig causal spatial prediction
(`params.dc_predict`, +0.2-2.6% ratio), (11) SIMD DCT-16 panel (decode +25-32%).**
Cumulative vs v1: crop512 +84%/+178%/+430%, dense +47%/+90%/+222% (at clustering; RDOQ
adds more on top). Decode ~2.4→~4.6 GB/s; encode ~1.7→~4.5 then back to ~2.5-3.2 GB/s once
clustering/RDOQ's two-pass landed (one-time compression cost; decode unaffected). Full
before/after numbers per lever are in git history (commits `12bdbaf`..`cfe4cee`) if needed;
not re-duplicated here.

**Honest negatives (entropy/quant side is well-tuned; cheap levers there are mined out):**
`hf_exp`/`dz_frac` tuning marginal + data-dependent (kept 0.65/0.80, env-overridable);
zero-tree parent context regressed 3-15%; hybrid-uint small-level tokens neutral-to-
negative; reconstruction-centroid already optimal at 0.40; more clusters (8→16) is a
bit-identical no-op (greedy merge already stops below the cap); richer raw context
(24→48) is a wash-to-negative (uncompressed context map overhead eats the gain at high q).
Payload breakdown (`FENIX_DCT_STATS`, 2026-06-30): category rANS 55-61%, raw mantissa+sign
27-41% (mantissa proven-uniform, only signs attackable), freq-table headers 1-4%,
DC/nsig/ctxmap each <6% — this is why the SOTA-research plan's rANS-table-signalling pick
was skipped: our compact-sparse-table + clustering already keeps headers to 1-4%. Real
ratio headroom is the **transform** itself (secondary low-freq transform, transform-
skip/DST, trained KLT) — needs a corpus/offline training, not yet started. Per forrest
2026-06-30: the ink-survivability metric and inverted task-quality track are **dropped**
(whole-volume RD — PSNR/SSIM/MAE/pct-err — is the metric; the bench already measures it).

**`.fxvol` v4→v5 container: all 5 phases implemented**, plus the native-dtype extension
(v5) and the `gather_box_u8`/`gather_box_f32` patch-gather fast paths for ML inference/
training feeders. `reserve_for` sizes the `MAP_NORESERVE` VA reservation from
dims+q (conservative compression-ratio lower bound) instead of a flat constant, so small
crops don't reserve the full 32 TiB ceiling. Optional later refinement noted in the header:
full copy-on-write page-table versioning (not started; current form is append-only LIVE +
repack-to-SEALED, which covers the known use cases).

**`.fxsurf` 2D codec (ADR 0010): implemented and in production use** for surface export
(9.7-29.5× vs raw tifxyz depending on tau, default tau=1/4 vox ≈16.1×). v1/v2 formats
rejected per the no-compat rule.

**Open / next real levers, rough ROI order:** (a) SIMD the dead-zone quant + widen rANS
4→8-way + a SIMD 16×16 transpose for the DCT x/z-pass gather (decode is still IDCT/
transpose-bound); (b) Tier-2 transforms for ratio (transform-skip/DST — probe first,
uncertain on smooth CT vs screen content; secondary-transform/KLT needs an offline corpus);
(c) CDEF-3D dering (deblock alone is done, dering not started); (d) GPU path for either
codec (interfaces deferred project-wide, ADR 0009); (e) full COW page-table versioning if
a concurrent-writer use case emerges. **Validation caveat:** the DCT-vs-wavelet verdict
and per-lever deltas still come from 2 regions — pull more diverse scroll regions
opportunistically before trusting the numbers outside that envelope. Open ADRs referenced
here: 0002 (codec+container), 0005 (retire wavelet), 0006 (fxvol v4), 0007 (FUSE, consumes
`.fxvol` — see `src/io/CLAUDE.md`), 0009 (GPU portability), 0010 (tile2d).
