# c3d / compress3d — codec research report

Source studied (read-only, vendored copy):
- `/home/forrest/villa/volume-cartographer/libs/c3d/c3d.h` (353 lines, public API)
- `/home/forrest/villa/volume-cartographer/libs/c3d/c3d.c` (5029 lines, single TU)
- `/home/forrest/villa/volume-cartographer/libs/c3d/CMakeLists.txt`

c3d is a **3D volumetric u8 compression codec** for larger-than-RAM X-ray/CT data
(Vesuvius scroll volumes), by SuperOptimizer. It is a single translation unit,
C23, libc-only, little-endian-only, panic-on-error (no status codes). It contains
**two independent codecs** sharing infrastructure:
1. a **lossy grayscale codec** (CDF 9/7 wavelet + dead-zone quant + rANS) — the
   primary subject and the one relevant to the fenix wavelet rewrite;
2. a **lossless multi-channel label codec** (octree + rANS), out of scope but
   noted.

Both share: MurmurHash3_x64_128 content hash, an 8-way interleaved rANS engine,
LEB128 varints, and the panic/assert scaffolding.

---

## 0. Fixed hierarchy constants (the "atoms")

| constant | value | meaning |
|---|---|---|
| `C3D_CHUNK_SIDE` | **256** | the codec atom: one encode/decode call = 256³ u8 |
| `C3D_VOXELS_PER_CHUNK` | 16,777,216 | 256³ |
| `C3D_BLOCK_SIDE` | 16 | caller-side RAM cache granularity (not the codec's) |
| `C3D_N_DWT_LEVELS` | **5** | 5 dyadic decomposition levels |
| `C3D_N_LODS` | **6** | LOD 0 = 256³ … LOD 5 = 8³ |
| `C3D_N_SUBBANDS` | **36** | 1 × LLL_5 + 5 levels × 7 detail subbands |
| `C3D_ALIGN` | 32 | required alignment of raw voxel buffers |

Note the **codec atom is 256³, not 64³**. fenix wants 64³ blocks — a deliberate
divergence. c3d's whole offset/LOD/subband table design is sized to 256³ and 5
levels; fenix at 64³ would naturally use 3–4 levels (64→32→16→8→4).

There is a separate u64 voxel key (`[lod:4][z:20][y:20][x:20]`, **planar, not
Morton**) used by the *caller's* archive layer, not internally by the codec.

---

## 1. The transform

- **Wavelet family:** **CDF 9/7** (the irreversible JPEG 2000 lossy wavelet),
  implemented via the **lifting scheme** (JPEG 2000 Part 1 Annex H cascade), not
  convolution. Coefficients (`c3d.c:831-836`):
  - α = −1.586134342059924, β = −0.052980118572961,
    γ = 0.882911075530934, δ = 0.443506852043971,
    K = 1.230174104914001, 1/K = 0.812893066115961.
  - Cascade: predict1(α) → update1(β) → predict2(γ) → update2(δ) → scale
    (even ×1/K, odd ×K).
- **Float, not integer.** Everything is `float` (f32). This is a *lossy* path;
  there is **no reversible 5/3 integer mode** and no lossless grayscale mode.
- **Separable 3D.** Each level runs 1D lifts along X, then Y, then Z, then
  recurses on the LLL octant (`c3d_dwt3_fwd`, `c3d.c:1306`). 5 levels on 256³ →
  LLL_5 occupies `[0:8,0:8,0:8]`.
- **Boundary handling:** whole-sample symmetric (mirror) extension at both ends:
  `x[-1]=x[1]`, `x[N]=x[N-2]` (`c3d.c:820-823`, applied as the doubled-tap edge
  updates in `c3d_cdf97_lift_fwd`).
- **In-place, deinterleaved layout.** After each 1D lift, samples are
  deinterleaved into `[evens(L) | odds(H)]` halves (`c3d_deinterleave`). So the
  36 subbands live as contiguous octant regions in one 256³ float buffer; a
  `c3d_subband_info` descriptor (`c3d.c:1671-1709`) gives each subband's
  `(level, kind, side, z0,y0,x0)`.
- **Subband ordering (canonical, coarsest-first):**
  - index 0 = LLL_5 (side 8)
  - indices 1..7 = level-5 details (side 8), 8..14 = level-4 (side 16),
    15..21 = level-3 (32), 22..28 = level-2 (64), 29..35 = level-1 (128).
  - Within a level the 7 detail kinds are ordered HHH, HHL, HLH, LHH, HLL, LHL,
    LLH (letter order ZYX), kinds 1..7. Each H adds +side on its axis.
- **SIMD:** the lifting deinterleave/interleave and the quant/dequant inner loops
  have AVX-512 / AVX2 / NEON / scalar variants. The DWT itself is also
  4-column-blocked (`_x4` variants) and OpenMP-parallel (`_team` functions; one
  `omp parallel` spans all 5 levels with `omp for` barriers chaining axes/levels).

## 2. Quantization

- **Dead-zone uniform (mid-tread) scalar quantizer**, per subband
  (`c3d_quant`/`c3d_dequant`, `c3d.c:1372-1389`):
  - `|c| < dz_half → 0`; else `sign·(floor((|c|−dz_half)/step)+1)`.
  - Dead-zone half-width = `dz_ratio · step`, with `dz_ratio = 0.55` for every
    subband kind currently (the per-kind table exists but is flat — found <0.02
    dB benefit because the R-D allocator already redistributes bits).
  - Dequant reconstructs at `dz_half + (|q|−1+α)·step` where **α ∈ [0.40,0.50]**
    is a per-subband Laplacian-optimal bin-offset, fit closed-form per subband
    (`α* = 1/u − 1/(eᵘ−1)`, u=step/β̂) and stored as one u8 per subband in the
    header. Default α: 0.45 (LLL), 0.40/0.375/0.33 by HF-axis count.
- **Per-subband step = `q · baseline · coeff_scale`**:
  - `coeff_scale` = global max|coeff| post-DWT (stored in header, absorbed into
    step so decode skips a normalize pass).
  - `baseline` = **perceptual / R-D weight** derived from CDF 9/7 synthesis gains
    squared (G_L²=2.08, G_H²=0.48), `1/w^softness`, softness=0.60, geomean-1
    normalized. Deep LF bands get fine steps, HF bands coarse.
  - `q` = the single global quality scalar (range 2⁻¹² … 2¹²).
- **Rate control (`target_ratio`):** bisection/Newton hybrid on `q`
  (`c3d.c:3211-3262`). The rate curve log(bytes) vs log(q) is ~linear slope
  −1.5; an EMA-tracked slope drives a log-space Newton step inside a maintained
  bracket; warm-start from previous chunk's q. Uses an O(1) **fine-histogram**
  rate estimator (1024 bins × 36 subbands prefix-sums, built once post-DWT) so
  trial steps don't re-quantize the whole buffer. Typically 1–3 iterations.
- **R-D allocator (two-pass calibrated Lagrangian, §Q3 v3, `c3d.c:3263-3307`):**
  pass 1 emits at the global q and measures actual per-subband bytes; pass 2 runs
  a Lagrangian (λ bisection) over a small grid of per-subband candidate steps,
  with calibrated rate (Shannon from the fine histogram, scaled by actual/est
  ratio) and synthesis-gain-weighted distortion, then re-emits with per-subband
  steps. Skipped when degenerate (steps within ±1%). ~+0.10 dB, ~9% encode cost.
  Disable via `C3D_NO_RD`.
- **~50× ratio** is just a target_ratio input; the codec spans roughly r≈5
  (near-lossless) to r≈200+. No hard max-error / strictly-near-lossless mode —
  it is rate- or q-driven, distortion follows.
- **Post-decode denoiser:** an optional separable 3×3×3 box blur blended with
  identity at strength α (ratio-dependent: 0 at near-lossless, mild at r≈50),
  applied at LOD 0 only. The encoder picks α and writes it to header byte 7; the
  decoder can be told to skip it (`c3d_decoder_set_denoise`). Buys ~0.03 dB.
- **`_masked` ("zero means ignore") variants:** voxels == 0 are treated as
  don't-care; replaced with the min non-zero value before DWT so air/material
  steps don't waste bits. Big win on ~40% air scroll data.

## 3. Coefficient / entropy coding

**Not** a zerotree/EZW/SPIHT/bitplane embedded coder. It is a **per-subband
quantize-then-rANS** scheme:

- **Symbol alphabet: 65 symbols** with **sign-prediction** mapping
  (`c3d_quant_to_symbol`, `c3d.c:1604`):
  - sym 0 = zero coeff (no sign).
  - sym 2k−1 / 2k = magnitude k (1..31), sign prediction correct / wrong.
  - sym 63 / 64 = escape (|q|≥32), sign correct / wrong; magnitude sent as a
    **LEB128 in a side "escape stream."**
  - Sign is predicted from the previous non-zero coeff at the **same (y,x) column,
    previous z** (`prev_sign_zy`) — CT coefficients correlate across slices, so
    "correct" symbols get high probability → fewer rANS bits.
- **Entropy coder: 8-way interleaved rANS** (ryg_rans_byte style; 32-bit state,
  byte renorm, lower bound 2²³), `c3d.c:237-663`. 8 independent lanes dealt
  round-robin, unrolled for ILP; symbol-0 fast path hoisted in both enc and dec.
  Per-subband frequency table is built from a histogram, normalized so freqs sum
  to M = 1<<denom_shift (denom_shift = 14 for LLL_5, 12 otherwise), serialized as
  `denom_shift, n_nonzero, n_nonzero×{sym, LEB128 freq}`.
- **Optional 2-table context model:** each subband may use two frequency tables
  selected by whether the same lane's previous symbol was zero (`_ctx` variants).
  The encoder estimates 1-table vs 2-table rate and picks the cheaper; a ctx_mode
  byte precedes the table(s).
- **Per-subband bitstream:** `u16 freq_table_size, freq_table, u32 n_symbols,
  u32 rans_block_size, rans_header(32B)+renorm, escape_stream`. All-zero subbands
  emit a 2-byte sentinel `0xFFFF`.
- **Determinism:** same-binary encode is byte-deterministic; cross-binary is not
  (float DWT). All quant/dequant SIMD paths are bit-exact vs scalar (truncating
  conversions, mask-select dead-zone). Build uses strict IEEE math (no
  -ffast-math) specifically to keep bytes reproducible.

## 4. Container / IO

c3d is a **block codec, not an archive.** It is purely in-memory (never touches
disk/fd/network). One call = one 256³ chunk → one self-contained byte payload.

**Chunk wire layout** (`C3D_CHUNK_MAGIC "C3DC"`, fixed header 388 B):
```
  0   "C3DC" magic (4)
  4   u16 format version (=1)
  6   reserved
  7   denoise alpha (u8, =round(α*400))
  8   f32 dc_offset           (chunk mean, removed before DWT)
 12   f32 coeff_scale         (max|coeff|, informational on decode)
 16   reserved/flags region … to 40
 40   qmul[36]   (144 B)      per-subband step (f32 each)
184   subband_offset[36] (144 B)  u32 byte offset of each subband in entropy region
328   lod_offset[6] (24 B)    u32 cumulative entropy size at each LOD boundary
352   alpha_per_subband[36] (36 B)  per-subband dequant α (u8 each)
388   entropy payload (variable), subbands in coarsest-first canonical order
```
- **Multiresolution / LOD is native and free.** Because subbands are stored
  coarsest-first, decoding LOD k means decoding only the first
  `c3d_n_subbands_for_lod[k]` subbands (36/29/22/15/8/1) and running
  `5−k` inverse DWT levels (`c3d_decoder_chunk_decode_lod`). `lod_offset[]` lets
  the caller fetch only the prefix bytes for a coarse preview.
- **Quality-scalable truncation (§T9):** if the supplied `in_len` is shorter than
  the full chunk (caller truncated for streaming), subbands past the available
  bytes are zero-filled instead of erroring. Quality degrades gracefully and
  monotonically — appending bytes only improves. This is an SNR-scalable
  embedded-ish property achieved at subband granularity (not bitplane).
- `c3d_chunk_inspect` (read header/LODs without entropy decode) and
  `c3d_chunk_validate` (structural check: magic, version, monotone offsets).
- An empty/uniform chunk reconstructs from the 388 B header alone (all-zero
  tables, lod0 == 0).

## 5. API surface, data structures, deps, threading

- **Stateless one-shot:** `c3d_chunk_encode(in, target_ratio, out, cap)`,
  `c3d_chunk_encode_at_q(in, q, …)`, `c3d_chunk_decode[_lod]`, plus `_masked`
  variants and `c3d_downsample_chunk_2x`.
- **Reusable contexts** (recommended): `c3d_encoder` / `c3d_decoder` with
  `*_new`/`*_free` and `*_chunk_encode[_at_q]` / `*_chunk_decode[_lod]`, plus
  batched `*_chunks_encode/decode`. NOT thread-safe per instance → one per worker.
  Encoder owns ~115 MiB scratch (coeff_buf 64 MiB + symbol/escape/rANS buffers +
  fine histograms); decoder ~80 MiB.
- **Dependencies:** libc + libm only. No zlib/jpeg/external entropy lib.
- **Threading:** OpenMP, parallelized **within a single chunk** over the 36
  subbands (dynamic,1 schedule) for both quant/encode and decode, and over axis
  passes in the DWT. Designed for one chunk per thread also via separate contexts.
- **Error model:** `c3d_panic`→`abort()` (overridable hook). No recoverable
  errors.

## 6. Performance

- Reusable contexts save 50–100 ms/chunk of alloc churn (header comment).
- Uniform-chunk fast path skips the DWT entirely (~80 ms saved) — critical since
  75–85% of masked scroll chunks are all-air or all-material.
- All-zero subband fast paths (2-byte sentinel) skip rANS for ~10–20 of 36
  subbands at r≥50.
- Rate-control estimator avoids per-iteration full emits; warm-start + Newton
  cuts bisection to ~1–3 iters. Exact ms/throughput numbers aren't in the source
  (only relative claims and PSNR deltas), but the design is clearly tuned for
  hundreds-of-MB/s-class CT throughput on AVX-512/NEON.

---

## 7. Recommendations for the fenix wavelet rewrite (greenfield, C++26, 64³)

**Carry over (these are the good, hard-won parts):**
1. **CDF 9/7 lifting in float, separable 3D, mirror boundary.** Correct, fast,
   standard, SIMD-friendly. The exact lifting coefficients and the doubled-tap
   symmetric-edge handling are reusable verbatim.
2. **Coarsest-first subband ordering → native LOD pyramid + byte-prefix
   truncation.** This is the single most valuable design idea and maps perfectly
   onto fenix's "LOD pyramid + per-block self-contained payload" requirement.
   Keep `subband_offset[]` + `lod_offset[]` so a coarse decode reads only a
   prefix. Keep §T9 quality-scalable truncation.
3. **Dead-zone uniform quant with per-subband step + Laplacian-α bin offset.**
   Simple, effective, far less complex than embedded coders. Keep the
   synthesis-gain-squared perceptual baseline weighting (G_L²/G_H²) — it is the
   bit-allocation backbone.
4. **8-way interleaved rANS + per-subband normalized freq tables + sign-predictive
   65-symbol alphabet with LEB128 escapes.** This is a clean, deterministic,
   high-throughput entropy stage. The cross-slice sign predictor and the optional
   2-table lane context are cheap wins worth keeping.
5. **Two-pass calibrated R-D allocator + log-space Newton rate control with
   warm-start.** Excellent rate accuracy at low cost; the fine-histogram O(1)
   estimator is the key enabler.
6. **Determinism discipline:** strict IEEE math, bit-exact SIMD-vs-scalar,
   same-binary reproducibility. Preserve as a hard invariant for an archive
   format.
7. **Self-contained per-block payload + cheap inspect/validate + uniform/all-zero
   fast paths.** Directly matches fenix's container model.

**Redesign / reconsider for fenix:**
1. **Block size 256³ → 64³.** Everything sized to 256³/5-levels/36-subbands must
   be reparameterized. 64³ with **3–4 levels** gives 22 (3-level) or 29 (4-level)
   subbands. At 64³ the LLL is tiny (4³ or 8³) — reconsider denom_shift and
   per-subband table overhead, which becomes proportionally larger on small
   blocks. Freq-table bytes (up to ~700 B/subband worst case, typically tens) and
   the 32 B rANS state header per subband are a real overhead tax at 64³; consider
   sharing/merging tables across the smallest subbands or a global table option.
2. **Target ratios 10×–1000×.** c3d tops out around r≈200 in practice. For the
   high end fenix will need: coarser steps, aggressive HF zeroing, possibly
   dropping whole HF levels (decode-as-LOD), and the freq-table/rANS-header fixed
   overhead per subband must be amortized or it dominates at 1000×. The all-zero
   subband sentinel and "emit fewer subbands" become the dominant levers.
3. **Container integration.** c3d has no archive layer at all (caller-owned). For
   fenix, fold the per-block payload into the shared archive container with the
   bitmap + LOD pyramid like the DCT-16³ codec — keep the c3d *block* payload
   self-contained but standardize the header to match the DCT sibling so one
   container holds both. Drop c3d's 388 B fixed header in favor of the shared
   format; keep the coarsest-first + offset-table concept.
4. **C++26 ergonomics:** replace panic/abort with the project's error policy;
   replace the malloc scratch pools + OpenMP-within-chunk with the fenix
   threading model (likely block-parallel, so per-block work can stay scalar +
   SIMD and parallelism lives at the archive level — simpler than c3d's
   subband-parallel-within-chunk). Replace the C macro SIMD kernels with
   std::simd / portable intrinsics wrappers.
5. **Possibly drop:** the post-decode box-blur denoiser (~0.03 dB, adds decode
   cost and a header byte) unless preview quality demands it; the per-kind
   dead-zone plumbing (currently flat 0.55); and `coeff_scale` from the decode
   path (already informational only).
6. **Consider but probably skip:** a reversible 5/3 integer / lossless mode — the
   near-lossless DCT-16³ sibling already covers that niche; fenix-wavelet should
   stay the high-ratio lossy preview/streaming path, which is exactly c3d's
   sweet spot.

**Out of scope but noted:** c3d also ships a lossless label codec (octree depth
constant + rANS + bitpacking, `C3D_LABEL_*`). Independent of the wavelet path;
ignore for the grayscale wavelet rewrite unless fenix also needs label volumes.
