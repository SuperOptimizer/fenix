# fysics — research report for a C++26 rewrite

**Subject:** `/home/forrest/taberna/third-party/fysics/` — a first-party (author:
SuperOptimizer) pure-C library of physics / image-processing kernels plus a
two-pass whole-volume preprocessing pipeline, a multi-resolution registration
stack, and a streaming process+export tool (`mca_export`) for Vesuvius Challenge
Herculaneum scroll CT volumes reconstructed by ESRF/BM18 + nabu (single-distance
Paganin phase-contrast tomography).

Read in full: `fysics.h`, `fft.c`, `paganin.c`, `dering.c`, `musica.c`,
`guided.c`, `phasecorr.c`, `noise.c`, `zdrift.c`, `downscale.c`, `zarr_io.c`,
`stream.c`, `register.c`, `pipeline.c`, `tools/fysics_process_main.c`,
`CMakeLists.txt`, `README.md`, `REGISTRATION.md`, `REG_REFINE.md`,
`docs/esrf-pipeline.md`. Skimmed: `tools/mca_export.c`.

---

## 0. Design philosophy and shape of the codebase

- **Pure C11, CPU-only, dependency-free core.** No SIMD intrinsics — the code is
  written deliberately flat (SoA, `restrict`, no aliasing, loop-carried
  dependencies removed) so `-Ofast -ffast-math -funroll-loops -march=native`
  auto-vectorizes the hot loops. This is an explicit design constraint, repeated
  in nearly every file header.
- **Built with `-ffast-math`** → `isnan`/`isinf` are unreliable; the code uses
  explicit validity masks instead of NaN sentinels (see `dering.c`).
- **Thesis (honest scope):** fysics *analytically inverts the known nabu
  reconstruction operators* (Paganin phase-retrieval low-pass + unsharp mask + the
  per-volume u8 export window) to restore contrast/sharpness, complemented by
  measured-from-data denoise / dering / z-drift. It explicitly does **not** claim
  super-resolution (FRC-verified: ~2.5–3× HF *contrast* lift, no SNR-limited
  resolution gain), and does **not** invert sinogram-domain operators (ring
  removal, FBP ramp, flat/dark — not recoverable from the reconstructed volume).
  `docs/esrf-pipeline.md` is the authoritative per-operator invertibility audit.
- **Geometry convention everywhere:** row-major, x fastest,
  `idx = (z*ny+y)*nx + x`, axis order `(z,y,x)`.
- **Streaming model:** fysics does per-chunk *math*; the caller owns chunk
  iteration + I/O. LOCAL ops (deconv/denoise/mask/dering-apply) run on one
  chunk + halo independently; GLOBAL ops (normalize, z-drift, ring detection)
  are TWO-PASS (accumulate tiny state → finalize → apply).
- **Threading:** OpenMP (the stated reason for the C rewrite of an earlier Python
  pipeline: "no GIL"). The kernels themselves are single-threaded and reentrant;
  `pipeline.c` parallelizes over tiles. `mca_export` uses pthreads
  (downloader pool + compute pool + bounded queue).
- **FFT dependency: OWN.** `fft.c` is a self-contained split-radix-ish FFT
  (radix-2 iterative Cooley-Tukey + a top-level radix-3 split). **No FFTW, no
  external FFT.** Sizes supported: `2^k` and `3·2^k`.
- **Build:** CMake. Core static `libfysics.a` + shared `libfysics.so` (always
  built with `FYSICS_S3` defined → needs `libcurl` + a `s3` target). Two CLIs:
  `fysics-process` (zarr→zarr, core-only) and `mca_export` (zarr/S3 → `.mca`
  archive, needs the external `matter_compressor` target). `s3` and
  `matter_compressor` are external targets provided by the parent `taberna`
  superbuild (no longer vendored).

---

## 1. The kernels — algorithm + math

### 1.1 FFT (`fft.c`)
- Iterative in-place radix-2 Cooley-Tukey with **precomputed per-size thread-local
  twiddle tables** (`__thread float *tw_re[32]`), built on demand. Tables make the
  butterfly an independent-iteration vectorizable loop (the old in-loop rotation
  recurrence was a loop-carried dep + accumulated drift).
- **Radix-3 split** (`fft1d_radix3`): for `n = 3·2^k`, DIT-by-residue into three
  m-point pow2 FFTs combined with cube-roots of unity; per-(n) twiddle + scratch
  cached thread-locally. This lets a 176-voxel halo'd tile pad to 192 not 256
  (~2.4× less FFT work).
- `fy_fft1d(re, im, n, sign)` split-complex (SoA); `sign=-1` fwd, `+1` inv, no
  normalization. `fy_fft3d` transforms each axis with a gathered scratch line.
  `fy_next_fft_size(n)` picks the smallest supported size ≥ n.
- **Note:** the Paganin deconv path (`paganin.c`) does NOT call `fy_fft3d`; it has
  its own optimized real-input (R2C/C2R Hermitian half-spectrum) driver inline.

### 1.2 Paganin phase retrieval + Wiener deconvolution (`paganin.c`) — the core
- Transfer functions:
  - `T_paganin(f) = 1 / (1 + delta_beta·λ·D·π·f²)`, λ = `1.23984199e-3 / energy_keV`
    µm, D = `distance_mm·1000` µm, f converted cycles/voxel→cycles/µm via
    `pixel_um` (auto-corrects a mm value passed as <0.1 "µm"). Matches nabu to
    ~1e-16.
  - `U_unsharp(f) = 1 + coeff·(1 - exp(-2π²σ²f²))` (gaussian unsharp).
  - `H = T_paganin · U_unsharp`, floored at 1e-4.
- **`fy_deconvolve`:** Wiener inverse `F_out = F_in · H/(H²+reg)`. `reg<=0` →
  `fy_auto_reg` (default 0.015). Gain hard-capped at `FY_MAX_DECONV_GAIN = 8.0`
  per-frequency so bad metadata can't blow up the noise floor.
- **`fy_deconvolve_matched`** (the pipeline's STORED default when deconv is on):
  inverts nabu's *measured effective* forward operator `H_fwd = G_psf · U_unsharp`
  where `G_psf = exp(-2π²σ_psf²f²)`. Key insight: nabu's unsharp already largely
  undid the Paganin blur, so the net residual volume blur is only ~1 voxel, not
  the ~9.8 vox naive Paganin implies — inverting full Paganin over-boosts. Tikhonov
  default 0.05. Supports **anisotropic PSF**: z frequency axis scaled by
  `(σz/σxy)²` (helical scans blur ~1.17× broader in z).
- **`fy_auto_deltabeta_scale`:** regime-dependent partial inversion (Gureyev
  analog). Discriminator = `H_nyq = T_paganin(0.5)`. Fine/strong-filter volumes
  (≤4.3 µm, H_nyq ≤ 0.0025) → 0.35× δ/β; coarse (≥8.6 µm, H_nyq ≥ 0.007) → 1.0×;
  linear ramp between. Measured over 90 cubes / 18 volumes.
- Shared core `deconv_radial`: reflect-pads to FFT-friendly size, runs the whole
  pipeline on the **Hermitian half-spectrum of real input** (two-for-one packed
  X R2C, Y/Z complex on hx columns, weight via a 4096-entry radial LUT over f²,
  inverse, crop). ~2× less FFT work + memory than full-complex.
- `fy_kernel_halo`: impulse-response decay extent of the filter (8..96 vox) → the
  tiling halo so region+halo processing is seam-free.

### 1.3 Ring artifact removal (`dering.c`) — detect-then-subtract, 2-pass
- Rings = angularly-invariant radial features centered on the **rotation axis**
  (metadata `rotation_axis_position`; BM18 = slice center). nabu does NO sinogram
  ring removal on these scans, so the volume carries unfiltered detector stripes.
- State: per `(z-slab, sector, radius-bin)` intensity sums/counts.
  - PASS 1 (`fy_dering_accumulate_u8`): accumulate from raw u8 tiles; per-(y,x)
    radius+sector plane-map computed once and reused across z (sqrt+atan2 were a
    hotspot). Per-thread single-slab scratch + locked merge.
  - FINALIZE (`fy_dering_finalize`): per slab+sector angular-mean radial profile →
    box high-pass → per-radius **sector sign-consistency vote** (a true ring sits
    at the same radius in every sector; a spiral papyrus wrap drifts in radius with
    angle and fails the vote). Ring estimate = median over sectors, clamped
    [min_amp, max_amp]. Verified PHerc0139 2.4µm: 267 ring radii, 93% ring energy
    removed in one pass.
  - PASS 2 (`fy_dering_apply`): subtract `ring[slab][rbin(y,x)]·scale` per voxel —
    a LOCAL op (tileable, no halo). Masked voxels (f≤0) untouched, clamp ≥0.
- Honest: a measured/gated complement, NOT a physics inverse. On by default in the
  pipeline, gated on ≥2 detected radii.

### 1.4 MUSICA contrast enhancement (`musica.c`) — per-slice viewing aid
- Vuylsteke & Schoeters (Agfa) multiscale contrast enhancement. 2D per-slice.
- Laplacian-pyramid analogue built via repeated 5-tap binomial blurs (cumulative
  octave: 1,3,7,15 passes per level); sublinear gain on detail coeffs
  `y = a·sign(x)·|x/a|^p` (p<1 boosts faint detail) with soft coring of
  small (noise) coeffs.
- **Mask/clip aware via normalized convolution:** only voxels strictly in (0,1)
  are real; masked (==0) and clipped (==1, i.e. u8 255) are excluded from the blur
  (`blur(value)/blur(weight)`) so black/white doesn't bleed across boundaries.
  Those pass through unmodified.

### 1.5 Guided filter (`guided.c`) — the recommended fast denoiser
- He–Sun–Tang O(N) edge-preserving filter, self-guided: `a = var/(var+eps)`,
  `b = mean·(1-a)`, `out = a·I + b` (a,b box-averaged). eps = range param from
  measured noise; radius r voxels. Apply AFTER deconv.
- Box filter = three separable 1D moving-sum passes, with precomputed reciprocals,
  X+Y fused per z-plane into an L2-resident buffer, Z slides whole X-rows. This is
  the pipeline's hottest kernel (~78% of runtime); heavily bandwidth-optimized.
- `fy_guided_denoise_ws` (caller-supplied `4·n` float workspace, avoids malloc
  churn in the per-tile loop). `fy_guided_denoise_fast_ws` (He & Sun 2015): compute
  a,b on an s-times *decimated* grid (radius/s), trilinearly upsample, apply at full
  res → ~s³ less box work, visually identical at s=2 (pipeline default).
- `fy_box_smooth`: plain single box pass (used for the air-cut scratch).
- `fy_guided_eps_for_noise`: `eps ≈ (3·noise_ref)²`.

### 1.6 Phase correlation + Mutual Information (`phasecorr.c`)
- `fy_phase_correlate`: sub-voxel translation via Fourier shift theorem. Mean-remove
  + separable Hann window → zero-pad to next pow2 → normalized cross-power spectrum
  `conj(F)·M/|·|` (phase-only → contrast/brightness robust) → inverse FFT → integer
  peak → **Foroosh (2002)** sub-pixel estimator (the phase-corr peak is a Dirichlet
  kernel, not a parabola; a 3-pt parabola biases ~0.1 vox). Accurate to ~0.01 vox.
  Returns shift (dz,dy,dx) + a normalized peak confidence in [0,1].
- `fy_mutual_information`: joint-histogram MI in nats over the in-bounds warped
  overlap (both scaled to [0,nbins) by own overlap min/max → invariant to monotone
  intensity change, unlike NCC which assumes linear). The multimodal gold standard;
  a drop-in alternative metric for cross-energy registration.

### 1.7 Noise model (`noise.c`)
- `fy_estimate_noise`: per-volume signal-dependent noise `var = g·I + b`. Local
  mean+variance in win³ windows (separable box sums), bin by intensity, take a LOW
  percentile of per-bin local variance (the noise floor; rejects edges), robust IRLS
  line fit. Primary robust output `noise_ref` = std at `ref_intensity` (slope often
  ill-conditioned). Measured 145 cubes/18 scrolls: level varies 1.5–3.3× scroll-to-
  scroll AT THE SAME resolution → must be estimated per-volume, not hardcoded.
- `fy_noise_aniso`: PSF z/xy ratio from lag-1 autocorrelation of the residual in
  flat 8³ blocks (`rho = exp(-1/(4σ²))` for a gaussian PSF). Feeds the matched
  deconv's anisotropic z inversion.

### 1.8 z-drift / shading correction (`zdrift.c`) — 2-pass streaming
- Removes the slow axial brightness gradient from beam-current drift (~13% drop,
  metadata `machineCurrentStart/Stop`). PASS1 `fy_zdrift_accumulate` → per-slice
  papyrus (above-threshold) sum/count; FINALIZE `fy_zdrift_finalize` → smoothed
  (window ~nz/20) per-slice multiplicative factor to the global papyrus mean;
  PASS2 `fy_zdrift_apply` multiplies. `fy_correct_zdrift` does it all in-RAM.

### 1.9 Downscale (`downscale.c`) — 2× LOD pyramid kernels
- Strictly within-cell (no halo, tiles independent), zero-preserving (all-zero cell
  → 0, so coarse levels are valid occupancy masks). `FY_DS_BOX` = 2×2×2 mean;
  `FY_DS_CBOX` = contrast-box, mean pushed toward the max-deviation voxel by `alpha`
  (keeps thin sheets/gaps visible at coarse zoom). Operates on u8. From
  volume-compressor.

### 1.10 Streaming/global stats helpers (`stream.c`)
- u8↔float; **u8↔physical attenuation** via the per-volume export window
  (`fy_u8_to_phys`/`fy_phys_to_u8`, exact linear inverse of nabu's windowing,
  `target_window_f32_min/max`) — the only way to get consistent physical units
  across volumes.
- Histogram state (256 bins, mergeable for parallel accumulation) + percentiles
  (`_inner` excludes bins 0 masked / 255 clipped). Global normalization
  finalize/apply. `fy_valley_depth` (bimodal split / Fisher discriminant for the
  air-cut). `fy_flat_noise` (median/low-percentile local std over bright blocks).

---

## 2. Registration stack (`register.c`, `REGISTRATION.md`, `REG_REFINE.md`)

Goal: align multiple CT scans of the SAME scroll at different
resolution/energy/time (e.g. PHerc0139 1.129µm vs 2.399µm) so they can be fused.
Three conceptual layers (Layer 3 = fusion, not in this lib).

- **Trilinear sampler** `fy_sample_trilinear` (OOB→0) is the shared primitive.
- **`fy_warp_affine`**: 3×4 affine M (12 doubles, row-major, (z,y,x), maps
  OUTPUT→INPUT voxel coords, "pull"/backward warp; translation col = M[3,7,11]).
- **LAYER 1 — affine/rigid intensity registration** (`fy_register_affine`):
  - Metric = **NCC over in-bounds overlap** (`fy_ncc_warped`), invariant to affine
    intensity change a·I+b → robust to cross-energy brightness/contrast (SSD would
    be wrong). MI (`fy_mutual_information`) is the documented multimodal upgrade.
  - Coarse-to-fine: up to 4-level gaussian pyramid (`fy_downsample2x`, σ~0.8 blur +
    2× decimate, stop <16). Per level: derivative-free **coordinate descent**
    (sweep each param ±step, accept improvements, halve steps on no-improvement).
  - Params: rigid = 7 dof `[rz,ry,rx, tz,ty,tx, log(s)]` (isotropic scale, about
    volume center); affine = 12. Translation rescaled by 2^level between levels.
  - Validation: self-registration recovers known transforms to ~0.7% MAD; survives
    gamma=1.8 contrast remap at NCC>0.97.
- **LAYER 2 — deformable Demons** (`fy_register_demons`):
  - Classic Thirion demons with **symmetric (active) forces** (uses both fixed and
    moving gradients), Cachier-normalized denominator (`K² = dynamic range²` guards
    the aperture/flat-region problem), fluid-then-elastic gaussian field
    regularization (`field_sigma`), coarse-to-fine pyramid with ×2 field upsample +
    value-doubling. Field `u=(uz,uy,ux)` on fixed grid = pull displacement;
    `fy_warp_field` applies it. NOT diffeomorphic (additive demons); SSD-style force
    → needs intensity normalization between energies.
  - `fy_register_full` = affine then demons.
- **`REG_REFINE.md` — the hard-won real-data result:** on real PHerc0139, the
  intensity-driven NCC/Demons path is non-discriminative on self-similar laminar
  papyrus. The working approach is **landmark-seeded local-patch refinement**:
  reconcile the shipped `transform.json` (axes are XYZ not ZYX; matrix maps
  moving→fixed, scale ~0.4701), resample, sample ~400 textured patches,
  `fy_phase_correlate` per patch → RANSAC-fit a global affine correction.
  Achieved ~0.73 coarse-voxel (1.76µm) accuracy; the residual is a REAL physical
  scan-to-scan warp floor, not an algorithm limit. **Demons makes the independent
  residual WORSE here** (snaps onto spurious self-similar matches) — documented
  negative result: do not run intensity-driven Demons on this material.

The registration stack is essentially a separate library that shares the sampler /
pyramid / FFT; it is NOT wired into the production `mca_export` pipeline (that is
single-volume preprocessing). It is the foundation for future multi-resolution
fusion.

---

## 3. Zarr I/O (`zarr_io.c`)

- **Zarr v2, raw uint8, no compression**, `dimension_separator '/'`, level `0/`
  with `.zarray` JSON + chunk files at `0/z/y/x` (C order). Missing chunk → all
  `fill_value` (sparse air padding). Tiny hand-rolled JSON int grabbers (no JSON
  lib).
- `fy_zarr` struct: root, shape[3] (z,y,x), chunk[3], fill. `fy_zarr_open` reads
  `.zarray`; `fy_zarr_create` writes one; `fy_zarr_read` assembles an arbitrary
  region from chunks; `fy_zarr_write_chunk` writes one output chunk.
- **Local chunks are mmap'd** (no intermediate copy; a 16-vox halo sliver faults
  only the pages it touches). mmap-failure falls back to fread into scratch.
- **Optional S3** (`#ifdef FYSICS_S3`, always on in the build): `s3://` roots fetch
  per-chunk via vendored `libs3` + `libcurl`, **anonymous/unsigned** requests
  (Vesuvius buckets are public), region-pinned. `fy_zarr_read` BATCHES a region's
  chunk GETs over a pooled connection set (`s3_get_batch`) for throughput; an
  ownership-stealing `fy_s3_get_own` hands back the libs3 body to avoid a double
  copy. **Hard distinction: 404 = legitimate sparse fill; any other failure is
  retried with backoff and then a HARD error** (a transient blip must never
  silently become air in a multi-TB output).

---

## 4. The pipeline (`pipeline.c`) and ESRF end-to-end flow

`pipeline.c` is the 2-pass whole-volume preprocessing orchestration (C + OpenMP;
a port of an earlier `superres/fysics_pipeline.py`).

### 4.1 Public pipeline API
- `fy_pipeline_cfg` — large config struct: physics (from metadata), per-stage
  toggles/params, and resolved calibration STATE (filled by `fy_calibrate`).
- `fy_run_pipeline(in_root, out_root, cfg, tile, verbose)` — full zarr→zarr 2-pass.
- `fy_calibrate(in_root, cfg, tile, verbose)` — run only PASS 1 (calibrate-only;
  `fy_run_pipeline` with `out_root==NULL`). Fills `cfg` incl. dering/zdrift/norm/
  dec-range state.
- `fy_process_chunk(zin, cfg, z0,y0,x0, tile, out, ...)` — process one inner tile
  with a halo'd read using calibrated cfg (thread-safe).
- `fy_process_buffer(cfg, u8buf, ...)` — the I/O-FREE half: process one inner tile
  from an already-read halo'd u8 buffer (this is what `mca_export`'s downloader/
  compute pool calls). Thread-local reused scratch (8·hn floats).

### 4.2 PASS 1 — unified calibration sweep (parallel, budget-sampled)
One OpenMP pass over budget-sampled 256³ "ptiles" (occupancy-guided: skip ptiles
whose coarse-pyramid chunks are all absent; a masked scroll is ~75% empty; I/O
budget default 200 GB spread over the occupied volume) accumulates:
- streaming/cheap (every sampled ptile): **norm histogram**, **per-z z-drift sums**
  (4×4 subsampled), **dering ring profiles** (locked per-slab merge).
- heavy stats (capped at HEAVY_MAX=256 central 128³ sub-blocks, ≥60% occupancy):
  **flat-noise + radius pairs**, **PSF sigma** (`measure_tile_psf_sigma`: ESF/LSF
  2nd-moment from subpixel-aligned air↔papyrus edges), **PSF anisotropy**
  (`fy_noise_aniso`), **scratch-denoised air histogram** (box smooth → valley).
- up to KEEP_MAX=32 sub-block copies retained for the post-sweep deconv-output
  range measurement.

Then commits: PSF median + **AUTO-DECONV gate** (matched deconv stored only when
psf_med ≤ 1.1 = sharp enough to recover real signal; else view-time only), halo
(from `fy_kernel_halo` if deconv on else 8), deconv global rescale range (seam-
safe), **radial guided-eps fit** `fn(r) = a + b·r` (flat-noise rises toward rim on
large scrolls; gated on slope/correlation/span), global `guided_eps`, noise floor,
PSF anisotropy ratio, **air-cut level** (physics window-floor → histogram valley by
`air_cut_aggr` dial), **dering gate** (≥2 ring radii), **norm gate** (span<0.40),
**z-drift gate** (coherence≥0.5 AND slope_frac≥0.05 AND metadata beam_drift≥0.05).

### 4.3 PASS 2 — parallel per-tile chain (`process_tile`)
Per input tile, halo-padded, in [0,1] float:
- (a) intensity corrections FIRST: dering_apply, zdrift_apply.
- (b) **deconv** (off by default for BM18; matched Wiener if gated on, else plain
  auto-reg). De-windows u8/255→physical µ before the inverse, re-windows after,
  optional global dec-range rescale, clamp [0,1].
- (c) **guided denoise** (fast subsampled s=2 by default; per-tile radial eps if
  the fn(r) fit is active). NOTE: denoise is OFF by default (`denoise_k<0`) — the
  white-noise floor is ~1 u8, nothing to remove.
- (d) **air-zero**: box-smooth scratch (on a 2× decimated copy) → histogram valley
  → texture-split threshold (uniform regions cut harder toward valley, textured
  sheet edges protected toward physics floor) → zero; optional connected-component
  despeckle of small isolated material islands (`air_cc_despeckle`, 6-connected BFS,
  boundary-touching components kept).
- (d.5) **recenter+stretch** with MUSICA headroom (global → seam-free).
- (e) optional **MUSICA** (per-slice, clip-aware via u8 rails).
- (f) final full-range stretch left to a global post-pass.
Output u8 quantization is **dithered by default** (hash of global voxel coord →
unbiased, seam-free, reproducible; kills banding in flat papyrus).
Then write the inner tile directly (tile == output chunk grid). Failed reads/
writes are counted and reported loudly — never silent fill.

### 4.4 The ESRF nabu chain and what fysics does about each operator
(from `docs/esrf-pipeline.md`, the per-operator invertibility audit; verified
against nabu 2026.1.0-alpha1 source.) nabu order: flat/dark → double-flat → intensity
norm → **Paganin phase retrieval (+unsharp)** → ring removal (sinogram) → distortion
→ **GHBP/FBP + ramp** → histogram rescale to u8.

| operator | domain | invertible from u8? | fysics |
|---|---|---|---|
| flat/dark, double-flat | projection | no | nothing |
| intensity norm / beam drift | projection (residual: volume) | residual only | z-drift correction, seeded from `machineCurrentStart/Stop` |
| ring/stripe removal | **sinogram** (NOT applied on these scans) | no | `fy_dering_*` detect-then-subtract (measured complement, not physics) |
| **Paganin (+unsharp)** | projection (commutes w/ FBP) | **yes, linear low-pass** | `fy_deconvolve` / `_matched` (Wiener inverse, partial δ/β on fine vols) |
| distortion/alignment, GHBP/FBP ramp | sinogram | no | nothing (does NOT invert ramp; tile seams = known gap) |
| **u8 export window** | post-recon, per-volume | **yes, exact linear** | `fy_u8_to_phys`/`fy_phys_to_u8` |

The single license to operate on the reconstructed volume at all is that
Paganin/TIE-Hom is linear+low-pass and **commutes with FBP**. Everything fysics
inverts is metadata-parameterized and post-recon-commuting; everything else is a
gated complement or left alone.

### 4.5 mca_export (`tools/mca_export.c`) — fused streaming export to `.mca`
- One tool: `mca_export <zarr|s3://...> <out.mc>` streams an uncompressed OME-zarr
  (local or S3) through `fy_calibrate` + the preprocessing chain into a
  **matter-compressor archive** (`.mc`, the external `matter_compressor` codec —
  zstd-based, 256³ chunks `MCC`) with all 8 LODs, bounded RAM, no intermediate zarr.
- Architecture ported from volume-compressor's `vc_export_stream`: occupancy from a
  coarse pyramid level (one tiny GET replaces ~10k HEADs, absent bands skipped);
  **downloader pool → bounded queue → compute pool** so S3 latency never blocks
  compute; work unit = (XY tile SB×SB, Z band) chunk-aligned + independent; each
  unit assembles a halo'd raw band → `fy_process_buffer` per 128³ tile → append L0
  256³ chunks → 2× box downscale → append L1 (lock-free appends); coarse tail
  (L2+) built from the archive itself (`mc_archive_read_region`), ~1.6% of data.
- Production-proven: 27 TB masked PHerc Paris4 2.4µm → 320 GB 8-LOD `.mca`,
  ~15 Gbit/s S3, resume journal, single accounted RAM budget, malloc_trim. The
  postmortem invariants (RAM accounting, fetch-gate progress token, pinning
  invariant, small bands beat big) are recorded in esrf-pipeline.md §postmortem.
- `tools/`: also `fysics_process_main.c` (the `fysics-process` CLI; reads
  `metadata.json` physics via a tiny `read_meta.py` popen stub, fills BM18 defaults,
  runs zarr→zarr), `fy_stats.c`, `fy_restretch.c` (the global post-pass stretch),
  `fy_noise.c`.

---

## 5. Public API surface + key data structures

Header `fysics.h` (522 lines, `extern "C"`). Groups:
- physics: `fy_physics` struct; transfer fns; `fy_auto_reg`/`_deltabeta_scale`;
  `fy_deconvolve`/`_matched`; `fy_kernel_halo`.
- FFT: `fy_fft1d/3d`, size helpers.
- streaming/global: `fy_hist_state`, u8↔float↔phys, histogram/percentile/norm,
  `fy_valley_depth`, `fy_flat_noise`.
- denoise: `fy_guided_denoise[_ws/_fast_ws]`, `fy_box_smooth`,
  `fy_guided_eps_for_noise`.
- contrast: `fy_musica2d`.
- noise: `fy_noise_model`, `fy_estimate_noise`, `fy_noise_aniso`.
- z-drift: `fy_zdrift_*`, `fy_correct_zdrift`.
- registration: `fy_sample_trilinear`, `fy_warp_affine/_field`, `fy_downsample2x`,
  `fy_ncc_warped`, `fy_register_affine/_demons/_full`, `fy_phase_correlate`,
  `fy_mutual_information`.
- zarr: `fy_zarr`, `fy_zarr_open/create/read/write_chunk`.
- downscale: `fy_ds_method`, `fy_downscale2x`.
- dering: `fy_dering` struct + `fy_dering_*`.
- pipeline: `fy_pipeline_cfg`, `fy_calibrate`, `fy_process_chunk/_buffer`,
  `fy_run_pipeline`.

Style notes: C error convention (`int` return, 0 = success, nonzero = failure);
out-params; caller-owned workspace buffers for hot loops; thread-local caches
(twiddle tables, per-tile scratch); generous in-header documentation (the header
is the primary doc).

---

## 6. What a C++26 rewrite should redesign

**Keep the math verbatim** — it is empirically tuned (90+ cubes, 18 volumes,
production-validated) and the numeric constants/gates are load-bearing. Port the
algorithms 1:1 and keep the tests as a golden-output harness (the C is tested
against naive DFT, FFT round-trip, the exact nabu formula, and a Python reference;
preserve that). The rewrite is an *ergonomics/safety/architecture* exercise, not an
algorithm change.

Concrete recommendations:

1. **Memory & buffers.** Replace the pervasive raw `malloc`/`free` + caller-owned
   `float *ws` workspaces and `__thread` scratch with RAII: `std::vector`/
   `std::unique_ptr`, a small arena/scratch allocator type, and `std::mdspan`
   (C++23, available in 26) for the (z,y,x) volume views — this kills the manual
   stride arithmetic (`(z*ny+y)*nx+x`) and the half-spectrum index juggling that
   is the bug-prone heart of `paganin.c`/`fft.c`. The 8·hn TLS buffer carving in
   `fy_process_buffer` becomes a typed scratch struct.

2. **Error handling.** Replace the `int` return + `goto done` + "return 16 on
   alloc failure" patterns with `std::expected<T,Error>` (C++23). The S3
   hard-error-vs-404 distinction (critical correctness invariant) should be a typed
   result, not an out-param int.

3. **Config.** `fy_pipeline_cfg` is a ~50-field mutable struct mixing inputs,
   tuning, and resolved state. Split into immutable `Physics`, `PipelineOptions`,
   and a separate `Calibration` result object (`fy_calibrate` returns it instead of
   mutating cfg). Use designated initializers / a builder; the current
   `memset(0)+field assigns` is error-prone.

4. **Threading.** OpenMP `#pragma omp parallel for` + `#pragma omp critical/atomic`
   over tiles can become `std::execution` parallel algorithms or a `std::jthread`
   pool with `std::atomic`/`std::barrier`; the mca_export pthread downloader/queue/
   compute pools map cleanly to a typed bounded `concurrent_queue` + jthreads +
   `std::stop_token`. Keep the kernels single-threaded and reentrant.

5. **Type-safe units & geometry.** delta_beta/energy/distance/pixel and the
   cycles-per-voxel↔cycles-per-µm conversions (with the mm-vs-µm auto-correct hack)
   are a strong-typedef opportunity (`Micron`, `KeV`, `CyclesPerVoxel`). The (z,y,x)
   axis order, the XYZ-vs-ZYX transform.json reconciliation, and the 3×4 affine
   convention are exactly the things that caused the real registration bug — encode
   them in types, not comments.

6. **FFT.** Keep the self-contained FFT (the radix-3/half-spectrum specialization
   is deliberate and dependency-freedom is a goal), but it can become a templated
   `fft<float>` with `std::complex` or an explicit SoA span type; the thread-local
   twiddle cache becomes a per-thread `static thread_local` table object. Consider
   whether to keep `-ffast-math` (the explicit validity-mask workaround for broken
   `isnan` is a smell — C++ `std::isnan` under fast-math has the same issue, so the
   mask discipline must be preserved consciously).

7. **I/O abstraction.** `zarr_io.c` mixes local mmap, S3 batch, and the
   `#ifdef FYSICS_S3` split. A `ChunkSource` interface (LocalMmap / S3Batch
   implementations) with a unified `read_region` would clean this up; keep the
   mmap-halo-sliver optimization and the batch GET.

8. **Keep dependency posture.** Core stays dependency-free (no FFTW). S3
   (libcurl/libs3) and matter_compressor stay optional/external, ideally behind
   interfaces so the core library is testable without them.

9. **Don't re-add the empirically ruled-out stages** (README "Ruled OUT": noise
   whitening, TV denoise, BM4D, aggressive deconv on fine vols, radial cupping,
   helical z-banding, repeat-scan averaging, saturation inpainting). A rewrite is
   the moment someone "helpfully" adds these back; the rule-outs are measured.

10. **Registration is a separate concern.** It is not in the production pipeline and
    has a documented real-data caveat (intensity Demons overfits laminar papyrus).
    Treat it as its own module/library in the rewrite; the production path is
    single-volume preprocess→export.
