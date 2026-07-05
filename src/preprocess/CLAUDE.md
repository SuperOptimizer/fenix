# preprocess — CLAUDE.md

## Purpose
CT preprocessing — the fysics lineage: invert known reconstruction operators to restore
contrast/sharpness, denoise, cut background, and (separately) cross-register scan pairs.
Ported from `github.com/SuperOptimizer/fysics`; the 2-pass streaming/out-of-core export
wrapper is still stubbed. See `docs/research/research-fysics.md`.

## Public API & key types
- **`fft.hpp`** — first-party FFT, no FFTW. `fft1d(span<cf32>, inverse)` in-place radix-2
  iterative Cooley-Tukey (power-of-two only, `FENIX_ASSERT`s it). `fft3d(span<cf32>, Extent3,
  inverse)` separable 3D (z then y then x passes). `next_pow2(n)`. Radix-3 / non-pow2 sizes
  not yet done — all FFT-based stages below require power-of-two dims.
- **`deconv.hpp`** — `unsharp_mask(VolumeView<f32>, sigma, amount)` cheap spatial (separable
  Gaussian via `core/filter.hpp`, no FFT, out-of-core-friendly). `gaussian_transfer(f, sigma)`
  frequency response. `apply_psf(VolumeView<const f32>, sigma)` forward Gaussian blur via FFT.
  `wiener_deconvolve(VolumeView<const f32>, sigma, reg=0.015)` — `H/(H²+reg)` Wiener inverse of
  a Gaussian PSF; the core invert behind fysics's Paganin/matched deconvolution (full
  TIE-Hom/nabu transfer function + per-volume regime gating not yet ported).
- **`dering.hpp`** — detect-then-subtract residual ring removal. `DeringParams` (center
  cy/cx default to slice center, `slab_z`, `ns` angular sectors, `hp_win`, `min_cnt`,
  `min_amp`/`max_amp` value-unit gates, `ss` subsample stride, `vote_slack`/`vote_dissent`).
  `dering_inplace<T>(VolumeView<T>, DeringParams)` → `DeringStats{n_rings, ring_energy,
  signal_var, ring_frac, mean_amp}` (intrinsic ring-energy-removed metric, no ground truth
  needed). 2.5D: 2D radial/angular geometry pooled over z-slabs; angular-sector
  sign-consistency vote so a true ring (same radius at every angle) is distinguished from the
  angularly-drifting papyrus winding (fails the vote). Self-registers the `dering` CLI stage
  (`run_dering`, in this header, not `preproc_cli.hpp`) with `iters`/`until`
  iterate-to-diminishing-returns.
- **`guided.hpp`** — `box_mean(VolumeView<const f32>, r)` separable reflect-boundary box
  filter. `guided_filter(VolumeView<const f32>, r, eps)` He-Sun-Tang self-guided denoise
  (`a = var/(var+eps)`, box-averaged); `eps ~ noise_std²`.
- **`musica.hpp`** — TRUE 3D MUSICA multiscale contrast (Vuylsteke & Schoeters). Detail
  namespace: `blur_axis`/`blur5_3d` (separable 5-tap binomial 1-4-6-4-1), `musica_gain(x, a,
  p, core)` sublinear gain with soft noise coring. Public: `musica_inplace(VolumeView<f32>,
  levels=4, p=0.7, core=0, vmax=255)` — mask-aware normalized-convolution pyramid, z-consistent
  (not per-slice, unlike the upstream fysics version); masked (==0) and clipped (==vmax) voxels
  pass through unmodified.
- **`aircut.hpp`** — `otsu_threshold<T>(VolumeView<const T>, lo, hi, nbins=256, stride=4)` and
  `otsu_threshold_u8(span<const u8>, stride=2)` (span variant for raw u8 patch buffers, e.g.
  the ML train-feed labeling path). `air_cut(VolumeView<T>, lo, hi)` zeros below the Otsu
  valley in place, returns the threshold. Papyrus CT is weakly bimodal (low background vs
  papyrus); real data isn't perfectly bimodal so the cut zeros some genuine low-value material
  near the threshold.
- **`phasecorr.hpp`** — `phase_correlate(VolumeView<const f32> a, b, f32* confidence=nullptr)`
  → `Vec3f` integer ZYX shift s.t. `b(x) ≈ a(x-s)`. Cross-power spectrum `conj(FA)·FB/|·|` →
  inverse FFT, peak = shift. Confidence normalized by live-bin fraction (`bestv * n / live`,
  fixed 2026-07-05 — sparse spectra were previously diluting the score). Integer-peak only;
  sub-voxel (Foroosh) not done.
- **`register_scans.hpp`** — `fenix register-scans <a> <b> <out_transform.json> um_a=<v>
  um_b=<v> [grid=128] [zflip=auto|0|1] [mask=0|1]`. Cross-scan registration for the ink-hunt:
  resamples both volumes' coarse pyramid levels onto a common physical pitch (trilinear,
  `detail::resample_cube`), phase-correlates (grid rounded up to pow2), auto-picks zflip by
  confidence when `zflip=auto`. `mask=1` binarizes both cubes via per-cube Otsu before
  correlating — needed when one scan is masked and the other isn't (raw correlation locks
  onto case-ring/exterior structure only one scan has). Prints `t_centroid` — a
  content-mismatch-robust material-centroid alternative start for the refine step. Emits a
  VC-compatible `transform.json` (uniform scale `um_a/um_b` + translation [+ z-mirror]).
  Inputs accept `.fxvol` or `cache@url` (`io::CachedVolume`) via `detail::RegVolume`.
  Self-registers directly (anonymous-namespace `register_stage` call, not the
  `FENIX_REGISTER_STAGE` macro used elsewhere in this dir).
- **`preproc_cli.hpp`** — shared CLI plumbing (`cli::load`/`cli::write`/`cli::opt`/`cli::parse`)
  + three stages: `deconv` (`run_deconv`; requires power-of-two dims), `denoise` (`run_denoise`,
  guided filter), `aircut` (`run_aircut`; optional manual `thr=` overriding Otsu when data isn't
  bimodal), `musica` (`run_musica`). `dering` registers itself from `dering.hpp` instead.
- **`preprocess.hpp`** — umbrella header pulling in all of the above; `run()` is still
  `stage_unimplemented("preprocess")` — there is no single orchestrated `preprocess` stage,
  only the standalone CLI subcommands above. `preprocess.cpp` is the split-build TU wrapper
  (unity build never compiles it).

## Inputs / outputs & formats
In: `.fxvol` (via `cli::load`/dering's own `load`); `register-scans` additionally
accepts `cache@url` for out-of-core gather. Out: **always `.fxvol`**, u8, round-clamped from
f32 (`static_cast<u8>(clamp(v,0,255)+0.5f)`) — dense buffers are computed in f32 but **never
written as f32/NRRD**; the archive encoder consumes the u8 source directly (see `no-f32-widening-u8`
memory note — never widen u8 CT for storage). `register-scans` writes a VC-style
`transform.json` (4x4 affine, scale + translation [+ zflip]), not a volume.

## Dependencies
Intra: `core` (Volume/VolumeView, parallel_for_z, Context, logging), `core/filter.hpp`
(Gaussian blur), `io` (`VolumeArchive`, `CachedVolume`), `codec` (`VolumeArchive`, `DctParams`).
Third-party: none (own FFT). libcurl/blosc2 transitively via `io`.

## Invariants & numerics
fast-math throughout; use validity masks (value <= 0 or == vmax), not NaN, for
masked/clipped voxels. Keep ported math **verbatim** — constants/gates are empirically
tuned (90+ cubes, 18 volumes from fysics). Only invert post-recon-commuting operators
(Paganin-style deconv); everything else is a gated complement or left alone (see
esrf-pipeline audit in the research report). All FFT-backed stages (`deconv`, `phasecorr`,
`register-scans`'s correlation grid) require power-of-two dims — callers pad/round up
(`register-scans` rounds `grid` up itself; `deconv`'s CLI stage hard-errors if dims aren't
pow2). Output is always u8 — `f32` compute buffers are round-clamped to `[0,255]` at write
time, never persisted.

## Performance notes
Guided-filter box mean and MUSICA's binomial blur are the hottest kernels — both separable,
parallelized over z (`parallel_for_z`). MUSICA holds ~8 full-volume f32 buffers (crop-scale
only for now; apron-tiling for out-of-core deferred). `dering`'s pass 1/2 are O(voxels) with
a small `(slab × sector × radius)` accumulator, subsample stride `ss` trades accuracy for
speed. FFT stages materialize a full `cf32` copy of the volume (2x memory of the f32 source).
The whole-scroll 2-pass streaming pipeline (calibrate → per-tile chain → bounded-RAM `.fxvol`
export) described in earlier versions of this doc is **not implemented** — everything here
operates on an in-RAM volume.

## Gotchas / pitfalls
Don't re-add the empirically ruled-out stages (TV/BM4D denoise, aggressive deconv on fine
vols — the fysics README rule-outs). Registration via intensity Demons overfits laminar
papyrus — don't; `register-scans` uses phase correlation + optional mask-Otsu instead.
Order matters when chaining CLI stages: **dering → deconv → denoise** (deconv amplifies
rings ~2x; denoise should run after deconv). `register-scans`'s `zflip=auto` confidence
was measured untrustworthy before the 2026-07-05 live-bin-fraction fix — force zflip
explicitly if the auto pick looks wrong, then refine (see `tools/inkhunt/refine_transform.sh`,
coordinate descent on surf-qc delta; PHercParis3 measured -3.7 -> +14.9). `mask=1` is
required, not optional, when correlating a masked scan against an unmasked one.

## Status & TODO
**Implemented + wired to CLI:** `dering`, `deconv`, `denoise`, `aircut`, `musica`,
`register-scans` — all six are real, tested-on-real-data stages (see measurements below).
Measured on 0125 (9.362µm): rings are only ~0.04-0.6% of the signal (structure-safe but a
minor lever); deconv +0.010 corr on the surface model, dering ~0, denoise slightly negative.
`register-scans` (2026-07-04, ink-hunt driven) has a known-fixed confidence bug (2026-07-05:
was diluted by dead spectrum bins on sparse content, now normalized by live-bin fraction).

**Not implemented (still stubbed):** the orchestrated `preprocess` stage/`run()` (currently
`stage_unimplemented`) — only standalone CLI subcommands exist; the full Paganin deconv
(physics δ/β from metadata, TIE-Hom transfer function, matched nabu operator, per-volume
regime gating — `wiener_deconvolve` is the generic core, not the matched operator); z-drift
correction; noise estimation (signal-dependent, per-volume); 2-pass streaming/out-of-core
export wrapper for whole-scroll volumes; radix-3/non-pow2 FFT; sub-voxel phase-correlation
refinement (Foroosh); a `dering` golden/synthetic test.
