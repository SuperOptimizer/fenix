# preprocess — CLAUDE.md

## Purpose
CT preprocessing — the fysics lineage: invert known reconstruction operators to restore
contrast/sharpness, denoise, and export raw scans into `.fxvol`. **STUB now, port later**
(after the unrolling vertical slice works). See `docs/research/research-fysics.md`.

## Public API & key types (planned)
- **Paganin/Wiener deconvolution** (matched, partial δ/β on fine vols), **dering**
  (detect-then-subtract), **guided denoise** (He–Sun–Tang, fast subsampled), **MUSICA**
  contrast, **z-drift** correction, **2× downscale** (box/contrast-box), **noise
  estimation** (signal-dependent, per-volume), **own FFT** (radix-2/3, no FFTW).
- **Registration** as a separate sub-stage (landmark + textured-patch phase-correlation →
  RANSAC affine; NCC/MI; **avoid intensity Demons on laminar papyrus** — documented
  failure). **Deferred.**
- The 2-pass streaming pipeline: calibrate (PASS1) → per-tile chain (PASS2) → export
  `.fxvol` (occupancy-guided, bounded RAM).

## Inputs / outputs & formats
In: OME-zarr raw scans (via `io`). Out: preprocessed `.fxvol`.

## Dependencies
Intra: `core`, `io`, `codec`. Third-party: none (own FFT). libcurl/blosc2 via `io`.

## Invariants & numerics
fast-math throughout (fysics is built for it; use validity masks not NaN). Keep the math
**verbatim** when ported — constants/gates are empirically tuned (90+ cubes, 18 volumes).
Only invert post-recon-commuting operators (Paganin + u8 window); everything else is a
gated complement or left alone (see esrf-pipeline audit in the report).

## Performance notes
Guided denoise box filter is the hottest kernel; halo'd tiles, per-thread scratch. The
export pipeline = download pool → bounded queue → compute pool.

## Gotchas / pitfalls
Don't re-add the empirically ruled-out stages (TV/BM4D denoise, aggressive deconv on fine
vols, etc. — the README rule-outs). Registration Demons overfits laminar papyrus — don't.

## Status & TODO
STUB (interfaces only this round). Port kernels after the unrolling slice. Registration
deferred.
