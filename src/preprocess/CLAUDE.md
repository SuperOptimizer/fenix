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
**Implemented + wired to CLI** (ported from `github.com/SuperOptimizer/fysics`): **`dering`**
(`dering.hpp` — detect-then-subtract residual rings; 2.5D = 2D radial geometry + z-slab pooling;
angular-sector sign-consistency vote so it never eats the angularly-drifting sheets; ring-energy
metric (`DeringStats`), loosenable vote (`vote_slack`/`vote_dissent`), `iters`/`until`
iterate-to-diminishing-returns), **`deconv`** (`deconv.hpp` wiener of a Gaussian PSF), **`denoise`**
(`guided.hpp` He-Sun-Tang guided filter) — CLI stages in `preproc_cli.hpp` (+ `dering` self-registers).
Order matters: **dering → deconv → denoise** (deconv amplifies rings ~2×; denoise after deconv).
Measured on 0125 (9.362µm): rings are only ~0.04–0.6% of the signal (structure-safe but a minor lever);
deconv +0.010 corr on the surface model, dering ~0, denoise slightly negative.
**Registration (2026-07-04, ink-hunt driven):** `register_scans.hpp` (`register-scans` —
cross-scan phase correlation of coarse pyramid levels on a common physical pitch -> VC
transform.json; prints `t_centroid` as the content-mismatch-robust alternative start;
KNOWN ISSUE: confidence degenerate, zflip=auto untrustworthy — force zflip and refine).
The refinement lives in tools/inkhunt/refine_transform.sh (coordinate descent on surf-qc
delta; PHercParis3 measured -3.7 -> +14.9).
TODO: a `dering` golden/synthetic test; the full Paganin (physics δ/β from metadata) deconv; 2-pass
streaming/out-of-core wrapper for whole-scroll; aircut/MUSICA/z-drift. Registration deferred.
