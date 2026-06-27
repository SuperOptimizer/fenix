# predictions — CLAUDE.md

## Purpose
Ingest and normalize ML- or classically-generated prediction fields and present them as
uniform **data terms** to `segment`/`winding`. The seam between learned/classical signal
and the geometry pipeline.

## Public API & key types
- Typed prediction fields: **sheet-probability**, **normals** (3 scalar archives nx/ny/nz),
  **distance-to-surface**, **winding-density (grad_mag)**, **direction** (double-angle).
- Normalization/conditioning to the conventions the fit expects (scale, validity, axis
  tagging); skeletonize predicted surfaces into **Track**s (via `geom`).

## Inputs / outputs & formats
In: prediction `.fxvol`s (from `ml` or external/classical), surfaces. Out: conditioned
dense fields + **Track** sets handed to `segment`/`winding`.

## Dependencies
Intra: `core`, `io`, `codec`, `geom` (skeletonize). Third-party: none (no torch here — ML
runs in `ml`; predictions just consumes its outputs as fields).

## Invariants & numerics
Predictions are stored via the lossy wavelet (accepted drift). Normals are axis-tagged
`Vec3`. Validity explicit.

## Performance notes
Mostly streaming field reads + light conditioning; out-of-core per block.

## Gotchas / pitfalls
This module **consumes** predictions; it does not run inference (that's `ml`). Either
ML-generated or classical (`segment` detector) fields enter here uniformly — the fit
shouldn't care which.

## Status & TODO
STUB. Open ADRs: prediction-field schema; track-extraction thresholds.
