# predictions — CLAUDE.md

## Purpose
Ingest ML- or classically-generated prediction fields (sheet-probability, distance,
winding-density, etc.) and normalize them into a uniform scalar convention before they
reach `segment`/`winding` as data terms. The intended seam between learned/classical
signal and the geometry pipeline — currently just the normalization primitives, not the
seam itself.

## Public API & key types
- `field.hpp`: free functions only, no field/type wrappers yet.
  - `enum class Norm { None, MinMax, Percentile }`
  - `Volume<f32> normalize(VolumeView<const f32> field, Norm scheme, f32 plo = 0.01f, f32 phi = 0.99f)`
    — maps a scalar field into `[0,1]`. `Percentile` sorts a copy of the field and clips
    to `[p1,p99]` (robust to outliers) before the linear map; `MinMax` uses true
    min/max; `None` copies through unchanged.
  - `Volume<u8> threshold(VolumeView<const f32> field, f32 t)` — binarizes a (normalized)
    probability field at `t`.
- `predictions.hpp`: stage entry point `Expected<int> run(std::span<const
  std::string_view>, Context&)`, currently `return stage_unimplemented("predictions")`.
  Self-registers via `FENIX_REGISTER_STAGE(predictions, "predictions stage (stub)",
  ::fenix::predictions::run)`.
- `predictions.cpp`: split-build TU only (`#include "predictions/predictions.hpp"`);
  not compiled in the default unity build.

Not yet present despite being planned: typed field kinds (sheet-probability, normals,
distance-to-surface, winding-density/grad_mag, direction/double-angle), axis-tagged
normals `Vec3`, validity tracking, skeletonize-to-`Track` conditioning, and any `.fxvol`
I/O — none of that is implemented here yet.

## Inputs / outputs & formats
Today: in-memory `VolumeView<const f32>` in, `Volume<f32>`/`Volume<u8>` out — no file
I/O in this module yet. Planned: prediction `.fxvol`s (from `ml` or external/classical
sources) and surfaces in; conditioned dense fields + `Track` sets out to
`segment`/`winding`.

## Dependencies
Intra: `core` only (`Volume`, `VolumeView`, `Extent3`, `Expected`/`Context` types).
No `io`, `codec`, or `geom` includes yet, despite being anticipated for track
extraction/skeletonize and `.fxvol` I/O. Third-party: none — no torch here (ML runs in
`ml`; predictions only ever consumes its outputs as fields).

## Invariants & numerics
`normalize`/`threshold` operate in f32 per project convention. `Percentile` mode copies
and sorts the full field (`O(n log n)`, extra `O(n)` buffer) — fine for now, will need an
out-of-core/streaming quantile approach once fields are volume-sized and read from disk.
No validity/NaN handling yet — callers must ensure the input field has no invalid voxels
baked in as sentinel values.

## Performance notes
Both functions are single dense passes plus (for `Percentile`) one full sort; currently
in-core only, no block/streaming behavior. Out-of-core per-block conditioning is planned
but not implemented.

## Gotchas / pitfalls
- This module **consumes** predictions; it does not run inference (that's `ml`).
- The doc previously described normals/direction fields, `Track` skeletonization, and
  `.fxvol` ingest as if implemented — as of this writing none of that exists in code;
  only the two normalization helpers above are real.
- Split-build TU (`predictions.cpp`) has nothing to add beyond the include — keep it
  that way; put all logic in the headers.

## Status & TODO
STUB. Implemented: `normalize` (MinMax/Percentile/None) and `threshold` on in-memory
scalar fields. Not implemented: stage `run` (returns `stage_unimplemented`), typed
prediction-field kinds, `.fxvol` ingest, `Track` extraction/skeletonize via `geom`,
validity/axis-tagging. Open ADRs: prediction-field schema; track-extraction thresholds.
