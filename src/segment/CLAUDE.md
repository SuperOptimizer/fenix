# segment — CLAUDE.md

## Purpose
The classical sheet-detection front end + the surface tracer that produce data terms and
**Patch** constraints for the unified winding fit. See `docs/research/research-core.md`
(segmentation), `villa-vc.md` (tracer), `segmentation-sota` notes in `research-build.md`.

## Public API & key types
- **Detectors** → per-voxel sheetness (scalar) + normal (vector), via a shared closed-form
  symmetric-3×3 eigensolver (`core`): **structure tensor**, **Hessian/Frangi** (plate),
  **OOF/Descoteaux** (resolves the ~1-voxel inter-wrap gap), **phase-symmetry**.
- **NLLS surface tracer** (VC-style patch growing): advancing/generational growth, affine
  extrapolation of new grid corners, data term = pull toward a thresholded surface-
  prediction field, normal-alignment + smoothness/dist regularizers; solved with our
  first-party Gauss-Newton/AdamW (**no Ceres**). Produces **Patch**es (surface grids).
- Optional signed-affinity graph + Mutex-Watershed/GASP partition (touching-wrap aware)
  as an alternative segmentation path feeding constraints.

## Inputs / outputs & formats
In: a volume and/or prediction fields (`predictions`), seeds/annotations (`annotate`).
Out: sheetness/normal dense fields, **Patch** sets (small `.fxsurf` grids), instance
labels (lossless `.fxvol`).

## Dependencies
Intra: `core`, `geom`, `io`, `codec`, `annotate`, `predictions`. Third-party: none.

## Invariants & numerics
Determinism of SNIC/MWS tie-breaks preserved (by index) where used. Detector σ scales are
**config-only Params (no baked defaults)** — tune per dataset (taberna noted σ_tensor≈1.5
over-smooths lone sheets). Normals are axis-tagged `Vec3` (no x/y/z-vs-z/y/x bug).

## Performance notes
Per-voxel, embarrassingly parallel (OpenMP); block + halo out-of-core. The eigensolver and
Gaussian blur are hot — use the shared `core` ones (one copy).

## Gotchas / pitfalls
The lever is the **affinity/sheetness field quality**, not the clustering algorithm (the
connectomics/SOTA lesson). Prevent wrap-merges at detection (signed/repulsive) rather than
repairing post-hoc. Don't duplicate eigensolver/blur/trilinear — use `core`/`geom`.

## Status & TODO
STUB. Open ADRs: tracer growth/accept-rollback policy; detector fusion; MWS-vs-fit role.
