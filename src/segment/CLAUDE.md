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

**RAM/speed (1024³ CT-sheetness trace: 47 GB OOM → 4 GB / ~20 s):**
- The whole-volume `structure_tensor` materializes 6 tensor-component volumes **plus** a per-voxel
  normal field at once — ~10× the input (~47 GB at 1024³ f32 → **OOM**). Use it ONLY on small/
  downsampled inputs that need the normal field (e.g. `compute_normal_field` on the `ds`-down cube).
- For the data-term path use **`structure_tensor_sheetness<T,Out>()`** — tiled (256³ + halo),
  sheetness scalar only, never the normal field; templated on in/out dtype (resident volumes **u8**,
  not f32); the Paganin/unsharp deconv is folded in per-tile (no full-volume blur copy). Peak RAM
  O(tile³). The tile loop is **serial with parallel work inside each tile**: parallelizing the tile
  loop nests `gaussian_blur`'s `parallel_for` and oversubscribes the cores (~10× slower). **Never
  nest OpenMP parallel regions** — parallelize one level.
- The CT term is a coarse fallback → compute it downsampled: **`ct_sheetness_coarse<T>(ct,cut,sig,
  ds)`** mean-downsamples by `ds`, runs sheetness on the ~ds³-smaller volume, gates by the air-cut,
  and returns a **ds-resolution** u8 (it also frees the full-res CT mid-call). The grower samples it
  via `DataField::ct_ds` / `GrowParams::ct_ds` — no full-res upsample, so the resident term is ds³
  smaller too. ds=2 → ~8× less structure-tensor work. `ct_sheetness_term` is the full-res-upsample
  variant for consumers that can't sample a coarse grid.
- Load big NRRDs with **`io::read_nrrd_u8`/`nrrd_max`** (streaming u8) so the full f32 is never
  resident. `core::gaussian_blur` is parallel (over lines) with a branch-free vectorizable interior.
- General rule: **any whole-volume multi-buffer pass (structure tensor, Hessian, OOF) is an OOM bug
  at scale — tile it with a halo; keep coarse terms coarse.**

## Gotchas / pitfalls
The lever is the **affinity/sheetness field quality**, not the clustering algorithm (the
connectomics/SOTA lesson). Prevent wrap-merges at detection (signed/repulsive) rather than
repairing post-hoc. Don't duplicate eigensolver/blur/trilinear — use `core`/`geom`.

## Status & TODO
STUB. Open ADRs: tracer growth/accept-rollback policy; detector fusion; MWS-vs-fit role.
