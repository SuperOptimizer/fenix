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
- **Multi-scale patch graph** (`patch_graph.hpp`): grow MANY seeds (`trace_volume`), then
  relate the patches as wraps of the spiral. `Patch` = a traced surface's valid cells with
  outward-oriented normals + per-cell confidence (persisted by the grower as `Surface`
  `normal`/`conf` channels). `build_patch_graph` computes, for each near pair: closest
  approach, **co-normality** `n_a·n_b`, **signed normal gap** `(b−a)·n_a`, and a
  **co-deformation residual** (local-frame, curvature-invariant); estimates the **wrap
  spacing** (median nearest-outward gap); and classifies each edge MERGE (same sheet,
  Δ=0) / LINK (adjacent wrap, Δ=±1) / CONFLICT. `merge_same_sheet` (union-find) collapses
  same-sheet patches; `assign_windings` gives every patch an integer winding via a
  **potential-DSU over ℤ** (highest-certainty edges first; a contradicting edge = a cycle
  conflict) — the discrete, exact form of thaumato's winding-angle relaxation.
  `analyze_patches` runs all three. The dense continuous view + the field-guided fill that
  repairs weak-prediction gaps live in `winding/patch_field.hpp` + `winding/cosegment.hpp`.

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

**Shape-targeted growth:** `GrowParams::uv_mask` (grid×grid, `id=v*grid+u`, 1=allowed) constrains the
BFS frontier to a target shape in the **(u,v) flattened domain** (the grid IS the pre-flatten
parameterization). Growth fills `mask ∩ {where the sheet exists}` and the result is clipped to the
mask. Mask is in grid cells (~`step` voxels each), anchored at the seed (grid centre). Useful for
fixed-size patch tiling, growing toward a seam, or matching an existing segmentation footprint.
Empty mask = unconstrained. (Demo: a Texas-shaped patch filled 87% of the mask, single component.)

## Gotchas / pitfalls
The lever is the **affinity/sheetness field quality**, not the clustering algorithm (the
connectomics/SOTA lesson). Prevent wrap-merges at detection (signed/repulsive) rather than
repairing post-hoc. Don't duplicate eigensolver/blur/trilinear — use `core`/`geom`.

**Tracer `surf_thresh` is a DIVISOR, and it competes with the CT term**: `value = max(pred/
surf_thresh, ct/ct_thresh)`. A *higher* surf_thresh DOWN-weights the prediction, so the coarse
(permissive) CT sheetness term takes over and growth sprawls/self-intersects. Measured on Paris 4
(seed 512³): 0.15→0.10 dropped selfX 0.078→0.029 and fragments 12→4 while keeping the surface on
*stronger* prediction. But going too low (≤0.06) makes seed establishment fragile (the snap peak
shifts off the 3×3 seed patch → some seeds yield valid=0). **~0.10 is the robust sweet spot** for
normalized 0..1 predictions; do NOT threshold ~0.

**"Rivers" (thin invalid channels) are real prediction dropouts along cracks**, not a threshold
artifact — they persist at every threshold. The shipped fix is the post-growth river-fill
(morphological closing + stretch filter). The *principled* alternative — `GrowParams::soft_gate`
(decouple geometry from data: carry weak-field cells by confidence-blended extrapolation + small
`max_bridge` budget + frequent `fit_every` ARAP, occupancy guard as the anti-wrap safety) — is
implemented but **experimental/off**: it does kill self-intersection (0.048→0.018, the old greedy-
bridge failure is gone) but currently FRAGMENTS (cmp 5→27) and distorts (sDir 0.151→0.204) more
than river-fill. Needs stronger ARAP governance + small-component pruning before it's a net win.

**Multi-scale patch graph — real-data notes** (`test_multiscale` on a 512³ prediction crop: 24
seeds → 16 clusters → coherent winding gradient, +1% valid from field fill):
- **Normal orientation is load-bearing.** Default orients normals by the umbilicus radial (correct
  for a full scroll). For a **crop**, the umbilicus is far outside and the radial flips across the
  crop, scrambling the signed gaps → set `PatchGraphParams::orient_global` (orient to the PCA
  dominant normal axis; the crop's sheets stack along one axis). Diagnosed via the per-patch mean
  normal (some patches flipped +y vs −y) — always sanity-check orientation before trusting gaps.
- **Wrap spacing can be small.** crop512 is densely wound (~3.4 vox/wrap); the gap histogram peaks at
  2–4 with clean 2×/3× tails at 8/11. The spacing estimate = median of per-patch nearest-outward
  co-normal gaps (NOT the median of all gaps, which the wrap-2/3 pairs bias upward).
- **Winding assignment must be SOFT.** Variable real spacing makes per-edge Δwrap noisy; a hard
  integer gauge flags every imperfect cycle. `assign_windings` solves the continuous least-squares
  winding (weighted Jacobi + robust reweight → snap), which recovers the global stack even when
  ~half the local links are individually ambiguous. `winding_conflicts` counts post-snap violations
  — it is **pessimistic** (counts low-certainty ambiguous edges); the coarse gradient can be correct
  while it's high. TODO: weight it by certainty; add the dense winding-density (lasagna) term for
  sub-spacing precision.

## Status & TODO
STUB core; the **multi-scale patch graph (`patch_graph.hpp`) is implemented + tested**
(`test_patch_graph`, `test_patch_field`, `test_cosegment`, `test_multiscale`). Open ADRs: tracer
growth/accept-rollback policy; detector fusion; MWS-vs-fit role.
TODO: make `soft_gate` net-positive (ARAP governance of bridged cells, component pruning); pick a
per-dataset `surf_thresh` default (≈0.10 for normalized predictions); stitch merged clusters into
single charts; spacing from CT autocorrelation; feed assigned windings into the `winding` fit.
