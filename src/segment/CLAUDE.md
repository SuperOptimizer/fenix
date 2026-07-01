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
- **CT-valley Δwrap (`ct_valley.hpp`) — the touch-proof winding fix.** The ML prediction FUSES adjacent
  wraps wherever they touch (common), collapsing the geometric gap so `round(gap/spacing)` mislabels
  adjacent wraps as `Δ=0` and a single fused "weld" transitively over-merges two whole wraps (paris4:
  0..3 windings of ~20). But the raw CT keeps the two wraps as distinct density PEAKS. So when a CT view
  is passed, `build_patch_graph<T>(sheets, umb, ct, gp)` sets `dwrap` = number of CT inter-wrap SADDLES
  crossed between two patches (`count_air_valleys`, by PROMINENCE not depth → catches shallow touch-gaps;
  spacing-free). `merge_same_sheet(g, min_support)` then adds a **consensus gate**: a real same-sheet seam
  sits in a dense mesh (common merge-neighbours) while a fused weld is a sparse bridge — demote
  unsupported bridges to Conflict so one weld can't collapse two wraps. The no-CT overload keeps the
  geometric path (synthetic tests bit-stable). With the tracer's `GrowParams::ct_barrier` (stop growth
  crossing a CT saddle), this lifts paris4 to ~0..13. The residual gap to ~20 is for the `winding` fit.

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

**Tracer speed — measured levers (paris4 512³ crop, pred+CT, 24 sheets):**
- The trace is **memory-working-set bound**, not compute bound. Both data volumes are already **u8**
  (`read_nrrd_u8` quantizes on load — confirmed near-lossless: the prediction needs its full u8 range
  for sub-voxel ridge snap, but quantizing pred below u8 wrecks it — **u4 ≈ 3× the folds**; the CT
  fallback tolerates ~u3, but ≤u2 breaks the Otsu air-cut, not the ridge). The two 128 MiB volumes
  (256 MiB) dwarf the ~16 MiB L3, so the snap line-scan's scattered trilinear gathers miss to DRAM.
- **`DataField::ct_skip`** (default 1.5) — per-LINE CT short-circuit in `snap_to_sheet`: scan the
  prediction alone; only if its ridge is weak (a possible crack) does the snap bring in the CT volume.
  Drops the entire second-volume gather on confident on-sheet snaps. **~6%**, quality-neutral. (The
  per-SAMPLE version is useless — most line samples are off-ridge where pred is weak and CT is needed.)
- **`GrowParams::arap_tol`** (default 0.15) — adaptive final-ARAP stop: end the outer loop when the
  **MEAN** per-vertex move < `arap_tol*step` (max move never converges — a few boundary vertices
  oscillate between snap targets forever). Fixed-12 ARAP **over-folds**; stopping at diminishing returns
  is strictly better on paris4 (fold 0.128→0.097, selfX 0.018→0.005, smoother dihedral, more coverage,
  0 winding conflicts) — heavy data-coupled ARAP partly undoes the growth-time injectivity guard.
- **`trace_volume_tiled<T>(f, ct, p, max_sheets, min_valid, seed_stride, seed_thresh, tile_core, halo)`**
  — cache-blocking + the OOC on-ramp. Partition into `tile_core`³ tiles + `halo`; COPY each
  (tile+2·halo) pred/CT sub-region into **contiguous** blocks (the packing is the win — a strided
  `crop` view still scatters across DRAM); trace sheet-FRAGMENTS in the tile; clip to the core;
  translate to global. Cores tile the volume (disjoint); fragments meet at seams and are re-stitched by
  the patch graph. **tile_core=128 → trace 25% faster + better local quality** (the ~11 MB×2 tile fits
  L3); **tile_core=256 is SLOWER** (56 MB tile spills L3 — the proof it's a cache effect). Caveat:
  fragmentation multiplies the patch count (24 sheets → ~220 fragments); the global stitch is then
  re-coheres by the **band-Eulerian winding** (`winding/patch_field.hpp`), not the discrete merge.
- **The tile loop is now MULTITHREADED (`parallel_for_dynamic` over disjoint cores) — the growth was the
  single-threaded wall-clock cost.** Tiles are independent (each has its own OccMap/normal-field/fragments,
  clipped to its core), so parallelizing them is bit-exact: the merge fills a per-tile slot vector in tile
  order → fragment order/ids/cells are IDENTICAL to serial (`test_trace_parallel` asserts this both ways).
  Anti-nesting: each tile body wraps its kernels in `core::SerialRegion` (thread-local `g_parallel_serial`
  → `parallel_for` runs serial), so the per-tile normal-field/ARAP/packing DON'T spawn nested OpenMP
  regions. **Measured (paris4 512³, 16 threads, tile_core=128): 83.6s → 2.0s ≈ 42×** — a compound win:
  removing the wasteful inner parallelism (per-tile kernels are far too small; fork/join × 417 fragments
  dominated) alone was ~7× (83.6→11.9s serial), then parallel tiles ~6× more. `schedule(dynamic,1)` (not
  static) because tile cost is wildly uneven (dense papyrus tiles dwarf air tiles); the 6× (not ~13×) is
  bounded by the heaviest single tile's critical path — **tile_core=64 rebalances to ~8× (1.0s)** at the
  cost of +2× fragmentation / −11% coverage (seam clipping), so 128 stays the quality default. Set
  `g_parallel_serial=true` around a `trace_volume_tiled` call to force it fully serial (A/B / determinism).
- **`trace_volume_streamed(pred_root, ct_root, full, ...)`** (`trace_stream.hpp`) — the OUT-OF-CORE form:
  identical tiling/growth/stitch (it shares `detail::trace_one_tile` with `trace_volume_tiled` — they
  differ only in how a tile block is obtained), but each tile's (tile+2·halo) pred/CT block is FETCHED
  from a zarr store via `io::read_zarr_region` (only the covering chunks are read) instead of cropped
  from a resident volume. The full multi-TB volume is therefore NEVER resident — peak RAM is one
  process-region block + the accumulating fragments. `test_trace_stream` proves the streamed fetch is
  voxel-identical to the resident crop and the streamed trace result equals the in-core one. (The per-
  tile body was factored into `detail::trace_one_tile` so the in-core and streamed tilers can't drift.)
- **`trace_volume_streamed_to_disk(..., out_dir)`** — the FULLY out-of-core tracer: streams tiles in
  from zarr AND streams fragments OUT to disk (one `.fxsurf` each via `io::write_fxsurf`, plus a
  `manifest.txt` of `name valid bbox` for a later OOC stitch). Only the current tile's block + fragments
  are ever resident, so peak RAM is bounded regardless of volume OR total-surface size — the last piece
  of the tracer's out-of-core memory bound. `test_trace_stream` proves to-disk == in-RAM and that
  reading every `.fxsurf` back recovers all cells (channels + manifest included).
- **`build_patch_graph` is parallel** — make_patch, KdTree builds, and the O(P²) pairwise metrics are
  per-element independent (per-row buffers; `KdTree::nearest` is `const` so concurrent same-tree queries
  are safe). ~3–5× (the tiled stitch was the bottleneck once fragmentation 9×'d the patch count). The
  pairwise is bbox+KdTree pruned, so it's ~O(P·neighbours) in practice, not O(P²).

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
- **Normal orientation is load-bearing — and must follow the manifold, not a global axis.** A signed
  gap needs consistently-oriented normals. A single global axis (or the umbilicus radial on a crop)
  CANCELS across a U-bend (+spacing on one limb, −spacing on the other → mean ≈ 0 → different wraps
  read as "same sheet"; this collapsed 60 patches into 1 cluster). Fix: keep each patch's RAW local
  normals (the grower's parallel-transported frame is consistent *within* a patch) and **propagate a
  consistent sign across the patch graph** (`SignDSU`: span the most-co-normal edges first). This is
  curvature-robust and needs no umbilicus. MERGE is then orientation-free (strong |n·n| + small
  |gap|). On the paris4 1024³ cube: clusters 1→54, winding conflicts 22→2.
- **Cap cells per patch** (`max_graph_cells`, default 8000) — the O(P²·cells) pairwise is 17 min on
  60 × 700k-cell patches, 8.9 s capped. The KdTree/field also subsample from these.
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
**Classical detection chain ported from taberna + tested:** `structure_tensor.hpp` (sheetness+normal,
tiled/OOC), `hessian.hpp` (Frangi plate), **`ced.hpp`** (coherence-enhancing diffusion — the fog fix:
diffuse ALONG confident sheets to close porosity, weakly ACROSS to preserve wraps; crop-scale, tile TODO),
**`ridge.hpp`** (NMS centerline — thick band → ~1-voxel crest along the normal). The chain is
CED → structure_tensor → ridge_nms → postproc/morph → eval/score. TODO ports: `affinity`+`partition`
(signed RAG + Mutex Watershed, the per-wrap instance path), the advancing-front `trace.c`, SNIC (rewrite
clean — taberna's is a verbatim MIT port). STUB core; the **multi-scale patch graph (`patch_graph.hpp`) is implemented + tested**
(`test_patch_graph`, `test_patch_field`, `test_cosegment`, `test_multiscale`). Open ADRs: tracer
growth/accept-rollback policy; detector fusion; MWS-vs-fit role.
TODO: make `soft_gate` net-positive (ARAP governance of bridged cells, component pruning); pick a
per-dataset `surf_thresh` default (≈0.10 for normalized predictions); stitch merged clusters into
single charts; spacing from CT autocorrelation; feed assigned windings into the `winding` fit.
