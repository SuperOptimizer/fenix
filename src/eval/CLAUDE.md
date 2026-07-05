# eval — CLAUDE.md

## Purpose
The quality-metric suite: segmentation, surface, topology, mesh, and deformation metrics
+ the official composite score, wired to two CLI stages (`eval`, `eval-set`). See
`docs/research/research-core.md` (eval), `villa-ml.md` (metrics), `spiral-v2.md`
(satisfaction metrics).

## Public API & key types
- `metrics.hpp`: `dice(a,b)`, `iou(a,b)` (binary overlap on `VolumeView<const u8>`
  foreground masks); `struct Voi{split,merge,total()}` + `voi(seg,gt)` — VOI over
  `s32` label volumes, **0 = ignore/background in both** (generic instance-seg VOI).
- `nsd.hpp`: `surface_voxels(mask)` (6-neighbour boundary extraction) + `nsd(pred, gt,
  tau)` — Normalized Surface Dice @ τ, EDT-based (DeepMind `surface_distance` port),
  symmetric, `[0,1]`.
- `score.hpp`: `struct VoiUnion{split,merge,total()}` + `voi_union(pred, gt)` — VOI
  restricted to the **union foreground** of two binary masks (26-conn CC per mask; 0 =
  "background cluster" where a mask is absent, NOT ignored — this is the composite's
  VOI, distinct from `metrics.hpp`'s generic `voi()`). `voi_score(total, alpha=0.3)` =
  `1/(1+alpha*VOI)`. `betti_f1(bp,bg)` + `topo_score(pred, gt, w={0.34,0.33,0.33})` —
  per-dim number-based Betti-F1 (via `topo::betti_numbers`), weighted-averaged over
  active dims (inactive dim = both zero → excluded). `struct Score{surface_dice,
  voi_score, topo_score, total}` + `official_score(pred, gt, tau=2.0)` — **the Kaggle
  composite**: `total = 0.30*topo_score + 0.35*surface_dice + 0.35*voi_score`.
- `deformation.hpp`: `jacobian_fold_fraction(dz, dy, dx, det_floor=tol::fold_det_floor)`
  — fraction of interior voxels where `det(I + grad(disp)) <= det_floor` (folded /
  non-invertible), central differences, ZYX component volumes. GT-free invertibility
  guard on a deformation field.
- `mesh_quality.hpp`: `struct MeshQuality{valid, components, holes, euler, edge_med,
  edge_cv, edge_p99_ratio, frac_long, frac_short, degen_tri, bad_angle, min_angle_deg,
  fold_detj, self_intersect, sdirichlet_mean, sdirichlet_p99, curvature_mean,
  normal_smooth_deg, boundary_frac}` + `analyze_mesh(const Surface&)` — intrinsic,
  GT-free grid-mesh quality: topology (components + 2D cubical Euler → hole count),
  edge spacing/tearing, triangle degeneracy/min-angle, orientation-flip (det-J proxy on
  a (u,v) grid), symmetric-Dirichlet distortion, 3-point curvature/normal-dihedral
  smoothness, self-intersection (uv-distant cells sharing a 3D bin), boundary fraction.
  All scale-normalized by median edge length.
- `eval.hpp`: the `eval` / `eval-set` CLI stages (below) + `detail::load_thresholded`,
  `detail::score_pair`, `detail::PairScore` — the plumbing that loads/thresholds a
  `.fxvol` pair and calls `official_score` + `dice`/`iou`.
- **NOT YET IMPLEMENTED** (still just named in this doc as future work, no header
  exists): adapted-Rand, clDice/center-surface Dice, critical-components
  (topological split/merge), winding consistency/monotonicity, self-intersection on a
  *deformation field* (mesh_quality's self-intersect is grid-local only), satisfied-
  patches/points (spiral-v2).

## CLI stages (eval.hpp, self-registered via `FENIX_REGISTER_STAGE`/`register_stage`)
- `fenix eval <pred.fxvol> <gt.fxvol> [--tau T] [--thresh X] [--gt-thresh X]
  [--band <fxsurf> [--band-off z y x] [--band-r N]] [--json]` — threshold both volumes
  to binary masks (default thresh = 0.5·peak, auto-detects 0..1 prob vs 0..255 codec
  scale) and print the Kaggle composite + Dice/IoU. `--band` rasterizes a shell around a
  `.fxsurf` mesh (`ml::rasterize_band_multi`) and zeros BOTH masks outside it — the
  referee for scoring a whole-volume prediction against a single-segment GT (see
  Gotchas). `--json` emits one machine-readable line (pred/gt paths, dims, tau,
  thresholds, official/surface_dice/voi_score/topo_score/dice/iou).
- `fenix eval-set <manifest.toml> <split> [--pred-dir D] [--gt-dir D] [--tau T] [--json]
  [--baseline B.json]` — score a whole calibration/validation/test split (aligned
  `pred`/`gt` path lists in a flat TOML, read via `core::Config`) and report per-pair +
  aggregate (mean/min/max/std) official score. `--baseline` is a **CLI-parsed-but-
  unwired reserved flag** — no regression-gate logic exists yet (TODO hook only).
  Manifest template + the 3-way split discipline (calibration/validation/test,
  disjoint by scroll+region — the overfitting firewall): `docs/design/eval-split.example.toml`.

## Inputs / outputs & formats
In: predicted vs GT binary/label volumes (`.fxvol` via `codec::VolumeArchive`), `.fxsurf` meshes (band masking, mesh_quality), point
collections (future winding metrics). Out: metric values → stdout log lines or one-line
JSON (`--json`) for baseline storage/regression gates; no persisted report format yet.

## Dependencies
Intra: `core`, `geom` (`connected_components`; NSD uses `geom::edt_squared`, not a
separate EDT), `topo` (`betti_numbers`), `codec` (`VolumeArchive`), `io` (`VolumeArchive`,
`read_surface`), `ml` (`rasterize_band_multi`, `VolumeSurfaceIndex` — only for `--band`).
Third-party: none.

## Invariants & numerics
- VOI/contingency-table math is integer-exact in the counting (parallel per-chunk local
  `unordered_map`s merged serially, deterministic regardless of thread count); the
  log-probability sums themselves are f64 (accumulation-sensitive, per root CLAUDE.md
  §2.3). Two DIFFERENT VOI definitions coexist by design: `metrics.hpp::voi` (0 =
  ignore in both masks, generic instance-seg) vs `score.hpp::voi_union` (0 = background
  cluster over the union foreground, what the composite score actually uses) — do not
  conflate them.
- `topo_score` currently uses the **number-based Betti-F1 proxy** (`min(b_pred,b_gt)`
  matching by count, not exact PH matching) — same caveat as taberna's dim-1/2 code;
  flagged as a refinement TODO, not silently claimed exact.
- Composite edge cases follow the metric definition, not a special-cased branch: both
  masks empty ⇒ all three components = 1 (total 1); one empty ⇒ surface_dice 0,
  topo_score 0 (k=0 Betti-F1 = 0), voi_score → small positive.
- Connectivity: 6-conn surface-voxel extraction in NSD, 26-conn CC in `voi_union`
  (per `docs/conventions.md`); asymmetric VOI emphasis is intentional — a cross-wrap
  merge is far worse than a split, so `merge`/`split` are reported separately even
  though the composite only consumes `.total()`.
- `load_thresholded` never densifies a u8-native `.fxvol` archive to f32 — `src_dtype()
  == u8` reads+thresholds `u8` natively (avoids a 4x transient RAM blowup; see
  `docs/review/2026-07-02/sweep-oom-ooc.md` and the root CLAUDE.md "no f32 widening of
  u8" convention). Non-u8 archives still go through an f32 threshold
  path — a real fix needs the streaming/chunked eval rework noted below.

## Performance notes
`metrics.hpp`/`score.hpp`/`nsd.hpp` are chunk-parallelized (`parallel_for`/
`parallel_for_z`, chunk count = `cpu_budget()`) with serial-merge accumulation — EDT
(`geom::edt_squared`) and connected-components dominate; both are single, shared
implementations reused from `geom`, not reimplemented here. `eval`/`eval-set` currently
load whole volumes into RAM (via `codec::VolumeArchive`) — fine for
eval-sized crops, but NOT out-of-core; a genuinely huge volume needs windowed/chunked
scoring (unimplemented — see Status).

## Gotchas / pitfalls
- **Single-segment GT cannot referee a whole-volume prediction.** Every correctly-
  detected but untraced wrap scores as a false positive outside the labeled segment
  (measured: both student and teacher scored SurfaceDice <0.2 on an otherwise-good
  block) — this is WHY `--band` exists; use it whenever GT covers less than the full
  prediction volume.
- Don't conflate `metrics.hpp::voi` (generic, ignore-on-zero) with
  `score.hpp::voi_union` (composite-specific, union-foreground) — they answer different
  questions and are not interchangeable.
- Don't ship a `topo_score` that silently claims exact Betti-matching when it's still
  the number-based proxy.
- NSD + winding-consistency did NOT exist in villa's harness — these are fenix-original
  additions to the suite, not ports; treat their calibration (τ, thresholds) as new
  and unverified against any prior baseline.
- `eval-set --baseline` is accepted on the CLI but does nothing yet — don't assume a
  regression gate exists just because the flag parses.

## Status & TODO
**Implemented + tested:** `metrics.hpp` (Dice/IoU/VOI), `nsd.hpp` (SurfaceDice@τ,
EDT-based), `score.hpp` (VOI-union, Betti-F1 TopoScore, **`official_score()`** —
the Kaggle composite, multithreaded), `deformation.hpp` (Jacobian fold-fraction),
`mesh_quality.hpp` (full GT-free grid-mesh quality suite), and the `eval` / `eval-set`
CLI stages including `--band` region-limited scoring and `--json` machine output.
Verified against the metric spec (discriminates merge→VOI↓, hole→TopoScore↓).

**Refinements TODO (taberna does these, ours approximates):**
- area-weight SurfaceDice by marching-cubes surfel area (`NSD_AREA[256]`) instead of
  raw boundary-voxel counts.
- tile TopoScore into 2×2×2 octants with dim-0 EXACT union-matching + 6-conn-fg Betti.
- exact dim-1/2 Betti-matching via `topo/cubical` (replace the number-based proxy in
  `betti_f1`).
- streaming/chunked `eval`/`eval-set` for out-of-core volumes (currently whole-volume
  RAM load via `VolumeArchive`).
- wire `eval-set --baseline` to an actual regression-gate comparison (flag exists,
  logic doesn't).

**Not started (named as future scope, no code yet):** adapted-Rand, clDice, critical-
components (topological split/merge), winding consistency/monotonicity metric,
satisfied-patches/points (spiral-v2). `jacobian_fold_fraction` covers deformation-field
invertibility but not deformation-field self-intersection or winding-specific checks.

Open ADRs: exact Betti-matching algorithm; VOI_score transform; per-metric tolerances.
