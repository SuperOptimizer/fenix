# winding — CLAUDE.md

## Purpose
**The heart of fenix.** The unified unrolling method: fit a global, invertible
(diffeomorphic) deformation that straightens the spiral scroll into an ideal Archimedean
spiral, so each wrap can be flattened. Blends Henderson's diffeomorphic spiral fit
(`spiral-v2`) + taberna's Eulerian winding field + VC tracer patches + thaumato
winding-graph ideas — as **one cohesive method, not switchable backends**. See
`docs/research/spiral-v2.md`, `research-core.md` (winding_field), `research-tools.md`,
`docs/design/winding-pilot-ablation.md` (first real-scroll fits, PHercParis4).

## Public API & key types
- **`SpiralModel`** (`spiral_model.hpp`) — the deformation `T: scroll↔spiral-space`, a
  composition of invertible factors: `umbilicus` shift ∘ **SVF flow** (`FlowField`,
  `flow.hpp`, RK4) ∘ global `AffineYX` (`expm2`, det>0) ∘ per-z-band residual
  `AffineStack affine_bands` ∘ radial `GapExpander gap` (per-winding scale) + scalar
  `dr_per_winding` + `winding_offset` (pure gauge). `to_canonical`/`to_scroll` are exact
  inverses. `winding_at(p)` = stepped wrap index (branch-cut at θ); `winding_cont(p)` =
  continuous winding (no θ term — smooth everywhere, the readout for corpus targets and
  holdout scoring). Persisted as `.fxmodel` (`model_io.hpp`, version 2).
- **`transforms.hpp`** — `Mat2`/`expm2` (closed-form 2×2 matrix exponential, always
  det>0), `AffineYX` (per-slice affine via expm), `AffineStack` (piecewise-constant
  z-banded residual affines — spiral-v2's per-slice affine, banded for capacity),
  `GapExpander` (monotonic per-winding radial scale, forward + searchsorted inverse).
- **`flow.hpp`** — `FlowField` (3 component volumes on a coarse lattice + `lat_lo`/
  `lat_scale` to map voxel↔lattice coords), `flow_point` (RK4 integrate, sign=±1),
  `flow_point_backward` (exact discrete adjoint of the unrolled RK4 — scatters into the
  velocity lattice via trilinear-stencil weights; the core of the differentiable fit).
- **`diffeo_fit.hpp`** — `fit_spiral_diffeo(model, targets, groups, rels, cfg)`: the
  full differentiable fit. Optimizes dr, global affine, `winding_offset`, per-band
  affines, per-winding gap logits, and the SVF flow lattice by analytic reverse-mode
  gradient (`winding_backward` mirrors the forward chain exactly, incl. the honest gap
  chain-rule through `GapExpander`), AdamW, **coarse→fine flow pyramid**
  (`DiffeoFitConfig::flow_pyramid`, halves the lattice to ≥4/axis and trilinear-upsamples
  between levels), Stage 0 (affine-only, flow frozen) → Stage 1 (unfreeze flow + bands +
  gaps). Per-constraint backward parallelizes over `threads` chunks with per-chunk grad
  buffers, reduced after. Loss = winding-target MSE + `lambda_cowind`·co-winding variance
  + `lambda_rel`·relative-winding MSE + flow L2/Laplacian + band/gap smoothness.
  `CoWindingGroup` (shared-but-unknown winding), `RelWindingConstraint` (`W(b)-W(a)=delta`,
  the "+1/+2/+3 radial line" annotation). Band params are **clamped** (±1.5 logits, ±512
  translation) — `expm2` overflows past `|logit|~89`, seen on real data (wind6).
- **`fit.hpp`** — `FitConstraint{scroll_pt, target_winding}`, `fit_spiral` (finite-diff
  dr+global-affine only fit) — the Stage-0 warm-start predecessor to `diffeo_fit.hpp`,
  still used standalone for quick smoke tests.
- **`spiral_fit.hpp`** — `spiral_fit_lsq`/`spiral_fit_from_field`: closed-form OLS
  Archimedean `r = a + b·θ_total` fit — a cheap global-pitch estimate, independent of the
  diffeo fit.
- **`winding_field.hpp`** — `winding_init` (analytic polar field, GT-free init/baseline),
  `monotonicity_violation` (ray-cast GT-free health metric: fraction of radial steps
  where winding fails to increase — the fold/wrap-overlap signal).
- **`relax.hpp`** — `relax`: red-black Gauss-Seidel Tikhonov smoothing of a noisy winding
  field (isotropic core the anisotropic/TV variants would build on).
- **Bridges** (segment/annotate/corpus → fit vocabulary):
  - `fit_bridge.hpp` — `patches_to_constraints`: `PatchGraph` with assigned integer
    `wrap`s → `FitConstraint` (conf≥`conf_min` cells) + `CoWindingGroup` per patch.
  - `corpus_bridge.hpp` — `corpus_to_constraints`: multi-wrap corpus meshes → per-cell
    CONTINUOUS `FitConstraint`s via BFS angle-unwrap (`unwrap_component`) + measured wrap
    spacing (`spacing_samples`) + per-component Archimedean base gauge (median
    `r/spacing - turn`). Also: `regauge_components` (EM re-gauging: shift each
    component by its post-fit residual median), `score_holdout` (generalization RMSE on
    never-fit meshes, gauge-free via per-component median removal), `finite_fm`/
    `median_of` (fast-math-safe NaN-dropping utilities used project-wide in this dir).
  - `anno_bridge.hpp` — `to_fit_inputs`: lowers an `annotate::AnnotationSet` (co-winding
    strokes + must-links, radial lines, normal hints) into targets/groups/rels; must-links
    DSU-union strokes so a labeled stroke propagates across its component.
- **`cosegment.hpp`** — Stage D EM loop (`cosegment_refine`): patch-graph analysis ↔
  field-guided hole fill/weak-cell correction, `CosegParams::eulerian` selects the
  band-Eulerian solve over the discrete graph for windings.
- **`patch_field.hpp`** — coarse Eulerian winding field FROM assigned patches
  (`build_patch_winding_field`, `fill_surface_from_field`) and the inverse — solving θ
  directly from patch normals (`build_eulerian_winding_field`, `assign_windings_from_field`),
  band-restricted (`FieldParams::band`) natural-BC gradient-matching solve (see Gotchas).
- **`stitch_stream.hpp`** — out-of-core winding stitch: `stitch_streamed` (z-slab sweep)
  and `stitch_streamed_3d` (full 3D tile-graph BFS alignment, RAM-bounded in all axes).
- **`trace_long.hpp`** — `run_trace_long` / CLI `trace-long`: spiral-guided long tracing,
  one seed grown as far as the data allows, gated by the fitted model's winding (in-core
  block mode AND a fully streamed mode over `CachedVolume`/zarr with on-demand window
  fetch — `segment::StreamGrower`, optional per-window ML inference via
  `ml::predict_surface_window`, else classical structure-tensor sheetness).
- **`wrap_label.hpp`** — CLI `wrap-label`: stamps each corpus-mesh cell with its absolute
  wrap index under the fitted model (gauge each component against the model, mask
  cells where mesh/model disagree by >`conf_tol`) → a `.wrapcolor` sidecar (`FXWCOL1`:
  u16 wrap per cell, `0xFFFF`=masked) — instance-label source for ML training, decoupled
  from the mod-k color choice (re-color without re-labeling).
- **`wrap_fill.hpp`** — CLI `wrap-fill`: DENSE per-voxel sheet-instance labels from
  `model.winding_cont` + a CT air-cut threshold — u8 `0`=air, `1..k`=papyrus (wrap mod k),
  `255`=ignore (near a wrap boundary, `|cont-round|>buffer`, or out-of-table). Exposes
  `wrap_fill_labels` as a reusable core (used by an ML feeder, not just the CLI).
- **`winding.hpp`/`winding.cpp`** — CLI `winding` (see below); `.cpp` is the SPLIT-build
  per-TU shim (`FENIX_SPLIT=ON`, ADR 0008) — unity build never compiles it.

### CLI: `fenix winding`
```
fenix winding surf=<fxsurf>... umb=y,x|<umb.toml> [bridge=corpus|patch] [stride=4] [rounds=2]
              [eulerian=1] [iters_affine=250] [iters_flow=500] [flow=12] [bstride=6]
              [conf=0] [spacing=0] [holdout=0] [abands=0] [flowz=0] [regauge=0] [gaps=0|-1=auto]
              [out=<fxmodel>]
```
Pipeline: load+subsample meshes → axis (straight `umb=y,x` or curved `umb=<toml>` from
`fenix umbilicus`) → bridge (`corpus`: per-cell continuous winding; `patch`: cosegment +
one integer per patch) → `fit_spiral_diffeo` → optional EM `regauge` rounds → honest
target-RMSE + optional holdout scoring → `write_fxmodel`. `gaps=-1` auto-sizes the
per-winding gap table from the observed winding range. Companion CLIs: `trace-long`,
`wrap-label`, `wrap-fill` (all self-registered stages, `winding.hpp` includes them all).

## Inputs / outputs & formats
In: corpus/tracer meshes (`Surface`, `io/surface.hpp`), a straight or curved
`annotate::Umbilicus` (TOML from `fenix umbilicus`), optionally an `AnnotationSet`
(co-winding strokes/radial lines/must-links, via `anno_bridge.hpp`), `PatchGraph`
(`segment`, via `fit_bridge.hpp`). Out: `.fxmodel` (hand-rolled binary, magic `FXMD`,
version 2, no back-compat — reject unknown versions), per-wrap `.fxsurf` (via
`flatten`'s `T⁻¹` sampling, not this dir), `.wrapcolor` sidecars (`FXWCOL1`), dense
u8 wrap-instance `.fxvol` labels (`wrap-fill`).

## Dependencies
Intra: `core`, `io`, `codec`, `geom`, `annotate`, `predictions`, `segment`, `ml` (only
`trace_long.hpp`'s optional streamed per-window inference, firewalled). Third-party:
none (the optimizer/AD is first-party — hand-rolled gradients + AdamW `core/optimize.hpp`;
~30-line SVF integrator; **no pyro/torchdiffeq**).

## Invariants & numerics
Invertibility/no-fold is structural (every factor: expm affine, monotone gap, SVF flow)
**and** soft (Laplacian/smoothness regularizers). f32 compute, f64 for optimizer/gradient
accumulation (`FitGrad` fields, `winding_backward`). Fast-math OK — fit quality is a
satisfaction metric (RMSE in windings), never bit-exact. `finite_fm`/`median_of`
(magnitude-bound NaN check) are load-bearing: `std::isfinite` constant-folds to `true`
under `-ffast-math`, and `nth_element` with NaN comparisons is UB (caused a real segfault
on diverged holdout residuals — see wind6 in the ablation doc).

## Performance notes
**Out-of-core is the key gap vs Henderson** (his is in-core/single-GPU ~19h) — but it now
has a first slice: `stitch_stream.hpp`'s z-slab and full-3D-tile stitches are OOC and
RAM-bounded; the diffeo FIT itself is still in-core (small lattices resident, all
constraints loaded). Roadmap: stream constraint mini-batches in z-tiles, accumulate grads
globally, coarse-global warmup first. GPU-target later (sampler + fit are prime
candidates). Per-constraint backward parallelizes over `hardware_concurrency()` threads
by default (`DiffeoFitConfig::threads`).

## Gotchas / pitfalls
- Don't build "multiple isolated unrolling backends" — it's one blended method.
- `shifted_radius` is the unifying quantity; keep winding-field and diffeomorphic-fit as
  two views of it, not two pipelines.
- Skip spiral-v2's cruft: ~150 hyperparameters, autoresearch/wandb, redundant loss
  variants, pyro/torchdiffeq.
- **`winding_at` (stepped) vs `winding_cont` (continuous) are NOT interchangeable.**
  Fitting `winding_at` against continuous corpus targets leaves an irreducible ±0.5
  sawtooth (wind4→5 in the ablation). Corpus bridge / holdout scoring / wrap-label /
  wrap-fill all use `winding_cont`. Patch-bridge integer targets use `winding_at`.
  `DiffeoFitConfig::continuous` selects which one the fit itself trains against.
- **The patch bridge is the wrong model for corpus meshes** — one integer-per-patch
  collapses a multi-wrap mesh to a single wrap. Use `bridge=corpus` for GP-style
  multi-turn segments; `bridge=patch` only for single-wrap tracer fragments.
  This is deliberately still a *choice at the winding stage*, not a data-format wart.
- **Axis error dominates everything else.** A straight-axis estimate 7.8k/4.9k voxels
  off the true curved umbilicus inflated measured spacing 4× (wind3→4). Always prefer a
  curved umbilicus (`fenix umbilicus`) over a straight fallback on real data.
- **holdout ≈ train ⇒ underfit, not overfit** — the opposite of the usual ML intuition;
  with too little model capacity (one global affine, coarse flow), both numbers are bad
  together. Add capacity (bands, flow resolution, gap logits) before regularizing more.
- Band params can blow up `expm2` (overflow past `|logit|~89` → inf → NaN windings) if
  unclamped — this happened on real data, not just synthetically; the clamp in
  `diffeo_fit.hpp`'s stage loop is load-bearing, don't remove it for "cleanliness".
- **`FieldParams::band` — restrict the Eulerian solve to a band around the data.** The
  full-domain `∇θ≈n/spacing` least-squares is ill-posed off-data (non-mean-zero
  divergence) → spurious far-field structure corrupts the integer readout. Multigrid was
  tried and REJECTED (residual converges but to the wrong answer — the formulation, not
  the solver, was wrong). `band ≥ spacing/ds` is the actual fix.
- **Boundary condition must be direct gradient-matching, not the divergence form** — a
  zero-flux BC clamps θ flat for volume-filling (dense) data (∇·b≈0 ⇒ θ→const ⇒ wraps
  collapse); matching the gradient directly (average over live neighbours only) gives the
  correct natural BC and recovers dense multi-wrap data.
- EM re-gauging is asymptotic, not exact — gate tests differentially (shift magnitude
  trend), never on an absolute loss/shift threshold alone; at regauge round 10 on real
  data the shift was still ~0.9-1.1 windings (not fully converged, but holdout still
  improved monotonically wind8→9→10).

## Status & TODO
**The differentiable diffeomorphic fit is implemented, validated, AND has now run
end-to-end on real scroll data** (PHercParis4, `docs/design/winding-pilot-ablation.md`,
2026-07-04): `fenix winding` on 20-25 corpus meshes with a curved umbilicus produced a
`.fxmodel` that `flatten mode=unroll` turned into the **first machine-unrolled,
ink-mapped strip produced end-to-end by this pipeline** (wind10 → 13824×1001 strip →
render-layers → predict-ink). Holdout RMSE went 11.16 → 2.06 windings (−82%) across the
ablation by adding: curved axis, continuous-target fitting, z-band affine + finer flow
capacity, EM re-gauging (regauge=10), and fitted per-winding gap logits — gauge EM and
the gap table were the two dominant levers.

Implemented + tested: `diffeo_fit.hpp` (analytic-vs-FD gradient gate, synthetic
deformed-spiral recovery), coarse→fine flow pyramid, per-z-band affine capacity, fitted
gap logits, `RelWindingConstraint`/rel-winding term, `anno_bridge.hpp` (annotation →
fit-input lowering, tested against held-out error halving), the patch graph + coarse
winding field + coupled fill + band-restricted Eulerian stitch (incl. CT-valley
touch-proof Δwrap; `test_patch_graph/field/cosegment`, `test_patch_valley`), the
out-of-core winding stitch (`stitch_streamed` z-slab + `stitch_streamed_3d` full 3D tile
BFS, both proven == the in-RAM whole stitch), `trace_long.hpp`'s spiral-guided long
tracer (in-core AND fully streamed modes), `wrap_label.hpp`/`wrap_fill.hpp` (instance-
label sidecars/dense volumes feeding ML surface training —
`docs/design/multiscale-instance-surface.md`). `fit.hpp::fit_spiral` (finite-diff,
dr+global-affine) remains as the cheap Stage-0 warm-start / smoke-test path.

Next (roadmap, expected-value order per the ablation doc):
1. **EM to full convergence** (shift < 0.05 — currently ~20-30 rounds needed, or a
   stronger inner refit per round).
2. Constraint balance across components (dense meshes currently dominate the loss;
   per-component weight normalization).
3. Per-wrap visual QC of the unroll strip itself (surf-qc on the unrolled surface, not
   just the input meshes).
4. More capacity (bands/flow resolution) only if 1-2 plateau first.
5. **Out-of-core the FIT itself** — currently only the winding *stitch* is OOC; stream
   constraint mini-batches in z-tiles for the diffeo fit, coarse-global warmup first.
6. Dense "lasagna" winding-density term + EM track re-assignment + sym-Dirichlet
   (spiral-v2 loss vocabulary not yet ported).

Open ADRs: coarse-to-fine + spring-anneal schedule; loss weights; OOC mini-batch
sampling; flow-lattice resolution.
