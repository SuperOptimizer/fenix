# tools/fullscroll — CLAUDE.md

## Purpose
Python lab for the **full-scroll model** (docs/design/full-scroll-model.md): one net
labeling every voxel — background/papyrus (+ recto/verso surfaces), sheet normal,
fractional winding coordinate w, ink — so the tracer collapses to level-set
extraction + integer-offset stitching, with `src/winding`'s global solve pinning
absolute wrap numbers.

## Pipeline map (one-sweep implementation, 2026-07-07, from the 16-agent design
## workflow: specs + adversarial reviews + completeness critic all applied)

data prep: `labelstore.py` (LabelStore v1: (value, confidence) u8 zarr-v2 dense
label container, coverage index) · `ctio.py` (tensorstore CT reads, block/halo
iterator, `fenix ingest-zarr` staging wrapper, PredStore = sweep output store
with done/CULLED brick ledger + provenance guards) · `mesh_ingest.py` (OBJ/PPM ->
theta-unwrapped, outward-oriented constraint point sets; never voxelized) ·
`fitfield.py` (fit v0: cylindrical winding-field fit — per-mesh integer offsets
via median-observation graph, monotone radial W profiles, dense Eulerian render)
· `semlabels.py` (tolerance-band recto/verso/papyrus + disagreement-driven conf)
· `inkproject.py` (2D ink back-projection along -n over a shell + 3D teacher
ingest; labeled-negative doctrine) · `texture.py` (real-CT profile/residual/air
bank) · `phantom.py` (+bank rendering — labels stay exact) · `feeder.py`
(manifest-driven mixed real+phantom IterableDataset, per-sample deterministic,
normal-aware flips/rot90).
training: `train.py` (phases phantom->real->qat; fp16+scaler / qat = no autocast
no scaler; EMA frozen at the QAT swap; live-net validation in qat; NaN-guard
resets accumulation; surface-dice gate vs pre-QAT ref) · `wrap_probe.py`.
inference: `lanes.py` (fp16cl / fp8 / int8-static via the SWAP path — never the
fp8_forward monkeypatch on a swapped net; fake analytic lane; NO TTA — flips
reverse winding chirality) · `sweep.py` (brick+margin sliding window, Gaussian
blend, exact-empty patch skip only, coarse-level prescreen -> culled markers).
solving: `solve.py` (block unwrap -> int16 kfield (rel = round(U-w)) + face
slabs -> max-weight-tree stitch + LS cycle refinement + conflict regions ->
umbilicus-anchored wrap_id int16 zarr; `eval-phantom` e2e gate) — kfield trick:
U is never materialized; wrap = rel + off + C + (w>=0.5).
orchestration: `pipeline.py` (fetch/labelgen/texture/train/sweep/solve/evaluate
stages over one TOML; `e2e-phantom` integration gate; label-free health gates:
edge keep rate > 0.9, conflict rate < 2%).

MEASURED (2026-07-07): solve.py eval-phantom 192^3 multiblock multiprocess =
wrap-id accuracy 1.0000 PASS; labelstore/ctio/losses/feeder CPU suites PASS;
physics losses ~0 on GT predictions (align 0.004, pitch 0.027).

## Public API & key types
- `model.py` — `FullScrollNet` (shared vesuvius ResEnc-UNet encoder + DecoderBody
  from `tools/ml-export/reference.py` + four 1x1x1 heads: sem 3ch / normal 3ch /
  wind 2ch sin-cos / ink 1ch), `decode_w`, `warm_start(net, surface_ckpt)`.
- `phantom.py` — `make_phantom(size, pitch, thickness, seed)`: Archimedean spiral
  phantom around a wandering umbilicus with DENSE exact GT for every head (incl.
  unwrapped `W` as the tracer oracle); `to_batch`.
- `losses.py` — `full_scroll_loss` (masked multi-task: BCE sem, 1-cos normal,
  sin/cos MSE wind, pos-weighted BCE ink), `head_metrics`.
- `train_smoke.py` — phantom training smoke; proves all heads learn.

## Inputs / outputs & formats
Phantoms now; real data later = CT patches + heterogeneous masked labels (surface
labels dense; w/normal rendered back from spiral fits — the self-distillation
flywheel; ink from projected annotations, papyrus-masked, ~4 orders sparser).
Deploy output = multi-channel `.fxvol` (see the design doc's channel table).

## Dependencies
torch, numpy, scipy (phantom blur), `dynamic_network_architectures` (via
reference.py). Inside the ML-island exemption.

## Invariants & numerics
- ZYX everywhere; normals point toward **increasing W** (outward, since
  W = r/pitch - theta/2pi grows with r) — orientation is part of the label.
- w is periodic: heads regress (sin 2*pi*w, cos 2*pi*w), never raw w (mod-1 wrap
  would poison MSE); circular metrics only.
- Every head loss is masked; no voxel is assumed fully labeled.
- Phantom gradient repair at the atan2 branch cut (|grad| > 0.25 zeroed) — the true
  W field is continuous there; without repair normals flip on one voxel column.

## Performance notes
Backbone identical to surface_recto_3dunet -> the whole fp8/int8 QAT, separable
student, TRT lane in `tools/ml-export/` applies unchanged; heads are noise.
Full-scroll sweep economics live in the design doc (~8.3T voxels).

## Gotchas / pitfalls
- Absolute wrap number is NOT a local property — never make it a class target; the
  net emits wrap-RELATIVE fields, the global solve integrates (design doc §2).
- `warm_start` maps `task_decoders.surface.*` -> `shared_decoder.*` and drops
  seg_layers; assert-no-unexpected keeps the mapping honest.
- Phantom `to_batch` normalizes per-sample (mean/std) like the real pipeline.

## Status & TODO
- Slice 1 (this dir): net + phantom + losses + smoke — implemented. Cold smoke
  (150 steps): papyrus Dice 0.97 held-out, all heads learning. Warm-started 400
  steps: train geometry converges (w MAE 0.021, normals 24 deg, papyrus 0.986)
  but held-out geometry DIVERGES (w MAE 0.19, normals 113 deg) with a fixed
  12-phantom pool, and plain BCE collapses the thin recto/verso shells.
  Trainer requirements learned: phantoms generated ON THE FLY (infinite pool),
  soft-Dice term for shell channels, longer schedule for ink.
- Slice 2: `unwrap.py` — weighted-LS unwrap (Ghiglia-Romero PCG, DCT-Poisson
  precond) + integer-offset block stitch: **SELFTEST PASS**, wrap-id acc 1.0000
  single-block AND stitched 2x2x2 at sin/cos noise 0.15, loop residuals 0.
  KEY FINDING: w must be supervised densely over the scroll interior (air gaps
  included, Eulerian view) — papyrus-only w leaves wraps as disconnected
  components in a block (acc 0.15). C++ port target: src/winding.
- Later: real-data feeders (masked labels from fits/meshes/ink projections),
  multiscale split (w head at coarser LOD), QAT the whole thing for deploy.
