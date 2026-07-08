# Full-scroll model — one net for papyrus / wrap / ink, then a trivial tracer

Status: design sketch (forrest, 2026-07-07). Companion to
`docs/design/model-registry.md`, the multiscale + instance-label surface-prediction
plans, and `src/winding/CLAUDE.md`.

## The idea

Replace the single-purpose surface-prediction net with **one network** that labels
every voxel of the scroll:

- **background vs papyrus**,
- **which wrap** the papyrus voxel belongs to,
- **whether the voxel is ink**.

Then the tracer's job collapses: it no longer searches for sheets in raw CT — it
extracts them from a field that already encodes them, and stitches wraps by scalar
continuity across blocks.

## The one non-negotiable reformulation

Absolute wrap number is **not a local property**. A 128³ patch at 2.4µm sees 3–5
sheet crossings; whether the middle one is wrap 214 or 215 depends on integration
from the umbilicus. So the net cannot emit absolute wrap ids — and doesn't need to.

It emits a **wrap-relative field**; absolute numbering is one global integration
constant plus conflict arbitration, done downstream. This is the same split both
predecessors converged on (thaumato's winding-angle graph, taberna's Eulerian
winding field) and it is exactly the data term `src/winding`'s fit wants.

## Output spec (per voxel, one forward sweep)

| head | channels | target |
|---|---|---|
| semantics | papyrus prob, recto-surface prob, verso-surface prob | subsumes today's surface net |
| local frame | sheet normal (oct-encoded 2ch or 3×i8) | direction of "inward one wrap" |
| winding coordinate | fractional wrap position `w ∈ [0,1)` (+ optional confidence) | taberna's `shifted_radius` fractional part; continuous across a sheet, +1 per wrap inward. **Supervised densely over the scroll interior — air gaps included** (the Eulerian-field view), not papyrus-only: papyrus-masked `w` leaves a mid-scroll block's wraps as disconnected components and block-local unwrap cannot link them (measured on phantoms: wrap-id acc 0.15 masked vs 1.00 dense) |
| ink | ink prob | sigmoid, papyrus-masked |

Stored as a multi-channel `.fxvol` (lossy tile-DCT per channel; normals + winding
coord need a tighter τ than the probability channels). All heads hang off the
shared 3D-UNet backbone as 1×1×1 convs — backbone cost is unchanged from the
current surface net, so the entire fp8/int8 QAT + separable-student + TRT lane
work applies verbatim.

## The tracer contract (why it becomes trivial)

Given `(papyrus, normal, w)`:

1. **Wrap surfaces are level sets.** Wrap k = the isosurface of the *unwrapped*
   winding scalar at integer level k. Within a block, unwrapping `w` is local
   integration along the normal (phase-unwrap with papyrus-masked confidence).
2. **Stitching = continuity.** Two blocks agree on wrap identity iff their
   unwrapped scalars differ by a constant integer on the shared halo — one int per
   block-pair, solved as a global offset graph (spanning-tree + loop-closure
   voting). This is the whole "stitch the wraps together" step.
3. **One anchor.** The umbilicus annotation pins wrap 0; everything else follows.

What remains genuinely hard (and stays in `src/winding`): regions where the net is
uncertain — fused/touching sheets, tears, missing papyrus — produce inconsistent
loop closures. Arbitration is a light graph solve over conflict regions, optionally
falling back to the full diffeomorphic spiral fit with these fields as dense data
terms (which makes the fit itself faster and more robust than fitting raw CT).

Dense per-voxel wrap ids then come from `wrap-label`/`wrap-fill` (already planned
in the winding module), and per-voxel ink rides along for free — the final artifact
is the labeled volume the user asked for: `(bg | papyrus, wrap k, ink?)`.

**Pipeline status (2026-07-07, `tools/fullscroll/`)**: the WHOLE pipeline is
implemented (one sweep, from a 16-agent design workflow + adversarial reviews):
label generation (mesh ingest → fit v0 → tolerance-band sem → ink projection →
texture bank), manifest feeder, phase trainer (phantom→real→QAT on the int8
lane), tiled inference sweep (brick ledger, lanes, prescreen culling), and the
out-of-core solver (kfield int16 + max-weight-tree stitch + umbilicus anchor).
The scaled solver passes its e2e oracle: **wrap-id accuracy 1.0000** on a 192³
phantom through the full production path. A fifth head (`interior`) was added —
the scroll-interior probability that is the solver's confidence channel. Not
yet run: real-data label generation (needs GP downloads + a PHerc0332
umbilicus), full training, real sweeps. See `tools/fullscroll/CLAUDE.md`.

**Prototype status (2026-07-07, `tools/fullscroll/`)**: the whole contract is
implemented and validated on synthetic spiral phantoms — weighted-LS phase unwrap
(Ghiglia-Romero PCG, DCT-Poisson preconditioner) + integer-offset stitch. With
sin/cos noise σ=0.15 and garbage `w` outside the supervised region: wrap-id
accuracy **1.0000** single-block and stitched 2×2×2, loop residuals exactly 0.

## Training: masked multi-task, heterogeneous labels

No voxel has all labels; every head's loss is masked to where its labels exist.

- **Semantics** — existing surface/papyrus labels (densest supply).
- **Winding coordinate + normal** — from GP-region ground-truth meshes with winding
  annotations, and (the scalable source) **from fenix's own spiral fits**: fit a
  region, render the fit back to dense `(normal, w)` fields, train on them.
  Self-distillation flywheel: fit → labels → better net → better fit.
- **Ink** — projected flattened-ink annotations only; tiny coverage, severe class
  imbalance → masked BCE/Dice, papyrus-masked, hard-negative mined.

## Label generation under imperfect surfaces

The surfaces we have are noisy, partial, and mutually inconsistent — they are NOT
used as voxel labels directly. Three principles:

1. **The fit is the label generator, not the meshes.** Individual meshes →
   global spiral fit → render dense `(w, normal)` from the FIT. The fit averages
   hundreds of overlapping noisy constraints, rejects outlier patches by
   residual, and its rendered field is smoother than any input mesh. A ±2-voxel
   mesh error at pitch ~14 vox is ~0.14 in w at that mesh, but the fit's field
   error is far smaller — and w is regressed, not classified, so residual label
   noise degrades gracefully.
2. **Labels are (value, confidence), never truth.** Fit residual → w/normal
   weight; overlap disagreement between segmentations → surface weight (or
   exclusion); teacher entropy → ink weight. The masked losses take continuous
   weights, so bad regions are down-weighted, not memorized. Surface channels
   supervise a tolerance BAND (the GT-band idea from tools/train), never exact
   one-voxel shells.
3. **Two label sources need no surfaces at all**:
   - *Phantom pretraining* (tools/fullscroll/phantom.py): perfect dense geometry
     labels, unlimited volume; close the domain gap by harvesting real-CT
     noise/texture patches into the renderer.
   - *Physics-consistency self-supervision* on unlabeled real CT: |grad w| along
     the predicted normal ~ 1/pitch and smooth; w level sets coincide with
     predicted surface shells; recto/verso sit at +-t/2 in w-units. These
     regularize everywhere and punish exactly the incoherence that noisy labels
     would otherwise teach.

**Ink labels**: project each flattened segment's 2D ink prediction back through
its surface mapping into a shell around the mesh (dilated along the normal by the
2D model's sampling depth); 3D ink preds are voxel-space teachers directly. Both
are noisy teachers → ink trains as distillation (soft targets), papyrus-masked,
confidence-weighted.

**Bootstrap order**: anchor on the GP banner region (highest-quality dense
segmentation available) for fit v0 → train v1 → v1 predictions widen/clean the
fit → render better labels → v2. Confidence weighting keeps early-cycle noise
from propagating.

Ink benefits from sharing the backbone: the head can condition on predicted
surface + normal ("sample texture along the normal at a recto surface") instead of
rediscovering surface geometry.

## Resolution split (multiscale)

- 2.4µm, 128³: 3–5 sheets/patch — right for semantics, surface, normal, ink.
- 9.6µm, 128³: 10–20 wraps/patch — right for the winding-coordinate head (more
  wrap context, and `w` needs no fine texture).

Same architecture, two LODs; the coarse `w` field upsamples to native for the
tracer (it is smooth by construction). Matches the existing multiscale design.

## Scale check

PHerc0332 level 0 is 33592×15761×15761 ≈ **8.3 T voxels**. One sweep of the whole
scroll at 128³ tiles ≈ 4.0 M tile inferences; at the current fp8 lane's ~40 ms/tile
on one 5060 Ti that's ~44 GPU-days dense — halo overlap, papyrus-bbox culling
(most of the volume is background), the int8/TRT lanes, and cheap 3090-class fleet
nodes are what make it practical. Backbone speed is everything; heads are noise.

## Open questions

1. Exact `w` parameterization near the umbilicus and at the outermost wrap
   (boundary conditions of the unwrap).
2. Verso/recto asymmetry: one signed coordinate or separate recto/verso surface
   channels + a sheet-interior coordinate?
3. Whether the ink head trains jointly from the start or is grafted on after the
   geometry heads converge (label supply differs by ~4 orders of magnitude).
4. Confidence calibration for the loop-closure voting (what the arbitration solve
   trusts).
