# Adaptive (per-patch, convergence-stopped) TTA is dead for surface_recto_3dunet

**Date:** 2026-07-06 · **Data:** PHerc0009B @ 8.64 µm native, PHerc0332 @ 2.4 µm native ·
**Setup:** predict-scroll mode=global, patch=256, overlap=0.5, .pt2 (AOTI) weights, 1024³
occupied bboxes, per-256³ sub-box MAE between tta=2 and tta=8 outputs (64 sub-boxes each).

## Idea being tested
Spend TTA variants adaptively: run 2, measure disagreement with the running mean, stop early
on "easy" patches and reallocate budget (up to the full 48-group) to "hard" ones. Hoped-for:
~2-3× cheaper whole-scroll teachers at equal label quality, or equal-cost better labels.

## Result: no exploitable structure
MAE(tta2 vs tta8) per 256³ sub-box, units of 1/255:

| corpus | min | p25 | p50 | p75 | p90 | max |
|---|---|---|---|---|---|---|
| 0009B @ 8.64 µm | 7.1 | 12.7 | 14.3 | 16.7 | 17.9 | 22.9 |
| 0332 @ 2.4 µm  | 9.2 | 12.6 | 14.3 | 16.1 | 17.9 | 22.0 |

1. **Resolution does NOT matter.** The distributions are near-identical on-grid (2.4 µm,
   the model's training spacing) and 3.6× off-grid. Variant disagreement is a property of
   the model, not of grid mismatch. (Hypothesis "TTA agrees more at native resolution" —
   falsified.)
2. **Disagreement is HOMOGENEOUS.** Easiest-to-hardest spread is ~2× around a high median —
   there is no low-disagreement population for a gate to accept. An honest ε (near the
   ~2/255 label noise floor) stops nothing; a loose ε degrades labels uniformly, which is
   just a global tta reduction with extra machinery.

## Consequences
- Whole-scroll teacher passes run **fixed tta=8** (the measured quality lever); budget for
  0332 @ 2.4 µm ≈ 3.5 days on a 4×3090 box (166 s per occupied 1024³, TRT+global+pipeline).
- The high uniform member disagreement is itself a datum: single-forward (tta=1/2) outputs
  are FAR from the tta=8 mean everywhere (median 14/255). Do not treat low-tta runs as
  approximations of teacher labels, and do not distill from them.
- If per-patch difficulty signal is ever wanted (e.g. for distillation loss weighting),
  variant VARIANCE at fixed tta=8 could still carry information — but it does not enable
  compute savings at inference.

## Related
`docs/design/training-pipeline.md` (teacher = high-TTA soft targets), memory
`fenix-tta-teacher-ablation` (tta=48 octahedral teacher, judge in soft space),
`2026-07-06` overlap A/B (overlap=0.25 rejected: checkerboard seams; 0.33 borderline).

## Addendum: disagreement localization (same day)
Per-voxel |tta2−tta8| on 0332 @2.4 µm (z=16896 plane of the 1024³ bbox), vs CT structure:
edges (top-quartile |∇CT|) 22.7/255 · interiors 7.4 · gaps/air 13.1 · corr(d,|∇CT|)=0.26 ·
47% of disagreement mass within 2 px of a strong edge (33% of area) · confident topology
flips 0.17% of voxels (5 blobs >100 px). Heatmap: whole sheet segments light up coherently
by ORIENTATION while similar neighbors stay dark; air shows variant-dependent phantom prob.

**Reading:** dominant components are (a) boundary re-discretization under the transforms
(benign; TTA-mean cancels it) and (b) equivariance failure (orientation-dependent features;
TTA-mean symmetrizes it). Genuine topological uncertainty is negligible → the tta=8 mean is
a TRUSTWORTHY teacher. Student consequence: train with full octahedral aug (feed supports
exact 48-sym) — a student that learns the symmetrized function likely needs NO inference
TTA (8× on top of architecture savings). Verify post-training with the 15-min tta2-vs-tta8
protocol.
