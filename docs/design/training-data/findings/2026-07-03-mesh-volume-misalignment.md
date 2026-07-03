# Finding: some corpus mesh resamples are misaligned with their named volume

- Date: 2026-07-03 · Status: CONFIRMED (per-mesh QC sweep in progress)
- Impact: GT-only training on the mixed corpus stalls at chance — misaligned meshes are
  label noise. KD runs masked this (dense teacher term dominates), so it survived every
  earlier shakedown.

## Symptom
Paris4 from-scratch rehearsal (76 meshes, 2.4 µm native, tri-state GT, aug=2, clDice):
flat at chance for 3k+ steps — CE ≈ 0.74, `sep` ≡ 0, near-constant output, tiny grads.

## Diagnosis chain (each step reproducible)
1. Ring inspection: CT@sheet-label ≈ CT@background-label (delta ±0–5 on u8) on BOTH the
   augmented train ring and the clean (aug=0) val ring → not an augmentation bug. Val
   also had all-zero-CT patches carrying ~95k sheet-labeled voxels.
2. `fenix slice` overlay: GT bands render as coherent smooth surfaces CUTTING ACROSS the
   visible papyrus sheets (z and x views).
3. Bright-CT-vs-sheet-label Dice = 0.044 (`fenix eval --thresh 90 --gt-thresh 200`) —
   near-zero overlap with actual papyrus.
4. Eliminated: axis permutation (all 6 tested against raw zarr — only our (z,y,x)
   mapping is even fully in-volume); corpus corruption (fresh tifxyz import from S3 is
   identical: same grid/valid/bbox); wrong-volume pairing (all 81 meshes are
   "-on-20260411134726", the volume we feed); meta.json (bbox+scale match the import).
5. Discriminator: CT sampled at exact mesh points, per mesh —
   - `20230702185753` (2023 GP segment): **+13.7** brighter than surroundings = ALIGNED.
   - `20231007101619`: deltas mixed/negative across its densest chunks = MISALIGNED.
   → per-mesh, not systematic. The upstream "-on-<volume>" resamples vary in quality.

## Root cause
Upstream: the per-volume mesh resamples published in `segments/<id>/mesh/
<id>-on-<volume>-<um>.tifxyz` are not all actually registered to that volume. Meshes are
traced on ONE scan (often the 7.91 µm canonical for older segments) and machine-resampled
to the others; some transforms are off. forrest's hypothesis (open, testable once the QC
sweep lands): FAIL meshes correlate with origin frames — e.g. traced on the 7.91 µm
volume and resampled with a slightly-wrong scale/affine into the 2.4 µm frame. Check:
FAIL-vs-segment-id-era, and whether a small global affine per FAIL mesh recovers
alignment (if yes, we can fix locally instead of dropping them).

## Remediation
- **`fenix surf-qc <ct|cache@url> <fxsurf...> [k=] [off=] [min_delta=]`** (2df7e00):
  CT at K surface points vs ±off voxels along the local uv-normal; aligned sheets are
  bright with dark gaps alongside. PASS/FAIL per mesh, exit!=0 on any failure.
  RUN IT ON EVERY PAIRS FILE BEFORE GT TRAINING.
- Rebuild train/val pairs from PASS meshes only; rerun the rehearsal.
- Report the FAIL list upstream (ScrollPrize) once the sweep completes.
- QC sweep results (all 110 Paris4 corpus meshes, k=60 off=12 min_delta=5): PENDING —
  append here when done.
