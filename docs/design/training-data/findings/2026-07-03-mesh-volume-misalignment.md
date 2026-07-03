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

## Hand-editing caveat (forrest, 2026-07-03) — part of the issue
Surfaces are HAND EDITED: proofreaders place them correctly through damaged/low-contrast
papyrus where CT brightness is ambiguous — a correct edited surface can score LOW on
brightness QC there, while a (wrong) model prediction scores high. So low delta ≠
misaligned, and over-filtering on brightness would systematically drop the hand-corrected
regions — the exact signal GT has that the teacher lacks. Three-tier model:
1. frame-level misregistration (offset/scale/affine) — the real defect; conservative QC
   threshold + visual overlay catch it; unusable as-is (or repairable, see below);
2. correct-but-dim hand-edited regions — MUST NOT be filtered; lenient threshold;
3. out-of-frame variants in the corpus dir (7.91/45/1.129 files) — never in pairs.
QC cut is set from the sweep's delta distribution calibrated to tier 1 only (the known-
bad mesh anchors the bad cluster); borderline = keep-with-flag, not drop.

## Era rule (forrest, 2026-07-03)
**Prefer 2025/2026 meshes, especially for Paris4** — those were traced NATIVELY on the
2.4 µm volume. Older segments (2023/2024) were traced on the 7.91 µm canonical scan and
machine-resampled into the 2.4 µm frame — the resample transform is where the
misalignment risk lives. Segment ids are date-prefixed (YYYYMMDD...), so the era filter
is a string prefix check in gen-pairs tooling.

## THE TRAINING KILLER (root-caused 2026-07-03, controlled experiment)
Isolation on the best-QC mesh (clean q=8 cache, aug=1, dice+CE only) still trained to a
constant ⇒ trainer-side. Slot forensics: CT gather EXACT (slot-vs-zarr MAE 2.6 = codec
noise); GT band on-sheet vs ALL voxels = **+12.5** (mesh meaningfully aligned!) but
on-sheet vs the BACKGROUND SHELL = **+1.3** ⇒ **the geometric "trusted background" shell
(±16 vox along the normal) lands ON the neighboring wraps** — at 2.4 µm wraps sit
~15-40 vox apart. Bright sheet texture gets labeled background as often as sheet → the
classes are statistically identical → constant collapse. Not a code bug: a labeling-
geometry assumption that only breaks at fine resolution + tight winding. (Secondary,
same forensics: this mesh runs ~10 vox off sheet-center — shift search doubled the
delta at (+10,0,+10) — the per-mesh QC/offset story remains real but was not the killer.)
FIX: the feeder INTENSITY-GATES the shell — background keeps only voxels darker than
halfway between patch mean and sheet mean; bright shell voxels → unlabeled-ignore.
Robust to untraced neighbor wraps AND modest mesh offsets.

## Remediation
- **`fenix surf-qc <ct|cache@url> <fxsurf...> [k=] [off=] [min_delta=]`** (2df7e00):
  CT at K surface points vs ±off voxels along the local uv-normal; aligned sheets are
  bright with dark gaps alongside. PASS/FAIL per mesh, exit!=0 on any failure.
  RUN IT ON EVERY PAIRS FILE BEFORE GT TRAINING.
- Rebuild train/val pairs from PASS meshes only; rerun the rehearsal.
- Report the FAIL list upstream (ScrollPrize) once the sweep completes.
- QC sweep results (all 110 Paris4 corpus meshes, k=60 off=12 min_delta=5): PENDING —
  append here when done.
