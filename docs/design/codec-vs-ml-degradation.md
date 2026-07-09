# DCT compression vs surface-prediction inference (2026-07-09)

How much does the fenix DCT-16 volume codec degrade ML surface prediction as
compression rises — and can compression-augmented training claw the accuracy back?

**TL;DR.** Off-the-shelf, the surface teacher tolerates the codec up to **q≈4 (~32×)**
with near-lossless inference, then falls off a cliff (SD@2 0.71 by q=128). A short
(2500-step) fine-tune on **real-codec-compressed inputs, self-distilled to the clean-input
teacher**, moves the safe limit to **q≈32 (~162×)** and holds **SD@2 0.91 even at q=128
(395×)** — an ~8× compression-headroom gain at matched fidelity. The learned robustness
**generalizes to compression levels never trained on** (q=64/128 got the largest clawback).
TTA is NOT a substitute: it polishes low-q but *hurts* at high q.

## Setup

- **Source**: PHercParis4 (Scroll 4) 2.4µm, level 0 `[75784,32693,32693]` u8. One
  **1024³ chunk** at origin z30000/y14000/x16000 (mean 75.8, std 33.9, 62% of voxels in
  the papyrus band — real sheet structure, uniform z-occupancy).
- **Codec sweep**: ingest at q=1 (near-lossless reference), then `fenix transcode` to
  q ∈ {2,4,8,16,32,64,128}. `fenix compare` for PSNR/MAE/max-abs; `fenix fxinfo` for ratio.
  New tool: **`fenix export-npy`** (decode .fxvol LOD0 → NumPy) is the .fxvol→array egress
  the Python side needed.
- **Inference**: `surface_recto_3dunet` teacher, fp16 throughout (no fp8/int8 — isolates
  the CODEC's effect). 128³ patches on a lattice over the cube, z-scored per patch.
- **Metric**: SD@2 (+ prob-corr, Dice) of each q's prediction vs the **q=1 prediction**.
  Measures prediction *stability under compression*, not vs ground truth (GT meshes not
  yet staged — deliberately out of scope; teacher-agreement is the codec-isolation metric).

## Result 1 — codec distortion (pure signal fidelity)

| q | ratio vs u8 | PSNR (dB) | MAE | max-abs |
|---|---|---|---|---|
| 2 | 19× | 44.8 | 1.10 | 20 |
| 4 | 32× | 41.0 | 1.75 | 40 |
| 8 | 55× | 37.7 | 2.53 | 60 |
| 16 | 94× | 34.9 | 3.47 | 76 |
| 32 | 162× | 32.1 | 4.72 | 93 |
| 64 | 264× | 29.6 | 6.22 | 142 |
| 128 | 395× | 27.3 | 7.98 | 173 |

Monotonic, well-behaved. The codec is doing its job; the question is what the ML does with it.

## Result 2 — single-pass inference degradation (base teacher)

| q | SD@2 | SD@2 **min** | corr | Dice |
|---|---|---|---|---|
| 2 | 0.9995 | 0.993 | 0.998 | 0.984 |
| 4 | 0.9924 | 0.862 | 0.991 | 0.964 |
| 8 | 0.9593 | **0.424** | 0.964 | 0.923 |
| 16 | 0.8960 | 0.375 | 0.921 | 0.877 |
| 32 | 0.8700 | 0.502 | 0.910 | 0.858 |
| 64 | 0.7884 | 0.388 | 0.865 | 0.807 |
| 128 | 0.7065 | 0.301 | 0.794 | 0.743 |

- **q≤4 (32×) is free.** SD@2 ≥ 0.99. Already better compression than the q=1 "reference".
- **q=8 is the knee.** Mean 0.96 looks fine but the **min collapses to 0.42** — damage is
  not uniform; it concentrates in thin/faint/high-frequency sheet patches where the codec's
  HF quant bites and the surface signal is already marginal. **Mean hides the cliff; watch min.**
- **q≥16 degrades steadily** to 0.71 at 395×.

## Result 3 — TTA (8-flip) is a low-q polish that backfires high

Corrected 8-flip TTA (q=1 TTA-vs-single agreement 0.92, sane). Δ = tta − single:

| q | single | tta8 | Δ |
|---|---|---|---|
| 4 | 0.9924 | 0.9994 | **+0.007** (min 0.86→0.99) |
| 8 | 0.9593 | 0.9677 | +0.008 |
| 16 | 0.8960 | 0.8483 | −0.048 |
| 32 | 0.8700 | 0.7684 | −0.102 |
| 128 | 0.7065 | 0.5103 | **−0.196** |

TTA helps at q≤8 (rescues low-q worst cases) but **hurts from q≥16**: the DCT operates on
axis-aligned blocks, so flipped views see *different* codec artifacts; averaging inconsistent
predictions smears the surface. TTA assumes label-preserving augmentation — heavy compression
breaks that. **TTA is not a fix for compression damage.**

## Result 4 — compression-augmentation training CLAWS IT BACK

**Fine-tune** (2500 steps, fp16, lr 2e-5, init from teacher): input = patch at a random
train-q ∈ {2,4,8,16,32} via the **real codec**; target = frozen teacher's prob on the
**clean (q=1)** patch (self-distillation, no GT). **Honest held-out**: train patches
x<512, eval patches x≥512 (disjoint) — clawback ≠ memorization. q=64/128 excluded from
training to test generalization.

| q | ratio | base SD@2 | **aug SD@2** | clawback | base min | aug min |
|---|---|---|---|---|---|---|
| 4 | 32× | 0.9849 | 0.9947 | +0.010 | 0.813 | 0.972 |
| 8 | 55× | 0.9533 | **0.9887** | +0.035 | 0.636 | 0.900 |
| 16 | 94× | 0.8613 | **0.9771** | **+0.116** | 0.327 | 0.857 |
| 32 | 162× | 0.8638 | **0.9565** | **+0.093** | 0.063 | 0.746 |
| 64 | 264× | 0.8104 | **0.9381** | **+0.128** *(unseen)* | 0.436 | 0.687 |
| 128 | 395× | 0.7827 | **0.9114** | **+0.129** *(unseen)* | 0.423 | 0.633 |

1. **The cliff is gone.** Safe limit q≈4 → q≈32; SD@2 ≥ 0.95 out to 162×, still 0.91 at 395×.
2. **Worst-case recovery is the story.** q=32 min 0.063 → 0.746; q=16 min 0.327 → 0.857.
   It rescued the catastrophically-wrong patches, not just the average.
3. **Generalizes to unseen compression.** q=64/128 never trained yet got the *largest*
   clawback — the net learned the DCT *artifact class* (blockiness/DC-banding/HF-loss), not
   a per-q lookup. Robustness extrapolates.
4. **Residual slope = deleted information.** Aug curve still declines 0.98→0.91 (q8→q128):
   sheets the codec dead-zoned to zero can't be trained back. Knee moved ~8×; floor stays <1.

## Caveats

- **Teacher-agreement, not ground truth.** This measures prediction stability under
  compression. The high-q end is where a net could learn to *hallucinate* the deleted sheet
  (looks like recovery vs teacher, wrong vs reality). Trust the mid-range (q≤32, signal
  intact) most; re-validate the high-q end with real GT when staged.
- **The built-in `augment.hpp compression()` is a STAND-IN** (8³ block-mean DC quant + HF
  attenuation), not our DCT-16 codec (16³ seams, RDOQ, deblock). This experiment used the
  **real codec** via transcode+export-npy. For the production training run, augment with the
  real codec, not the stand-in.
- fp16 inference throughout (no fp8/int8) to isolate the codec.

## Takeaway for deployment

Compress-aug is a large, cheap lever. Folding **real-codec** compression augmentation into
the surface training run should let the deploy codec run **~8× more aggressive (q≈32 vs q≈4)
at no measured inference cost** — a direct multi-TB storage/bandwidth win for whole-scroll
inference. Do it as part of the next real-GT training run (GT improvements in prep), and
re-validate the high-q end against meshes.

## Artifacts

`/tmp/dctsweep/`: `codec_metrics.csv`, `infer_metrics.json`, `tta_metrics.json`,
`aug_metrics.json`; `sweep_infer.py`, `sweep_tta.py`, `aug_finetune.py`; the q=1 + q{2..128}
`.fxvol`/`.npy` volumes. New CLI: `fenix export-npy` (src/io/io.hpp).
