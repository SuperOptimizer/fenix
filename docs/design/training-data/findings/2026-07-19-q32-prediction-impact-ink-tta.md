# q=32 compressed-CT impact on surface/ink prediction + ink TTA ablation

Date: 2026-07-19 · Box: Hot Aisle 1× MI300X VF (enc1-gpuvm015), ROCm 7.2.4, torch
2.13.0+rocm7.1 · Scroll: PHerc0332 2.4 µm (20251211183505) · bbox z16384 y7168 x7168 + 1024³
(TTA sweep: z16640 y7424 x7424 + 512³) · tta=8 batch=24 patch=256 mode=global q=32 unless
noted. Artifacts: ~/fenix-pod-archives/hotaisle-mi300x-20260719/ (preds + renders + logs).

## 1. Raw S3 CT vs dct3d-q=32 community-export CT → prediction deltas

Same bbox, same weights, only the CT source differs (raw open-data zarr vs
dl.ash2txt.org/community-uploads/forrest/exports dct3d q=32 zarr-v3 sharded).
Compare is in prediction space (u8 0–255):

| model | PSNR | MAE | max-abs | visual |
|---|---|---|---|---|
| surface_recto_3dunet | 23.22 dB | 12.57 | 142 | same topology; ridges softer, faint inter-sheet haze |
| ink_3d_dino_guided | 24.11 dB | 3.87 | **255** | dimmer; some blobs attenuated or **missing entirely** |

Verdict: **surface survives q=32** (amplitude/sharpness changes only — usable for bulk
prediction and winding). **Ink does not**: max-abs 255 = detections fully flip. The ink
model is the fine-tune candidate (train-feed already has a `compression` aug op);
surface fine-tune is optional polish. Consistent with the earlier q=32 CT finding
(inter-sheet fuzz, not sheet corruption) — the fuzz lands exactly where the ink net
reads surface texture.

## 2. Ink TTA ablation (tta=1 / 8 / 48, 512³, local ctcache CT — no network)

Runtime 7 s / 27 s / 139 s. vs tta=8: tta=1 MAE 2.45 (PSNR 27.7), tta=48 MAE 2.78
(PSNR 27.5) — tta=48 differs from tta=8 as much as tta=1 does (averaging jitter, not
convergent refinement).

**Every detection present at tta=48 is already present at tta=1** — TTA buys ink
smoothing/denoising, never new or removed detections (unlike surface, where TTA was the
biggest measured quality lever). No-GT caveat: this is change-magnitude + visual
evidence on one 512³ region.

Bulk-ink recommendation: first pass (pre-fine-tune, reconnaissance) at **tta=1** (8×
cheaper, finds the same spots); definitive post-fine-tune pass at tta=8. tta=48 never
pays.

## 3. Feed/perf ladder on a thin-CPU box (8 vCPU)

Per 1024³ bbox, tta=8 surface: raw-S3 + eager **554 s** → q=32 export + eager **337 s**
(GPU-bound end-to-end) → q=32 export + AOTI .pt2 (batch=16, --no-max-autotune) **286 s**
(cumulative 1.94×). Raw-zarr fetch on this box degenerated to serial ~1 MB/s
single-chunk GETs with the GPU at 0% between groups — the compressed exports are the
correct bulk feed wherever the box is CPU- or network-thin. predict-scroll grew
`occ=<zarr-root>` (+ a size cap) because the exports ship only levels 0/1 and the
occupancy loader would otherwise try to fetch a TB-scale level 1 (crash: silent, via
-fno-exceptions terminate on bad_alloc).

.pt2 export validation vs eager: surface corr 0.999969, ink corr 0.999997 (--ink lane
added to export_aoti.py).
