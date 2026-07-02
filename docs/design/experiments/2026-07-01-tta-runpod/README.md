# TTA / inference-perf experiments — 2026-07-01 (RunPod RTX 5090)

Raw logs + driver scripts behind the findings in
[`../../ml-accel-and-distillation.md`](../../ml-accel-and-distillation.md) §7 and the
`perf(ml)` / `perf(eval)` commits. Archived here before the pod was shut down. The scroll
crops and prediction `.fxvol`s these ran on were NOT kept (re-pullable from the open-data
S3 zarr; teacher weights re-pullable from HF `scrollprize/surface_recto_3dunet` — see
`tools/ml-export/README.md`).

Hardware: RTX 5090 (32 GB, sm_120), 27-CPU cgroup budget. Crops: two disjoint 512³
PHercParis4 (A: z≈37k mid-scroll; B: z=50000), plus 1024³ for perf.

## Perf (lossless inference + metric multithreading)
- `eval_baseline_1024.log` — serial-metrics eval on 1024³ (2m46s baseline before the CC/EDT/Betti/VOI/NSD multithreading).
- `prof_infer.log` / `prof_sampler.log` — profiled 1024³ inference BEFORE the RAM/copy round (fwd 68.6s, scatter 2.9s, RSS 11.7 GB, VRAM 28.9 GB).
- `prof_infer2.log` / `prof_sampler2.log` — AFTER (fwd 67.0s, scatter 0.9s, RSS 7.6 GB): separable Gaussian weights + fp16 transfers.

## TTA teacher ablation (which augmentation combo → distillation teacher)
- `tta_ablation.sh` / `tta_ablation.log` — the 8-combo × 2-crop run grid (t0..t48ms, timings).
- `tta_eval.sh` / `tta_scores.log` — mask agreement vs the t48ms reference.
- `tta_gain.sh` — TTA-vs-single-pass gain (distance to best estimate): +0.17/+0.19 official, soft MAE ~3.5× lower.

## Arbitrary-rotation TTA (`rots=`)
- `rot_runs.sh` / `rot_runs.log` — r4/r8/t8r4 on both crops.
- `rot_eval.sh` / `rot_eval.log` — the cross-family divergence: rotation ensemble converges to a DIFFERENT mean than octahedral (MAE ~0.081) → real z-anisotropy + interpolation blur.

## Noise + tile-offset TTA (`noise=` / `offsets=`)
- `nz_runs.sh` / `nz_runs.log` — n4/n8/o3/t8n4 on both crops.
- `nz_eval.sh` / `nz_eval.log` — verdict: neither adds signal beyond octahedral (model is already noise/grid-robust; its dominant error mode is orientation). Octahedral t48 is the single-model TTA ceiling.
