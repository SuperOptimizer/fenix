# Surface student KD: half-res b64 beats full-res b48; dice 0.64 (Thunder A6000, 2026-07-22)

Companion to [2026-07-20-ink-compression-finetune.md]. Goal: a ~12× smaller surface
student (like the ink student) to rewrite the bulk-surface fleet budget. Method identical
to the ink KD lane: `finetune_ink_compression.py --task surface` — teacher =
`surface_recto_3dunet` at tta=8 on raw CT (`NAME.steacher.npy`), paired {raw, q32}
patches, anchored soft BCE + one-way consistency, zscore input norm (the C++ surface
contract). Data: 25 train regions + r12288a val (768³ PHerc0332). Referee =
`student_infer_eval.py --task surface`: sliding-window 128³/stride-96, sheet-dice@0.5
vs teacher>0.5 (blob recall is meaningless for sheets; MAE/PSNR reward blur — dice is
the scoreboard).

## Results (val region r12288a, q32 input unless noted)

| student | params | steps × batch | val MAE | sheet-dice@0.5 | best-thr dice | infer/768³ |
|---|---|---|---|---|---|---|
| v1: base=64 **stem=2** | 22.7M | 12k × 24 | 27.9 | **0.640** (raw 0.641) | 0.640 @128 | 61 s |
| v2: base=48 **stem=1** | 12.8M | 12k × 6 | 35.97 | 0.476 (raw 0.478) | 0.545 @96 | 64 s |

Threshold sweep (q32, pred>thr vs teacher>128): v1 = 0.640/0.637/0.544/0.430 at
128/96/64/32; v2 = 0.476/0.545/0.470/0.383. Dimming explains part of v2's headline gap
(it peaks at thr 96) but not the ranking — v2 loses at every threshold.

Visual (z=384 slices, `~/fenix-renders/hotaisle-smoke/surf_{teacher,stu1,stu2}_z384.png`):
teacher = thin crisp continuous ridges. v1 = correct wrap topology, mildly blurred/dimmer
ridges, faint sliding-window seams. v2 = mushier, more background haze, broken ridges —
worse on every axis. Both students are compression-invariant by construction
(q32-vs-raw dice spread ≈ 0.001–0.002).

## The stem design rule, revised

The 07-21 hypothesis was "strided stems for blob tasks (ink), full-res stems for ridge
tasks (surface — 3–5 voxel ridges blur at half-res)". **Not confirmed at fixed
wall-clock.** The full-res stem's activation memory forced batch 6 (42 GiB) vs the
half-res net's batch 24 (17 GiB), so v2 saw ~¼ the samples in the same 12k steps and
was still far from converged (val MAE falling through the end). The batch/throughput
cost of full-res hurt more than half-res blur did — and full-res bought no inference
speed either (64 s vs 61 s: the b48 full-res stem alone rivals the whole b64 half-res
net in FLOPs). Rule as amended: **half-res stem + trilinear upsample is the default for
BOTH tasks; a full-res stem must justify a ~4× effective-data cut, which it didn't.**

## 24k continuation + Gaussian blending (same day)

Resuming v1 12k→24k (fresh cosine tail, batch 24) moved val MAE 27.9 → 26.6 and dice
0.640 → **0.646**; Gaussian window blending (σ=patch/4, `--gauss`) added another
+0.008 → **0.654** (raw 0.655; q32-vs-raw spread still ≈0.001). Diminishing returns:
the 22.7M student converges at **dice ≈ 0.65 vs teacher**. More steps are not the
lever; if 0.65 is insufficient for the winding data term, the next levers are wider
students (b96 s2) or more/harder training regions — but measure winding-side first.

## Status / next
- Open: is dice ~0.64-vs-teacher sufficient for the winding data term? (The consumer is
  the diffeomorphic fit's dense term, which wants ridge position more than calibrated
  amplitude — needs a winding-side ablation, not more KD polish.) Gaussian-blended
  window averaging (vs uniform) would remove the visible seams cheaply.
- ~~No production loading path yet~~ **CLOSED**: `tools/ml-export/export_student_ts.py`
  packages EMA checkpoints as .ts for fenix's JitNet path with zero C++ changes (surface
  exports cat([0, logit]) so softmax ch1 == sigmoid; ink exports the raw 1-ch logit).
  End-to-end smoke: `fenix predict-surface r12288a.q32.fxvol surf_student_b64.ts out 128
  0.25` → **dice 0.6492** vs teacher — matches the python referee (0.6462 uniform).
  Deploy gotcha: the .ts path invokes TorchScript's runtime kernel fuser, which dlopens
  `libnvrtc-builtins.so` — absent from fenix's loader path when linking a pip libtorch.
  Fix: `env LD_LIBRARY_PATH=<dist-packages>/nvidia/cu13/lib fenix predict-...` (scoped
  per-command; .fxweights inference never needed this).
