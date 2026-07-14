# Distilled surface-prediction student — sizing & training plan (2026-07-08)

> **STALE PLAN — distillation is DEAD (2026-07-13).** The teacher traces at half our
> from-scratch students' recall (arbiter table below stands as the record); rungA distill
> reached only ~70% of its already-weak teacher. Production = from-scratch StudentUNet
> base-16 on the graded GT corpus (M8 recipe, studentH), deployed via TRT fp16.
> Authoritative record: memory/student-lineage-verdicts.md. Keep for history; do not
> execute the plan below.

Teacher: `surface_recto_3dunet` ResEnc-UNet, **142.2M params** (the "280" is the
fp16 checkpoint's MB). 7 encoder stages, widths 32-64-128-256-320-320-320,
blocks 1-3-4-6-6-6-6, SE per block; surface decoder = transpconv + 1 stacked
conv per stage. Deployed today via the int8 resident lane at **41 ms/patch**
(128³, 5060 Ti, SD@2 0.9989 vs fp16).

## The sizing insight

Params and milliseconds live in DIFFERENT places:

| | stages 0-2 (128³-32³) | stages 3-6 (16³-2³) |
|---|---|---|
| params | ~8M | ~120M + decoder |
| runtime at 128³ | **dominant** (memory-bound, huge M) | minor (TC-dense) |

So: cutting deep blocks/width shrinks the checkpoint but not the sweep time;
cutting shallow width cuts time ~quadratically. The student must do BOTH —
params matter for the fleet's VRAM/load times, ms matter for the full-scroll
sweep (PHerc0332 @2.4µm ≈ 8.3T voxels; at 41 ms/patch a full pass is ~45 GPU-h
— the student is what makes whole-scroll inference routine).

## Candidate ladder (train all three, pick by gate)

Param model validated against the real net (142.4 computed vs 142.2 actual).
FLOP ratio = compute proxy at 128³ (time upper bound; memory-bound floor means
realized speedup is lower, measure per rung).

| rung | widths (6 stages) | blocks | params | FLOPs vs T | expected ms/patch* |
|---|---|---|---|---|---|
| A "swift" | 16-32-64-128-160-160 | 1-2-3-3-3-3 | **14.4M (10×)** | 5.5× | ~10-14 |
| C "middle" | 24-48-96-128-192-192 | 1-2-2-3-3-3 | 19.5M (7×) | 2.9× | ~14-18 |
| B "safe" | 32-64-128-192-256-256 | 1-2-3-4-4-4 | 46.5M (3×) | 1.4× | ~25-30 |

*floor set by the norm/tail memory passes, not conv FLOPs; measure, don't trust.

- **6 stages, not 7** (deepest 4³ at 128³): stage 7 adds 33M params for a 2³
  bottleneck — negligible receptive-field gain, the decoder skip does nothing
  at that size.
- **Primary bet: rung A.** Dense per-voxel KD is a strong signal; 10× is inside
  the range segmentation students routinely absorb. C is the fallback, B the
  "can't-fail" rung. Sweep cost ~3 h/rung (50k steps @ ~218 ms) — train all.

## Architectural changes: almost none, deliberately

KEEP (the whole int8/TRT/fxweights stack transfers unchanged):
- BasicBlockD/CDNR module topology, InstanceNorm+LeakyReLU, k=3, stride∈{1,2},
  transpconv decoder, SE blocks (fp16 CastConv, cheap). The fused-kernel swaps,
  structural consumer binding, epilogue stats, resident PreQuantI8 path, and
  the C++ .fxweights adapter all key on this structure.
- **Channel widths: multiples of 16, prefer 32** (MMA tiles; sign bitpack needs
  C%8==0; block-sparse packs are 32-wide groups). All rungs above comply
  except rung C's 24/48 — if C is trained, round to 32/48 (both %16==0) — 24
  is %8-legal but wastes MMA lanes; use 32-48-96-128-192-192.

DO NOT (measured dead ends, do not relitigate):
- Separable/factorized convs (2.6× SLOWER — memory-bound 1D passes eat the
  FLOP cut; sep_student.py verdict).
- Attention/transformer blocks at 128³ (memory-bound, breaks int8+TRT lanes).
- Ternary/BitNet (int8 speed at best on GPU), int4/W4A8 (no conv3d path).

CONSIDER LATER (orthogonal, after a rung passes):
- 2:4 sparsity on the deep stages for the 3090/TRT lane (deploy-only win).
- precision_assign re-run per rung (teacher: ALL-int8 passed outright;
  student's thinner channels may want 1-2 fp8/fp16 layers — the tool decides).

## KD training protocol (everything already built)

- **Recipe: QAT in deploy precision from step 0** — `--int8qat --bwd-fp8`
  (sm120) / `--bwd-fp16` (Ampere, zero bwd quantization). Delayed scaling for
  fwd activations, fresh amax for all bwd operands (measured doctrine).
- **Teacher on GPU1, CUDA-graphed, 1-step pipeline** (475 ms/iter floor,
  teacher fully hidden behind the ~218 ms student step).
- **Init: L1 channel slicing** of the teacher — for each student conv take the
  top-|w| teacher channels (warm start beats random for width-pruned students;
  cheap to implement in the harness).
- **Loss:** per-voxel KL (T=2) on surface logits + Dice on GP fit labels
  (confidence-weighted, from the labelstore) + stage-wise feature hints:
  1×1 fp16 adapter convs at the 3 decoder skips, MSE to teacher features —
  hints matter at ≥7× compression, drop them for rung B.
- **Data:** real CT crops (the e2e crop feeder), NOT phantoms; add flip TTA of
  the teacher's targets only if the plain run gates below bar.
- **Schedule:** 50k steps Adam 1e-4→cosine, ~3 h/rung on GPU0.

## Gates (per rung, in order)

1. e2e loss-curve twin sanity (existing harness, 1500 steps).
2. **SD@2 ≥ 0.995 + corr ≥ 0.995 vs TEACHER int8 output** on held-out real CT
   patches (deploy-precision student vs deploy-precision teacher).
3. `trace-eval` on GP segment meshes — the downstream metric that actually
   matters (surface completeness/accuracy, not voxel agreement).
4. ms/patch on the int8 resident lane (5060 Ti) + TRT int8 engine (3090):
   rung A must land ≤ 15 ms/patch to justify itself over C.

## Open questions

- Does the 41→~12 ms win survive the norm/tail memory floor? (Measure rung A's
  fused-lane profile; if norm passes dominate, the N-aware batched path
  becomes the next lever — amortize fixed cost over N patches.)
- Warm-start value: ablate L1-slice vs random init on 5k steps (one afternoon).
- Whether the student should also emit the ink/normal heads (multi-task student
  for the full-scroll net) — out of scope here; this plan is the surface net.

## Run 1 results (rung A, 50k steps, 2026-07-08)

- **Capacity CONFIRMED**: fp16 lane KD 0.0024-0.014 — near teacher parity at 14.4M.
- **Pure int8-QAT UNSTABLE at this width**: stable ~0.03 to step 31k, degraded to
  0.17-0.21 by 44k, ended 0.08. Thin channels (16-64) are quantization-fragile:
  per-tensor scales average over few channels; outliers dominate.
- Training speed: 66-73 ms/step (3× the 218 full-size); teacher hidden.
- Weights of run 1 discarded (harness pre---save); identical rerun with
  --save models/students/rungA_50k.pth in flight.
- **Deploy path forward**: fp16-trained student + precision_assign mixed
  precision (deep/wide layers int8, thin shallow layers fp8/fp16), NOT pure
  int8-QAT. Gate SD@2 both variants when the saved artifacts land.

## Run 1 gate (rungA_50k.pth, SD@2 vs teacher-int8, 128³ 5060 Ti)

| lane | fp16 SD@2 | int8-resident SD@2 | ms/patch |
|---|---|---|---|
| **student16** (fp16-trained → quantized) | 0.9434 | 0.9404 | 16.2 |
| **student8** (pure int8-QAT) | 0.8666 | 0.8650 | 16.2 |

Reading:
- **fp16-train-then-quantize beats int8-QAT by ~0.075 SD** — QAT's thin-width
  instability (run-1 note) carries into the deployed weights. **Do not QAT the
  student**; train fp16, quantize post-hoc (only 0.003 SD lost fp16→int8).
- **int8-resident student = 16.2 ms/patch = 2.5× the teacher's 41.1 ms**, same
  128³ patch. That's the compression win.
- **BUT SD@2-vs-teacher = 0.94, below the 0.995 parity gate.** At 10× fewer
  params this is expected; 0.94 is the *distillation-agreement* number, not
  ground truth. Per the gate ladder the real arbiter is trace-eval on GP
  meshes — 0.94 agreement may still trace acceptably, or may not. Decision
  pending: (a) trace-eval rung A on GP meshes before judging, and/or (b) step
  up to rung C (19.6M, 7.3×) for a fidelity/speed midpoint, and/or (c)
  precision_assign to recover SD on the thin shallow layers. Rung A alone does
  not clear strict teacher-parity.
- Gate loader note: student8 was saved post norm-swap (`gamma`/`beta`, conv
  biases norm-absorbed) — load requires renaming back + zero-filling the
  absorbed biases (InstanceNorm is shift-invariant, so bias is a no-op).

## Trace-eval arbiter (2026-07-11, `tools/ml-export/rung_trace_eval.py`) — PLAN OBSOLETED

Ran the promised arbiter on the 12 Paris4 holdout crops (graded-corpus GT meshes,
same protocol as `tools/train/trace_eval_run.py`):

| model | recall@2 mean/min | recall@4 mean/min |
|---|---|---|
| teacher (surface_recto_3dunet) | 0.107 / 0.000 | 0.242 / 0.013 |
| rungA (student16 lane) | 0.076 / 0.016 | 0.173 / 0.096 |
| **StudentUNet studentG (M8, base 16)** | **0.252 / 0.162** | **0.535 / 0.440** |
| StudentUNet studentM (M10 multi-scroll, base 32) | 0.236 / 0.153 | 0.504 / 0.380 |

rungA ≈ 70% of its teacher end-to-end — consistent with the 0.94 SD agreement. But
the finding that matters: **the teacher itself traces at half the recall of our
from-scratch graded-corpus StudentUNets of the same size class** (base-32 ≈ 11M vs
rung A 14.4M). Distilling this teacher is distilling a worse model than we already
train from scratch. Caveat: single protocol/domain (Paris4 2.4µm crops, our GT
convention); the teacher may retain an edge elsewhere. Decision anyway:
**retarget the deploy-precision pipeline (int8-resident / fp8 / TRT lanes) at the
StudentUNet lineage** — port the tri-dtype resident kernels to that arch instead of
pushing more KD rungs off surface_recto_3dunet.
