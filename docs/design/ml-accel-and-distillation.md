# ML inference acceleration + distillation — design notes

Status: **planning**. Author intent: make `predict-surface` (and later ink) dramatically faster
without losing surface quality — via quantization (int8/fp8), a distilled smaller student, and
test-time augmentation (TTA) — AND build the metrics to *prove* each step helps.

This doc is grounded in a code audit (2026-07-01) + hardware probing on the RTX 5090 box. It
records what exists, what the hardware/toolchain actually supports, and a staged plan.

---

## 0. The load-bearing finding: we cannot currently measure quality

Everything below is gated on this. Today fenix has **no way to score a surface prediction against
a reference**:

- The eval metric *primitives* exist and are unit-tested — `official_score` (0.30·TopoScore +
  0.35·SurfaceDice@2 + 0.35·VOI), NSD/SurfaceDice (`eval/nsd.hpp`), VOI (`eval/metrics.hpp`),
  Betti (`topo/betti.hpp`), GT-free MeshQuality (`eval/mesh_quality.hpp`).
- BUT: the `fenix eval` stage is a **stub** (`eval/eval.hpp:15`), none of these has a CLI entry,
  none is fed a (prediction, GT) pair by any harness, there is **no ground-truth surface dataset**
  referenced, TopoScore is only the number-based Betti *proxy* (exact cubical PH is claimed in
  docs but absent), and the "TTA helps +0.1–0.17" measurement lives in session memory, not code.
- There is **no in-tree training** (no loss/optimizer/backward) — the `ml` module is inference-only.
  Distillation therefore also needs a training loop built (or done out-of-tree in Python).

**Rule: no accel/distill change lands without a metric that shows its quality delta.** Phase 1 is
building that metric harness. Without it, "is int8 good enough?" and "does TTA help?" are unanswerable.

---

## 1. Hardware / toolchain reality (measured on the box)

RTX 5090 = **sm_120 (Blackwell)**, driver 580.159, torch **2.8.0+cu128**, cuDNN 9.

| dtype | torch 2.8 exposes it? | works for our net (3D **conv**)? |
|---|---|---|
| fp16 | yes | yes (current baseline) |
| bf16 | yes | yes — but **same speed as fp16** (25 vs 26 ms microbench); no win |
| int8 | yes (`torch.ao.nn.quantized.Conv3d`) | quantized conv3d module exists; PTQ path viable |
| fp8 e4m3/e5m2 | yes (dtypes + `_scaled_mm`) | **NO — cuDNN has no fp8 conv** (`getCudnnDataTypeFromScalarType() not supported for Float8_e4m3fn`). fp8 only does matmul, not conv. |
| fp4 (`float4_e2m1fn`) | **NO** (not in torch 2.8) | n/a |

**Consequences:**
- Our net is **conv3d-heavy**, so the dtypes that matter are the ones with fast conv kernels.
- **Raw torch/cuDNN gives us int8 conv but NOT fp8/fp4 conv.**
- **TensorRT is the unlock for fp8/int8 3D convs on Blackwell** — TRT 10.x ships hand-tuned
  low-precision 3D conv kernels for sm_120 that cuDNN lacks. TRT is **installable** here (driver
  580 supports it; `torch.onnx` export works; TRT/torch-tensorrt not yet installed).
- **torch upgrade** (2.9+/nightly): would add newer quant APIs and possibly `float4`, but does NOT
  fix the cuDNN fp8-conv gap — that's a kernel-library gap, not a dtype-exposure gap. Upgrade only
  buys us much if we also go TensorRT. Recommendation: **install TensorRT first**; upgrade torch
  only if a specific API (e.g. `torch.export` → TRT, or fp4) requires it.
- **Driver upgrade (580 → 595): NOT possible in-place, and NOT needed.** The box is a Docker
  container (`/.dockerenv`); the NVIDIA driver is a host-injected kernel module (580.159.03), no
  `lsmod`/module access — the host (RunPod) controls it. Getting 595 means **selecting a RunPod
  template that ships it**, not upgrading inside the container. And it buys nothing on the critical
  path: driver 580 already supports CUDA 13, sm_120, and TensorRT 10.x with fp8/int8 (TRT needs
  driver ≥545 for Blackwell). The blocker is cuDNN's missing fp8 conv, which TRT solves on 580.
  → Flag 595 for the next instance if convenient; do not block on it.

---

## 2. What exists to build on

- **Net** (`nets/resenc_unet.hpp`): 7-stage ResEnc-UNet, features `32/64/128/256/320/320/320`,
  blocks `1/3/4/6/6/6/6`, **InstanceNorm3d** (per-instance, NOT BatchNorm → no static norm-fold;
  PTQ calibration must handle per-instance stats), scSE attention. **Trained at patch 256³** (from
  `config.json`: `ps256_bs2`). All feature channels are multiples of 16 (tensor-core friendly);
  only the 1-ch input and 1/2-ch output convs are unaligned (tiny FLOP fraction).
- **Weights** (`.fxweights`, `weights.hpp`): dense f32/f16/bf16, keyed by name, **no scale/zero-point
  fields**, reader **hard-rejects version≠1**. Quant needs: new dtype codes + a scale companion +
  a version bump. Clean insertion points: `load_into` (casts every tensor to the module dtype,
  weights.hpp:96) and `net->to(dtype)` (infer.hpp:87).
- **TTA**: octahedral (`tta` count over the 48-element axis-symmetry group, infer.hpp:211-249,
  serial-only — batching disabled when tta>1) + multiscale (`scales`, `predict_surface_scales`,
  infer.hpp:299-330, mean-fuse, stacks with octahedral). Both implemented; **neither is measured**.
- **Reference harness** (`tools/ml-export/reference.py`): gen input → run authoritative PyTorch net
  → run C++ net → cmp. This validates the C++ *reimplementation* is bit-exact. **Reuse it as the
  quant-error harness** (teacher-vs-quantized numeric delta) — but note it measures numeric
  equivalence, not surface *quality*.
- Perf baseline (all lossless, overlap=0.5 kept): 1024³ 116s→76s via decode-once-u8
  + batch=3 + prep/forward pipelining. GPU is compute-saturated (~500W/575W, full boost). fp16 is
  the floor for lossless.
- **Second lossless round (measured, full profiled 1024³ run):** separable Gaussian blend weights
  (the tile grid is Cartesian ⇒ wacc = Wz·Wy·Wx exactly — dense weight volume eliminated), parallel
  normalize, fp16 H2D+D2H (CPU-side casts, same rounding), no input clone, source freed pre-encode.
  Wall 72s→68s (fwd 68.6→67.0s, scatter 2.9→0.9s, decode-in 0.4s, write-out 1.4s); **peak RSS
  11.7→7.6 GB (−35%)**; VRAM ~29 GB (unchanged, batch=3). Output scored vs the previous binary's
  prediction: official = 1.00000 on every component — result-preserving confirmed. Wall is now 98%
  GPU forward: the lossless ceiling is reached; further speed comes from Phase 3 (distillation).

---

## 3. Staged plan

### Phase 1 — Measurement harness + overfitting guards (BLOCKING; do first)
Goal: a reproducible number for "how good is this prediction," on data we did NOT tune on. Without
it nothing else is decidable, and *with a bad split* we'd fool ourselves.
1. **`fenix eval` CLI** wiring the existing primitives: `fenix eval <pred.fxvol|.nrrd> <gt>
   [--metrics official,nsd,voi,dice,betti]` → prints the composite + components. (Implement the stub
   `eval/eval.hpp`.)
2. **A ground-truth set.** Options: (a) proofread surface labels from ScrollPrize/villa (import via
   the VC importer); (b) treat the **teacher (fp16, full TTA)** prediction as a pseudo-GT for
   *relative* quant/distill deltas (cheap, available now, measures "did we drift from the teacher"
   not "absolute correctness"). Use (b) immediately for accel work; pursue (a) for absolute quality.
3. **Three disjoint data splits — the overfitting firewall.** This is non-negotiable for quant + KD:
   - **Calibration set** — crops used to fit int8/fp8 activation ranges (PTQ) or as the KD training
     input. The quantizer/student *sees* these.
   - **Validation set** — crops used to *pick* hyperparameters (which layers stay fp16, batch, KD
     temperature, #TTA). We tune against these; never report them as the result.
   - **Held-out test set** — crops touched ONLY for the final number. If val and test diverge, we
     overfit the val set. Different scrolls/regions per split (spatial leakage is the trap: adjacent
     256³ crops share papyrus structure — split by *scroll and distant region*, not random crops).
4. **Baseline JSON + regression gate** (the open ADR: bench baseline storage): store the fp16-teacher
   metrics per test crop; any accel/distill model is scored against it, and a CI gate flags a drop
   beyond tolerance. Prevents silent quality regressions.
5. **TTA ablation** on val+test: `tta=0/8/48`, `scales={},{1.2},{0.8,1,1.2}` → metric delta +
   wall-time. Answers "does our TTA help" with committed numbers (supersedes the session-memory
   claim). **Overfitting check for TTA itself:** confirm the TTA gain holds on the held-out test set,
   not just where it was first observed (the "+0.1–0.17" was measured on one region 0125).

**Overfitting-prevention principles that carry through every phase:**
- **Report only held-out-test numbers.** Calibration/val numbers are for tuning, never for claims.
- **Split by scroll + distant region**, not random crops (spatial autocorrelation → leakage).
- **Quant calibration set ≠ eval set.** A PTQ range fit on the eval crops would flatter itself.
- **Distillation: watch student-vs-teacher gap ON TEST.** A student that matches the teacher on the
  KD-training crops but drifts on test has memorized, not generalized. Early-stop on val, report test.
- **Prefer per-channel over per-tensor quant scales** (less prone to a single crop's outlier setting
  a bad global range).
- **Sanity floor:** every candidate must also pass the GT-free `MeshQuality`/`analyze_mesh` checks
  (fold fraction, connectivity, self-intersection) — a model can score OK on voxel metrics while
  producing topologically broken surfaces; those GT-free checks catch it without any labels.

> **Direction update (forrest, 2026-07-01):** we are **skipping explicit post-training quantization**
> (Track A/B below) and going **straight to distillation** — train a smaller student (with mixed /
> "auto" per-layer precision as needed) rather than quantize the existing teacher — then **pretrain on
> new data** once forrest's new training set is ready (in prep, not ready yet). The Phase 2 quant tracks
> are kept below as reference/fallback, but Phase 3 (distillation) is the active path. The measurement +
> overfitting harness (Phase 1) is the prerequisite for both and is unchanged.

### Phase 2 — Quantization (reference/fallback; deprioritized per the direction update)
Two tracks; pick by measured quality/speed:
- **Track A — int8 PTQ via torch** (`torch.ao` quantized Conv3d). Calibrate per-channel weight scales
  + per-tensor activation scales on the eval crops (InstanceNorm means calibration must be per-patch).
  Extend `.fxweights` (dtype code 9=int8 + a scale tensor per quantized weight, version bump).
  Expected ~1.5–2× on convs; measure quality drop with `fenix eval` vs the fp16 teacher.
- **Track B — TensorRT engine (int8 AND fp8).** Export the net (ONNX via `torch.onnx`, or rebuild the
  graph in TRT's API), build an int8/fp8 engine with a calibrator over the eval crops. This is the
  only path to **fp8 conv** on Blackwell. Cost: breaks "we own the graph" (the C++ `torch::nn` net
  stays as reference/fallback; TRT engine is a second, optional backend behind the ml firewall).
  Highest expected speedup (2–4×), most integration work.
- **Decision gate:** run both on the eval set; keep whichever hits the speed target within an agreed
  quality tolerance (e.g. official_score drop < X, SurfaceDice@2 drop < Y). If int8 alone clears the
  bar, skip TRT/fp8 complexity.

### Phase 3 — Distillation (a smaller student)
- **Prereq: an in-tree (or Python out-of-tree) training loop** — none exists today. Fastest start:
  distill **out-of-tree in Python** (teacher = the ScrollPrize checkpoint at fp16 + full TTA;
  student = a narrower/shallower ResEnc), export the student to `.fxweights`, run it in the existing
  C++ inference path (no C++ training needed).
- **Student design knobs:** fewer stages or narrower features (e.g. `24/48/96/192/256/256/256`),
  fewer blocks, drop scSE (measure), **per-layer dtype ("auto types")** — keep sensitive layers
  (first conv, seg head, scSE) in bf16/fp16, quantize the bulky 320-ch mid stages to int8/fp8. The
  `.fxweights` per-tensor dtype code already supports mixed precision by tensor.
- **TTA-as-distillation-signal (your idea):** distill the student to match the *TTA-ensembled teacher*
  (48-way octahedral + multiscale mean). The student learns the augmentation-averaged target, so at
  inference the student needs *less* TTA to match teacher quality → compounding speedup. Measure:
  student@tta=0 vs teacher@tta=48 on the eval set.
- Metric gate at every step (Phase 1 harness).

---

## 4. How the accel levers interact (important — they're not independent)

Ordered by what we learned this session:
- **overlap = 0.5 is fixed** (it directly improves surface quality; never traded for speed). All
  speedups must be at fixed overlap.
- **batch=3** is the VRAM sweet spot at fp16/256³ (28/32 GB). **Under int8/fp8 the activation memory
  shrinks → a LARGER batch fits → re-tune batch after quantizing.** Batch and precision are coupled.
- **channels_last_3d (NDHWC): a LOSS in raw torch** on the full net (layout-thrash from scSE/adds/
  norms outweighs the conv win; microbench lied). BUT **TensorRT chooses layouts internally** — under
  a TRT engine, channels_last is TRT's problem, not ours; don't force it.
- **cuDNN autotune (`setBenchmarkCuDNN`): a wash** at fp16 (heuristic already optimal). Under int8 it
  *may* matter more (more algo choices) — re-test if we go torch-int8. Under TRT it's irrelevant (TRT
  does its own tactic search at engine-build time — that IS the autotune).
- **CUDA graphs:** won't help fp16 (compute-bound; launch overhead already amortized by batching).
  Could help int8 (smaller/faster kernels → launch overhead becomes relatively larger) — re-evaluate
  post-quant. Under TRT, engine execution is already graph-like.
- **prep/forward pipelining (done):** orthogonal to precision — keeps helping regardless. As the
  forward gets faster (quant), the CPU prep becomes a *larger* fraction → pipelining matters MORE,
  and we may need to also overlap `scatter` and speed up the u8→f32 gather.
- **patch size:** 256³ is the training size = quality-optimal; smaller is worse (quality + launch
  overhead), bigger drifts from training + VRAM. Keep 256³ unless a retrained/distilled student
  changes the training patch.

**Net interaction summary:** the clean stacking is **{fixed overlap 0.5} × {distilled student} ×
{int8-or-fp8 via the backend that has fast 3D conv kernels = TensorRT} × {re-tuned batch} ×
{pipelining} × {minimal-TTA because the student absorbed the teacher's TTA}**. cuDNN-autotune and
channels_last are subsumed by TensorRT; CUDA graphs are a post-quant re-check.

---

## 5. Open decisions
- Install **TensorRT** now? (yes if we want fp8/int8 conv speed; it's the only path.)
- Upgrade **torch**? (only if a needed API demands it — TRT-via-torch.export, or fp4 later.)
- **Ground truth**: import ScrollPrize proofread labels, or run teacher-as-pseudo-GT for relative
  deltas first? (Recommend pseudo-GT immediately + import GT in parallel.)
- Distillation **in-tree (build a C++ training loop) vs out-of-tree (Python)**? (Recommend Python
  first — inference stays C++, no training infra needed in-tree yet.)
- Acceptable **quality-loss tolerance** for accepting a quant/distill model (needs the metric first).

---

## 6. Status (Phase 1 partially built, 2026-07-01)

**Done + committed:**
- `fenix eval <pred> <gt> [--tau --thresh --gt-thresh --json]` — single-pair composite + components.
  Verified: self-compare = 1.00000; batch=1 vs batch=3 = 0.99986 (quantifies the batching speedup as
  lossless-to-tolerance).
- `fenix eval-set <manifest.toml> <split> [--pred-dir --gt-dir --tau --json]` — run a named split
  (calibration/validation/test), per-pair + mean/min/max/std aggregate. The overfitting firewall.
- `docs/design/eval-split.example.toml` — documented split template (split by scroll+region).

**Metric speed (fixed):** the shared primitives are now multithreaded — EDT per-line, CC via
z-slab-local union-find + serial boundary merge + parallel relabel (label ids bit-identical to the
serial scan), Euler/Betti per-plane partial sums, VOI/NSD per-chunk local accumulation with
integer-exact merges. Measured on the RunPod box (27-CPU cgroup budget; parallel_for clamps there,
never the 256 host cores): single 1024³ eval **2m46s → 23.3s (7.1×)**, identical scores; 2×1024³
eval-set 48s; 2×256³ eval-set **6.5s → 1.6s**. 1024³ evals are now practical.

**Still TODO for Phase 1:** real ground-truth import (or a committed teacher-as-pseudo-GT manifest);
`--baseline` regression-gate wiring in eval-set (the CLI hook exists).

## 7. TTA ablation for the distillation teacher (measured 2026-07-01)

Question: which augmentation combo generates distillation soft targets, at what cost. No real GT yet,
so the measurement is **agreement with the maximal ensemble** (t48ms = octahedral-48 × scales {1,1.2})
on two spatially disjoint 512³ PHercParis4 crops (A: z≈37k mid-scroll; B: z=50000, 13k voxels away).
Ranking was picked on A and CONFIRMED unchanged on B.

| combo | augs | walltime/512³ | mask-official vs ref (A / B, thr 0.5) | soft MAE vs ref (A / B) |
|-------|------|--------------|-----------------------------------|------------------------|
| t0    | none               | 8 s    | 0.657 / 0.568 | 0.088 / 0.088 |
| s12   | rescale ×1.2       | 16 s   | 0.653 / 0.631 | —     / —     |
| ms    | scales {1,1.2}     | 21 s   | 0.677 / 0.667 | 0.074 / 0.071 |
| t8    | 8 mirror flips     | 61 s   | 0.692 / 0.667 | 0.061 / 0.069 |
| t24   | 24 octahedral      | 176 s  | 0.792 / 0.693 | 0.033 / 0.045 |
| t8ms  | flips × 2 scales   | 201 s  | 0.734 / 0.670 | 0.049 / 0.054 |
| t48   | 48 octahedral      | 348 s  | 0.825 / 0.758 | 0.025 / 0.035 |
| t48ms | 48 oct × 2 scales  | 1169 s | (reference)   | (reference)   |

**Findings:**
1. **The base model is strongly non-equivariant**: single members disagree wildly (t0-vs-t8 mask
   official = 0.49). Single-pass predictions are high-variance; TTA is a big lever, not smoothing.
2. **Soft ensembles converge ~1/√N** (MAE 0.088→0.061→0.033→0.025 for 1/8/24/48 members). The
   binarized-mask metrics amplify sub-voxel threshold drift on thin sheets and look non-converged
   (t24-vs-t48 mask agreement only 0.76–0.85) — for KD, **soft-space distance is the relevant
   measure** (the student trains on soft targets), and there the ensemble stabilizes.
3. **Octahedral members beat multi-scale per unit cost on BOTH crops**: t24 (176 s, MAE 0.033/0.045)
   dominates t8ms (201 s, MAE 0.049/0.054). Scales add little unique signal toward the converged
   ensemble; s12 (grid-matching rescale alone) is barely better than t0.
4. **Recommended teacher for bulk soft-target generation: t48** (full octahedral, no scales) — within
   MAE 0.025–0.035 of the 3.4×-more-expensive t48ms. Budget-bound alternative: **t24** at half the
   cost, MAE 0.033–0.045. Skip multi-scale for target generation.
5. Caveat (documented, not hidden): agreement-with-ensemble cannot prove the ensemble is *correct* —
   members sharing augmentations with the reference correlate by construction (mask columns). The
   final teacher choice gets re-validated against real GT when the new training data lands; the
   soft-convergence result (finding 2) is the part that stands regardless.

**How much does TTA help vs a single raw pass** (distance to the t48ms best-estimate reference, both
crops): single pass → full octahedral moves **mask-official +0.17/+0.19** (0.657→0.825 / 0.568→0.758)
and shrinks **soft MAE ~3.5×** (0.088→0.025 / 0.088→0.035, i.e. **−72%/−60%** of the single-pass
deviation; +6.8/+5.5 dB PSNR). Diminishing returns past 24 members (24→48 buys only ~0.007 MAE). This
is a *distance-to-best-estimate* number, not distance-to-truth, and the proxy flatters TTA (shared
augmentations) — but its magnitude matches the earlier labeled +0.1–0.17 correlation gain, so the
effect is real, large, and consistent across two disjoint regions. It is exactly the gap the distilled
single-pass student is meant to capture.

### Arbitrary-rotation TTA (z-axis, continuous angle) — measured 2026-07-01

Added `rots=deg,...` (rotate about the scroll axis via GPU trilinear grid_sample → predict → rotate
back → validity-weighted mean-fuse). Tested r4 {0,22.5,45,67.5}, r8 (8 angles), t8r4 (flips × 4 rot).
The point was to sample orientations the 90°-only octahedral group can't reach.

**Finding — the two augmentation families converge to DIFFERENT means (both crops):** the rotation
ensemble self-converges fast (r4-vs-r8 MAE 0.025/0.014) and the octahedral ensemble self-converges
(t8-vs-t48 MAE 0.052/0.048), but **rotation-vs-octahedral disagree by 0.081/0.082** — nearly as far as
a single raw pass sits from either. If arbitrary rotations only re-sampled the same variance pool, the
families would agree; they don't. Two causes, both actionable: (a) **real z-anisotropy** — continuous
in-plane rotation mixes x/y through interpolation and produces a *systematically* different (not just
noisier) result, direct evidence the z axis is physically special and full-3D-rotation invariance
should NOT be imposed; (b) **interpolation blur** — every non-90° angle costs two trilinear resamples,
biasing the rotation mean toward thicker/blurrier sheets (an artifact, not signal).

**Consequences:**
- **Teacher target generation: stick with octahedral (t48), do NOT add arbitrary rotations.** t8r4
  (flips×rot, 263 s) lands FURTHER from t48 (MAE 0.064/0.066) than plain octahedral members — it adds
  disagreement, not refinement, at flip-level cost.
- **Student training augmentation: DO include small arbitrary rotations** — the model's sensitivity to
  them is a defect to train out (modulo the real anisotropy), via full flip+rotation augmentation +
  KD from the octahedral teacher + a rotation-consistency loss. Impose in-plane+flip invariance but
  keep the z axis distinguished (don't force full SO(3)).
