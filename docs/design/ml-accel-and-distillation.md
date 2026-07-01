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
- Perf baseline (this session, all lossless, overlap=0.5 kept): 1024³ 116s→76s via decode-once-u8
  + batch=3 + prep/forward pipelining. GPU is compute-saturated (~500W/575W, full boost). fp16 is
  the floor for lossless.

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

### Phase 2 — Quantization (needs Phase 1 to judge acceptability)
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
