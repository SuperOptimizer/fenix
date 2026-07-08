# fp8 conv3d on sm120 (consumer Blackwell) — spike verdict

Status: 2026-07-06, MEASURED on 2× RTX 5060 Ti (sm_120, 16 GB), CUDA 13.3 / torch
2.9.1+cu128 / Triton 3.5.1 / CUTLASS 4.6. Author: Claude + forrest.

## TL;DR — the parked verdict is overturned

ADR 0010 and `model-registry.md` recorded quantized conv3d as "dead in the whole NVIDIA
toolchain," with custom CUTLASS fp8/fp4 conv3d kernels parked as "pioneering genuinely
unsupported territory." That verdict was about the **export toolchain** (ONNX→TRT/torchao),
and it still holds for that path. But a **hand-written fused fp8 conv3d does win**, decisively:

| Path | Speedup vs cuDNN fp16 conv3d | Notes |
|---|---|---|
| TRT fp16 (adopted, ADR 0010) | 1.53× | the current production bar |
| explicit im2col + cuBLASLt fp8 GEMM | **0.14–0.21×** (7× SLOWER) | im2col materialization buries the win |
| **fused Triton fp8 implicit-GEMM, single layer (kernel only)** | **2.86–4.11×** | tensor-core ceiling, no im2col |
| **fp8-resident 4-layer chain (fused requant epilogue)** | **2.85× (C=320) – 4.13× (C=128)** | the realistic inference number |

Numerics throughout: **corr 0.9993** vs f32 conv3d, ~3.5–3.9% max relative error,
per-tensor symmetric e4m3 scaling (amax/448). Good enough for this network (matches the
fp8 GEMM's own corr-0.9993 vs fp32).

## The findings that got us there

1. **sm120 fp8 e4m3 tensor cores compute correctly on consumer Blackwell.** `torch._scaled_mm`
   (cuBLASLt) fp8 GEMM: corr 0.9993 vs fp32. The hardware is not the blocker.
2. **CUTLASS has no sm120 conv collective.** Conv collectives ship for sm90 (GMMA/TMA) and
   sm100 (UMMA/tcgen05) only — both datacenter instructions that don't exist on GeForce
   silicon. So conv3d on sm120 MUST be unfold+GEMM or a hand-written implicit-GEMM; there is
   no drop-in CUTLASS conv path. (The shipped sm120 fp8 GEMM examples are all block-scaled
   `OpClassBlockScaledTensorOp`; example 87a even asserts out of the box — a per-tensor dense
   fp8 sm120 GEMM must be hand-assembled from the collective builder.)
3. **The isolated fp8 GEMM is 3.5× faster than cuDNN fp16 conv** (45–103 TFLOP/s, rising with
   channel width). The compute ceiling is large and real.
4. **Explicit im2col destroys it** — materializing the 27×-inflated fp8 column tensor + the
   per-tap cast is memory-bound and 7× slower than cuDNN. This is the whole reason the naive
   unfold+GEMM route (and the export toolchain's implicit assumption) looks dead.
5. **Fusion recovers the ceiling.** A Triton implicit-GEMM that gathers the 27 taps inside the
   K-loop (never materializing im2col) hits the 2.9–4.1× kernel ceiling. Autotuning tile
   shape / warps / stages barely moved it (1.07→1.17× end-to-end) — because the residual
   overhead was NOT the kernel.
6. **The residual overhead is format conversion, and it amortizes to zero.** Per-layer
   f32→fp8 cast + channels-last permute costs ~as much as the kernel. Keeping activations fp8
   channels-last BETWEEN layers (requantize fused into the epilogue) removes it: the 4-layer
   fp8-resident chain runs at the kernel ceiling (2.85–4.13×).

## What this means

- **fp8 conv3d is a viable ~3× inference win over cuDNN fp16 (~2× over the TRT fp16 path)** on
  consumer Blackwell — IF the network runs fp8-resident (activations stay fp8 channels-last
  across layers, static per-layer amax calibration, requant in-kernel). Layer-at-a-time fp8
  with f32 round-trips is NOT worth it.
- The win grows with channel width and is largest on the mid/deep ResEnc-UNet stages
  (128–320 ch), which dominate the surface/ink net cost.

## Reproduce

- `tools/ml-export/fp8_conv3d_spike.py` — explicit im2col + cuBLASLt fp8 (finding 3–4).
- `tools/ml-export/fp8_conv3d_triton.py` — fused single-layer implicit-GEMM, autotuned
  (finding 5); the profiling split (kernel vs cast/permute) is finding 6.
- `tools/ml-export/fp8_conv3d_chain.py` — fp8-resident 4-layer chain, fused requant epilogue
  (the headline 2.85–4.13× number).

## Accuracy over depth — the gate (2026-07-06, follow-up)

Per-layer corr 0.999 doesn't prove the network survives fp8: error compounds through the
residual stream. Measured a fp8-vs-fp16 depth sweep (fused convs in fp8, InstanceNorm +
residual add in f32 — the sensitive-path recipe; `tools/ml-export/fp8_depth_real.py`):

| depth | RANDOM weights (pessimistic) | **REAL trained surface_recto_3dunet stage-3 weights** |
|---|---|---|
| block 1 | corr 0.9970 | **corr 0.9999** (maxrel 2.6%) |
| block 5 | corr 0.9777 | **corr 0.9996** (maxrel 4.3%) |
| block 8 | corr 0.9608 (maxrel 23%) | — (real stage has 5 resident blocks) |

**Trained weights barely compound** — corr stays 0.9996–0.9999 across the deepest (256-ch)
stage; the alarming 0.96 was a random-weight artifact (unstructured filters, no learned
activation smoothness). This clears the biggest risk: fp8 is accurate enough for the surface
net. Still owed: a true end-to-end SurfaceDice@2 check on a full forward (all stages + decoder
+ skips) vs the fp16 teacher, on real CT, against the ≥0.998 bar. Building-block accuracy is
green; whole-net number pending.

Op coverage validated (`tools/ml-export/fp8_op_validate.py`): 3×3×3 s1/s2, 1×1×1, +LeakyReLU
all corr ≥ 0.9985 vs torch. InstanceNorm is two-pass (conv→deq→per-channel stats→norm) because
its per-instance mean/var is only known post-conv; the affine could later fold into the requant
epilogue once stats are computed, saving a pass.

## Whole-net end-to-end gate (2026-07-06) — PASSES with fine-grained scaling

Ran the ENTIRE reference net (`surface_recto_3dunet`, 71 3×3×3 convs) with those convs routed
through fp8, on REAL Scroll-1 CT patches (fetched via `fenix ingest-zarr` from the open-data
bucket, 2.4µm level 0), vs the fp16 autocast reference. Tool: `tools/ml-export/fp8_forward.py`
(monkeypatches `nn.Conv3d.forward`; stem / transpose-conv / 1×1×1 SE+head stay fp16).

**Per-tensor fp8 is too coarse for the whole net** — passes on some patches, fails on others:

| patch | per-tensor SurfaceDice@2 | **fine-scale SurfaceDice@2** |
|---|---|---|
| 1 | 0.9998 ✅ | 1.0000 ✅ |
| 2 | 0.9972 ❌ | 0.9998 ✅ |
| 3 | 0.9876 ❌ | 0.9993 ✅ |

**Fine-grained scaling clears the ≥0.998 bar on every patch** (corr 0.9996–0.9999). The recipe:
**per-output-channel weight scale + per-token (per-voxel) activation scale** — the standard fp8
accuracy recipe, and exactly what sm120's block-scaled tensor cores implement in hardware. Error
is dominated by the *encoder* (65 of 71 convs); keeping the decoder fp16 does NOT fix it
(0.9876→0.9917 on patch 3), so layer-selection is not the lever — scale granularity is.

Two important caveats on this number:
- The fine-scale result is SIMULATED in the forward hook (quantize→f32 conv→dequant) to prove
  the accuracy lever; the Triton kernel currently does per-tensor only. Adding per-channel/
  per-token scaling to the kernel (and re-confirming the 2.85–4.14× speed holds) is the next
  step (task open).
- On SYNTHETIC noise input the net is out-of-distribution (decision margin ~0.5 logits) and even
  per-tensor fp8 "fails" at 0.87 — an artifact, not signal. Only in-distribution real CT gives a
  meaningful gate number. Always test fp8 accuracy on real scroll CT.

## Max-performance sprint — FINAL: 109 ms → 40.0 ms (2.72×), SD 1.0000

Goal escalated to absolute maximum fp8 performance. Whole-net 128³ forward, fair baseline
(fp16 autocast + channels_last_3d = 109 ms; NCHW eager was 199 ms):

| step | ms | what |
|---|---|---|
| naive fp8 hook | 168 | per-conv quantize+dequant round trips |
| resident blocks + fused norm-act-quant | 95→64 | ONE glue kernel between convs; killed `.float()` materializations in quantize/var_mean (−31 ms!) |
| float-scale caches (no `.item()` syncs) | 59.5 | 70+ hidden syncs/forward removed; also enables CUDA-graph capture |
| **cube-tiled v2 conv kernel** | 50.1 | program = 3D voxel cube → 27-tap gather hits L2 (~10× reuse); flat K-loop over (tap,cin); 1.6–1.9× vs cuDNN-CL per layer where v1 was 0.9–1.7× |
| fused block tail | 46.6 | scSE-apply + residual + lrelu + dual fp16/fp8 emit in ONE pass; successor map lets a block tail quantize directly for the next block's conv1 |
| conv-epilogue norm stats | 44.3 | per-CTA column partials via 64-way-spread atomics — no separate stats pass |
| fused decoder | 41.0 | transpconv k2s2 = ONE fp16 GEMM, pixel-shuffle decoded in the quant-cat kernel's ADDRESSING (never materialized); cat never exists in fp16 |
| small-tile autotune configs | **40.0** | deep stages (M=512/64/8 rows) finally fill the GPU; BK<=64 only (BK=128 crashed a GPU off the bus) |

CUDA graph capture works (all sync-free) but adds only ~0.3 ms — the pipeline is
kernel-bound, not launch-bound. Accuracy after every step: corr 0.9998, SurfaceDice@2
1.0000 vs the fp16 reference.

Hard-won correctness lessons (all verified, all cost real debugging):
- **Triton 3.5.1/sm120 miscompiles the v2 kernel for TD ≥ 4 cube tiles** (corr 0.85–0.99;
  TD ∈ {1,2} exact). Config list prunes them.
- **Autotune benchmarking pollutes atomic-accumulator args** — benchmarks run the kernel
  dozens of times into the same ps/pq buffers. Fix: `reset_to_zero=[...]` in @autotune.
  Made worse by a persistence trap: skip-projection 1×1 convs weren't hook-calibrated, so
  their scales got recorded during the polluted first resident forward and stayed wrong.
- **Conv bias is absorbed exactly by InstanceNorm** (post-conv, spatially constant) — but a
  **transpconv's bias is NOT** (it enters the next conv's INPUT; zero-padding makes its
  contribution border-dependent). Fold transp bias into the GEMM output instead.
- The resident path is **N=1 only** (InstanceNorm stats pooled per-channel over the whole
  fold); production parallelism comes from `predict-scroll gpuworkers=`, not batch.

Remaining wall-clock (per-stage, eager-inflated): stage1 13.4, decoder 9.2, stage0 7.6,
deep stages ~5 (M=512/64/8 rows underfill the GPU — small-tile autotune configs in flight),
stage2+3 7.5. Candidate next levers: grid swizzle for A-tile L2 reuse, SE gates fused into
the norm kernel, fp8 sSE/transpconv GEMMs, multi-stream skip/main overlap.

## Training + fp4 — MEASURED verdicts (2026-07-07, post-reboot)

**fp8 training WORKS and is fast:**
- grad correctness vs f32 autograd: y/dx/dw all corr 0.9993 (the fp8 noise floor)
- fwd+bwd micro-bench vs fp16 cuDNN autograd: **2.84× (C=64), 2.19× (C=128) faster**
- optimizer sanity: Adam on an MSE distillation converges 0.477 → 0.0095 in 30 steps
- TRAP (cost a debug session): scale math MUST be f32 — in fp16, `amax/448` underflows to
  0 for small gradients (fp16 min subnormal 6e-8) → 0/0 NaN on the first real loss whose
  MSE grads are ~1e-7. corr checks can't catch it (scale-invariant); only a live
  optimizer loop did.

**fp4 verdict: dead on this hardware via Triton.** `tl.dot_scaled` (fp8 act × MX-fp4
weight) is bit-EXACT vs reference (corr 1.000000) but **2.2× SLOWER than plain fp8
`tl.dot`** at the same shape — it lowers to emulation, not native block-scaled MMA, on
sm120/Triton 3.5.1. Since fp4's other benefit (weight bandwidth) is irrelevant for conv
(weights are L2-resident), **fp8 is the precision sweet spot for conv3d on consumer
Blackwell**. fp4 remains a candidate for GEMM-heavy nets (dinovol ViT) via CUTLASS's
blockscaled path (examples 79/87), not via Triton, and not for conv.

### Authoring notes (written 2026-07-07 while GPUs were down)

Goal expanded to full fp8/fp4 training + inference. Authored and CPU-validated while the
GPUs were down (see incident below):

- **fp8 backward** (`fp8_train.py` + kernels in `fp8_conv3d_op.py`): `Fp8Conv3d` autograd
  Function. dgrad of a stride-1 same-pad conv IS a forward conv with mirrored taps +
  swapped channels (`pack_weight_dgrad_fp8` — identity verified exactly vs autograd), so
  the v2 kernel does dgrad unchanged. wgrad = per-tap `dyᵀ@x_shift` GEMM with split-M f32
  atomic accumulation (`_wgrad_f8` — identity verified exactly). Saved-for-backward
  activations are the fp8 bytes themselves: **2× activation-memory saving** on top of
  speed. Strided convs fall back to torch autograd (6 in the net).
- **fp4 weights** (`fp4_conv3d_op.py`): MX-format e2m1 packed 2/byte + e8m0 scale per
  32-K-block (`pack_weight_fp4`, roundtrip corr 0.993 ≈ intrinsic 4-bit noise), consumed
  by a cube-tiled conv via `tl.dot_scaled` (fp8 activations × fp4 weights — weights are
  the bandwidth-heavy conv operand). `fp4_probe_dot.py` must run FIRST on GPU: it decides
  whether sm120 dot_scaled lowers to native block-scaled MMA or bf16-upcast emulation
  (if emulated, Triton fp4 is pointless → CUTLASS ex79 blockscaled GEMM is the fallback).
- Production autotune latency: `dump_tuned`/`load_tuned` pin per-shape best configs.

### GPU incident (2026-07-06 late)
Aggressive autotune configs (BLOCK_K=128, and a 1D-grid swizzle variant) caused
`cudaErrorLaunchFailure` → **GPU1 fell off the bus** ("Unknown Error"), which then wedged
CUDA init system-wide (even CUDA_VISIBLE_DEVICES=0). Configs pruned (BK ≤ 64, no swizzle,
TD ≤ 2). Recovery needs `sudo nvidia-smi --gpu-reset -i 1` or a reboot. Triton 3.5.1 on
sm120 is genuinely fragile at the edges of its config space — treat every new config
family as suspect until corr-checked (the autotuner benchmarks by SPEED only).

## Open work before production (not done in this spike)

- **Hand-assembled CUTLASS dense sm120 fp8 GEMM** as the kernel core, if Triton leaves ceiling
  on the table at other shapes / with cluster/pipeline tuning; the Triton kernel already meets
  the isolated-GEMM ceiling, so this is a ceiling-chasing exercise, not a correctness one.
- **Full fp8-resident net**: fold every conv3d + norm/act/scSE into the fp8 dataflow, static
  amax calibration pass, a `Fp8Net` adapter behind the ML firewall (mirrors `TrtNet`/`AotiNet`)
  so `fenix predict-surface` can dispatch it. The per-layer requant epilogue is prototyped in
  the chain kernel.
- **Accuracy at the network level**: DONE — see the whole-net gate above. Per-tensor fails on
  ~2/3 of real patches (0.988–0.9998); FINE-GRAINED scaling (per-channel wt + per-token act)
  passes all (≥0.9993). Next: implement fine scaling IN the Triton kernel (task open) and
  re-confirm speed.
- **stride-2 / 1×1×1 convs**: DONE in the op library (`fp8_conv3d_op.py`, validated corr
  ≥0.9985). Transposed convs (decoder upsample) still fall back to fp16 — need a scatter-based
  gather; cheap enough (6 of them) to leave fp16 for now.
- **fp4 (nvfp4)**: sm120 supports it (CUTLASS ex 79); plausibly another ~1.5–2× on top for the
  backbone if accuracy holds — the ViT/dinovol fp4 note in model-registry.md applies to conv
  too now that the fused path works.
```

## Stack upgrade + max-perf plan (2026-07-07, torch 2.12.1 / triton 3.7.1 / CUDA 13)

Main env upgraded 2.9.1→2.12.1 (bundles triton 3.7.1, cu13 wheel stack, cuDNN 9.20).
Measured on upgrade day:

- **Correctness: everything survives.** Per-op grads 0.9993 (all conv variants), transpconv
  1.0000/0.9993, whole-net (258/258 modules fp8, `fp8_netgrad.py`) fwd logit corr 0.99938,
  median grad corr 0.9992 over 391 params.
- **Transpconv fp8 training DONE** (`Fp8TranspConv3d` in `fp8_train.py`): fwd = one fp8 GEMM
  `[M,Cin]@[Cin,8·Cout]` + pixel-shuffle, bwd = two fp8 GEMMs via `torch._scaled_mm`
  (dims padded to 16 rows — `_pad16`). The net now has ZERO non-fp8 conv-like layers in
  training ("kept 0").
- **fp4 re-verdict on 3.7.1: STILL EMULATED** (`fp4_probe_dot.py`: bit-exact, 2.33× slower
  than fp8 dot). Shelved again; re-probe on the next Triton bump.
- **TD≥4 re-verdict on 3.7.1: partially fixed, still broken** (`fp8_td_probe.py`: (4,8,8)
  and (8,8,8) now exact, but (4,4,8)/(8,4,4) corr 0.925). The bug class is alive → keep
  TD ≤ 2.
- **BASELINE SHIFT (the big fallout): cuDNN 9.20 sped fp16 conv3d training 3–4×**
  (C=64 fwd+bwd 8.7→2.0 ms). fp8 now LOSES the per-op training micro-bench (0.63–0.80×).
  The deficit is overhead, not tensor-core math (fp8 kernel times unchanged: 3.2→3.1 ms).

### The overhead ledger (why per-op fp8 training is slow today)

Per conv per step the naive path pays: (a) `pack_weight_fp8` every fwd + `pack_weight_dgrad_fp8`
every bwd (weights only change at `opt.step()`); (b) `_q(xm.float(), sx)` — a full f32
materialization of every activation just to quantize; (c) `float(sx)`/`float(sdy)` `.item()`
syncs — ~500 pipeline stalls per step at 258 modules; (d) unfused norm/act between convs;
(e) launch overhead on tiny deep-stage kernels (no CUDA graph).

### Phase plan (each phase gated by measurement; gates: median grad corr ≥0.999,
### e2e loss curve matches fp16 twin, SurfaceDice@2 ≥0.998, no bench regression)

- **P0 — honest baselines on the new stack** (in flight): `fp8_resident_bench.py`
  (inference # vs re-measured CL-fp16 baseline + SD gate), `fp8_train_e2e.py` (whole-net
  step time vs fp16-autocast twin).
- **P1 — training overhead removal (mechanical):**
  - P1.1 quantize-on-update weight cache: `Fp8Conv3dLayer` caches `(w8, sw, wg8, swg)`
    keyed on param version; refresh after `opt.step()` (hook). Kills (a).
  - P1.2 fused fp16→fp8 quant kernel (amax reduce + quant, no f32 copy) for x in fwd and
    dy in bwd. Kills (b).
  - P1.3 device-side scales: kernels take a scale POINTER (tiny kernel change) or delayed
    scaling (previous-step amax, Transformer-Engine style). Kills (c) → unblocks graphs.
- **P2 — training step structure:**
  - P2.1 CUDA-graph the whole train step (static buffers; autotune pre-warmed +
    pinned via `dump_tuned`/`load_tuned`; Adam `capturable=True`). Kills (e).
  - P2.2 fp8-resident training forward: reuse the fused inference blocks (conv-epilogue
    stats + `fused_norm_act` emitting fp8) inside a custom autograd Function per CDNR
    block; save fp8 acts + norm stats; hand-written norm/act backward. Kills (d) — the
    structural 2×, same trick that took inference 109→40 ms.
  - P2.3 widen wgrad/dgrad autotune ON GPU0 with the crash-safe protocol (one config
    family at a time, corr-check after each; GPU1 stays quarantined).
- **P3 — memory→throughput:** fp8 saved acts (+P2.2) ≈ half the activation memory →
  ~2× batch/patch at 16 GB; report samples/sec, not just ms/step. Optional probe: fp4
  STORAGE-ONLY saved activations (dequant at wgrad; needs its own grad-corr gate).
- **P4 — inference re-baseline on 3.7.1:** re-autotune everything (new compiler = new
  best configs), re-run the 109→40 ms table's endpoints, re-pin tuned configs per
  (GPU, triton) pair.

Fallback positions if fp8 step-time ties fp16 after P1–P2: fp8 still wins samples/sec at
fixed memory (P3), and inference keeps its independent win regardless.

### P0/P1 measured (2026-07-07 evening)

- **P0 whole-net step time (pre-P1)**: fp8 532 ms vs fp16-autocast 266 ms (0.49×) —
  40-step self-distillation @128³ real CT.
- **P1 (all three: weight cache + fused quant + tensor scales) — correctness unchanged**
  (per-op 0.9993 everywhere), per-op fwd+bwd vs cuDNN 9.20: 0.63→**0.94×** (C=64),
  0.80→**1.04×** (C=128); whole-net step 532→**405 ms** (0.66× vs the twin). The
  remaining 139 ms over the twin is between-op overhead → P2 (graphs + fused resident
  training forward).
- **e2e harness traps found by the first NaN run**: (1) `kl_div(reduction="batchmean")`
  at batch=1 is a SUM over ~2M voxels → grads 1e6× → both lanes NaN'd; use per-voxel
  mean. (2) The fp16 twin needs a GradScaler; the fp8 lane does NOT — dynamic per-tensor
  dy quantization is a built-in per-layer loss scale. (3) An UNPERTURBED student copy of
  the teacher is a degenerate fixed point: true grad = 0, so fp8's grad noise
  random-walks the weights (loss drifts up) while the fp16 twin's exact-zero grads sit
  still. `--perturb` (identical relative weight noise on both students) restores a
  signal-dominated comparison.
- **Whole-net grad-corr structure** (128³ real CT, all 258 modules fp8): logit corr
  0.99938; weights median ≈0.999; the below-0.9 tail is (a) SE fc layers — M=1 GEMMs on
  pooled tensors, quant noise doesn't average (candidate: keep SE fp16, ~0 FLOPs), and
  (b) stage-5 convs (corr 0.83–0.87) — wgrad reduces over only 4³=64 voxels, inherent
  small-M statistics. 59 params NaN-corr = a dead block (stages.6.blocks.5, zero grad in
  BOTH nets — checkpoint quirk, unreachable from the surface head). Conv biases are
  norm-absorbed → near-zero grads → corr meaningless (155 of them).
- **Autotune latency**: a fresh process re-sweeps fwd+dgrad+wgrad (~20 min at 128³).
  `dump_tuned`/`load_tuned` now cover ALL autotuners (multi-kernel JSON, legacy format
  accepted); fp8_train_e2e.py auto-loads/saves `~/.cache/fenix-fp8-tuned-train.json`.

### P2/P3 measured verdicts (2026-07-07 late) — training perf on sm120 + cuDNN 9.20

- **P2.1 CUDA-graphed train step: no-op.** 408 vs 405 ms — the step is kernel-bound at
  128³, same as inference (+0.3 ms). Graph mode works (in-graph weight packing via
  `set_graph_mode`; the host-side `_version` cache can't run during replay) — keep for
  small-patch regimes, irrelevant at 128³.
- **P2.3 wgrad autotune widening: +7 ms** (405→398). wgrad (83 ms/step, 21%) is
  atomics/M-bound, not config-starved. Structural ideas if ever needed: fewer splits at
  small M, e5m2 dy, per-shape splits.
- **Profile of the 398 ms step**: 114 ms (28%) NCDHW↔[M,C] permute-contiguous copies
  around every conv (both directions), 83 ms wgrad, 35 ms conv fwd+s1-dgrad, ~22 ms
  InstanceNorm, 13 ms DtoD, rest elementwise. The copies are inherent to running [M,C]
  kernels inside an NCDHW-autograd net.
- **P2.2 fused CDNR training block (fp8_train_fused.py): built, validated, and a NET
  LOSS in Stage-A form** — 469 ms (vs 398 unfused), peak mem 7.89 vs 7.71 GiB, grads
  0.9128 vs 0.9378 weights-median. The torch-side glue (f32 xhat materialization, sign
  mask, extra lrelu alloc, separate stats/quant passes) exceeds what fusing the norm
  saves. It only pays as ONE dual-emit kernel (norm+act+fp16-out+fp8-out+sign, and the
  bwd chain fused likewise) — future work. Traps found and fixed on the way, valid for
  any future attempt: (a) saving POST-act fp8 forces lrelu inversion in bwd → 100×
  noise on the negative side; save PRE-act; (b) branch-by-reconstructed-sign flips
  ~0.4% of near-zero elements → save exact sign bits; (c) reconstructing
  xhat=(pre-β)/γ amplifies fp8 noise by 1/γ (real γs get small → whole-net dgamma
  corr 0.46) → save xhat directly, fp8 for M>64k, fp16 below.
- **P3 memory: the 2× activation promise does NOT materialize in the unfused path** —
  fp8 11.0 vs fp16 10.55 GiB peak at batch 2. The fp16 [M,C] layout transients and
  norm-saved fp16 tensors offset the fp8 saved-activation win. Requires the same full
  residency as the speed gap.
- **GRAD-CORR GATE CORRECTION**: earlier whole-net medians (0.9992/0.9996) were
  NaN-poisoned (59–89 dead params from the unreachable stages.6.blocks.5 block break
  `statistics.median`). HONEST numbers vs f32 reference at 128³: unfused weights-median
  **0.9378**, fused 0.9128 (per-stage 0.87–0.97, worst = decoder-tail norms + SE M=1
  layers). The dominant term is lrelu branch divergence: fp8 conv noise flips which
  side of 0 borderline pre-activations land on vs the reference — the gradient is
  exact FOR THE FUNCTION ACTUALLY COMPUTED, it just isn't the reference's gradient.
  The meaningful gate is therefore BEHAVIORAL: the e2e loss curve matches the fp16
  twin (0.024 vs 0.027 at 40 steps) — fp8 training converges equivalently.

### BOTTOM LINE (training, this hardware/toolchain) — REVISED after the second sprint

**fp8 training reached parity with cuDNN-9.20 fp16-autocast and wins on memory**:
269 vs 266 ms/step (0.99×) at 128³, peak 7.24 vs 7.71 GiB, converging identically.
The second sprint's levers (532 → 269 ms total, 1.98×):
- P1 overhead removal (quantize-on-update weight cache, fused quant, zero syncs): −127 ms
- removing forced `.contiguous()` on layer outputs/grads (CL-layout views flow): −23 ms
- **`--normfuse`: layout-native fp8 norm+act** (`Fp8NormAct` in fp8_train.py — one
  dual-emit kernel: post-act fp16 + xhat fp8 at a FIXED scale (xhat is unit-variance,
  no amax pass) + exact sign bits; two-pass backward): −99 ms.
  TRAP that cost 218 ms before being found: reduction kernels with a (N, C-blocks)-only
  grid serialize ~2M rows through ONE CTA at shallow stages — S-split grid + f32
  atomics fixed it (494 → 276 ms).
- wgrad SPLITS autotuned per shape (was fixed 8): −7 ms.
Use `fp8_train_e2e.py --normfuse` (should become the default trainer config).

Final 266 ms profile: **wgrad 70.6 ms (26.5%) — the one big lever left**; conv fwd+dgrad
35 ms (efficient); `_in_norm_act_train` 6.2 ms for all 75 norms (the fusion is cheap
now); quant+amax ~14 ms; ~50 ms fragmented elementwise tail (SE/residual/casts).
wgrad findings: config space + SPLITS autotuned (exhausted); the strided dy load is NOT
the problem — `tl.trans`-in-dot is now CORRECT on Triton 3.7.1 (fragility class fixed;
`WGRAD_TRANS` flag in fp8_conv3d_op.py, left off — measured no speed change).

### wgrad v2 + the WIN (2026-07-07 night, second pass)

`_wgrad_f8_v2`: ONE GEMM `dW[Cout, 27·Cin] = dyᵀ @ im2col(x)` with M as the K-loop —
the B gather is exactly the fwd kernel's input gather, tap/cin decompose once per CTA
(v1 redid the spatial decomposition per tap per chunk). Bit-exact vs
`torch.nn.grad.conv3d_weight` (corr 1.00000), and at net shapes it took the step
**269 → 247 ms**.

**fp8 training now BEATS cuDNN-9.20 fp16-autocast: 247 vs 266 ms (1.08×), peak
7.24 vs 7.71 GiB, identical convergence.** Full sprint: 532 → 247 ms (2.15×).
**Chunked-im2col + cuBLASLt GEMM wgrad: MEASURED DEAD END** (fp8_wgrad_gemm_probe.py:
corr 1.00000 but 0.10×/0.25×/1.00× vs triton-v2 at shallow/mid/deep shapes — the 27×
materialization traffic drowns the MMA-rate gain; same physics as the spike's im2col
OOM, just slower instead of bigger). Fused-gather Triton v2 is the right structure;
beating it needs a hand-written CUTLASS sm120 implicit-GEMM wgrad collective (upstream
has none) — multi-day, est. ≤20 ms.

### Tail fusion (2026-07-07, last) — **232 ms, 1.14×, 7.14 GiB**

`Fp8Tail` + `swap_tails_fp8` (fp8_train.py): the whole BasicBlockD tail —
scSE-apply `h*(gc+gs)` + residual + lrelu — is ONE kernel each direction
(`_tail_train_fwd/_tail_train_bwd`), replacing 3–4 eager passes + their saved fp16
tensors with h-fp8 + sign bits; dgc/dgs flow back into the torch-side SE subgraph so
the gate params train normally. −15 ms and −0.1 GiB on top of everything else.
Both fusions are now the HARNESS DEFAULT (`--no-normfuse`/`--no-tailfuse` to disable).

**FINAL: 532 → 232 ms (2.29×) — fp8 training is 1.14× FASTER than cuDNN-9.20
fp16-autocast with 7.14 vs 7.71 GiB peak.** Last unimplemented candidates:
producer-fused amax (~10 ms, needs cross-layer successor plumbing à la the inference
`_NEXT8` map) and the CUTLASS wgrad collective (≤20 ms, multi-day).
Verdict: **train in fp8 when memory-bound (bigger batch/patch at 16 GB), either way at
no speed cost; deploy in fp8 regardless** (inference 2.78×, SD@2 1.0000).
