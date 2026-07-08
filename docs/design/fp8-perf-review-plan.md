# fp8 perf deep-review — verified optimization plan (2026-07-07)

Produced by a 40-agent review workflow (5 lenses -> adversarial verification ->
synthesis); every item verified against source + the measured dead-end list in
fp8-conv3d-sm120.md. Baseline: 232 ms/step vs 266 fp16-autocast.

## IMPLEMENTATION LEDGER (updated as items land)

- **Tier 1 SHIPPED (232 → 206 ms, 1.29×)**: db fused-dtype reduce on the contiguous
  view; norm-absorbed conv biases dropped (exact absorption); SE convs → CastConv3d
  fp16 (the grad-corr tail is GONE — weights-median rose to 0.94); needs_input_grad
  guard kills the stem's dead dgrad+pack; wgrad configs widened (128-wide tiles, s3).
- **Item 6 SHIPPED (206 → 196 ms, 1.36× — under 200)**: side-stream wgrad drain
  (`set_wgrad_stream`/`join_wgrad_stream` in fp8_train.py; e2e arms post-warmup,
  joins before opt.step; requires zero_grad(set_to_none=True); off in graph mode).
- **Item 9 SHIPPED (196 → 195 ms)**: conv-epilogue stats bound to the following norm
  (swap_norms_fp8 binds bias-free Fp8Conv3dLayer siblings; one-shot stats handoff;
  N=1 only, in_stats fallback). Only −1 ms — the in_stats pass was already partly
  hidden under the side-stream drain — but it removes critical-stream work and is
  correctness-neutral (0.9376). Keep.
- Correctness after all of the above: weights-median 0.9376–0.9386, per-op 0.9993.
- **Cumulative session sprint: 532 → 195 ms (2.73×), 1.36× over fp16-autocast.**
- **Teacher channels_last SHIPPED (wall-clock)**: teach() ~199 → ~113 ms; wall
  575 ms/iter. TRAP: convert the teacher AFTER the student deepcopies — a pre-copy
  convert leaks CL weights into the fp16 twin and slows its cuDNN path 266→329 ms.
- **Item 7 (W-aligned wgrad) SHIPPED (195 → 193 ms, 1.38×)**: bit-exact at all probe
  shapes; split boundaries rounded to whole output rows, scalar (n,od,oh) per chunk
  when BLOCK_K | Wo, runtime-uniform branch with generic fallback.
- Remaining: items 4 (dgrad_s2 parity skip), 8 (decoder cat fusion), 10 (delayed
  scaling, multi-day staged), 11 (sign bitpack), 12 (SE glue rework), 13 (interior
  fast path), 14 (CUTLASS wgrad — projected floor ~165 ms).

## 3090-CLASS LANES (2026-07-08, Tier A/B/C program — sm86 fleet has NO fp8)

- **v2 conv kernel is now TRI-DTYPE**: fp8 / fp16 (unit scales) / int8 (`IS_INT`:
  int8 dot → int32 acc → f32 rescale). The whole fused-inference structure (norm+act,
  tails, epilogue stats, CL residency, graphs, pins) is inherited by all three lanes.
- **PROBE VERDICT (GPU1): int8 is MORE accurate than fp8 for this net** — per-layer
  corr 0.9999 vs 0.9993, depth-8 chain 0.99853 vs 0.99305 (uniform 256 levels beat
  e4m3 on normalized bounded activations). fp8 passed SD@2 1.0000 whole-net → int8
  gate near-certain. fp16 lane exact (1.0000). Speed on sm120: int8 wins at deep
  stages (0.37 vs cuDNN 0.64 ms), eager-quant overhead dominated shallow — fixed
  with `quantize_i8_fused` (INT8 path in _quant_kernel). On sm86 int8 TC = 2× fp16.
- Tools: `fp16i8_probe.py` (lanes probe), `pack_weight_i8`/`pack_weight_f16`/
  `quantize_i8_fused` in fp8_conv3d_op.py. TensorRT 11.1 (cu13) reinstalled (Tier A).
- GPU1 now power-capped 150 W + lgc 2200 (forrest) — safe for lane dev; it hit 85°C
  vs GPU0's 60°C on identical sweeps pre-cap (supports the marginal-cooling theory).
- **int8-QAT SHIPPED — now FULL int8 (fwd AND bwd), converging**: `set_int8_qat(net)` /
  e2e `--int8qat`. Forwards run the deployment int8 kernel (versioned `_i8cache`
  packs + `quantize_i8_fused`); backward is int8 too — dy quantized to int8
  (amax/127), dgrad via the v2 kernel with `pack_weight_dgrad_i8`, wgrad_v2 with
  int32 accumulate (`IS_INT` in all three kernels). 199 ms/step (1.33×), loss curve
  normal — int8 GRADIENTS work. On sm120 int8 TC = fp8 rate (parity); on Ampere/3090
  this makes the ENTIRE training step tensor-core-fast — the fleet can QAT-train
  ladder rungs in the exact precision they deploy in. Epilogue-stats binding works
  under int8 too (stats read the f32-rescaled acc).
  TRAP: v1/v2 wgrad kernels share a signature shape — a constexpr added by matching
  the param block can land on the wrong kernel (KeyError 'unrecognised' at launch).
- **int8 WHOLE-NET GATE: PASS — SD@2 0.9983, corr 0.9968 (real CT, 156 int8 convs).**
  The road there is the whole story (all measured):
  (1) symmetric per-tensor: SD **0.744** — post-LeakyReLU activations are heavily
      skewed (negatives all ×0.01 near zero); uniform codes sized by amax starve them.
      The depth-8 probe (0.9985) was MISLEADING — its global re-normalization hid the
      skew/outlier structure. fp8 never had this problem (subnormals).
  (2) + per-channel weight scales (WS_VEC epilogue): 0.754 — necessary, not sufficient.
  (3) + AFFINE activations (zero-point; `quantize_i8_affine`, wsum epilogue correction,
      zp-fill for padding taps): **0.9983 PASS**. This is the canonical TRT/ONNX int8
      recipe, now in OUR kernel where the export toolchain failed to deliver it.
  Training keeps symmetric acts (bwd has no zp correction; loss-curve validated);
  inference switches to affine automatically under no_grad in the int8-QAT layer.
- **int8 STATIC CALIBRATION SHIPPED: 130.4 → 57.6 ms (2.3×), SD@2 0.9987 PASS** on the
  power-capped GPU1 — `int8_calibrate(net, patches)` records dynamic-affine (scale, zp)
  per layer, freezes, steady state quantizes via `quantize_i8_static` (zero reduces/
  syncs). The dynamic aminmax+item() pairs were most of the int8 forward.
- **SPARSITY WIRED (both kinds)**: (a) 2:4 sparse-QAT (`set_sparse24`/`--sparse24`,
  ASP top-2-of-4, fixed mask, STE) — trains toward TRT sparse-TC deploy (no speedup in
  OUR kernels: Triton cannot emit mma.sp). (b) BLOCK sparsity (`set_blocksparse`/
  `--bsparse R`) — 32-wide (tap,cin)-groups pruned layer-globally in the PACK layout;
  fwd conv K-chunks and wgrad N-blocks SKIP pruned groups (exact: zeros) — real
  speedup in our kernels on any GPU; accuracy is the experiment, KD recovery is the
  mechanism. Composable: --int8qat --sparse24 --bsparse all stack.
- **2:4+int8 CONVERGENCE SMOKE PASSED** (20 steps, GPU0): sparse-int8-QAT loss tracks
  the fp16 twin (0.047 vs 0.058 @ step 19); median step 298 ms vs fp16 268 ms — the
  fixed-mask multiply costs ~100 ms over plain int8-QAT (199 ms) in training, paid only
  to earn TRT sparse-TC inference on Ampere. Verdict: 2:4 is deploy-viable.
- **PRECISION AUTO-ASSIGN: ALL-INT8 PASSES** (precision_assign.py, GPU1): all-int8
  static-affine SD@2 = 0.9986 >= 0.998 gate with 0/156 layers promoted to fp8; the
  most sensitive layer (decoder stage-4 conv) gains only +0.0011 from fp8. Deploy
  config = pure int8 everywhere (57.6 ms static-cal on capped GPU1); no
  mixed-precision map needed. Map at ~/.cache/fenix-precision-map.json.
- **BLOCK-SPARSE SWEEP (int8-QAT lane, 20 steps)**: NOT a training-time win — 0.5:
  285 ms, 0.25: 297 ms vs plain int8-QAT 199 ms (the autograd mask multiply costs more
  than K-chunk skipping saves at 128^3/32-wide groups; 50% skip clawed back only
  ~13 ms). Accuracy: 0.25 recovers immediately (loss 0.04 vs twin 0.06 @19); 0.5 does
  NOT (0.33 vs 0.06) at this horizon. Verdict: deploy-side-only at <=25%, and only
  worth revisiting if the mask is folded into packed weights (kill the autograd
  multiply) or for inference K-chunk skip.
- **SEPARABLE-STUDENT KD: MEASURED DEAD END (via cuDNN)**: 3x3x3 -> (3,1,1)+(1,3,1)+
  (1,1,3) on the full teacher (142 convs factorized, 142.2M -> 93.2M params): 702 ms
  vs 265 ms dense (2.6x SLOWER — three sequential memory-bound 1D passes eat the 3x
  FLOP cut), and KD loss plateaus 3x above the dense control (0.163 vs 0.053
  last-quarter mean @60 steps). Revisit only with a fused custom separable kernel AND
  evidence the accuracy gap closes with longer KD; neither is currently justified.
- **STEP-0 PROBE (int8 workflow's blocking finding, CONFIRMED 2026-07-07)**:
  `torch.is_grad_enabled()` is ALWAYS False inside `autograd.Function.forward`
  (torch disables grad mode there by design) — the "training = symmetric acts"
  branch in Fp8Conv3d.forward is DEAD. Measured branch counts over 2 QAT steps:
  affine 152 (all forward act quants), sym 152 (all BACKWARD dy quants), static 0.
  So int8-QAT training forward has been dynamic-AFFINE all along: (a) train/deploy
  mismatch is only dynamic-vs-static scales; (b) wsum is consumed every training
  step (lazy-wsum item halves); (c) `quantize_i8_affine`'s aminmax .item() sync
  fires per conv per training step — freezing static cal during QAT would kill
  those syncs AND make training deploy-exact. 48-agent optimize plan + full tier
  list: /tmp task wwva43w0g output (Tier-1 exit projection: infer 57.6→~45 ms
  capped, KD wall 306→~205 ms/iter via GPU1 teacher offload).
- **TIER-1 BATCH SHIPPED (2026-07-07 night)**: (1) dead-amax guard (sx computed only
  in the fp8 branch); (2) int8 epilogue-stats binding re-enabled (AFFINE zp precedes
  STATS in the kernel); (3) **s2 dgrad LIVE GRADIENT BUG fixed**: dgrad_s2 reduces
  over Co so per-channel [Co] scales cannot apply — kernel silently used swi[0] for
  every channel (measured 10.0% rel error per call); fix = per-TENSOR forward-layout
  repack stashed in the dgrad cache slots (rel err 1.0% = int8 floor, corr 0.99995);
  s1 dgrad pack upgraded to per-ci [Ci] scales (WS_VEC fires). (4) **STATIC
  CALIBRATION IN TRAINING** — the big one: int8-QAT step had regressed 199->285 ms
  when the affine fallback landed (Step-0: training fwd silently took dynamic affine
  = one aminmax .item() HOST SYNC per conv per step, ~85 ms across 76 convs; ruled
  out: cache poisoning, --cl, stats binding — profile showed ~100 ms/step of
  elementwise glue + sync stalls). Fix: pass frozen i8_cal in training too +
  int8_calibrate at harness start. **Result: 285 -> 207 ms/step (1.29x over fp16
  267), loss healthy (0.009-0.049 vs twin 0.08-0.14), and training fwd is now
  bitwise deploy-exact.** Remaining Tier-1: CL avg-pool kernel, inference-mode
  norm/tail, N-keyed autotune, GPU1 teacher offload (KD wall lever). Full plan:
  docs/design/int8-optimize-plan.md.
- **TIER-1 COMPLETE (2026-07-08)**: (5) CL avg-pool kernel (`avgpool2_cl`, fwd+bwd,
  bit-exact 0.0 diff vs AvgPool3d; swap in swap_convs_fp8); (6) inference-mode
  norm/tail dispatch (module-level no_grad -> in_stats/in_norm_act + block_tail —
  no xhat/sgn/h8 stores, no sh amax). **int8 inference: 44.0 ms/patch on GPU0,
  SD@2 0.9989 PASS** (was 57.6 on capped GPU1). (7) N added to all 5 autotune keys
  (one-time retune, cache re-pinned). (8) stale weight-scale reuse (refresh every
  16 repacks; pack_weight_i8(scales=)). (9) **GPU1 teacher offload + 1-step
  pipeline + CUDA-graphed teacher** (`--teacher-gpu 1`): KD wall 587 -> 475 ms/iter
  = the exact floor (teacher 100% hidden; production single-student wall ~207 ms
  = the plan's 306->205 projection). Two traps found on the way: (a) copying the
  target back at LAUNCH time enqueues a dep-on-GPU1 copy ahead of the student
  kernels on GPU0's default stream — serializes the whole step (wall 605);
  copy at CONSUMPTION after sync(1). (b) eager teacher launch burns ~50+ ms of
  HOST time that cannot overlap — CUDA-graph the teacher (one replay call).
  Loss curves unchanged throughout (0.026-0.035 vs twin 0.13).
- **TIER-2 RESIDENT PORT (norm->conv edges) SHIPPED (2026-07-08)**: `in_norm_act`
  gains EMIT_I8 (consumer's frozen affine (scale,zp), bitwise-parity rounding incl.
  fp16 pre-round); `PreQuantI8` carrier + `Fp8Conv3dLayer._forward_prequant` (no
  permute, no quantize); STRUCTURAL consumer binding in `_bind_consumers` (traced
  tensor-identity binding is UNSOUND: non-module consumers — identity skip, SE
  gates — share the tensor invisibly, and id() reuse after GC fakes hits; also
  CDNR keeps dead alias twins (.conv/.norm) next to all_modules — bind through
  all_modules or a never-calibrated twin gets bound). 32 edges bound; **int8
  inference 42.9 ms/patch GPU0, SD@2 0.9989 PASS**. Tail/cat edges remain
  (multi-consumer — need per-consumer dual emission).
- **TIER-2 INT8 CAT FUSION SHIPPED (2026-07-08)**: `quant_cat2_i8` (cat+affine
  quantize in one kernel, bitwise-parity rounding) + `swap_decoder_cats` binding
  on Decoder/DecoderBody (auto from swap_tails_fp8; inference-only dispatch, train
  falls back to torch.cat) feeding the stage conv's PreQuantI8 fast path.
  **int8 inference: 41.1 ms/patch GPU0, SD@2 0.9989 PASS** (cumulative from 57.6
  capped / 48+ GPU0-class at day start). Training re-gated: 207 ms, loss normal.
  Tier-2 remaining: fused int8 dy emission (train), sign bitpack (memory/batch-3),
  flip-TTA batched driver (quality), tail multi-consumer resident edges.
- **LONG-RUN GATE CAUGHT STATIC-CAL DRIFT (2026-07-08)**: 1500-step int8-QAT twin —
  the int8 lane DIVERGES with a one-shot frozen calibration (loss 0.003 -> 0.31
  climbing while the fp16 twin holds ~0.01): activation ranges drift with the
  weights and the frozen (scale, zp) stops covering them. Invisible in <=14-step
  smokes. FIX: periodic recalibration (`--recal-every`, default 250 — one forward
  on the current crop + refreeze, ~0.2 s per event). Doctrine: any static-cal QAT
  trainer MUST recalibrate periodically; tools/fullscroll/train.py inherits this
  requirement for its qat phase.
- **QAT SCALE-FRESHNESS FORENSICS COMPLETE (2026-07-08)** — 1500-step KD plateaus:
  one-shot static 0.31 DIVERGING; recal-250 0.15-0.20 climbing; recal-50 ~0.14;
  dynamic affine ~0.06-0.08 STABLE (fp16 twin 0.01). Verdict: the misfit is
  CROP-TO-CROP range variation, not temporal drift — no frozen-cal cadence fixes
  it. Sync-free dynamic (quantize_i8_dyn: device-tensor scale AND zp; ZP_PTR in
  _quant_kernel + _conv3d_f8_v2 with exact-int32 zp*wsum epilogue) reaches the
  dynamic plateau at 282 ms — revealing the 78 ms dynamic premium is the aminmax
  REDUCE PASS (full extra activation read x76 convs), NOT host syncs. Sign
  bitpack rode along in these runs (kernels live, loss curves unchanged).
  NEXT LEVER: producer-fused amax + DELAYED SCALING — norm/tail kernels emit
  min/max atomically for free (they already touch every element); conv quantizes
  with the 1-step-stale range x headroom => dynamic-quality scales at ~207 ms.
  Residual ~0.06-vs-0.01 plateau = int8 GRADIENT noise floor (dy int8) — separate
  experiment: dy-fp8-with-int8-fwd on sm120.
- **DELAYED SCALING GATE: PASS (2026-07-08, 1122/1500 steps — run killed by session
  teardown, evidence sufficient)**: `quantize_i8_delayed` (last step's observed
  range x 1.5 headroom; OBS min/max atomics fused into the SAME `_quant_kernel`
  pass, so freshness costs zero extra reads) holds the dynamic plateau —
  0.047-0.10 oscillating, step-1122 loss 0.066, NOT climbing — at **205 ms/step**
  (dynamic 282, fp16 twin 267 => 1.30x). This is now the default int8-QAT training
  path (`self._dyn` always passed when i8; static cal serves only no-grad
  forwards). Frozen-cal recal cadence tuning is CLOSED — delayed supersedes it.
  Pending: dy-fp8 noise-floor 1500-step run (`--bwd-fp8`) to attribute the
  remaining 0.06-vs-0.01 gap to int8 dy vs int8 fwd quantization.
- **TIER-A TRT PROBE PASSED** (trt_probe.py, GPU1, torch 2.12 dynamo ONNX export →
  TRT 11.1): trt-fp16 48.6 ms/patch @128^3 = 1.81x over eager fp16 (gate 1.5x). The
  engine path is healthy on the new stack. Custom fp8 resident (40 ms) still wins on
  sm120; TRT is the 3090/Ampere deploy lane (sparse-int8 engines next there).
  Fixes that were needed: onnxscript install; parser.parse_from_file (dynamo exporter
  externalizes weights); modelopt QDQ int8/fp8/fp4 exports still broken (fp4 ext
  won't compile) — not pursued, quantized-TRT was a verified dead end.
- NEXT: block-sparse ratio sweep (speed + loss-recovery vs dense), int8 resident-path
  port (affine emission in norm/tail kernels) + CUDA-graph the sync-free int8 fwd,
  Tier-A trt_probe, Tier-B fp16 resident bench, 3090-pod re-tune. Bigger board:
  separable-student KD (3x FLOPs, composes with everything), per-layer precision
  auto-assignment; ternary/BitNet = int8-speed on GPUs (curiosity only); int4/W4A8 =
  no conv3d path (dinovol lever).

# fp8 Training Step — Final Ranked Optimization Plan

Baseline: 232 ms/step @128³ (fp16 twin: 266 ms). All items verified against code; duplicates merged across the four lenses. Ranked by revised_ms per unit effort (hours ≈ 0.5 day-equiv, day = 1, multi-day = 3). Every item ends with its validation gate; the standing protocol (one config family at a time on GPU0, `dump_tuned` after, corr-gate vs `torch.nn.grad` references, e2e loss-curve twin) applies throughout.

---

## Tier 1 — hours-scale, do first (~22 ms)

**1. Bias bundle: fused-dtype db + drop norm-absorbed conv biases — ~13 ms, hours** *(merges 5 findings across kernel-micro / memory-traffic / autograd / numerics / scheduling lenses)*
- **What:** (a) Replace `dy.float().sum(dim=(0,2,3,4))` with `dy.sum(dim=(0,2,3,4), dtype=torch.float32)` at `fp8_train.py:102` and `:183` — kills the full f32 materialization of dy (micro-benched on-device: 12.93 → 2.43 ms across all real dy shapes, ~8-10 ms realized). (b) In `swap_convs_fp8`, drop bias entirely (bias=None, `requires_grad_(False)`) for convs whose sibling successor is InstanceNorm — exact absorption, already proven by the inference path (`fp8_forward.py:127`, SD@2 1.0000); kills the eager fwd `y = y + bias` pass (~5 ms) and db for ~155 convs. **Keep** transpconv biases (documented border trap, `docs/design/fp8-conv3d-sm120.md:146-148`), seg-head and SE biases; where bias stays, compute db from the contiguous `dym` view (`dym.sum(0, dtype=f32)`; transpconv: `dyg.sum(0).view(8,Co).sum(0)`).
- **Where:** `fp8_train.py:74-75, :102, :158-159, :183`; `swap_convs_fp8` (successor-is-norm structural assert, must run before `swap_norms_fp8` replaces the norm modules).
- **Gate:** (a) is numerically identical — grad-corr confirms. (b) via e2e loss-curve twin (bias grads are already pure noise; per-param corr meaningless).

**2. Take the SE subgraph (cSE fc1/fc2, sSE) out of the fp8 conv path — ~5 ms, hours**
- **What:** In `swap_convs_fp8`, use the already-available (currently unused) module `name` to skip children under `squeeze_excitation`; SE stays plain fp16 `nn.Conv3d`. Removes ~900 launches of M=1/Co=1 tensor-core-hostile fp8 GEMM chains, plus sSE's full amax+quantize pass over every block output. Design doc already flags this; grad correctness improves (kills the below-0.9 corr tail).
- **Where:** `fp8_train.py:413, :425-433`.
- **Gate:** grad-corr per SE param + step-time and **peak-memory** measurement (fp16 sSE retains the fp16 block output for wgrad, slightly eroding the memory win).

**3. Widen `_cfgs_wg2` for `_wgrad_f8_v2` — ~5 ms, hours**
- **What:** Add BLOCK_CO=128 / BLOCK_N=128 config families and num_stages 3/4 to the wgrad autotune set (currently capped at 64 / stages=2). Keep BLOCK_K ≤ 64 — the crash family was BK=128 + the 1D swizzle, not wide output tiles. SMEM/register budgets fit sm120.
- **Where:** `fp8_conv3d_op.py:471-481`.
- **Gate:** one family at a time on GPU0, corr-gate vs `torch.nn.grad.conv3d_weight`, `dump_tuned`. Worst case: configs lose the sweep, no-op.

**4. `_dgrad_s2_f8`: parity tap-skip + widen the post-crash config stub — ~3 ms, hours**
- **What:** Hoist the (kz,ky) parity test to a block-uniform scalar and skip the ≥15/27 tap iterations that currently run full MMAs on all-masked zeros (runtime guard: `W % BLOCK_M == 0`, fallback for W≤32; guard must key on the autotuner-chosen BLOCK_M). Separately widen its ultra-conservative config set (BM=64/BK=32/w4/s2 only) with BM=128, BK=64, warps 8, stages 3.
- **Where:** `fp8_conv3d_op.py:313-316, :343-365`.
- **Gate:** existing dgrad corr-gate; skipped taps contribute exact zeros.

**5. `ctx.needs_input_grad[0]` guard on dgrad — ~1.5 ms, hours**
- **What:** Stem conv computes and discards a full-res 128³ dx for a `requires_grad=False` leaf input every step, plus a dead `pack_weight_dgrad_fp8`. Return `dx = None` when not needed; skip the dgrad weight pack for such layers (also guard the `dx.to(dy.dtype)` for None; mirror in `Fp8TranspConv3d.backward`).
- **Where:** `fp8_train.py:91-99, :175-176, :211`.
- **Gate:** none needed beyond a smoke run — standard autograd contract, graph-safe.

## Tier 2 — day-scale (~24 ms)

**6. Side-stream wgrad drain — ~15 ms, day** *(highest single day-scale win)*
- **What:** After the shared dy8 quantize, record an event and launch `fp8_conv3d_wgrad` (and the transpconv dW `_scaled_mm`) on a dedicated side stream; single event-join on the main stream before `opt.step()`. Valid because `zero_grad(set_to_none=True)` means nothing reads dw before the step. wgrad is atomics/latency-bound while dgrad is MMA-dense — genuinely complementary. Do **not** do per-op joins and do not expect overlap under saturated shallow-stage conv grids (verified scoping constraint) — the ceiling is ~12-18 ms, not "hide all 60."
- **Where:** `fp8_train.py:100-101, :178-180`; dw allocation must move under the side-stream context (`fp8_conv3d_op.py:547`), `record_stream()` on x8/dy8.
- **Prereq:** pre-pin autotune via `load_tuned` before enabling (concurrent kernels pollute the benchmark-based tuner). Breaks graph-mode capture — acceptable (P2.1 was a no-op).
- **Gate:** e2e grad-corr + loss-curve twin; measure co-execution on one stage pair before wiring all convs.

**7. Wo-aligned scalar M-decomposition in `_wgrad_f8_v2` — ~4 ms, day**
- **What:** constexpr W_ALIGNED path: when `Wo % BLOCK_K == 0` and chunks are row-aligned (true at the dominant shallow stages), od/oh/n_ become block-constant scalar div/mods and zz/yy collapse to scalar+[BN] broadcasts, removing per-chunk vector integer ALU and most of the 9-term mask across ~2000 K-iterations. Launcher picks path per shape; deep stages keep the current path.
- **Where:** `fp8_conv3d_op.py:517-531`.
- **Gate:** bit-identical numerics — corr-gate vs `torch.nn.grad.conv3d_weight`; watch W_ALIGNED × autotuned-BK interaction.

**8. Decoder `quant_cat_shuf` training wrapper — ~4 ms, day (optimistic; pair with #10)**
- **What:** Autograd Function wrapping the existing fused inference kernels `_quant_cat_shuf`/`_quant_cat2` for the training decoder: removes the fp16 cat materialization + its two re-reads, the pixel-shuffle permute-reshape copy fwd and inverse-shuffle `.contiguous()` bwd, and cat's slice-copy backward. New backward kernel is the address-mirror of the existing forward.
- **Where:** `reference.py:112` path via `fp8_train.py:154-157, :169-171, :178-179`; kernels at `fp8_conv3d_op.py:1031-1052`.
- **Needs:** a scale for the fused output before amax is known — take delayed scaling from item #10, or a cheap two-source amax (eats ~0.7 ms of the win). Extend kernels for transpconv bias + device inv_s.
- **Gate:** per-edge grad-corr; the cat/shuffle is linear so no numerics traps.

**9. Conv-epilogue stats (`want_stats=True`) in the training path — ~2.5 ms, day** *(merges 3 findings)*
- **What:** Training ignores the existing STATS epilogue (`fp8_conv3d_op.py:249-256, :279-298`, already a measured −1.8/−2.3 ms win in inference) and instead re-reads every conv output via `in_stats` (`fp8_train.py:267`, 75 norms/step). Bind (Fp8Conv3dLayer, Fp8NormActLayer) pairs at swap time; conv hands (mean, rstd) to the norm. **Sequence after item 1b:** with bias dropped, epilogue stats are exact with zero correction (else correct mean by +bias). N=1 only; keep `in_stats` fallback for batch>1 and non-conv-fed norms (decoder cat). Stash stats on the layer, not the output tensor; plumb the norm's eps.
- **Where:** `fp8_train.py:73, :267`; `swap_norms_fp8`.
- **Gate:** tolerance check mean/rstd vs `in_stats` on one step, then e2e twin; re-check conv fwd kernel time (epilogue atomics at shallow stages).

**10. Producer-fused amax → dual-emit fp8 with delayed scaling — ~10 ms total, multi-day, staged** *(merges 7 findings: producer-fused amax ×3, dual-emit ×2, weight-derived fixed scales, e5m2 dy, tail-xh sharing)*
This is the endorsed known-unimplemented item, refined into a staged attack on the ~14 ms quant+amax budget:
- **Stage A (hours, ~3 ms):** weight-derived fixed scales for norm-act outputs — amax(y) ≤ max_c(|γ_c|·16+|β_c|), cached keyed on `gamma._version` like `_packed()`; consumer conv/tail skips `_amax_scale` entirely (no read pass at all). Log actual_amax/bound per layer first; fall back where >8× loose.
- **Stage B (day, ~4-5 ms):** per-CTA `tl.max(|v|)` + one f32 `tl.atomic_max` epilogue in the four owned producer kernels (`_in_norm_act_train` y16 @ `op.py:766`, `_tail_train_fwd` @ `:913`, `_in_bwd_apply` @ `:843`, `_tail_train_bwd` @ `:937/:949`) and/or fused into `_quant_kernel` with previous-step (TE-delayed) scale. Nonneg f32 atomic_max is exact; masked lanes must contribute `tl.where(mask, |y|, 0)` (loads use other=0 → pre=β corruption otherwise); per-step buffer reset via the want_stats pattern. Covers the dy side that Stage A can't.
- **Stage C (multi-day, ~+4-5 ms):** producers dual-emit the fp8 bytes the next conv needs (one extra ~1B/elem store from hot registers), killing the standalone `quantize_fp8` passes; successor map à la inference `_NEXT8`, resolved at swap time. Dual-consumer edges (tail y16 → conv1 + skip) keep the fp16 emit. Fold in the tail-xh sharing (~1 ms + ~140 MB saved-tensor): reconstruct h = γ·(xh·s)+β from the norm's saved fp8 xhat — multiplication by γ, the *safe* direction (trap (c) is the 1/γ division). Do **not** attempt the x8/xh8 saved-tensor dedup — they are different tensors (post-act y vs xhat); reconstructing xhat from y8 is the documented 1/γ trap.
- **Enabler (hours, probe-gated, ~2 ms):** e5m2 for dy (`fp8_train.py:90/:173`, clamp ±448→±57344, `.to(tl.float8e5)`) — makes delayed/stale dy scales range-safe. **Probe first** (fp4_probe_dot pattern): mixed e5m2×e4m3 `tl.dot` must show corr-parity AND unchanged kernel time (Triton has form emulating exotic dot dtypes on sm120); also probe the transpconv `_scaled_mm` lowering.
- **Gate:** the full behavioral battery — per-op grad-corr, e2e 40-step loss-curve twin, per-layer saturation counter (~0), SD@2 where applicable. Delayed dy scaling weakens the dynamic-quant-as-loss-scale property; margin (amax·1.25) + max-over-last-K history; bootstrap step 0 with the two-pass path.

## Tier 3 — trims, do opportunistically (~7 ms)

**11. Sign-mask bit-packing (8 signs/byte along C) — ~3 ms, day.** `fp8_conv3d_op.py:769/:781/:917/:959` write, `:812/:840/:935` read — ~2-2.4 GB/step of uint8 sgn traffic, 8× reducible; all C and BLOCK_C are multiples of 8 so each CTA owns whole bytes. Preserves the exact-sign-bit trap by construction (bits identical). Also ~0.6 GB off peak memory. Gate: trivial corr (zero numerics change).

**12. SE glue traffic — ~2 ms, 1-2 days.** Re-scope **after** item 2: the correct GAP replacement is `mean(out) == beta2` identically (out is post-InstanceNorm) — zero passes, grad lands on β; folding rank-1 gate grads into `_tail_train_bwd` needs the hand-derived dgs_pre chain (two-pass tail bwd). Per-block grad-corr gate. Do not use the conv-epilogue-stats mechanism here (pre-norm stats ≠ GAP — verified wrong).

**13. Block-uniform interior fast path in `_conv3d_f8_v2` — ~1.5 ms, day.** One scalar interior test before the K-loop, mask-lite body for the ~68% interior CTAs (must also exclude N-fold boundary tiles). Fwd/s1-dgrad only — wgrad K-chunks cross W boundaries, not block-uniform. Could be ~0 if compare ALU overlaps MMA: **measure, keep-or-revert**. Corr-gate mandatory (same sm120 fragility family as the TD≥4 miscompiles).

## Tier 4 — the big swing, last

**14. CUTLASS sm120 implicit-GEMM wgrad collective — up to ~35-50 ms if the ≤20 ms wgrad estimate holds; multi-day, high implementation risk.** Verified: `_wgrad_f8_v2` has no removable redundant traffic — this is the only remaining lever on the 55-70 ms wgrad block beyond items 3/6/7. No upstream sm120 wgrad collective exists. Zero numerics risk (f32 accumulate). Sequence all bandwidth items first; note items 3+7 and the side-stream hiding (6) are partially *superseded* by this if it lands.

## Wall-clock only (does not move the 232 ms number — report wall/iter separately)

- **Teacher forward pipeline/offload — ~150 ms/iter, day.** `teach(x)` is a serial NCHW-eager fp16 forward (~199 ms measured) with zero dependence on the current step: compute teach(x_{i+1}) during step i, ideally on GPU1 with **plain cuDNN fp16, no Triton autotune** (Xid 79 was hardware — do not trust pinned configs to protect it; watch for Xid events). Cheaper orthogonal: teacher via the fp8 resident path (40 ms, SD@2 1.0000, N=1 only) or channels_last (109 ms).
- **Pinned-memory crop prefetch — ~2 ms/iter, hours.** Measured 2.19 ms/step stall, not the claimed 8. Only if harness wall time matters.

## Do NOT do (verified)

- Stochastic rounding (f32 masters/accum/atomics make it useless here; revisit only for fp4 storage-only saved activations).
- Per-op dgrad∥wgrad stream joins; overlap under shallow-stage conv grids (saturated).
- Re-litigating the measured dead ends: im2col+cuBLASLt wgrad, tl.trans, TD≥4, step CUDA-graphing, fp4 dot_scaled.
- DDP on GPU1 (hardware Xid 79, bus-fall wedges CUDA system-wide), sSE-gate fp8 storage (hits the backbone gradient), _pad16 hoist (identity, ~0 ms).

## Projected best-case step time

Overlap-corrected sums (quant+amax budget capped at its ~14 ms ceiling across items 2/8/10; side-stream hiding recomputed against the post-micro-opt wgrad):

| Milestone | Step time | vs fp16 (266 ms) |
|---|---|---|
| Baseline | 232 ms | 1.15× |
| + Tier 1 (~22 ms) | ~210 ms | 1.27× |
| + Tier 2 (~24 ms, overlap-corrected ~20) | ~190 ms | 1.40× |
| + Tier 3 (~6 ms) | ~184 ms | 1.45× |
| + CUTLASS wgrad (net +~19 after superseding 3/6/7's overlap) | **~165 ms** | **1.61×** |

Realistic committed target without CUTLASS: **~185-190 ms** (1.4× over fp16). Best case with CUTLASS landing at its estimate: **~165 ms** (1.6×). Wall-clock per iteration additionally drops by ~150 ms once the teacher is pipelined.
