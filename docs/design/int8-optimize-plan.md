# int8 lane deep-optimize — 48-agent workflow plan (2026-07-07)

# fenix int8/fp8 lane — implementation plan

## STEP 0 — blocking 30-minute probe (do before anything else)

Two independent verifications found that `torch.is_grad_enabled()` is **always False inside `autograd.Function.forward`** (torch 2.12.1), making the symmetric branch at `fp8_train.py:83-87` dead code — a live probe showed int8-QAT training forward actually takes the **dynamic-AFFINE fallback** (lines 94-101). The ledger's "training is symmetric" belief is wrong in the opposite direction: the real train/deploy mismatch is **dynamic-affine vs static-affine**.

**Action:** add a one-line print/counter in each branch of `Fp8Conv3d.forward`, run 3 QAT steps, record which branch fires. This settles:
- whether `wsum_i8` is dead in training (Tier-1 item 2's wsum half **evaporates** if the affine branch is live — wsum is consumed every step);
- Experiment 0's framing (what mismatch we're actually quantifying);
- the design of the deploy-matched QAT item (it may be a smaller step than believed).

Update `docs/design/fp8-perf-review-plan.md` with the result either way.

---

## TIER 1 — do now (all CONFIRMED, gain/effort ≥ ~2 ms per half-day)

Ranked by measured-gain-per-effort. Every kernel change gets the standard per-op corr gate + SD@2 ≥ 0.998 re-run (sm120 house rule).

| # | Item | File/function | Gain | Effort |
|---|---|---|---|---|
| 1 | **Channels-last avg-pool kernel** (kills the 6.51 ms `.contiguous()`; measured 6.82 ms overhead on GPU0, replacement is bit-exact vs AvgPool3d) | new `_avgpool2_cl` in `fp8_conv3d_op.py`; swap the 6 `AvgPool3d` skip modules in `fp8_train.py` `_fp8_block_forward`/skip path (:533) | **~5.5-6 ms/patch infer**; ×8 under TTA | 1 day |
| 2 | **Dead-amax guards**: guard `sx = _amax_scale(xm)` at `fp8_train.py:69` with `if i8packed is None` (verified-safe degenerate form), **plus the same fix for `Fp8Tail.forward:501`'s dead sh amax** (remaining ~15-25% of the 3.08 ms AbsMaxOps bucket) | `fp8_train.py:69`, `:501` | **~2.5-3 ms/patch infer** | 2 lines + gate rerun |
| 3 | **Inference-mode norm/tail kernels** (stop writing xhat-fp8/sgn under no_grad). Dispatch at **module level** — `Fp8NormActLayer.forward` and `_fp8_block_forward`, mirroring `fp8_train.py:317` — NOT inside `Function.forward` (dead branch, per Step 0). Wire existing `_in_norm_act` (op.py:897); add store-deleted `_tail_fwd` clone of `_tail_train_fwd` | `fp8_train.py` `Fp8NormActLayer.forward`, `_fp8_block_forward`; `fp8_conv3d_op.py` new `_tail_fwd` | **~2.5-3 ms/patch infer** (norm ~2.1 + tail ~0.6 + skipped amax/allocs) | half day |
| 4 | **Re-enable conv-epilogue stats binding in the int8 lane**: change the `i8 is None` condition at `fp8_train.py:305`. AFFINE zp correction verified to precede STATS accumulation in `_conv3d_f8_v2`, so stats are correct. Gate: per-layer mean/rstd corr vs `_in_stats` | `fp8_train.py:305` | **~1.8-2.3 ms/patch infer** (fp8-lane precedent), ~1 ms/step train | hours |
| 5 | **Weight-repack diet**: (a) lazy `wsum_i8` with None-sentinel in `_i8cache` — **contingent on Step 0**; guard must also cover the no-grad uncalibrated fallback; (b) stale per-channel weight scales, refresh every K=16 steps, **force fresh repack at calibration/export** | `fp8_train.py` `_packed_i8` (:293-301); `pack_weight_i8` (op.py:340) | **2-4 ms/step train** (wsum's `.to(int32)` materializes ~570 MB); amax-skip half (~0.5-1.3 ms) survives regardless of Step 0 | half day |
| 6 | **s1 dgrad per-channel weight scales — pack-side only** (WS_VEC auto-fires, zero kernel change) **and fix the live s2 mis-scaling bug**: `_dgrad_s2_f8` scalar-loads `swi[0]` from a per-channel [Co] tensor (op.py:447) — every int8 stride-2 dgrad is distorted by swi[co]/swi[0]. Fix = new per-ci dgrad-s2 pack + epilogue vector lane | `pack_weight_dgrad_i8` (op.py:369); `_dgrad_s2_f8` + new pack fn | **correctness** (live gradient bug), 0 ms | s1: hours; s2: half day |
| 7 | **N in autotune keys + per-N tuned cache** — prerequisite for ALL batch>1 work; batch-2 numbers today embed batch-1 tile configs | five decorators at `fp8_conv3d_op.py:33,175,392,486,568`; `dump_tuned/load_tuned` (:695/:710) | measurement correctness + low-single-digit ms at N≥2 | hours + retune sweep + corr gate |
| 8 | **Teacher offload to GPU1, 1-step pipeline** — biggest wall-clock lever in the whole set. **Gate first:** measure capped-GPU1 eager fp16-CL teacher forward; must be ≤ ~190 ms to hide under the 193 ms student step; fallback = int8-static teacher (57.6 ms measured, needs KD-quality gate vs fp16-teacher control). Spot-check: recompute 1 target/~200 steps on GPU0, assert KL < eps | `fp8_train_e2e.py` KD loop (teacher_device, prefetch stream, cudaEvent) | **KD wall ~306 → ~200-215 ms/iter (~1.4-1.5×)** | 1-2 days |
| 9 | **Experiment 0** (see accuracy section) — gates all affine-QAT work | `fp8_train_e2e.py` + temp hooks | information; <1 GPU-hour | half day |
| 10 | **Ledger bookkeeping**: close "CUDA-graph the static forward" as measured-unnecessary (idle 0.97 ms/it, max gap 9 µs); record the DDP verdict (vanilla DDP loses to teacher-offload for KD; side-stream/Reducer race documented; manual 2:1 allreduce is the only valid shape if ever needed); record the Step-0 dead-branch finding | `docs/design/fp8-perf-review-plan.md` | closes 2 open items | 30 min |

**Tier-1 exit state (capped GPU1): int8 infer 57.6 → ~44-46 ms/patch; int8 train 199 → ~194-196 ms/step (or ~192 if wsum survives Step 0); KD wall ~306 → ~205 ms/iter.**

---

## TIER 2 — next (build after Tier 1 gates pass)

1. **Producer-fused int8 dy emission with delayed scaling** (ledger item 10 Stage B, int8-symmetric variant). `_in_bwd_apply`/`_tail_train_bwd` EMIT_I8 + atomic-max amax slot; two-pass bootstrap, headroom 1.25-2×, saturation-triggered fallback. Gain **3-5 ms/step** (not 5-8 — backward-only coverage, dres keeps fp16 dual-emit). Gate: KD loss-curve twin match ≥60 steps + weights-median corr.
2. **int8 resident-path port** (EMIT_I8+AFF in `_in_norm_act`/`_block_tail`/`_norm_act_quant`). **~3-4 ms/patch**. Real port work: `_cdnr_fused` must thread int8 pack/wsum/AFFINE — "reuse as-is" understates it. Do together with Tier-1 #3 kernel work if schedule allows.
3. **int8-affine decoder cat fusion** (`_quant_cat2`/`_quant_cat_shuf` INT8/AFF emit, standalone port feeding the conv IS_INT lane — don't drag in the fp8 resident harness). **~2.3-2.8 ms/patch**. Replicate `_quant_kernel` round/clamp+zp exactly for bitwise parity.
4. **Flip-8 TTA batched driver** (`tta_int8` in `fp8_forward.py`) + **tta_chunk memory governor + N-sweep**. Requires Tier-1 #7 (N keys). Measure N∈{1,2,4,8} peak-mem/knee; per-variant SD@2 ≥ 0.998 gate; note transpconv fp8 scales become batch-pooled. Speed ~parity with sequential; value = quality (+0.1-0.17 corr per C++ evidence) + harness for #2/#3 gains (**TTA-8 lands ~350-380 ms capped after #2/#3**).
5. **TTA-coverage calibration** — but **amend `int8_calibrate` finalize to min/max-UNION across observations** (current max-scale-wins does NOT guarantee coverage); broaden to multi-patch. Deliverable: per-variant SD@2 flip-sensitivity table.
6. **Batch-2 QAT measurement** (`--no-fp16-twin` flag; gate on samples/s). After #7. Expected 1.05-1.15× per-sample; decision-grade even if flat.
7. **Sign-mask bitpack** (ledger item 11): −0.6 GiB×B + ~3 ms/step. **Strike the y16-recompute fallback — the retained fp16 it targets doesn't exist.** Batch-3 fit (~13.1-13.5 GiB) is projected, verify by measurement.
8. **Deploy-matched affine QAT forward (STE, dual-quantize)** — **gated on Experiment 0 + Step 0**. Two mandatory fixes: symmetric x8 must *replace* the current save (memory-neutral), and no amax offset exists (sx still feeds bwd). Cost +3-5 ms/step; SD@2 upside +0.0003-0.0013.
9. **Calibration observer upgrade (EMA/percentile, multi-patch)** — cheap (<30 min GPU); real deliverable is the **first held-out measurement of static-cal generalization** (current 0.9987 is train-on-test) + per-layer scale-variance map.
10. **Probes**: per-Cout dy amax spread (20 steps; build the wgrad vector lane only where spread >4×, cost hideable on the wgrad side stream) and **dy underflow audit** (20-step hooks off the timed lane; honest prior is "no widespread starvation" — run as de-risking for delayed scaling, not as the corr-gap explanation).
11. **Batch>1 static-int8 inference** (per-sample stats epilogue, batched SE/tail gc indexing, lift N==1 asserts) — fold into the TTA driver work; realistic ~2 ms/patch-equivalent + TTA parity with the C++ TRT path.
12. **Eval/calibration sidecar on GPU1** — as the *delivery vehicle* for Experiment 0's per-checkpoint drift curves in GPU1 slack behind the teacher. Zero GPU0 gain today (no in-loop stalls exist); checkpoint GPU0 frequently (Xid-79 wedges the box).

---

## PARK (with reason)

- **QAT recalibration schedule** — insurance for Tier-2 #8 which hasn't landed; drift never measured. Exception: the **zp-as-device-pointer** plumbing is moot since CUDA-graphs were measured-unnecessary — park entirely.
- **Octahedral rot16/oct48 TTA** — contingent on flip-8 gate passing; variants run serially at today's 57.6 ms (rot16 ~0.92 s, oct48 ~2.76 s/patch). Revisit only if flip-8 quality demands more and the resident port has landed. Property-test aug→identity→deaug for all 48 elements first.
- **DDP-QAT on GPU0+GPU1** — measured-inferior in expectation to teacher-offload (~1.25-1.6× wide-error vs 1.4-1.5×), GPU1 gradient-corruption blast radius, side-stream/Reducer race. Closed by Tier-1 #10; reopen only if a non-KD real-label phase materializes, and then as manual asymmetric 2:1 allreduce-after-join, never torch DDP.
- **Zero-comm soup/selection** — same-basin premise unsupported (corr evidence is from identical-seed smoke runs); GPU1 is spoken for by the teacher. Keep as an idle-GPU1 ablation mode only.
- **3090-fleet DDP plan** — future hardware. When it arrives: manual post-join flat allreduce with per-layer event bucketing (the naive after-join ordering exposes ~25-30 ms of comm), per-(GPU,triton) tuned pins, ~0.85 efficiency at 4×.

---

## ACCURACY EXPERIMENTS — exact protocol + gates

**Experiment 0 (mismatch quantification)** — half day, <1 GPU-hour, runs on GPU1:
- (a) On the int8-QAT lane, snapshot every 20 steps for 3×60 KD steps; score each snapshot under (i) static-affine deploy (per-snapshot `int8_calibrate` re-run) and (ii) dynamic quant (needs a small force-flag under no_grad — per Step 0, "dynamic" may mean dynamic-affine, not symmetric; label accordingly). Also score the fp16-twin weights under static-affine.
- (b) Per-layer SQNR on one real-CT patch across all 156 layers: training quantizer vs frozen affine; rank by delta.
- **Gate:** if QAT-vs-twin gap under affine deploy < 0.0005 SD@2, **kill Tier-2 #8 and the recalibration item**; budget one longer confirmation run before killing (60-step horizon is suggestive, not conclusive).

**Standing gates for every change:** per-op corr ~0.999 (wgrad 0.9993 baseline), KD loss-curve twin match ≥60 steps, weights-median corr no regression from 0.9376, deployed SD@2 ≥ 0.998 (0.9987 baseline), every new kernel-config family corr-checked before trusting speed numbers.

---

## INTERACTIONS / ORDERING CONSTRAINTS

1. **Step 0 → Tier-1 #5 (wsum), Exp0, Tier-2 #8.** If training runs affine-dynamic, wsum is live every step and only the amax-skip half of #5 ships.
2. **Tier-1 #7 (N keys) strictly before** batch-2 measurement, TTA batched driver, batch>1 inference, N-sweep.
3. **Do not double-count amax savings**: Tier-1 #2 == ledger item 10 Stage A for the int8 lane == the "amax deletion" inside Tier-2 #8. Book it once (Tier-1 #2).
4. **Tier-1 #1/#2/#3 land before the TTA driver** — TTA multiplies fixed waste ×8.
5. **Tier-1 #4 (stats binding) is N==1-only** — it helps batch-1 deploy but is inert at TTA-8; the N-sweep must control for this (~1 ms path difference) or it misattributes scaling.
6. **Tier-1 #3 and Tier-2 #2 share kernel surface** (`_in_norm_act` becomes the EMIT_I8 host) — sequence #3 first, #2 extends it.
7. **Teacher offload (Tier-1 #8) occupies GPU1**; the eval sidecar runs in its slack; the soup idea is foreclosed while it runs.
8. **CL pool int8-emit sub-variant** waits for the resident port and carries a quantize-order (pool∘Q vs Q∘pool) numerics gate — ship fp16-emit now.
9. dy-side changes (delayed scaling, per-Cout dy) touch buffers consumed by the wgrad side stream — respect `record_stream`/`join_wgrad_stream` ordering; probes off the timed lane (enable after i==3).

---

## PROJECTED END-STATE

Assumptions: capped GPU1 runs at ~65-75% of uncapped-class (GPU0/3090-class); "uncapped-class" scales the capped measurement by ~0.72.

| Metric | Today (measured) | After Tier 1 | After Tier 1+2 |
|---|---|---|---|
| int8-QAT train, ms/step (GPU0) | 199 | ~194 (192 if wsum dead) | **~188-191** (+ delayed-scale dy) |
| int8 infer, ms/patch, capped GPU1 | 57.6 | ~44-46 | **~38-41** |
| int8 infer, ms/patch, uncapped-class | ~41 (est.) | ~32-34 | **~28-31** — beats TRT fp16's 48.6 by ~1.6× |
| TTA-8, ms/patch, capped GPU1 | 461 (8× sequential) | ~395-405 (fixed-waste ×8 removed, batched) | **~350-380** (~44-48/variant) |
| KD wall, ms/iter (production single-lane) | ~306 | **~200-215** (teacher on GPU1, 1.4-1.5×) | same; + ~1.05-1.15× per-sample if batch-2 measures positive |
| Deploy SD@2 | 0.9987 (train-on-test, 1-patch cal) | held-out multi-patch number (unknown today) | 0.9987 → up to ~0.999+ if Exp0 justifies affine QAT; else confirmed-benign and closed |

Critical path: **Step 0 probe → Tier-1 #1/#2/#3/#4 (inference stack, ~3 days) ∥ #8 teacher offload (~2 days) → #7 N-keys → Exp0 → Tier 2.** Total Tier-1 effort ≈ 1.5 engineer-weeks; every item independently gated and revertible.