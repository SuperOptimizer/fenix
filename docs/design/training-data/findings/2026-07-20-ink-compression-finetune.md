# Ink compression fine-tune: 95% blob recovery on q=32 CT (Thunder A6000, 2026-07-20)

Follow-up to [2026-07-19-q32-prediction-impact-ink-tta.md] (q=32 CT flips ink detections).
Method: self-distillation — student init = released `ink_3d_dino_guided`, frozen teacher =
same net at tta=8 on RAW CT; student trained on paired {raw, q32} patches at identical
coordinates. Loss = soft BCE vs teacher (mild pos_w=2) + one-way consistency
`|p_q32 − p_raw.detach()|`. Data: 6× 768³ PHerc0332 regions (3.9–7.3% ink-positive),
5 train + 1 val (r12288a). 6000 steps, batch 14, patch 128, lr 2e-5 cosine, EMA 0.999.
Script: `tools/train/finetune_ink_compression.py` + `prep_ink_ft_data.sh`.

## Headline result (held-out region, tta=8, vs teacher = orig-on-raw)

Blob-level recall (connected components ≥200 vox of teacher prob>0.5):

| pass | thr 0.5 | thr 0.25 | thr 0.125 |
|---|---|---|---|
| orig on q32 | 122/257 | 153/257 | **180/257 (70%)** |
| ft on q32 | 138/257 | 208/257 | **244/257 (95%)** |
| ft on raw | 182/257 | 237/257 | 255/257 (99%) |

**q=32 compression genuinely destroys ~30% of ink detections for the released model**
(absent at any threshold, not just low-confidence). The fine-tune recovers blob survival
to **95%** — but with globally deflated confidence (detections ride lower on the prob
scale; renders visibly dimmer/thinner). PSNR-vs-teacher on q32: 20.88 → 21.47 dB; MAE
worsens (5.82 → 7.17) from diffuse low-prob response — MAE/PSNR alone mislead here;
blob recall at swept thresholds is the honest referee.

## Failure modes found on the way

- **pos_w=10 ink-weighted BCE mis-calibrates** (run 1): val MAE-vs-teacher rose
  monotonically on both inputs. Dense SOFT teacher targets don't need anti-collapse
  weighting (that lesson belongs to hard sparse GT); keep pos_w ≤ 2.
- **Meet-in-the-middle** (run 2): with symmetric BCE, the consistency objective is
  satisfied by both halves converging to a dimmer compromise (val spread 2.04 → −0.14
  while raw MAE rose 10.06 → 10.74). Fix shipped for later runs: `anchor_w=2` upweights
  the raw-half BCE so raw stays pinned and q32 does the moving. Alternative/cheaper:
  post-hoc gain calibration on the val region.

## Generalization + calibration (late-session additions)

- **Fresh-region check (r22528a, z≈22.5k — never seen in training/val; 190 teacher
  blobs):** orig-on-q32 114/137/**149 (78%)** vs ft-on-q32 98/142/**166 (87%)** at thr
  0.5/0.25/0.125. The recovery GENERALIZES (+9 pts at the operating threshold) though
  with a smaller margin than the val region (70→95). At strict thr 0.5 ft is WORSE —
  the dimming signature again: **ft maps must be consumed at thr ≈ 0.125**.
- False-positive blob load at the thr≈0.125 operating point: UNMEASURED (box killed
  mid-computation; the ft net's thr-0.5 new-blob count was 57 vs orig's 43 on the val
  region — expect FP growth at the lower threshold; measure before any bulk triage
  consumes these maps as detections rather than review candidates).
- **Global logit gain/bias calibration is a DEAD END:** the fitted correction (a=1.07,
  b=−0.50) dims everything — the model's error is spatially bimodal (blobs too dim AND
  background haze too bright), which no monotone 1D recalibration can fix (blob recall
  at thr 0.5 dropped 140→113 after "calibration"). Lower threshold at consumption is
  the correct operating point; a haze-aware fix belongs in the next training run
  (heavier anchor or a background-suppression term), not post-hoc.

## Day-2 results (2026-07-21): FP precision + 29-region student

- **FP at the thr≈0.125 operating point (vs teacher):** ft full net 268/407 false (66%)
  val region, 300/481 (62%) fresh region — **~2 of 3 predicted blobs are spurious** at
  the threshold that delivers the recall. The haze crystallizes into fake blobs.
- **29-region student (12k steps, ~4.5 h):** final val MAE 11.14 (vs 6-region 13.7; full
  net floor 10.6). Blob recall on q32: **173/224/243 of 257** — recall parity with the
  281M ft net (243 vs 244) and LESS dimming (173 vs 138 at strict thr). 5× data closed
  the entire recall gap at 12.4× fewer params / ~10× speed.
- **BUT student FP is worse: 538/636 false (85%)** — haze grows as capacity shrinks.
  Ranking at the operating point: ft net ~34% precision, student ~15%, both funnel-only.
- Net doctrine: recall is solved, precision is NOT — background-suppression loss term
  (extra BCE weight where teacher≈0) is mandatory for the next ink iteration; pipeline
  precision comes from cross-model consensus + raw-CT confirmation of candidates.
- Referee reproducibility: rebuilt box (day 2) reproduced day-1 referee numbers exactly;
  ~30 GPU processes ran wedge-free under fenix_retry/watchdog harness.

## Day-3 (2026-07-22): bg-w student — precision doubled, not solved

v3 student = v2 recipe + `--bg-w 3` (BCE ×3 where teacher < 0.02; 12k steps, batch 24).
Val MAE 9.83 — best of any student AND below the 281M ft-net floor (10.6). Referee (q32):

| | recall thr128/64/32 | pred blobs @32 | false | precision |
|---|---|---|---|---|
| v2 (bg-w 1) | 173/224/**243** | 636 | 538 | 15% |
| v3 (bg-w 3) | 158/212/**233** | 545 | 392 | **28%** |

bg-w trades ~10 recall blobs (95%→91% survival) for a near-doubling of precision —
worth it for a triage lane, but haze is only wounded. The global term upweights ALL
background, mostly voxels the student already gets right. Next lever (settled): mined
hard negatives — feed a trained student's own false-blob coordinates back into the
sampler as upweighted negative patches; and hard positives from missed teacher blobs.
Raw-input numbers: recall 171/223/244, 361/497 false — same story.

## Deblocking dead end

3ddct's decode-side deblock filter (`deblock.h`, +0.4 dB on its own corpus) is a no-op
here: on the q=32 exports, seam-plane MAE is only 9% above off-plane — the damage is
in-block quantization/ringing, not blocking. Global PSNR moved −0.08 dB. Dropped.

## Ops (Thunder Compute)

- The shim **wedges CUDA streams nondeterministically** (process asleep, log frozen;
  fake 100% GPU readout). Two triggers seen: any CUDA op from a second thread
  (prefetcher must be pure CPU), and spontaneous (~1 in 3 runs). Standing fix:
  `~/train_watchdog.sh` — stall detector (240 s log freeze) + kill + relaunch with
  `--resume`; full 6000-step run then completed with zero manual intervention.
- A6000 training throughput: 2.06 s/step at batch 14 / patch 128 / 37 GiB (281M net).

## Artifacts / next

- `~/ink_ft_compression.pth` (ema+model), `~/ink3d_ft.fxweights` (converted) on the box;
  archive before killing it.
- **Student KD result (same night):** StudentUNet base=64 **stem_stride=2** (half-res net,
  upsampled output — the full-res stem dominated FLOPs; stride-2 stem = ~10× faster per
  sample than the b64 stride-1 variant AND ~4× faster than the 281M teacher), 22.7M
  params, anchored paired loss, from scratch, 8000 steps @ 0.86 s/step batch 24 (~2 h).
  Referee (same protocol): student-on-q32 PSNR 17.55, MAE 10.67, blob recall
  **106/169/210 of 257** at thr 0.5/0.25/0.125 — i.e. **82% blob survival: beats the
  ORIGINAL 281M net on compressed CT (70%), trails the fine-tuned one (95%)**, at ~10×
  inference speed. Compression-invariant by construction (val spread ≈0 from step 1).
  Next levers: more regions (6 is thin from-scratch), longer runs; GT-refereed eval
  before production trust. Deployment shape: student = cheap triage lane, ft full net =
  quality lane.
- Watchdog lesson: the stall detector's 240 s window MUST be paired with a trainer
  heartbeat (steps 1–5 print eagerly) — cold-start warmups >240 s of log silence caused
  repeated false-positive kills that mimicked shim wedges. The shim ALSO wedges real
  fresh CUDA processes ~50% of the time (timeout+retry wrappers are mandatory for
  every GPU process on Thunder, not just trainers).
- Bulk-ink doctrine unchanged: recon at tta=1 with the ft net on compressed exports is
  now defensible (95% blob survival); definitive pass post-calibration/GT-referee.
