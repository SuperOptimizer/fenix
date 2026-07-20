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
- Student KD run (StudentUNet base=64, 22.7M = 12.4× smaller, same data + anchored loss,
  20k steps) queued overnight under the watchdog — tests whether ink survives the shrink;
  needs GT-refereed eval (ink-hunt segments) before production trust, and more regions if
  it plateaus (each ~8 min prep).
- Bulk-ink doctrine unchanged: recon at tta=1 with the ft net on compressed exports is
  now defensible (95% blob survival); definitive pass post-calibration/GT-referee.
