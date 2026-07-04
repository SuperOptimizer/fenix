# v2 from-scratch rehearsal: verdict — pipeline validated

**Goal** (forrest): pretrain from scratch on PHerc Paris 4 @ 2.4 µm to prove the whole
training platform works and that from-scratch approaches the pretrained model.

## Setup
- Corpus: 45 QC-passed 2.4 µm meshes (delta ≥ +1), 5 held-out val meshes; streaming
  zarr→cache feed (q=32), patch=128, bg_frac=0.15, aug=2 (full policy), Otsu-disciplined
  tri-state labels, clDice 0.2, aux material head 0.2, warmup 1000, AdamW 3e-4 cosine,
  base=32 student, batch 8, 25k steps on the community 5090. No teacher (alpha=0).
- Referee: cleanest val mesh 20260603145540 (QC delta +11.4), 512³ block, GT band
  thickness 6, **band-limited eval r=24** (single-segment GT scores untraced wraps as FPs
  otherwise — the referee flaw finding).

## Result (band-limited, same block, same GT)

| model | official | SurfDice@2 | VOI | Topo |
|---|---|---|---|---|
| v2 from-scratch (best ckpt, step 11k) | 0.448 | 0.180 | 0.832 | 0.313 |
| pretrained production recto (tta=8) | 0.460 | 0.199 | 0.798 | 0.371 |

From-scratch lands within ~3% official / ~10% SurfDice of the production teacher after
25k steps — **the training platform (feed → labels → losses → val → export → predict →
eval) is validated end-to-end.** Best val checkpoint was step 11k; the last 14k steps
bought nothing on val (select on val_sep_sma, not steps).

Contrast with v1 (same arch, pre-fix): collapsed to constant (labels), then predicted
sheet-everywhere at volume level (sampler coverage gap). The three fixes that made v2
work: Otsu-disciplined labels, bg_frac random draws, band-limited eval as the referee.

## Follow-ups
- KD phase (alpha>0) A/B should now beat GT-only — the teacher sweep decision is next.
- Per-mesh CE telemetry ({out}_meshloss.json) is live from the next run — use it to rank
  the 45-mesh corpus and cross-check the QC gates.
- Absolute SurfDice vs a thin single-mesh band understates both models identically; use
  these numbers only comparatively.
