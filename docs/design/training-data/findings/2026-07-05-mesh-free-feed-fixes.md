# 2026-07-05 — mesh-free KD feed: three bugs fixed before first real run

Context: the 0009B self-distillation loop (no meshes/GT — upscaled teacher preds are the
only target; `- <ct> <teacher.fxvol> um=8.64` pairs). Smoke-trained 150 steps end-to-end
on forlindesk2 after these fixes; KD loss fell 2.785 → 2.676, no NaNs, RAM flat.

## 1. Occupancy build was a RAM bomb on LOD0-only teachers
The first cut gathered "the coarsest teacher LOD ≤ 512³" to build the origin-sampling
occupancy mask. predict-scroll outputs are **LOD0-only at full-scroll extent** — no such
LOD exists, the search fell through to LOD0, and the "coarse" gather became a dense
full-scroll decode (>100 GB for 0009B; would have been the second RAM-out of the day —
the first was `fenix slice` on a full-extent sparse archive, same root shape).
**Fix:** occupancy now comes from LOD0 **chunk coverage** (`coverage(0, c)` page-table
reads, zero decode, zero RAM risk): Real chunk == teacher wrote sheet signal there.
Also errors out when the teacher has 0 Real chunks — an empty/uncommitted prediction
(exactly what the killed smoke run left behind) now fails loudly at feed startup instead
of silently training on air.

## 2. Reject sampling missed ~96% of draws on sparse occupancy
The origin sampler reject-sampled 24 attempts for an occupied patch center, else took the
last (random) candidate. The 1024³ smoke teacher occupies 0.18% of the chunk grid →
~4% hit rate → ~96% of "occupied" draws silently degraded to random air patches.
**Fix:** direct sampling — collect the occupied-cell index list once, draw a cell
uniformly, jitter within the cell. Exact, deterministic, no misses at any sparsity.

## 3. No deep-air patches in pure mesh-free mode
Occupancy-only sampling never shows the student all-air patches — the exact failure
measured on the mesh path 2026-07-04 (net predicts 'sheet' everywhere at volume
inference), which is why mesh draws have `bg_frac`. **Fix:** `bg_frac` of free draws
take a fully random position; the teacher decodes to fill=0 there = clean air KD target.

## Telemetry
Pure-KD runs (GT all-ignore) logged `sep nan~nan` — the one learning signal watched
during training. train.py now falls back to the **teacher's verdict** for sep when there
are no GT sheet voxels (sheet = teacher>0.5, bg = teacher<0.1).

Commits: 5dea5b4 (coverage occupancy + empty-teacher guard), 28df845 (bg_frac air draws),
190a119 (direct occupied-cell sampling), 3921748 (teacher-fallback sep), 026fb87
(predict-scroll local-.fxvol source — the teacher now decodes CT from disk, 0 network).
