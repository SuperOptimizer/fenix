# 2026-07-04 — Multi-gate verdict: surface models are statistically indistinguishable; block variance dominates

## What ran
`gate_multi.sh`: 5 models × 10 PHercParis4 eval blocks (512³, band-limited eval vs GT
meshes), the variance-corrected replacement for the single-block gates that had been
driving training decisions.

## AGGREGATE (mean ± std over 10 blocks)

| model   | official        | surfdice        |
|---------|-----------------|-----------------|
| recto (upstream) | 0.6112 ± 0.2584 | 0.5000 ± 0.3448 |
| v4 (105k steps)  | 0.6017 ± 0.2631 | 0.5508 ± 0.3084 |
| v3final          | 0.6016 ± 0.2633 | 0.5508 ± 0.3080 |
| v2               | 0.6007 ± 0.2636 | 0.5508 ± 0.3080 |
| v3best (2.5k)    | 0.5978 ± 0.2655 | 0.5601 ± 0.2978 |

## Verdict
1. **No model separation.** All five are within ~0.01 official / ~0.06 surfdice of each
   other with per-block std ~0.26–0.34. At n=10 nothing here is significant. The earlier
   single-block rankings (v3best@2500 "beats" v4@105k etc.) were sampling noise —
   the variance mandate was correct.
2. **Our training runs (v2 → v3 → v4, 105k steps, all label-quality interventions) did
   not measurably move the aggregate.** Fine-tuning steps and schedule are NOT the
   bottleneck. Consistent with the knee-at-25-30k observation and with every label-QC
   finding: quality is **label/data-limited**.
3. **Upstream recto trades surfdice for official** (0.611/0.500 vs our ~0.60/0.55) —
   our runs are slightly better calibrated on surface distance, upstream on the official
   composite; neither difference clears the noise floor.
4. **Block variance (±0.26) is the headline number.** Any future gate must be ≥10 blocks;
   single-block deltas under ~0.15 are unreadable.

## Implications
- Stop investing in surface-model *training* until labels materially improve (trust
  grids / surf-repair / consensus meshes) — per the strategic pivot, surface models are
  the only thing we still train, and even they are data-gated now.
- The eval harness itself (multi-block aggregate) is the durable win; wire it into CI
  as the regression gate (eval-set 12c).
