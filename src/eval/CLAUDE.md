# eval — CLAUDE.md

## Purpose
The quality-metric suite: segmentation, surface, topology, and unrolling metrics + the
official composite score. See `docs/research/research-core.md` (eval), `villa-ml.md`
(metrics), `spiral-v2.md` (satisfaction metrics).

## Public API & key types
- Segmentation/surface: **NSD** (Normalized Surface Dice, DeepMind port), **VOI**,
  **adapted-Rand**, **surface-Dice**, **clDice**/center-surface Dice, **critical-
  components** (topological split/merge).
- Topology: **TopoScore** via `topo` (cubical PH + Betti) — build toward **exact**
  Betti-matching, not taberna's dim-1/2 proxy.
- Unrolling (mostly GT-free): **winding consistency/monotonicity**, **det-J fold-
  fraction** (invertibility), **self-intersection**, **satisfied-patches/points**
  (spiral-v2).
- The **Kaggle composite** (`0.30·TopoScore + 0.35·SurfaceDice@2 + 0.35·VOI_score`).

## Inputs / outputs & formats
In: predicted vs GT label/surface volumes, `.fxmodel`/`.fxsurf`, point collections. Out:
metric values → the run stats JSON; tables/reports.

## Dependencies
Intra: `core`, `geom` (EDT/CC), `topo` (PH/Betti), `io`, `annotate`. Third-party: none.

## Invariants & numerics
Metrics computed bit-exact in integer where defined (VOI/Rand contingency tables);
surface/NSD with explicit tolerances. Connectivity per `conventions.md`. EDT from `geom`
(one copy). Asymmetric VOI emphasis (a cross-wrap merge is catastrophic — VOI_merge≫split).

## Performance notes
EDT/CC dominate; reuse `geom`. Windowed/cropped evaluation for large volumes.

## Gotchas / pitfalls
NSD + winding-consistency did NOT exist in villa's harness — implement fresh. Don't ship a
TopoScore that silently uses the proxy where exact is claimed.

## Status & TODO
**Implemented + tested:** `nsd.hpp` (SurfaceDice@τ, EDT-based), `metrics.hpp` (Dice/IoU/VOI),
`topo/betti.hpp` (b0/b1/b2), and **`score.hpp`** — the **Kaggle composite** `official_score()` =
`0.30·TopoScore + 0.35·SurfaceDice@2 + 0.35·VOI_score` (VOI over the union foreground, VOI_score
=1/(1+0.3·VOI), TopoScore = weighted Betti-F1). Verified against the metric spec (discriminates
merge→VOI↓, hole→TopoScore↓). **Refinements TODO (taberna does these, ours approximates):** area-weight
SurfaceDice by marching-cubes surfel area (`NSD_AREA[256]`); tile TopoScore into 2×2×2 octants with
dim-0 EXACT union-matching + 6-conn-fg Betti; exact dim-1/2 Betti-matching via `topo/cubical` (vs the
number-based proxy). Open ADRs: exact Betti-matching algorithm; VOI_score transform; per-metric tolerances.
