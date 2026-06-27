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
Full suite. Open ADRs: exact Betti-matching algorithm; VOI_score transform; per-metric
tolerances.
