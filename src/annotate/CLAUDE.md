# annotate — CLAUDE.md

## Purpose
Human/auto annotations that constrain the unrolling: umbilicus, point collections, winding
seeds, link constraints. See `docs/research/research-core.md` (annotate), `villa-data.md`.

## Public API & key types
- **Umbilicus** — polyline (one (y,x) per z); load/save + **auto-estimate** from a
  material mask (per-band centroid → angular-variance refinement → smooth).
- **PointCollection** — points + winding annotations (absolute/relative) + **must/cannot-
  link** constraints (cannot-link → hard repulsive edge in segmentation).
- Winding seeds (Dirichlet) for the winding field / fit.

## Inputs / outputs & formats
**TOML files** (umbilicus, collections, links), parsed by the shared `core` config reader.
Consumed by `winding`, `segment`, `eval`.

## Dependencies
Intra: `core`, `io`. Third-party: none.

## Invariants & numerics
ZYX, strong-typed coords. Umbilicus is the cylindrical-frame origin — the highest-leverage
input; treat its accuracy as load-bearing.

## Performance notes
Tiny data; trivial. Auto-estimate is a cheap per-z scan.

## Gotchas / pitfalls
Cannot-links must reach the segmentation partition as hard constraints. Relative vs
absolute winding semantics matter (absolute calibrates scale, relative links).

## Status & TODO
STUB. Open ADRs: annotation TOML schema; auto-estimate robustness.
