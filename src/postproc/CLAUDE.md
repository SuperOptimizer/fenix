# postproc — CLAUDE.md

## Purpose
Binary surface-mask cleanup — the Kaggle-winning post-processing toolkit. See
`docs/research/research-core.md` (postproc), `surface-detection-kaggle` notes in
`research-build.md`.

## Public API & key types
- **Morphology** (via `geom`): iterated 3×3×3 majority filter (= discrete curvature flow,
  the single biggest score gain), dust removal (small-CC), hole fill (flood bg from
  border), pinhole plug, ball open/close, normal-aware in-plane close, fragment connect.
- **Sheet repair** — per-CC height-map rebuild over a PCA tangent plane (Laplace inpaint
  gaps → watertight disk).
- **Topo surgery** — PH-guided tunnel filling: window screen via `topo` Betti → localize
  H1 loops (`cubical_features` on inverted window) → precise dilate-fill (vs blunt global
  dilation that kills SurfaceDice).

## Inputs / outputs & formats
In: binary/label surface masks (`.fxvol`). Out: cleaned masks (`.fxvol`).

## Dependencies
Intra: `core`, `geom` (morphology/CC/EDT), `topo` (Betti/PH). Third-party: none.

## Invariants & numerics
Double-buffered alias-safe morphology. Connectivity per `conventions.md`. Repairs must not
tank SurfaceDice (precise > blunt).

## Performance notes
Mostly OpenMP per-voxel; sheet repair is per-CC (parallel across components). Out-of-core
via windows for topo surgery.

## Gotchas / pitfalls
fenix's thesis is to prevent wrap-merges at segmentation (signed affinity + the
diffeomorphic fit) rather than repair post-hoc — postproc is for mask-output/eval paths
and the Kaggle surface-detection task, not the primary unrolling correctness.

## Status & TODO
**Implemented + tested** (ported from taberna `morph.c`/lineage): `cleanup.hpp`
(`remove_small_components`, `fill_holes`) + **`morph.hpp`** (`majority_filter` = iterated
27-neighbour median/surface-tension flow, `connect_fragments(r)` = bridge distinct components across
a gap WITHOUT thickening, `plug_pinholes`, `ball_{dilate,erode,close,open}`). TODO: `inplane_close`
(normal-aware), **sheet_repair** (PCA-height-map watertight rebuild — HARD), **topo_surgery**
(PH-localized tunnel fill, needs `topo/cubical`). Open ADRs: PCA-height-map limits on curved porous
ridges (failed in taberna); windowed surgery params.
