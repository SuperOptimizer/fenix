# geom — CLAUDE.md

## Purpose
The shared classical geometry/algorithm toolkit used across segment/postproc/eval/
flatten/winding. Replaces villa's vendored cc3d/edt/dijkstra3d/ECL-MaxFlow + kimimaro/
skimage with one first-party set. Implemented **in full** (not stubbed).

## Public API & key types
- **EDT** — exact squared-Euclidean (Felzenszwalb-Huttenlocher), signed + unsigned.
- **Connected components** (cc3d-style, label compaction), connectivity explicit
  (default fg 6 / bg 26), `region_betti`/cc helpers shared with `topo`.
- **Morphology** — dilate/erode/open/close (ball SE), majority filter, hole fill.
- **Marching cubes — MC33** (topologically correct isosurface).
- **Mesh** — tri/quad, verts/faces/normals/UV/named-attrs; cleanup (self-intersection
  via SAT on KD-tree, largest-CC); **OBJ (v/vt/vn) + PLY read/write**.
- **KD-tree**, **Dijkstra3D**, **maxflow/min-cut (Boykov-Kolmogorov + Dinic)** (touching-
  sheet cut, lasagna widest-path), **3D skeletonization** (TEASAR/kimimaro geodesic +
  Lee thinning option, for track extraction).

## Inputs / outputs & formats
In/out: `Volume<T>`, label volumes, `Mesh`, point sets. OBJ/PLY files for mesh interop.

## Dependencies
Intra: `core` (Volume, Vec, arena, parallel_for). Third-party: none.

## Invariants & numerics
Connectivity stated explicitly per call (default fg6/bg26). EDT is exact. Deterministic
tie-breaks where it matters (label ids by index). f32/f64 per `conventions.md`.

## Performance notes
These are workhorses on huge volumes — parallel + cache-friendly + out-of-core-aware
(operate on blocks + halo). Maxflow and skeletonization are the heaviest; profile.

## Gotchas / pitfalls
Unify primitives here — do NOT let modules re-roll EDT/CC/eigensolvers (the taberna
3×-eigensolver smell). `Mesh` is for interop/isosurface; the primary surface is the
`.fxsurf` grid, not a mesh.

## Status & TODO
Full toolkit is an early milestone. Open ADRs: skeleton parameters; maxflow variant
selection per use.
