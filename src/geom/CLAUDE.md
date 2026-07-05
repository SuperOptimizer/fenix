# geom — CLAUDE.md

## Purpose
Shared classical geometry/algorithm toolkit used across segment/postproc/eval/flatten/
winding/ml. Replaces villa's vendored cc3d/edt/kimimaro/skimage-ish bits with one
first-party set. `geom.hpp` itself (the stage entry point) is a **stub** — the
underlying primitive headers it includes are real, implemented, and in active use by
other modules (see call sites below). No dijkstra3d, maxflow, or skeletonization exist
yet despite being on the original roadmap — dropped from scope below until built.

## Public API & key types
- **`edt.hpp`** — `edt_squared(VolumeView<const u8> seed) -> Volume<f32>`: exact squared
  Euclidean distance transform, separable per-axis (Felzenszwalb-Huttenlocher lower
  envelope of parabolas). **Unsigned only** — no signed variant exists yet.
- **`connected_components.hpp`** — `connected_components(VolumeView<const u8> mask, Conn
  = Conn::Six) -> CcResult{Volume<s32> labels; s32 count}`. `enum class Conn {Six=6,
  TwentySix=26}`. Parallel z-slab union-find + serial boundary merge + parallel relabel;
  labels are bit-identical to a serial scan (root = first-occurrence voxel in z,y,x
  order, compact ids assigned in ascending-root order).
- **`morphology.hpp`** — `majority_filter(mask, iters=1, thresh=14)` (27-neighborhood
  vote per iteration, iterated ping-pong buffers — "discrete curvature flow" cleanup),
  `dilate(mask, conn=Six)`, `erode(mask, conn=Six)`. **No `open`/`close`/`hole_fill`
  yet** — compose dilate+erode manually if needed.
- **`marching.hpp`** — `marching_tetrahedra(VolumeView<const f32> field, f32 iso) ->
  Mesh`. **Marching tetrahedra, not marching cubes** — each cube split into 6 tets
  sharing the 0-7 diagonal, each tet has 16 unambiguous sign cases, so the surface is
  watertight without MC's 256-case ambiguity table. Triangle-soup output (coincident,
  unindexed shared vertices).
- **`mesh.hpp`** — `struct Mesh {vertices: vector<Vec3f>; tris: vector<array<s32,3>>;
  normals; colors (per-vertex RGB); uvs (per-vertex, array<f32,2>)}`. `write_obj`,
  `read_obj` (fan-triangulates polygons, resolves OBJ 1-based/negative-relative face
  indices, ZYX<->XYZ swap on v/vn, parses `vt` into `uvs` — needed because VC segment
  meshes share the same index between `v` and `vt`), `write_ply` (binary
  little-endian, optional per-vertex RGB). No PLY reader yet.
- **`kdtree.hpp`** — `class KdTree(std::span<const Vec3f>)`, `.nearest(Vec3f) -> s64`
  (index or -1), `.point(s64)`. Median-split build, branch-and-bound query with
  backtracking. Used for nearest-neighbor patch/point association (`segment/
  patch_graph.hpp`, `winding/patch_field.hpp`), not yet for mesh self-intersection
  cleanup (no such cleanup pass exists in `Mesh` yet).
- **`rtree.hpp`** — `struct Box3f{zlo,zhi,ylo,yhi,xlo,xhi}` (ZYX, f32, empty when
  `hi<lo`), `class BoxRTree` (packed STR bulk-load, `kFan=16`, no insert/delete —
  built once from `vector<pair<Box3f,u32>>`, `.query(Box3f) -> vector<u32>` appends
  matching ids). Mirrors VC3D's surface-patch spatial index; used by `ml/
  surface_index.hpp` for multi-mesh (multi-segment) box queries over a training chunk.

**Not present** (previously documented, doesn't exist): signed EDT, MC33, mesh
self-intersection/cleanup/largest-CC, Dijkstra3D, maxflow/min-cut (BK/Dinic),
skeletonization (TEASAR/kimimaro/Lee thinning), `region_betti` (that lives in `topo`,
not re-exported here).

## Inputs / outputs & formats
`Volume<T>`/`VolumeView<T>` and label volumes (`core`), `Mesh`, `Vec3f` point spans,
`Box3f`/id pairs. File I/O: OBJ (read+write) and PLY (write-only) for mesh interop —
consumed by `io/import_obj.hpp` (VC segment OBJ + `transform.json` -> `.fxsurf`) and
`segment/trace_surface.hpp` (writes traced-surface PLY).

## Dependencies
Intra: `core` (Volume, VolumeView, Vec3f, Expected/Errc, parallel_for/parallel_for_z,
cpu_budget). Third-party: none. `geom.hpp` (the stage) additionally pulls in all the
above headers but is otherwise a stub — see Status below.

## Invariants & numerics
- Connectivity stated explicitly per call; default fg is `Conn::Six` (per
  `docs/conventions.md`); eval/postproc call sites pass `Conn::TwentySix` explicitly
  where that's the intended semantics.
- EDT is exact (not an approximation), computed in squared distance to avoid a sqrt
  pass; `edt_big = 1e18f` stands in for +inf (fast-math forbids real `INFINITY`).
- CC labels are deterministic: smaller-index root always wins in `unite`, so results
  don't depend on thread scheduling or slab count.
- `Mesh::vertices` are `Vec3f` in **ZYX** order; OBJ I/O swaps to/from XYZ at the
  file boundary (`v x y z` on disk == `Vec3f{z,y,x}` in memory).
- f32 primary throughout; f64 nowhere in this module currently (no accumulation-heavy
  reductions here yet).

## Performance notes
- EDT and CC both parallelize: EDT per orthogonal-axis line (per-task scratch buffers,
  sized to the volume's max axis length), CC via z-slab-local union-find (`cpu_budget()`
  slabs) with a serial boundary-merge pass (cheap: only slab-boundary planes) and a
  parallel read-only relabel.
- Morphology ops (`majority_filter`, `dilate`, `erode`) are a single full-volume pass
  per iteration/op, 27- or 6/26-neighborhood, `parallel_for_z` over z; ping-pong buffers
  in `majority_filter` avoid in-place hazards.
- None of the current primitives are out-of-core themselves — they take a full
  `VolumeView` in RAM. Block+halo tiling is the caller's responsibility (e.g. eval/
  postproc call these per-chunk).
- `BoxRTree` bulk-load is O(n log n) (STR sort into z/y/x slabs then bottom-up pack);
  query is ~`log_16(n)` node visits. Rebuilt fresh per use — no incremental update path,
  intentional (surfaces are immutable once loaded).

## Gotchas / pitfalls
- Unify primitives here — do NOT let other modules re-roll EDT/CC (the taberna
  3×-eigensolver smell this module exists to prevent).
- `marching_tetrahedra`'s quad-case triangulation: the 4-crossing enumeration order
  from `tedges` yields cycle `c0-c1-c3-c2` (c2/c3 swapped vs adjacency) — the true
  diagonal is `c0-c3`, not the naive `c0-c2`. Get this wrong and you get inverted/
  self-intersecting quads silently.
- `Mesh` here is triangle-soup/interop, not deduplicated or indexed — no shared-vertex
  welding, no self-intersection repair, no largest-CC filter. If a caller needs those,
  it currently has to build them itself; don't assume `geom` provides mesh cleanup.
- The primary surface representation downstream is the `.fxsurf` grid, not `Mesh` —
  `Mesh` is for isosurface extraction / OBJ-PLY interop / ink-hunt import, not the
  winding/flatten pipeline's native format.
- `read_obj` requires vertices to precede faces in the file (rejects forward face
  references) and requires `vt`/`v` to share index per vertex (matches VC segment
  meshes; doesn't handle independently-indexed OBJ texcoords).
- `geom.hpp`'s `run()` is `stage_unimplemented` — there is no `fenix geom` CLI
  subcommand; every primitive here is a library call used by other stages, not a
  user-facing entry point.

## Status & TODO
Implemented and in real use: EDT (unsigned squared), connected components, majority
filter + dilate/erode, marching tetrahedra, Mesh + OBJ read/write + PLY write, KD-tree,
packed R-tree. Stubbed/absent: the `geom` CLI stage itself; signed EDT; MC33 (isosurface
uses marching tetrahedra instead — no plan to add MC33 on top); mesh cleanup
(self-intersection removal, largest-CC, vertex welding); PLY read; hole-fill/open/close
morphology; Dijkstra3D; maxflow/min-cut (BK/Dinic); 3D skeletonization (TEASAR/kimimaro/
Lee thinning). These last four were on the original roadmap for touching-sheet
separation and track extraction but have no code or ADR yet — treat as unscheduled, not
"in progress," until a design lands in `docs/design/` or `docs/adr/`.
