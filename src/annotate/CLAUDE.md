# annotate — CLAUDE.md

## Purpose
Human/auto annotations that constrain the unrolling: the (straight, in-memory)
umbilicus, the auto-fit curved umbilicus, co-winding strokes, radial winding lines,
normal hints, link constraints. The `gui` viewer (see `src/gui/CLAUDE.md`) is the main
producer of the constraint document; `winding`, `segment`, `eval` consume. See
`docs/design/viewer-annotation.md`, `docs/research/research-core.md` (annotate),
`spiral-v2.md` (rel_winding / PCLs).

## Public API & key types
- **Umbilicus** (`umbilicus.hpp`) — SoA polyline (`z,y,x` vectors, sorted ascending by
  z); `center(zq)` (linear interp, clamped) / `polar(vz,vy,vx, radius, theta)` gives the
  cylindrical frame; `estimate(material, threshold)` auto-estimates from a material mask
  (per-z centroid + 3-tap z smoothing, straight-line quality only).
- **Curved-umbilicus fit** (`umbilicus_fit.hpp`) — `umbilicus_from_sheets(sheets, params)`:
  fits the axis from traced segment meshes instead of annotation. At every z the sheets
  wrap around the core with across-sheet normals pointing radially at it, so per z-band
  the axis is the least-squares intersection of (cell, normal-line) pairs — a 2x2 normal-
  equations solve in (y,x), IRLS-reweighted (Tukey, `tukey_c`) against curvature/edge
  outliers, unsupported bands inherit the nearest solved neighbour, then 3-tap z-smoothed.
  `UmbilicusFitParams{band, stride, irls, tukey_c}`. `save_umbilicus`/`load_umbilicus`
  are versioned TOML (`kUmbilicusVersion = 1`, `[umbilicus]` section, flat `z=`/`y=`/`x=`
  arrays), atomic temp-rename write. Self-registers the **`umbilicus` CLI stage**:
  `fenix umbilicus surf=<fxsurf>... out=<umb.toml> [band=1024] [stride=4] [irls=3] [tukey=200]`
  — no annotation needed, the corpus measures its own axis; this replaces a straight
  axis, the dominant winding-gauge error over a 70k-z scroll (feeds `winding umb=<toml>`).
- **AnnotationSet** (`annotation.hpp`) — the constraint document:
  - `CoWindingStroke` — a contiguous set of points on ONE winding (patch extracts,
    fibers, a kollesis, drawings — the winding scalar is constant along a sheet arm, so
    any-size same-sheet sets are valid). `StrokeKind{generic,patch_extract,trace_extract,
    fiber,kollesis,drawing}`. Optional absolute `winding` label.
  - `RadialLine` — ordered wrap crossings walking radially outward, each with an integer
    `offset` relative to the first (+1/+2/+3…). Optional absolute `base_winding`.
  - `NormalHint` — sparse across-sheet normal (for the dense lasagna term, winding P4).
  - `StrokeLink` — must-link (same winding; merged by the bridge) / cannot-link
    (different windings; hard repulsive edge for segmentation).
  - `save_annotations`/`load_annotations` — versioned TOML (`kAnnotationVersion = 1`,
    `[stroke.N]`/`[radial.N]`/`[normal.N]`/`[link.N]` sections; points flattened as
    `[z,y,x, z,y,x, …]`), unknown version rejected, atomic temp-rename write.
- **extract_trusted** (`extract.hpp`) — carve co-winding strokes out of an existing
  traced `Surface`: conf≥τ ∧ valid mask → uv 4-neighbour erosion (borders erode by
  design) → 4-connected components → min-cell filter → uv stride subsample → hard cap
  (`max_points`, uniform thinning). "A patch doesn't have to be good everywhere to be
  useful." `ExtractParams{conf_min, erode, min_cells, stride, max_points}`.
- The annotation→fit lowering lives in **`winding/anno_bridge.hpp`** (`to_fit_inputs`),
  NOT here — annotate must not depend on winding (winding depends on annotate; also
  consumed by `winding/cosegment.hpp`, `corpus_bridge.hpp`, `spiral_fit.hpp`,
  `spiral_model.hpp`, `winding_field.hpp`, `winding.hpp`).
- **`annotate.hpp`** — the `annotate` CLI stage itself is still a **stub**
  (`stage_unimplemented`); it just pulls in `annotation.hpp`/`extract.hpp`/
  `umbilicus.hpp` and self-registers. The real annotation-authoring surface is the
  `gui` viewer, not a CLI subcommand; `umbilicus_fit.hpp` registers its own separate
  `umbilicus` stage (see above) which IS implemented.

## Inputs / outputs & formats
**TOML files** read via the shared `core::Config` reader; written by a first-party
writer (atomic temp-rename in both `annotation.hpp` and `umbilicus_fit.hpp`). Two
independent format versions: `kAnnotationVersion = 1` (annotation.hpp) and
`kUmbilicusVersion = 1` (umbilicus_fit.hpp) — mismatched version on either is rejected,
no migration. Umbilicus fit input: one or more `.fxsurf` traced-segment meshes (via
`io::read_fxsurf`); output `.fxvol`/`.fxsurf` consumers are `winding` (`umb=<toml>`).

## Dependencies
Intra: `core` (Config, Surface/`core::surface.hpp`, Vec, VolumeView), `io`
(`io::read_fxsurf`, umbilicus-fit stage only). Third-party: none.

## Invariants & numerics
ZYX, f32. Umbilicus is the cylindrical-frame origin — the highest-leverage input; a
straight umbilicus over a 70k-z scroll is the dominant winding-gauge error, which is why
`umbilicus_from_sheets` exists as a corpus-measured alternative to the straight/manual
one. Radial-line `offset` is per-point relative to `points[0]`; sizes must match (load
rejects mismatch). Link indices are validated against the stroke count on load. The
umbilicus-fit normal equations drop rays whose normal is too axial (`nyx < 0.5`, i.e.
its (y,x) projection is unstable) and bands with `<32` supporting rays (inherit a
neighbour instead of solving).

## Performance notes
Tiny data; trivial for annotation.hpp/extract.hpp/umbilicus.hpp (extraction is O(cells)
BFS on a uv grid). `umbilicus_from_sheets` is O(total uv cells / stride²) to build rays
+ O(rays) per IRLS round per z-band — linear, single-pass over the input meshes at
`stride` subsampling; not parallelized (small relative to the meshes' own tracing cost).

## Gotchas / pitfalls
- Cannot-links must reach the segmentation partition as hard constraints; the fit bridge
  only uses must-links (merging co-winding components).
- Relative vs absolute winding semantics matter: absolute calibrates the global scale,
  relative constrains differences only. A must-link to a labeled stroke PROPAGATES the
  absolute winding across the component (see anno_bridge).
- Extraction gates on validity alone when the surface has no conf channel.
- Two unrelated "umbilicus" paths exist: `Umbilicus::estimate` (fast in-memory centroid
  estimate from a raw material mask, straight-ish, no TOML) vs `umbilicus_from_sheets`
  + `save_umbilicus`/`load_umbilicus` (curved, corpus-measured, versioned TOML, its own
  CLI stage). Don't conflate the two — different quality tiers, different call sites.
- `annotate` (the CLI subcommand) and `umbilicus` (the CLI subcommand) are two different
  self-registered stages living in this directory; `annotate` is unimplemented, don't
  assume a green `fenix annotate ...` run means the document-authoring path works — that
  path is exercised via the `gui` viewer + the header APIs directly, not the stub stage.

## Status & TODO
Implemented + tested (`tests/test_annotate.cpp`): AnnotationSet TOML roundtrip + version
gate, `extract_trusted`, and (in winding) the bridge + rel-winding fit term
(`bridge_lowers_annotations`, `bridge_musklink_propagates_absolute_winding`,
`rel_winding_fit_recovers`). Curved-umbilicus fit (`umbilicus_from_sheets` + TOML IO +
`umbilicus` CLI stage) implemented, not yet covered by `test_annotate.cpp` (no dedicated
test file seen for it). `gui` viewer v1 (stroke/radial/link tools, ridge-snapped radial
clicks, TOML save/load) is done per `docs/design/viewer-annotation.md`. Still stub: the
`annotate` CLI subcommand itself (`annotate.hpp::run`). Next: worker-thread renders +
progressive refine in the viewer, surface↔plane intersection overlays, undo/delete,
fiber extraction from predictions, Winding seeds (Dirichlet) TODO. Open ADRs:
auto-estimate / umbilicus-fit robustness.
