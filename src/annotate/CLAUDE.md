# annotate — CLAUDE.md

## Purpose
Human/auto annotations that constrain the unrolling: umbilicus, co-winding strokes,
radial winding lines, normal hints, link constraints. The viewer (`gui`) is the main
producer; `winding`, `segment`, `eval` consume. See `docs/design/viewer-annotation.md`,
`docs/research/research-core.md` (annotate), `spiral-v2.md` (rel_winding / PCLs).

## Public API & key types
- **Umbilicus** (`umbilicus.hpp`) — polyline (one (y,x) per z); `center`/`polar` +
  **auto-estimate** from a material mask (per-z centroid + z smoothing).
- **AnnotationSet** (`annotation.hpp`) — the annotation document:
  - `CoWindingStroke` — a contiguous set of points on ONE winding (patch extracts,
    fibers, a kollesis, drawings — the winding scalar is constant along a sheet arm, so
    any-size same-sheet sets are valid). Optional absolute `winding` label.
  - `RadialLine` — ordered wrap crossings walking radially outward, each with an integer
    `offset` relative to the first (+1/+2/+3…). Optional absolute `base_winding`.
  - `NormalHint` — sparse across-sheet normal (for the dense lasagna term, winding P4).
  - `StrokeLink` — must-link (same winding; merged by the bridge) / cannot-link
    (different windings; hard repulsive edge for segmentation).
  - `save_annotations`/`load_annotations` — versioned TOML (v1), unknown version rejected.
- **extract_trusted** (`extract.hpp`) — carve co-winding strokes out of an existing
  traced `Surface`: conf≥τ ∧ valid mask → uv erosion (borders erode by design) →
  connected components → min-area filter → stride subsample. "A patch doesn't have to be
  good everywhere to be useful."
- The annotation→fit lowering lives in **`winding/anno_bridge.hpp`** (`to_fit_inputs`),
  NOT here — annotate must not depend on winding (winding depends on annotate).

## Inputs / outputs & formats
**TOML files** read via the shared `core::Config` reader; written by a first-party
writer (atomic temp-rename). Format version `kAnnotationVersion = 1`; mismatched
versions are rejected (no migration). Sections: `[stroke.N]`, `[radial.N]`,
`[normal.N]`, `[link.N]`; points flattened as `[z,y,x, z,y,x, …]`.

## Dependencies
Intra: `core` (Config, Surface, Vec). Third-party: none.

## Invariants & numerics
ZYX, f32. Umbilicus is the cylindrical-frame origin — the highest-leverage input.
Radial-line `offset` is per-point relative to `points[0]`; sizes must match (load
rejects mismatch). Link indices are validated against the stroke count on load.

## Performance notes
Tiny data; trivial. Extraction is O(cells) BFS on a uv grid.

## Gotchas / pitfalls
- Cannot-links must reach the segmentation partition as hard constraints; the fit bridge
  only uses must-links (merging co-winding components).
- Relative vs absolute winding semantics matter: absolute calibrates the global scale,
  relative constrains differences only. A must-link to a labeled stroke PROPAGATES the
  absolute winding across the component (see anno_bridge).
- Extraction gates on validity alone when the surface has no conf channel.

## Status & TODO
Implemented + tested (`tests/test_annotate.cpp`): AnnotationSet TOML roundtrip + version
gate, extraction, and (in winding) the bridge + rel-winding fit term. Umbilicus TOML
load/save still TODO (only in-memory + estimate today). Winding seeds (Dirichlet) TODO.
Next: viewer annotation tools (`gui`), assisted radial-line snapping, fiber extraction
from predictions. Open ADRs: auto-estimate robustness.
