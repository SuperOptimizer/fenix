# Viewer + constraint annotation — design

Goal: an interactive viewer (VC3D-style) whose primary job is producing **fast, highly
accurate constraint annotations** for the tracer and the diffeomorphic spiral fit. We do
not need parametric-surface editing; we need slice/surface *viewing* that is fast enough
to annotate against, plus tools that make trustworthy annotations cheap.

## The constraint vocabulary (what annotations must express)

From spiral-v2 / ADR 0003, in fit terms (`winding/diffeo_fit.hpp`):

| Annotation | Fit lowering | Notes |
|---|---|---|
| Co-winding stroke (patch extract, fiber, kollesis, drawing — any contiguous same-sheet set) | `CoWindingGroup` (winding-variance) | winding is constant along a sheet arm, so any-size sets qualify |
| … with absolute winding label | `FitConstraint` per point | absolute calibrates the global scale |
| Radial line (+1/+2/+3 crossings) | consecutive-pair `RelWindingConstraint` | no absolute needed; also yields local spacing (winding-density) samples |
| … with base winding | `FitConstraint` per point | |
| Must-link | merge co-winding components (labels propagate) | |
| Cannot-link | hard repulsive edge (segmentation; not a fit term) | |
| Normal hint | pass-through for the dense lasagna normals term (P4) | |
| Umbilicus | cylindrical-frame origin (existing) | |

Layering: the document (`annotate/annotation.hpp`, versioned TOML) is Qt-free and
fit-free; the lowering lives in `winding/anno_bridge.hpp` (winding already depends on
annotate, never the reverse); the GUI only edits the document.

## Fast trusted-annotation generation (the point of the tool)

- **`annotate::extract_trusted`** (done): carve only the good portions out of existing
  patches/traces — conf∧validity mask, border erosion, connected components, subsample.
  Small-but-certain beats big-but-doubtful.
- **Assisted radial line** (viewer tool, planned): click near the umbilicus and drag
  outward on a slice; each wrap crossing snaps to the nearest sheet-prediction/CT ridge
  along the ray, auto-incrementing the offset — a +1..+N line in seconds.
- **Kollesis strokes**: a full-height same-wrap stroke on the front surface of the sheet
  is just a `CoWindingStroke{kind=kollesis}` spanning z — first-class, no special case.
- Planned: fiber extraction from predictions; extracting clean parts of surface preds.

## Viewer architecture (three layers)

1. **`src/view` — Qt-free slice/composite engine** (next): renders from
   `codec::VolumeArchive` (LOD pyramid, `gather_box_f32`, SIEVE byte-budget cache).
   - Axis-aligned xy/xz/yz + oblique plane reslice at zoom-appropriate LOD, progressive
     refine (coarse LOD immediately, sharpen as chunks land).
   - Surface-normal composite views (offset range along `Surface` normals; reduce modes
     mean/max/min/alpha/beer-Lambert — VC3D's `Compositing` set, minus the cruft).
   - Viewport-distance prioritized async prefetch (the piece the archive lacks).
   - Headless-testable + benchable; the CLI `slice`/`render` stages reuse it.
2. **`src/gui` — Qt shell**: 4 linked panes (xy/xz/yz + surface/composite), crosshair
   sync, wheel slice-scroll, annotation overlays + editing tools, surface↔plane
   intersection curves. Firewalled behind `FENIX_GUI` as today.
3. **`src/annotate` — the document** (done): types + TOML, extraction.

## Status

- Done: annotation model + TOML (v1), `extract_trusted`, `RelWindingConstraint` fit
  term, `anno_bridge::to_fit_inputs`, tests (`test_annotate.cpp`).
- Next: the `src/view` engine, then the Qt shell + tools.
