# fenix — Glossary

Domain terms for the Vesuvius scroll-unrolling problem and fenix's architecture.

## Scroll domain
- **Scroll** — a single CT-scanned carbonized Herculaneum papyrus, wound in a spiral.
  Identified by `(scroll_id, energy, resolution[, variant])`. Target: **PHerc Paris 3**
  (≈70k×40k×40k voxels, ~33% dense connected ROI).
- **Segment** — a region of one papyrus sheet that has been traced/flattened. A timestamp
  id in the villa world.
- **Umbilicus** — the scroll's central axis: a polyline (one (y,x) per z). The reference
  for every "radial/outward" direction and the origin of the cylindrical coordinate
  system the unrolling is built on. Highest-leverage human annotation.
- **Sheet / wrap** — the papyrus is one sheet wound into many concentric **wraps**.
  Adjacent wraps *touch* but must not be merged — the central difficulty.
- **recto / verso** — the front/back faces of a sheet (see `conventions.md` for the sign).
- **Winding number / winding field** — a continuous scalar per voxel = how many wraps out
  from the umbilicus. Its level sets are sheets. The **continuous** view of "which wrap."
- **shifted_radius** — `‖yx‖ − θ/2π·dr` in spiral-space; equals `winding · dr`. Turns
  "which wrap" into a smooth scalar. The quantity unifying the winding field and the
  diffeomorphic spiral fit.
- **Diffeomorphic spiral fit** — a global *invertible* map T: scroll→spiral-space that
  straightens the spiral into an ideal Archimedean one (Paul Henderson, `spiral-v2`).
  Invertibility ⇒ no fold-over. The deformation backbone of fenix's unrolling.
- **Gauge / gauge freedom, regauging** — the fit's absolute winding is only defined up to
  a per-component additive constant (the "gauge"): shifting every target in a connected
  component by the same integer number of windings leaves the fit's residuals unchanged.
  **Regauging** (EM re-gauge) picks that constant per component from the residual median
  so it doesn't fight the fit. See `winding/corpus_bridge.hpp` (`regauge_components`).
- **Wrap mod k / gauge-invariant CE** — an ML target/loss design for instance labels: the
  *absolute* wrap index isn't locally inferable from a small patch (a training crop spans
  only ~1 wrap, so it can't see which absolute winding it's on), but `wrap mod k` (a small
  cyclic color, k+1 classes incl. background) is. **Gauge-invariant cross-entropy** takes
  the min CE over the `k` cyclic shifts of the predicted coloring against ground truth —
  the loss judges *relative* ordering/adjacency of colors, not an arbitrary absolute
  phase, so the model isn't penalized for picking a consistent-but-different phase.
- **Instance rings / wrapk** — training-data mode that paints per-cell **instance
  colors**: class `1 + (wrap mod k)` per papyrus voxel (`0` = background, `255` = ignore),
  built from the fitted model's wrap partition + mesh-measured sheet placement
  (`train.py --wrapk`). See `docs/design/multiscale-instance-surface.md`.
- **Unroll / flatten** — produce, per wrap, a flattened (u,v) surface; sample the volume
  along it (± depth layers) to make the image stack ink detection reads.
- **Ink detection** — the downstream ML step that reads flattened layer stacks to reveal
  text. Out of fenix's core, but fenix's `ml/` includes it; its input contract drives the
  render output.

## Data / geometry
- **LOD** — Level Of Detail / resolution level; **0 = full res**, higher = coarser.
- **Chunk** — the 64³ unit of network IO and codec transform; the base addressable block.
- **Coverage tri-state** — per chunk: **NOT_SURE** (ask remote), **ZERO** (air, no blob),
  **REAL** (has data). Lets streaming skip air without fetching.
- **OME-Zarr** — the canonical on-disk/cloud volume format (multiscale, chunked, Blosc).
  fenix reads it (v2/v3/sharded) and transcodes into its own archive.
- **Patch** — a small traced surface piece (a grid of 3D coords); a fit constraint.
- **Track** — a skeletonized polyline through a predicted surface; a fit constraint.
- **Point collection (PCL)** — annotated points with winding labels + link constraints.
- **Surface / tifxyz(-successor)** — a (u,v) grid storing the 3D coord of each surface
  point + validity + named channels. fenix's `.fxsurf`.
- **Structure tensor / Hessian / OOF** — classical sheet-detection front ends producing
  per-voxel sheetness (scalar) + normal (vector). OOF resolves the ~1-voxel inter-wrap gap
  the Hessian merges.
- **Signed affinity graph** — a region-adjacency graph with attractive (same wrap) and
  repulsive (touching different wrap) edges; partitioned by Mutex Watershed / GASP.
- **wrap-fill** — the winding stage that turns the fitted diffeomorphic model into
  **dense per-voxel sheet-instance labels**: the model partitions space into wraps, the
  papyrus mask colors which voxels count. See `src/winding/wrap_fill.hpp`,
  `docs/design/multiscale-instance-surface.md`.
- **Flow-lattice pyramid** — the diffeo fit's coarse-to-fine solve: the SVF/ODE-flow
  field is represented on a multiscale lattice pyramid, solved coarsest-first, each level
  initializing the next — stabilizes convergence and cuts cost vs fitting the finest
  lattice directly.
- **Streaming frontier** — the out-of-core sheet-growth mechanism in `segment/`: the
  tracer advances across block/tile boundaries by carrying its growth frontier forward
  block-to-block rather than requiring the whole volume resident, so tracing scales past
  single-block memory.
- **Winding gate** — a per-step guard in long-range tracing that rejects a proposed step
  if its model-predicted winding jumps too far from the gauge-transported running
  estimate (an EMA alone was observed to leak ~1.6 windings before the gate was added) —
  keeps the traced wrap "pinned" to the correct sheet instead of drifting onto a neighbor.

## fenix architecture
- **Stage** — a pipeline unit: a typed node (declared in/out) + a self-registering CLI
  subcommand. The recipe DAG composes them.
- **Recipe (`.fxrecipe`)** — a TOML pipeline definition (ordered stages + params + io).
- **Project (`.fxproj`)** — the workspace grouping a scroll's volumes/surfaces/models/
  annotations/recipes/runs (the `.volpkg` successor).
- **Container codecs** — one separable all-float **DCT-16 tile** codec (3D volumes in
  64³ tiles = 4³ DCT blocks sharing tile-global rANS tables, + a 2D tile variant) as the
  sole lossy transform codec, + a **lossless** codec (rANS + filters, for labels/priors).
  The CDF 9/7 wavelet core was **retired** (ADR 0005) after the DCT-16 beat it on
  ratio@quality *and* speed across the measured range; LOD is served by an explicit
  multiscale pyramid, not wavelet subbands. See `src/codec/CLAUDE.md`.
- **Artifacts** — `.fxvol` (volume), `.fxsurf` (surface), `.fxmodel` (deformation/spiral
  model), `.fxproj` (project), `.fxrecipe` (recipe).

## Predecessors (study, don't copy — all MIT)
- **taberna** — our C predecessor (winding-field pipeline + matter-compressor + fysics).
- **villa** — ScrollPrize monorepo (volume-cartographer, thaumato, vesuvius(-c), c3d, …).
- **matter-compressor** — taberna's lossy DCT codec + `.mca` archive (codec lineage; its
  integer DCT-16 is rewritten as fenix's all-float DCT-16, kept alongside the c3d wavelet;
  container ideas carried into `.fxvol`).
- **c3d / compress3d** — the user's 3D CDF 9/7 wavelet codec (the wavelet lineage).
- **fysics** — taberna's CT preprocessing (Paganin deconv, dering, denoise, registration).
- **ThaumatoAnakalyptor (TA)** — villa's auto full-scroll winding-graph segmentation.
- **Volume Cartographer (VC/VC3D)** — villa's interactive tracer→flatten→render tool.
- **spiral-v2** — villa branch with Henderson's diffeomorphic spiral fitting.
