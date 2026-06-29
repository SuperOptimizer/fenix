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

## fenix architecture
- **Stage** — a pipeline unit: a typed node (declared in/out) + a self-registering CLI
  subcommand. The recipe DAG composes them.
- **Recipe (`.fxrecipe`)** — a TOML pipeline definition (ordered stages + params + io).
- **Project (`.fxproj`)** — the workspace grouping a scroll's volumes/surfaces/models/
  annotations/recipes/runs (the `.volpkg` successor).
- **Container codecs** — one CDF 9/7 **wavelet** core (2D + 3D, bitplane-progressive) +
  a **lossless** codec (rANS + filters, for labels/priors). See `src/codec/CLAUDE.md`.
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
