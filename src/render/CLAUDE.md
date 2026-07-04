# render — CLAUDE.md

## Purpose
Sample the volume along a flattened surface to produce the layered texture stack that ink
detection consumes. See `docs/research/villa-vc.md` (Render/Compositing), `villa-thaumato.md`
(mesh_to_surface), `villa-ml.md` (ink input contract).

## Public API & key types
- For a (flattened) surface, build per-output-pixel 3D coords + normals + validity, then
  sample the volume at **± N normal-offset layers** (`base + normal*zStep*i`) into a layer
  stack, using `core` trilinear/tricubic samplers (chunk-cache-aware, out-of-core).
- **Compositing: mean/max first** (more modes — MIP/alpha/beer-lambert/first-hit-iso/
  gradient-mag + transfer LUTs — later). Config-driven Params (layer count ~±32, step,
  bit depth, mode).

## Inputs / outputs & formats
In: `.fxsurf` (flattened) + a volume (`.fxvol`). Out: a **layered surface-volume** stored
via the **2D wavelet codec** in a `.fxvol` (the ink-detection input). Optional PNG/JPEG.

## Dependencies
Intra: `core`, `io`, `codec`, `geom`. Third-party: none (GUI/VTK is separate).

## Invariants & numerics
Output is the ink contract — keep the layer-stack semantics (depth ordering, ±offset,
mask) compatible-in-spirit with TimeSformer-style consumers. f32 sampling.

## Performance notes
Per-pixel × per-layer sampling is the bulk cost — vectorized, chunk-cache-aware, parallel;
out-of-core over surface tiles. Prime later GPU candidate (the sampler).

## Gotchas / pitfalls
The render texture is **not** a plain TIFF dir — it's a `.fxvol` (2D codec) for uniform
tooling, with our IO providing the ink-detection read path.

## Status & TODO
**Implemented (2026-07-04):** `bake.hpp` (`surf-bake` — CT->papyrus texture via the view
composite, percentile-windowed grayscale JPEG mapping 1:1 to uv cells; cache inputs get an
ensure() band prefetch); `layers.hpp` (`render-layers` — the ink-detection layer stack:
(nlayers, nv, nu) u8 via per-cell stencil-normal sampling, z REPLICATE-PADDED to the DCT
block — a zero-pad step rings 6-8 counts into outer real layers, measured; step is in
CURRENT-volume voxels: canonical-2um-era models want step=1 on 2.4um, 7.91um-era models
want step=3.296). TODO: more composite modes, transfer LUTs, GPU sampler, tiled
out-of-core for whole-segment stacks beyond RAM.
