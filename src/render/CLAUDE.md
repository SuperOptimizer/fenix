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
STUB. mean/max + layer stack first. Open ADRs: layer-count/step defaults; compositing
mode set; transfer-function LUTs.
