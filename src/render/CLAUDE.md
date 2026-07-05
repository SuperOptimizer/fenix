# render — CLAUDE.md

## Purpose
Sample a volume along a flattened surface (or winding field) to produce the texture/layer
products downstream consumers need: the ink-detection layer stack, a segment's papyrus
base-color texture, and a whole-volume unrolled image. See `docs/research/villa-vc.md`
(Render/Compositing), `villa-thaumato.md` (mesh_to_surface), `villa-ml.md` (ink input
contract). The real-time interactive 3D renderer (`view-surf`/`view-vol`, GigaVoxels-style
paging) lives in `src/gui/` + `docs/design/render3d.md` / ADR 0011 — **not** this module;
this module's `bake.hpp` supplies that renderer's baked segment texture.

## Public API & key types
Three independent CLI stages, all self-registered (`FENIX_REGISTER_STAGE` /
`register_stage`):
- **`render-layers`** (`layers.hpp`, `run_render_layers`) — the ink-detection layer-stack
  renderer. `fenix render-layers <ct.fxvol|cache@zarr-url> <fxsurf> <out.fxvol>
  [layers=65] [step=1] [q=8]`. Per uv cell: `ml::detail::stencil_normal` gives the local
  normal, then samples `view::BlockSampler::trilinear` at `layers` offsets along it into
  an `(pad_z, nv, nu)` `Volume<u8>` (layer index = Z axis, matching the per-layer-TIFF
  shape upstream ink stacks expect). Rows parallelized (`parallel_for` over `v`), one
  `BlockSampler` per thread.
- **`surf-bake`** (`bake.hpp`, `run_surf_bake`) — bakes a segment's grayscale papyrus
  texture from CT for real-time display. `fenix surf-bake <ct.fxvol|cache@zarr-url>
  <fxsurf> <out.jpg> [lo=-4] [hi=4] [step=1] [mode=mean|max|beer] [q=92]`. Delegates the
  actual sampling/compositing to `view::render_surface_composite` (shared with the GUI's
  slice engine), then does 1st/99th-percentile windowing to u8 and writes a grayscale
  JPEG (1:1 with uv cells) via `io::write_jpeg`.
- **`render`** (`render.hpp`, `run`) — the simplest end-to-end path, no surface/segment
  needed: `fenix render <in.fxvol> <out.fxvol> [pitch=8] [samp=4]`. Estimates a
  material threshold (mean intensity), runs `annotate::Umbilicus::estimate`, then
  `winding::winding_init`, then `unroll()` (below). Despite the CLI usage string still
  writes a `.fxvol` (the only volume container fenix writes; NRRD was removed project-wide).
- **`unroll()`** (`unroll.hpp`) — flatten-by-winding-field: forward-scatters each CT voxel
  into a `{1, H, W}` image by `(z -> row, winding -> column)`, accumulating a running mean
  per bin (`UnrollParams::samp` = output columns per unit winding). No surface/mesh
  involved; this is the whole-volume "straighten the spiral" path, a different lineage
  from the surface-based layer/bake stages above.
- **`render_surface()`** (`surface_render.hpp`) — an earlier in-core prototype: builds
  tangents from one-sided differences over the `Surface` coord grid (skipping invalid
  neighbours), cross-product normal, then samples `±num_layers` at `step` into a
  `RenderResult{stack, mask}`. **Superseded by `view::render_surface_composite` +
  `view::BlockSampler`** (the chunk-cache-aware, out-of-core streaming path used by
  `render-layers`/`surf-bake`). Kept only for `tests/test_surface_render.cpp` and pulled
  in transitively by `render.hpp`; not wired to any CLI stage. Do not extend it — extend
  the `view::` sampling path instead.

## Inputs / outputs & formats
- In: `.fxsurf` (flattened surface, `io::read_fxsurf`) + a volume — either a local
  `.fxvol` (`codec::VolumeArchive::open`) or a `cache@<zarr-url>` (`io::CachedVolume`,
  chunk-cache over a remote OME-zarr).
- Out: `.fxvol` for both `render-layers` (u8 layer-stack volume, DCT-16 tile codec,
  `q` quality knob) and `render` (f32 unrolled image, a 1-deep volume); JPEG for
  `surf-bake` (`io::write_jpeg`, 1 component, quality `q`).
- **No wavelet anywhere** — the CDF 9/7 wavelet was retired project-wide (ADR 0005); all
  `.fxvol` output here goes through the DCT-16 tile codec (`codec::VolumeArchive` +
  `codec::DctParams`). (Prior revision of this doc said "2D wavelet codec" — stale.)
- `cache@url` inputs are prefetched with `CachedVolume::ensure()` over a coarse
  (every-4th-cell) uv sweep, padded by the max sample offset + 2, before the dense
  per-cell sampling loop runs — the composite/trilinear samplers read the archive
  directly and must never observe absent-as-air.

## Dependencies
Intra: `core`, `io` (`cached_volume`, `surface`, `jpeg`), `codec` (`archive`), `annotate`
(`umbilicus`, only for the `render` stage), `winding` (`winding_field`, only for `render`),
`view` (`composite`, `sampler` — the shared streaming sampler/compositor), `ml` (`surf_qc`
for `stencil_normal`). Third-party: none (GUI/VTK is a separate module, `src/gui/`).

## Invariants & numerics
- Output of `render-layers` is the ink-detection contract — keep layer-stack semantics
  (Z = layer index, ± offset ordering, replicate-not-zero padding) compatible with
  TimeSformer-style consumers.
- **`step` units differ from prior docs' assumption**: it is voxels in *this* volume's
  spacing, not a physical constant. Upstream ink models trained at 7.91 µm/1-voxel
  spacing need `step≈3.296` when rendering from 2.4 µm rescans (same physical spacing);
  fenix-native models trained at 2.4 µm want `step=1`.
- `render-layers` **replicate-pads** the z axis up to the next multiple of 16 (the DCT
  block size) instead of zero-padding — measured: zero-padding rings 6-8 counts into the
  outer *real* layers on a synthetic gradient stack.
- Decode/fetch failure during sampling is a **hard fail**, never silent air:
  `BlockSampler::failed()` is checked after each row and `render-layers` returns
  `Errc::io_error` if ever set.
- f32 sampling/accumulation throughout; final layer stack quantizes to u8 (`+0.5f`
  round, clamped to [0,255]); `surf-bake`'s JPEG windowing clamps to [0,1] before *255.
- `surf-bake` requires ≥64 valid pixels post-composite or hard-errors (a near-empty
  surface would otherwise produce a degenerate percentile window).

## Performance notes
Per-pixel × per-layer/offset sampling is the bulk cost — `render-layers` parallelizes by
row (`parallel_for` over `v`) with a per-thread `BlockSampler` (memoizes the last two
16³-ish blocks touched, since a trilinear tap cluster usually straddles at most two).
Both `render-layers` and `surf-bake` `reserve_cache()` several GiB on the archive before
sampling. Out-of-core via the cache-backed archive + `CachedVolume` prefetch; GPU sampler
remains a future candidate (real-time volumetrics already went this route in `src/gui/`,
this module hasn't yet).

## Gotchas / pitfalls
- `render.hpp`'s CLI usage string still says "flattened NRRD image" — it's stale; the
  actual output is `.fxvol` (NRRD writing was removed project-wide).
- `surface_render.hpp`/`render_surface()` is legacy/in-core and NOT the sampling path used
  by any registered CLI stage — don't confuse it with `view::render_surface_composite`,
  which is what `render-layers`/`surf-bake` actually call.
- Two unrelated "layer stack" concepts exist in this dir: `unroll.hpp`'s whole-volume
  winding-field scatter (`render` stage) vs `layers.hpp`'s per-surface-cell normal-offset
  stack (`render-layers` stage). Don't conflate their params (`samp` vs `layers`/`step`).
- `step=` semantics are volume-relative, not physical — see Invariants above; picking the
  wrong step silently mismatches an upstream model's training spacing without erroring.

## Status & TODO
**Implemented (2026-07-04):**
- `render` — umbilicus estimate -> winding field init -> `unroll()` -> `.fxvol`.
- `unroll.hpp` — mean-accumulate winding-to-column scatter.
- `bake.hpp` / `surf-bake` — CT -> papyrus base-color JPEG via `view`'s streaming
  mean/max/beer_lambert composite, percentile-windowed to u8, cache-aware prefetch.
- `layers.hpp` / `render-layers` — the ink-detection `(layers, nv, nu)` u8 surface volume,
  stencil-normal sampling, replicate-padded to the DCT-16 block, hard-fails on any
  decode/fetch error.
- `surface_render.hpp` — superseded in-core prototype, retained for its unit test only.

**TODO:** more `surf-bake`/layer composite modes (MIP/alpha/first-hit-iso/gradient-mag +
transfer LUTs — `view::CompositeMode` already has `min`/`alpha`/`beer_lambert` beyond
mean/max; only mean/max/beer are exposed via `surf-bake`'s CLI so far), GPU sampler,
tiled out-of-core for whole-segment layer stacks beyond RAM, reconciling `render`'s
winding-field unroll path with the surface-based layer/bake lineage (currently separate
code paths with no shared abstraction).
