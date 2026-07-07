# view — CLAUDE.md

## Purpose
The Qt-free viewer ENGINE: interactive-rate slice + surface rendering streamed straight
off a `.fxvol` archive. The `gui` module is a thin Qt shell over this; everything here is
headless-testable and reusable by CLI export. Built for the annotation workflow (see
`docs/design/viewer-annotation.md`) — fast panes to annotate constraints against.

## Public API & key types
- **`SliceEngine`** (`slice_engine.hpp`) — construct over a `codec::VolumeArchive` OR
  any `codec::VolumeSource` (reserves the block cache; the source must outlive it).
  With an `io::CachedPyramid` source the same render path streams a remote zarr:
  first touch fetches + recompresses into the disk cache, later renders are local.
  - `render(SliceSpec)` — axis-aligned xy/xz/yz pane: ONE edge-clamped `gather_box_f32`
    of the covering LOD rect, then bilinear resample (parallel rows). `SliceSpec` =
    axis + slice + view center + zoom (px per LOD-0 voxel) + output size.
  - `render_oblique(ObliqueSpec)` — arbitrary plane (origin + unit du/dv), per-pixel
    trilinear through the block cache.
  - `pick_lod(zoom)` — archive-pyramid LOD so one LOD voxel ≥ one output pixel.
  - `prefetch_around(spec, k)` — queue this viewport's tile ring ± k tile-steps along
    the normal on the background warmer.
  - `SliceImage` — f32 pixels + `pixel_to_volume`/`volume_to_pixel` (the annotation
    tools' coordinate bridge) + `to_u8(lo,hi)` windowing.
- **`render_surface_composite`** (`composite.hpp`) — the surface pane: per surface cell,
  march offsets along the across-sheet normal and reduce: **mean / max / min / alpha
  (front-to-back) / beer_lambert**. Streams via the block cache (no whole-volume RAM);
  falls back to coord-grid tangent normals when the surface has no normal channel.
- **`BlockSampler`** (`sampler.hpp`) — per-thread trilinear sampler with a 2-block memo
  over `block16`; decode failure latches `failed()` (checked per image, never per voxel).
- **`Prefetcher`** (`prefetch.hpp`) — jthread pool + priority heap of 64³ tiles;
  `begin_batch()` drops stale work when the viewport moves; `drain()` for tests.

## Inputs / outputs & formats
In: any `codec::VolumeSource` — a local `.fxvol` (`codec::VolumeArchive` via
`codec::ArchiveSource`) or a streaming source (`io::CachedPyramid`), `core::Surface`.
Out: in-memory f32 images. No disk formats of its own.

## Dependencies
Intra: `core`, `codec` (incl. the `codec::VolumeSource` interface — view never sees
`io`; streaming sources are injected by the caller). Third-party: none (no Qt/VTK
here — that is `gui`).

## Invariants & numerics
ZYX; all public positions are **LOD-0 voxel coordinates** — the engine scales by 2^lod
internally. Codec is lossy: rendered values are tolerance-correct, never bit-exact.
A chunk decode failure fails the RENDER with an error — it must never silently render
as air (absent-vs-failed rule). Prefetch is best-effort and swallows errors (the render
path re-reports them).

## Performance notes
Axis panes are gather-bound: one `gather_box_f32` per view (block-major, one cache lock
per 16³ block), then a parallel bilinear over rows. Oblique/composite are trilinear-bound
through the 2-block sampler memo. The prefetcher hides tile decode behind scroll. LOD
selection keeps work ∝ output pixels, not volume size. GPU path deferred (interface-level
candidates: the composite march + oblique sampling).

## Gotchas / pitfalls
- `SliceSpec.zoom` is px per LOD-0 voxel; `pick_lod` floors to the pyramid — don't
  double-scale coordinates (only the engine divides by 2^lod).
- The prefetcher warms whole 64³ tiles via a representative `block16`; requesting per-16³
  blocks would decode the same tile 64×.
- `Surface` cells without valid neighbours give degenerate tangent normals → the cell is
  skipped (invalid in the output), not rendered garbage.

## Status & TODO
Implemented + tested (`tests/test_view.cpp`, 6 cases): axis panes vs direct decode,
zoomed-out LOD pick, oblique==axis cross-check, composite modes, prefetch cache-warming,
pixel↔volume roundtrip. Consumed by the `gui` module's v1 4-pane viewer (done — see
`docs/design/viewer-annotation.md`). Split-build TU: `view.cpp` (unity build does not
compile it; header-only otherwise).
TODO: progressive refine (coarse-LOD immediate + sharpen callback) for the GUI worker;
surface↔plane intersection curves (for overlays); tricubic option; bench baselines.
