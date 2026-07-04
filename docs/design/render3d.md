# render3d — the multiscale real-time volumetric renderer ("seeing the scroll")

**Goal:** fly through a whole scroll (75784×32693×32693 @ 2.4 µm) in real time: zoom from
the full spiral to a single fiber **in an instant**, with progressive refinement — coarse
LOD immediately, sharper bricks streaming in over frames (eventual consistency, never a
hole, never a stall). Segment meshes render as PBR surfaces in the same scene. This is
the visual substrate for 3D papyrology (raking light, measurement, face labels, picking).

Companions: ADR 0011 (backend choice), `view-vol`/`view-surf` (the VTK v1s this
supersedes for scale), `src/view/` (the CPU slice engine whose LOD/prefetch design this
generalizes to 3D).

## VTK leverage (forrest 2026-07-04: "leverage VTK as much as we can")

VTK stays the HOST for everything it is good at: render window/interactor, trackball,
mesh PBR pass (view-surf machinery), text/HUD/widgets, depth compositing, picking
scaffolding. What VTK cannot do is the volume core — vtkSmartVolumeMapper wants one
resident vtkImageData; out-of-core paging, per-ray LOD, and partial residency need our
own engine. So the GigaVoxels core is a CUSTOM MAPPER inside a VTK scene, built in two
rungs:
- **Rung 1 — clipmap stack (pure stock VTK):** N nested vtkVolumes around the camera
  focus, level k = LOD k at 2^k spacing, each a resident brick box; async re-gather +
  double-buffered swap on camera moves. Instant zoom + progressive refinement with ZERO
  custom GPU code. Ships first; also IS the eventual far-field fallback.
- **Rung 2 — page-table mapper:** custom vtkOpenGLGPUVolumeRayCastMapper shader
  replacements sampling a brick ATLAS through a page table with ancestor fallback +
  a request-feedback pass (virtual-texturing style FBO readback — works within macOS
  GL 4.1). True per-ray LOD, sparse residency.

## Architecture (GigaVoxels/clipmap lineage, adapted to the fxvol pyramid)

### Data model
- **Brick = the archive's native 64³ chunk**, at every pyramid LOD. No re-chunking:
  the codec already stores an explicit LOD pyramid of 64³ tiles; decode-to-brick is the
  existing `gather_box_u8` path.
- Bricks are uploaded with a **1-voxel apron** (66³ effective, duplicated from
  neighbors at upload) so hardware trilinear filtering never bleeds across bricks.

### GPU residency
- **Brick pool**: one large 3D texture atlas (u8), e.g. 8×8×8 slots of 66³ ≈ 147 MB at
  512 slots; sized by a VRAM budget knob. LRU-evicted.
- **Page table**: per-LOD sparse map volume-brick-coord → atlas slot. Implementation: a
  small 3D texture per clipmap level around the camera + a fallback hash buffer for
  far-field lookups. Every entry also stores the finest RESIDENT ancestor so lookup
  misses resolve in O(1) to a coarser brick — **the eventual-consistency invariant:
  every ray sample always has SOME resident data (the top LODs are pinned).**

### Raycaster (compute kernel)
- Per pixel: ray–volume intersect, march with step ∝ current LOD voxel size.
- **LOD selection = cone footprint**: projected pixel size at distance t picks the
  mip level (exactly 3D mipmapping); clamped by residency (sample the finest resident
  ancestor from the page table).
- Sample: hardware trilinear inside the atlas brick; transfer function (window + density
  + papyrus color ramp) as a small 1D LUT texture; gradient shading (central differences
  in-atlas) for the lit look; early-out at opacity ~1.
- **Request feedback**: on ancestor-fallback the kernel appends (brick, lod, priority ∝
  screen coverage) to a GPU request buffer; CPU reads it back each frame.

### Streaming loop (CPU)
frame N: drain request buffer → dedup → priority heap (view/prefetch.hpp pattern) →
async decode workers (codec block cache; CachedVolume for S3-backed archives) → upload
ring (shared-memory on Apple Silicon: zero-copy staging) → page-table update → frame N+k
renders sharper. Camera motion invalidates gracefully: no stall, coarse first.

### Surfaces in the same scene
Segment meshes (strided fxsurf triangles, PBR material, baked CT texture from
`surf-bake`) render in a raster pass sharing the depth range with the volume pass
(volume raycast reads the surface depth buffer to composite correctly). Raking-light
control carries over from `view-surf`.

## Backend: VTK/GL now, Metal underneath when the ceiling bites (ADR 0011)
- **Metal is APPROVED** (forrest 2026-07-04) via **metal-cpp** (Apple's official
  header-only C++ binding, Apache-2.0, vendored) — no Objective-C in the tree.
- Deployment order: rung 1+2 ship on VTK's OpenGL path (max leverage, portable to the
  Linux rig). macOS GL is capped at 4.1 (no compute/SSBO) — when that ceiling limits us
  (atlas size, feedback latency, AO), the volume pass moves to a Metal kernel rendering
  into an IOSurface-shared texture that the VTK scene composites; the scene, meshes,
  and interaction never leave VTK.
- The engine core (brick cache, page table, request heap, LOD policy, transfer function)
  is API-agnostic C++ shared by all rungs.

## Phasing
- **P0** this doc + ADR 0011. ✅
- **P1** clipmap stack in stock VTK (`view-scroll`): nested LOD volumes, async
  re-gather workers, double-buffered swap; meshes overlaid (view-vol machinery). Exit:
  whole-scroll → close-up zoom, coarse-instantly + sharpening-in-seconds, 60 fps motion.
- **P2** page-table mapper (GL shader replacements): atlas + page table + ancestor
  fallback + feedback readback. Exit: per-ray LOD, sparse residency, no clipmap popping.
- **P3** shading polish (gradient PBR-ish, jittered sampling, optional AO), surface
  depth compositing, raking light; Metal volume pass via IOSurface if GL 4.1 limits.
  Exit: "segments look good" at any scale.
- **P4** papyrology tools: picking (ray → brick → LOD-0 voxel), measurement, face-label
  overlay volumes (recto/verso as a second channel), annotation hooks.

## Budgets & invariants
- Never block the render thread on IO or decode. Never render a hole (top LODs pinned).
- ZYX on disk; the atlas is XYZ-natural — the upload does the swizzle once.
- u8 native end-to-end (no f32 widening — house rule).
- Absent-vs-failed: a failed decode keeps the coarse ancestor AND surfaces an on-screen
  diagnostic count; it must never silently render as air.
- Perf targets (M-series): ≥60 fps at 1600×1200 during motion; ≤2 s to visual
  convergence after a full-scroll → close-up jump on warm disk cache; S3-cold limited by
  bandwidth, never by stalls.
