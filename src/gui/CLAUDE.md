# gui — CLAUDE.md

## Purpose
Everything under the GUI firewall: the Qt6 **4-pane slice viewer + constraint
annotator** (feeds the tracer + spiral fit — `docs/design/viewer-annotation.md`), plus
four native-VTK **real-time 3D viewers** that are rung 1 of the "seeing the scroll"
volumetric-rendering effort (`docs/design/render3d.md`, ADR 0011). Firewalled, opt-in
(`-DFENIX_GUI=ON`, preset `gui`); the core pipeline builds and runs without it.

## Public API & key types
Five stages, all registered only under FENIX_GUI:
- **`fenix view <vol.fxvol | zarr-root-url> [--surf <s.fxsurf | tifxyz-dir-or-url>] [--anno a.toml] [--cache dir]`**
  (`viewer.hpp`) — the Qt 4-pane window (`ViewerWindow`): xy/xz/yz `SlicePane`s +
  composite `SurfacePane` (`panes.hpp`), crosshair-linked (any pane or the surface sets
  the shared cursor; every pane follows). The volume may be a local `.fxvol` OR an
  OME-zarr multiscale root (local dir or open-data http(s)/s3 URL) — remote streams
  through `io::CachedPyramid` (fetch once → per-LOD `.fxvol` recompress cache under
  `--cache`, default `io::default_cache_dir()`); `--surf` takes a `.fxsurf` or a
  (remote) tifxyz segment dir, transcode-cached via `io::cached_surface`. Rendering is
  `view::SliceEngine` / `view::render_surface_composite` over `ViewerState::src`
  (a `codec::VolumeSource*`) — the GUI never touches codec/io directly except via
  `ViewerState` (`state.hpp`, the single storage coupling point). **VC3D navigation:** wheel = zoom at cursor,
  shift+wheel = slice ± step (shift+G/H adjusts step 1..100), ctrl+wheel = zoom,
  shift+=/− = zoom keys, arrows = pan 64 px, drag/right-drag/middle-drag = pan,
  alt+right-drag = window/level, M = fit, R = focus voxel under mouse, X = recenter all
  panes, space = annotation overlay toggle, C = cycle composite mode, F1 = keybind help,
  double-click = finish edit. Panes focus on hover. **Annotation tools** (toolbar):
  co-winding **stroke** (kind: generic/patch/trace/fiber/kollesis/drawing; optional
  absolute-winding label), **radial line** (+1/+2/… crossings, clicks snap to the local
  intensity ridge), **link** (click two strokes; shift = cannot-link). Save/load =
  versioned TOML via `annotate/`. Composite modes: mean/max/min/alpha/beer-Lambert.
- **`fenix view-chunk [umb=y,x] <chunk.qcchunk...>`** (`chunk3d.hpp`) — GPU-raycast the
  CT of a `.qcchunk` (fenix qc-chunk: raw u8 CT + rasterized band u8) with the mesh band
  as a red `vtkFlyingEdges3D` isosurface; `others=128` renders blue (all segments touching
  the chunk, not just the primary). `umb=` enables face labels: rewritten taberna
  `air_trace` in 3D — material voxel with air within 3 steps along the (chunk-constant)
  umbilicus-outward yx radial → recto (green), inward → verso (cyan); air = below the
  chunk's own Otsu threshold. Keys: n/p chunk, b band, o others, f faces, `[ ]`/`{ }` CT
  window, r reset, q quit.
- **`fenix view-surf <fxsurf> [tex=baked.jpg] [stride=N] [roughness=R]`** (`surf3d.hpp`)
  — real-time PBR render of a segment: uv grid → triangle mesh at `stride` (tcoords
  always address the full-res texture regardless of geometric stride), CT-baked
  papyrus texture (`surf-bake`) as PBR base color, one movable **raking key light**
  (arrows = azimuth/elevation, +/- = intensity) plus a dim headlight fill. Keys: t
  texture on/off, g wireframe, r reset, q quit.
- **`fenix view-vol <fxvol> [z0 y0 x0 d [h w]] [lod=auto] [max_mb=512] [surf=<fxsurf>...] [stride=N] [surfoff=z,y,x]`**
  (`vol3d.hpp`) — one resident brick (whole archive or a region) GPU-raycast via
  `vtkSmartVolumeMapper`, auto-picking the coarsest LOD whose region fits `max_mb`
  through the explicit pyramid; segment meshes overlay as PBR actors in the same scene
  (`detail::surf_actor`, shared with `scroll3d.hpp`). `surfoff` shifts mesh world coords
  into a cropped fxvol's local frame (eval blocks). v1: single brick, no streaming.
- **`fenix view-scroll <cache-prefix>@<zarr-root> [levels=0,3,5] [box=288] [top=5] [surf=<fxsurf>...] [stride=4]`**
  (`scroll3d.hpp`) — whole-scroll multiscale rendering, render3d rung 1: a **clipmap
  stack** of stock `vtkVolume`s — a pinned coarse top level (whole scroll, always
  resident: never-a-hole floor) plus nested finer boxes re-centered on the camera focus.
  Each non-top level is gathered on a background `std::thread` (`ClipLevel`: front/back
  u8 buffers + atomic `ready`/`busy` flags), swapped into the displayed `vtkImageData`
  on a 150 ms `vtkCommand::TimerEvent` (`apply_swaps`) — coarse instantly, sharp seconds
  later, eventual consistency. Data streams per-LOD through `io::CachedVolume`
  (`<prefix>_l<k>.fxvol@<zarr-root>/<k>`): first visit pulls from S3, later visits hit
  local disk. Keys: `[ ]`/`{ }` window, d/D density, f re-center clipmap on focus, s
  surfaces, r reset, q quit.

## Inputs / outputs & formats
In: `.fxvol` (+ explicit LOD pyramid), `.fxsurf`, annotation TOML, `.qcchunk` (raw
header + u8 CT + u8 band payload, ML QC tool), zarr-backed cache prefixes (view-scroll),
baked JPEG textures (`surf-bake` output). Out: on-screen rendering only (view-chunk/
view-surf/view-vol/view-scroll do not write files); `view` additionally writes
annotation TOML.

## Dependencies
Intra: `core`, `codec`, `io`, `view`, `annotate`, `preprocess` (aircut/Otsu, for
view-chunk face labels). Third-party: **Qt6 Widgets** (the 4-pane `view` shell) +
**VTK** (`RenderingOpenGL2`, `RenderingVolumeOpenGL2`, `InteractionStyle`,
`FiltersCore`, `RenderingFreeType`, `IOImage`/JPEG reader) for all four native-VTK
viewers. This is the only place in the tree either links. VTK is no longer reserved for
"a later pane" — it is a fully-adopted second GUI toolkit alongside Qt (ADR 0011: "leverage
VTK as much as we can" for the host/scene/interaction/mesh-PBR layer; a first-party
engine only where VTK can't do out-of-core volume paging).

## Invariants & numerics
Pure visualization/annotation — does not participate in producing the unroll. Qt/VTK
must not leak into any non-GUI TU (fenix.hpp firewalls the include). **No Q_OBJECT / no
moc** — header-only widgets; wiring is lambda connects to stock-widget signals only.
**Qt↔std string bridges go through UTF-8 `char*`** (`toUtf8().constData()` /
`QString::fromUtf8`), never `toStdString`/`fromStdString` — a libstdc++-built system Qt
against our libc++ build breaks at the std::string ABI. **VTK headers use
typeid/throw** — the GUI TU alone is compiled with RTTI+exceptions enabled (see
CMakeLists `FENIX_GUI` block); the rest of the binary stays `-fno-exceptions
-fno-rtti`; nothing VTK-typed may cross the firewall. u8-native throughout the 3D
viewers (no f32 widening of CT — house rule); ZYX on disk, VTK axes are XYZ so every
viewer swizzles exactly once at upload/actor-build.

## Performance notes
- `view` (Qt panes): **fully async + adaptive** — the GUI thread never renders, and a
  render job never blocks on the network. Two `core::WorkerPool`s owned by `run_viewer`
  (stopped BEFORE the window dies): `render_pool` (slice renders + surface composite;
  per-pane latest-request-wins, one job in flight) and `io_pool` (blocking loads — the
  `--surf` fetch/transcode runs after first paint). Render jobs use the BEST-EFFORT
  engine paths (`SliceEngine::render_available`, composite `best_effort`): strictly
  local data, per-chunk coarser-LOD fallback, misses scheduled on the source's
  background fetcher. A 16 ms `QTimer` tick re-renders any pane whose last frame was
  incomplete once `ready_generation()` advances (edge-triggered — an idle sharp
  viewport does no work; eventual consistency at up to 60 fps, "streaming…" badge while
  converging). Workers post images back via `QMetaObject::invokeMethod(widget, functor,
  QueuedConnection)` (no Q_OBJECT needed); `ViewerState::surface` is a
  `shared_ptr<const Surface>` so an in-flight composite keeps its snapshot across a
  reload. `prefetch_around` warms the scroll direction after every render. PERF
  (2026-07-07, whole-scroll streamed 2.4 µm): the driver must set
  `KMP_BLOCKTIME=0`/`OMP_WAIT_POLICY=passive` for `view*` BEFORE `init_thread_limits()`
  (libomp spin-wait was ~60% of CPU), and `CachedPyramid::reserve_cache` gives every
  level half the budget (a depth-halving split starved coarse levels into per-frame
  viewport re-decode). Design informed by VC3D's deleted CAdaptiveVolumeViewer stack
  (villa 7cbbd36be→557766a1b) — kept: fallback ladder, render-discovers-missing
  scheduling, edge-triggered repaint coalescing; skipped: its custom lock-free cache
  tiers and 4-stage pipeline (it was reverted for complexity).
- `view-scroll`: the only viewer with background streaming today — one worker thread
  keeps every non-top `ClipLevel` centered on `want_focus` via `CachedVolume::gather_box_u8`,
  double-buffered (`back`→`front` swap on the render thread only, never mid-frame);
  transient fetch failure just skips a pass and keeps the coarse ancestor (never air).
  `view-vol`/`view-chunk`/`view-surf` are single-shot loads — no streaming, no LOD
  paging beyond `view-vol`'s one-time coarsest-fit pick.
- render3d roadmap (ADR 0011): rung 1 = the clipmap stack (`view-scroll`, shipped);
  rung 2 = a custom `vtkOpenGLGPUVolumeRayCastMapper` shader replacement (brick atlas +
  page table + ancestor fallback + feedback readback) for true per-ray LOD; Metal via
  vendored metal-cpp is the approved escalation path if macOS GL 4.1 becomes the
  ceiling — not started.

## Gotchas / pitfalls
- **The Qt/VTK firewall:** both are parsed in exactly ONE top-level TU —
  `apps/gui.cpp` (mirrors `ml/inference.cpp` for libtorch), added to the fenix target
  under FENIX_GUI in both unity and split builds. `fenix.hpp` does NOT include gui; the
  driver TU stays Qt/VTK-free and that TU's parse compiles in parallel with everything
  else. No moc needed anywhere (no Q_OBJECT).
- Stroke erasure invalidates link indices — only ever erase the just-appended stroke
  (`ViewerState::finish_edit` relies on this).
- macOS dev builds: growth-path `std::string` appends (`operator+` chains) inside VTK
  call sites can abort under the homebrew-LLVM-headers + system libc++ dylib + VTK dyld
  weak-coalescing combination — `scroll3d.hpp`'s cache/zarr path construction uses
  single-shot `snprintf` into a stack buffer instead, deliberately.
- `vtkSmartPointer`/`vtkNew` wrapping externally-owned buffers (`wrap_u8` in
  chunk3d.hpp, the `front`/`buf` arrays elsewhere) always pass `save=1` to
  `vtkUnsignedCharArray::SetArray` — VTK must never try to free memory it doesn't own;
  the backing vector must outlive the `vtkImageData`.
- mimalloc is off on macOS builds (static override cannot interpose system dylibs —
  affects this module because VTK/Qt bring plenty of their own allocations).
- Canonical GUI env is the Chimera docker `Dockerfile.gui` image (source-built Qt6/VTK
  against musl+libc++, `build-qt6.sh`/`build-vtk.sh`); a system Qt/VTK works for dev
  smoke builds subject to the string-ABI rule above.

## Status & TODO
**Implemented:** `view` (Qt 4-pane + full annotation toolset, smoke-tested: offscreen
launch + stage registration, interactive use on desktop). `view-chunk` (CT volume +
band/others/recto/verso isosurfaces). `view-surf` (PBR mesh + raking light + baked
texture). `view-vol` (single-brick LOD-picked volumetric render + mesh overlay).
`view-scroll` (clipmap stack, background streaming worker, never-a-hole pinned top
level) — render3d rung 1, shipped.
**TODO:** `view` — worker-thread renders + progressive refine, surface↔plane
intersection curve overlays, umbilicus editing, point delete/undo, screenshots.
render3d — rung 2 page-table mapper (brick atlas, GPU feedback readback, true per-ray
LOD); Metal backend via metal-cpp if GL 4.1 ceiling is hit; surface depth compositing
with the volume raycast pass; picking/measurement/face-label-as-volume-channel (P4 in
render3d.md). Open ADRs: viewer↔annotate integration (pre-render3d); ADR 0011
(render3d backend) — accepted, rung 2 not started.
</content>
