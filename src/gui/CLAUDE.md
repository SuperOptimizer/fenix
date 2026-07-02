# gui — CLAUDE.md

## Purpose
The interactive viewer + **constraint annotator**: a 4-pane desktop app for inspecting
volumes/surfaces and producing the annotations the tracer + spiral fit consume
(`docs/design/viewer-annotation.md`). Firewalled, opt-in (`-DFENIX_GUI=ON`, preset
`gui`); the core pipeline builds and runs without it.

## Public API & key types
- **`fenix view <vol.fxvol> [--surf s.fxsurf] [--anno a.toml]`** — the `view` stage
  (registered only under FENIX_GUI).
- **4-pane window** (`viewer.hpp` ViewerWindow): xy/xz/yz `SlicePane`s + the composite
  `SurfacePane` (`panes.hpp`), crosshair-linked (any pane or the surface sets the shared
  cursor; every pane follows). Rendering is `view::SliceEngine` /
  `view::render_surface_composite` — the GUI never touches codec/io directly except via
  `ViewerState` (`state.hpp`, the single storage coupling point).
- **Interaction:** wheel = slice scroll (shift ×10), ctrl+wheel = zoom, drag/middle-drag
  = pan, right-drag = window/level, double-click = finish edit.
- **Annotation tools** (toolbar): co-winding **stroke** (kind: generic/patch/trace/
  fiber/kollesis/drawing; optional absolute-winding label), **radial line** (+1/+2/…
  crossings, clicks snap to the local intensity ridge — the assisted part), **link**
  (click two strokes; shift = cannot-link). Save/load = versioned TOML via `annotate/`.
- Composite modes for the surface pane: mean/max/min/alpha/beer-Lambert.

## Inputs / outputs & formats
In: `.fxvol`, `.fxsurf`, annotation TOML. Out: on-screen rendering + annotation TOML.

## Dependencies
Intra: `core`, `codec`, `io`, `view`, `annotate`. Third-party: **Qt6 Widgets only** for
the current shell (VTK reserved for a later GPU volume-render pane; never in the core
pipeline). This is the only place C++ links Qt/VTK.

## Invariants & numerics
Pure visualization/annotation — does not participate in producing the unroll. Qt must
not leak into any non-GUI TU (fenix.hpp firewalls the include). **No Q_OBJECT / no moc**
— header-only widgets; wiring is lambda connects to stock-widget signals only.
**Qt↔std string bridges go through UTF-8 `char*`** (`toUtf8().constData()` /
`QString::fromUtf8`), never `toStdString`/`fromStdString` — a libstdc++-built system Qt
against our libc++ build breaks at the std::string ABI (found the hard way).

## Performance notes
Panes render synchronously in paint (the engine's LOD pick keeps it interactive);
`prefetch_around` warms the scroll direction after every render. TODO: move renders to
a worker thread + progressive coarse→fine refresh for huge volumes over S3.

## Gotchas / pitfalls
- **The Qt firewall:** Qt is parsed in exactly ONE top-level TU — `apps/gui.cpp`
  (mirrors `ml/inference.cpp` for libtorch), added to the fenix target under FENIX_GUI
  in both unity and split builds. `fenix.hpp` does NOT include gui; the driver TU stays
  Qt-free and the Qt parse compiles in parallel. No moc needed because no Q_OBJECT.
- Stroke erasure invalidates link indices — only ever erase the just-appended stroke
  (`ViewerState::finish_edit` relies on this).
- Canonical GUI env is the Chimera docker `Dockerfile.gui` image (source-built Qt6/VTK
  against musl+libc++, `build-qt6.sh`/`build-vtk.sh`); a system Qt works for dev smoke
  builds subject to the string-ABI rule above.

## Status & TODO
**v1 implemented**: 4-pane viewer + annotation tools as above, smoke-tested (offscreen
launch + stage registration; interactive use on a desktop). TODO: worker-thread renders +
progressive refine; surface↔plane intersection curve overlays; umbilicus editing; point
delete/undo; VTK volume-render pane; screenshots. Open ADRs: viewer↔annotate integration.
