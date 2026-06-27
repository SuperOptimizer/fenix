# gui — CLAUDE.md

## Purpose
A desktop viewer for inspecting volumes and surfaces. Firewalled, opt-in
(`-DFENIX_GUI=ON`); the core pipeline builds and runs without it. See
`docs/research/research-tools.md` (taberna GUI), `villa-vc.md` (VC3D GUI).

## Public API & key types
- **4-pane viewer:** 3 orthogonal slice MPR panes + 1 **VTK GPU volume-render** pane.
  Loads our **`.fxvol`** volume archive + **`.fxsurf`** surface container.
- Window/level, colormaps, blend modes (composite/MIP/MinIP/avg/iso), slice scroll, LOD
  auto-pick (slice ≤ ~2048, volume ≤ 256³ for GPU upload).
- A single, isolated coupling point to the codec/IO (like taberna's `volume_source`) so
  the rest of the GUI is decoupled from storage internals.

## Inputs / outputs & formats
In: `.fxvol`, `.fxsurf`. Out: on-screen rendering (and screenshots). Read-only **now**;
umbilicus/point-collection **editing later**.

## Dependencies
Intra: `core`, `io`, `codec`. Third-party: **Qt6 + VTK** (GUI-only; never in the core
pipeline). This is the only place C++ links Qt/VTK.

## Invariants & numerics
Pure visualization — does **not** participate in producing the unroll. Must not pull
Qt/VTK into any non-GUI translation unit.

## Performance notes
GPU volume ray-cast via VTK; CPU slice blit. Streams LODs from the archive.

## Gotchas / pitfalls
Note the single-TU rule is for the **core** pipeline; the GUI is a separate opt-in target
(Qt's moc + VTK need their own build). Keep storage coupling in one file.

## Status & TODO
Stub behind `-DFENIX_GUI=ON`. **Image: `Dockerfile.gui`** — both Qt6 (qtbase) and a minimal
VTK are now **built from source against musl + libc++** by `cmake/scripts/build-qt6.sh` +
`build-vtk.sh` (the same scripts CMake's `auto`/`source` resolver uses). Qt6 6.8.1 builds
clean; VTK 9.3.1 builds minimal (volume ray-cast + OpenGL2 + Qt support) with EGL/offscreen,
legacy-GL compat symlinks (musl mesa has no GLVND split), and `-fchar8_t` (bundled fmt needs a
real `char8_t` under libc++). CMake resolves them as `fenix::qt6` / `fenix::vtk`. See
`docs/design/docker.md`. Next: the 4-pane viewer; annotation editing later. Open ADRs:
viewer↔annotate integration.
