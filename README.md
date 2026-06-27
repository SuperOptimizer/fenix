# fenix

Virtually unrolling an entire Vesuvius Challenge scroll — a greenfield, header-only,
single-translation-unit **C++26** system. A ground-up rewrite of `~/taberna` and parts of
`~/villa` (volume-cartographer, thaumato, c3d, spiral-v2, vesuvius).

> **Read [`CLAUDE.md`](CLAUDE.md) first.** It is the authoritative project guide for both
> humans and agents. Every subdirectory has its own `CLAUDE.md` with module-specific detail.

## At a glance
- **C++26, LLVM/Clang only** (libc++ + lld + llvm-* tools; no GCC/GNU). Developed entirely
  inside a **Chimera Linux** Docker image (musl + LLVM userland).
- **Header-only**; one real `.cpp` (`apps/driver.cpp`) is the only translation unit; stages
  self-register; includes are transitive.
- **One unified, out-of-core diffeomorphic unrolling method** (blending Henderson's spiral
  fit, taberna's winding field, VC's tracer, thaumato's winding graph).
- **One wavelet codec** (CDF 9/7, 2D+3D, bitplane-progressive) + a lossless codec, in a
  64³-chunk `.fxvol` archive that streams to whole-scroll scale.
- **Fast by default**, minimal dependencies (curl, zlib, blosc2, mimalloc; Qt/VTK for GUI;
  libtorch for ML), everything else written ourselves.

## Build & run (inside the container)
```sh
./bootstrap.sh build        # configure + build the release preset in the Chimera image
./bootstrap.sh test         # build + run tests (debug preset)
./bootstrap.sh shell        # dev shell; then: cmake --preset release && cmake --build --preset release
./build-release/fenix help  # list subcommands
```

## Layout
`src/` — header-only modules (core, geom, io, codec, preprocess, predictions, annotate,
segment, winding, flatten, render, eval, topo, postproc, ml, gui), each with a `CLAUDE.md`.
`apps/driver.cpp` — the single TU. `tools/` — CLI subcommands. `tests/` — unit/golden/fuzz/
bench. `docs/` — conventions, glossary, ADRs, research, design.

## Status
Scaffolding + a compiling stub skeleton. See each module's `CLAUDE.md` (§Status) and
[`docs/adr/`](docs/adr/) for what's implemented vs stubbed. License: MIT.
