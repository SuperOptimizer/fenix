# apps — CLAUDE.md

## Purpose
Holds **`driver.cpp`** — the single real translation unit of the entire program. It is the
ONLY `.cpp` the compiler ever sees for the core build (a unity build).

## Public API & key types
- `driver.cpp` = `#include "fenix.hpp"` (the umbrella header at `src/fenix.hpp`, include
  dir = `src/`; transitively pulls
  in every module) + a `main()` that parses argv, builds a `Context`, and dispatches to the
  **stage registry**. It hard-codes nothing about individual stages.

## Inputs / outputs & formats
CLI: `fenix <subcommand> [opts]` and `fenix run <recipe.fxrecipe>`. Subcommands self-
register from their modules; `driver.cpp` never enumerates them.

## Dependencies
Intra: everything (via `fenix.hpp`). Third-party: whatever the linked modules need
(libcurl/zlib/blosc2/mimalloc; libtorch if ml enabled; Qt/VTK only if `-DFENIX_GUI=ON`).

## Invariants & numerics
This is THE translation unit. Keep it tiny: parse → Context → registry → run. Adding a
stage must never require editing this file (self-registration).

## Gotchas / pitfalls
Don't add a second `.cpp` to the core build. Tests/fuzz/bench are separate TUs that include
the same headers.

## Status & TODO
Stub `driver.cpp` (registry dispatch + `--help`/`info`/`run`). Open ADRs: subcommand UX.
