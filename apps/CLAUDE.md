# apps — CLAUDE.md

## Purpose
Holds the program's entry-point TUs: **`driver.cpp`** (always) and **`gui.cpp`** (only
under `-DFENIX_GUI`). In the default (UNITY) build `driver.cpp` is the ONLY `.cpp` the
compiler sees for the core binary.

## Public API & key types
- `driver.cpp` — two build modes, chosen by `#if defined(FENIX_SPLIT)`:
  - **UNITY (default):** `#include "fenix.hpp"` (umbrella header at `src/fenix.hpp`,
    include dir = `src/`) pulls in every module transitively; every stage
    self-registers via this one TU.
  - **SPLIT (`-DFENIX_SPLIT=ON`, dev-only):** includes only `codec/archive.hpp` +
    `core/config.hpp` + `core/core.hpp` (what `main()` itself touches). Each module
    is compiled separately as its own TU at `src/<mod>/<mod>.cpp` (NOT
    `src/units/`), globbed by CMake, linked in, self-registering via static
    initializers at process startup. CMake enforces a sync check: every module
    `#include`d by `src/fenix.hpp` must have a matching `src/<mod>/<mod>.cpp` or the
    split build hard-fails at configure time (silently dropping a stage is worse
    than a build error). See ADR 0008.
  - `main()` parses argv, strips verbosity flags (`-v`→debug, `-vv`/`-vvv`/`--trace`→
    trace, `--quiet`→error; else default info or `$FENIX_LOG_LEVEL`), calls
    `fenix::init_thread_limits()` (clamps OpenMP team size to the cgroup CPU quota,
    not host `nproc` — a container-thrash guard) before any parallel region, then
    dispatches: `help`/`-h`/`--help`, `info <artifact>` (inspect any `.fxvol`,
    printing dims/chunks/codec params + coverage real/zero/absent counts), `run
    <recipe.fxrecipe>` (loads TOML, runs each `stages[]` entry through the registry
    with its own `<stage>.args`), or `fenix::find_stage(cmd)` for every other
    subcommand. Adding a stage never touches this file.
  - `gui.cpp` — THE Qt/VTK TU (the GUI firewall, mirroring `ml/inference.cpp` for
    libtorch): `#include "gui/gui.hpp"` only, nothing else. Qt/VTK headers are
    parsed here and nowhere else; compiled+linked in only when `-DFENIX_GUI=ON`,
    with `-frtti -fexceptions` on this TU alone (the rest of the binary stays
    `-fno-exceptions -fno-rtti`). Registers the `view` stage.

## Inputs / outputs & formats
CLI: `fenix <subcommand> [opts]`, `fenix info <artifact.fxvol>`, `fenix run
<recipe.fxrecipe>`. Subcommands self-register from their modules; `driver.cpp` never
enumerates them (the `help` listing walks `fenix::registry()`).

## Dependencies
Intra: everything (via `fenix.hpp` in UNITY; core/config/codec-archive directly in
SPLIT). Third-party: whatever the linked modules need (libcurl/zlib/blosc2/mimalloc
always; libtorch + optionally TensorRT under `-DFENIX_ML`; Qt6/VTK only under
`-DFENIX_GUI`, isolated to `gui.cpp`).

## Invariants & numerics
Keep `driver.cpp` tiny: parse → Context → registry → run. Never add a second
general-purpose `.cpp` to the core (UNITY) build. In SPLIT mode, every module
`#include`d by `fenix.hpp` needs its `src/<mod>/<mod>.cpp` (CMake fails the
configure step otherwise) — `core` is exempt (driver includes it directly in both
modes) and `gui` is exempt when `FENIX_GUI` is off.

## Gotchas / pitfalls
- Don't add a second `.cpp` to the core (UNITY) build; tests/fuzz/bench are separate
  per-file TUs that include the same headers, not part of the `fenix` binary target.
- The SPLIT/UNITY sync check is the guard against a stage silently vanishing from
  dev (split) or GUI/ML builds while unity CI stays green — don't bypass it by
  excluding a module without discussing.
- `gui.cpp` and `ml/inference.cpp` are the only two `-frtti -fexceptions` TUs in the
  whole codebase; nothing typed by Qt/VTK/libtorch should cross back into the
  `-fno-exceptions -fno-rtti` core.

## Status & TODO
`driver.cpp` and `gui.cpp` are both implemented and in active use (`info`, `run`,
and dispatch to real stages — see `tools/CLAUDE.md` for the current stage roster).
Open ADRs: subcommand UX; ADR 0008 covers the split-build TU layout in detail.
