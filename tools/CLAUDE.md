# tools — CLAUDE.md

## Purpose
The thin CLI surface — one `fenix` multi-command binary (NOT many small binaries like
taberna's ~55 tools / VC's ~45). Each subcommand is a thin wrapper over a library stage.

## Public API & key types
- Subcommand registrations: `ingest, preprocess, segment, predict, anno, wind, flatten,
  render, eval` (per stage) + `run` (recipe), `import` (VC interop), `info` (inspect any
  artifact, `--json`), `view` (GUI), `bench`. Each `{name, help, run(args, Context)}`
  self-registers; the actual logic lives in the corresponding `src/` module.

## Inputs / outputs & formats
CLI args + config/recipe (TOML). Verbosity `-v/-vv/-vvv` + `--quiet`. Outputs land in the
project/run dir (manifest + stats JSON + logs + outputs).

## Dependencies
Intra: the stage modules + `core` (arg parser, registry, Context). Third-party: via modules.

## Invariants & numerics
Subcommands are thin — no business logic here; they parse + call the library. They compile
into the single `driver.cpp` TU via the umbrella header (header-only, self-registering).

## Gotchas / pitfalls
Keep tools thin; resist accumulating experimental one-off binaries (taberna's tool sprawl
was a documented smell — current-vs-abandoned confusion). New experiments = a subcommand
or a test, not a new binary.

## Status & TODO
Stub the subcommand registrations matching the module stubs. Open ADRs: subcommand naming.
