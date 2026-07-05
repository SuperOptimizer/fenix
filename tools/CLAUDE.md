# tools — CLAUDE.md

## Purpose
Two different things live here under one directory:
1. **The CLI surface of the `fenix` binary itself** — one `fenix` multi-command binary
   (NOT many small binaries like taberna's ~55 tools / VC's ~45). Each subcommand is a
   thin self-registering wrapper (`FENIX_REGISTER_STAGE`) over a library stage; the
   subcommand definitions live in `src/<mod>/*.hpp`, not under `tools/`.
2. **Off-binary research/ops tooling** — Python + shell scripts (mostly ML-island:
   training, dataset QC, ink-hunting, fleet orchestration) that drive or inspect the
   `fenix` binary from outside. This half has had the most recent activity; it is
   pre-approved to use NVIDIA/ML-ecosystem deps freely (see root CLAUDE.md §2.5
   ML-island exemption) and is explicitly exempt from the "thin/no-sprawl" rule below —
   these are working research scripts, not core CLI subcommands.

## Public API & key types

### fenix binary subcommands (self-registered, current roster)
`ingest`, `preprocess`/`deconv`/`denoise`/`aircut`/`musica`/`dering` (fysics
lineage), `segment`, `annotate`/`umbilicus`, `predictions`, `winding`, `flatten`
(unroll: fitted `.fxmodel` → per-wrap `.fxsurf`), `render` (unroll a `.fxvol` to a
flattened NRRD), `postproc`, `topo`, `codec`, `geom`, `eval`, `ml` (libtorch
inference; sub-verbs include `predict-surface`, `run-raw`, `train-feed`,
`ingest-band`, `band-blocks`), `view` (Qt6 4-pane viewer, GUI build only) — plus the
driver-level `help`/`info`/`run` handled directly in `apps/driver.cpp` (see
`apps/CLAUDE.md`). Several stages are still stubs (`preprocess`, `segment`, `codec`,
`geom`, `postproc`, `predictions`, `topo`, `annotate` top-level) with the real logic
living in sibling headers under the same module (e.g. `preproc_cli.hpp`,
`dering.hpp`, `umbilicus_fit.hpp`). Each is `{name, help, run(args, Context)}`;
verify the live roster with `grep -rn FENIX_REGISTER_STAGE src/` — this list drifts
as modules land.

### off-binary tooling (`tools/<area>/`)
- **`tools/train/`** — the KD training loop: `train.py` (PyTorch shrunk ResEnc-UNet
  student, bf16 autocast, KD-from-teacher + Dice/CE + optional soft-clDice, EMA
  weights, torchao QAT hook, checkpoint/resume), `feed_reader.py` (reads the
  `fenix train-feed` shm ring), `audit_persist.py` (checkpoint/telemetry audit),
  `inst_train_supervisor.sh` (per-ring instance-mode training supervisor),
  `tame_feeders.sh` (pins one feeder process per ring — fixes a supervisor
  restart storm). Consumes the torch-free C++ data plane (`fenix train-feed`); no
  Python at inference time (see `tools/ml-export/`).
- **`tools/ml-export/`** — offline venv (not part of the C++ build) that converts
  published ScrollPrize PyTorch checkpoints into fenix's `.fxweights` format:
  `introspect.py` (dump tensor names/shapes), `convert_weights.py` (checkpoint →
  `.fxweights` + TOML registry entry), `reference.py {gen,run,cmp}` (authoritative
  PyTorch reference used to validate the C++ reimplementation numerically),
  `probe_precision.py` / `trt_probe.py` / `build_engine.py` (TensorRT fp16/int8
  engine build + precision probing — see ADR 0010). Full recipe in
  `tools/ml-export/README.md`.
- **`tools/inkhunt/`** — ink-detection model hunting/review: `hunt.sh` (orchestrates
  a multi-model render pass), `consensus.py` (cross-architecture agreement maps),
  `ink_gallery.py` (review gallery), `refine_transform.sh` (coordinate-descent
  refinement of a segment's transform against the surf-qc delta).
- **`tools/labelqc/`** — training-label quality control: `gallery.py` /
  `chunk_viewer.py` (triage UI, incl. a hand-rolled WebGL2 volume raycaster),
  `apply_verdicts.py` (ingests human triage verdicts back into the label set).
- **`tools/fleet/`** — multi-pod GPU orchestration: `pod_bootstrap.sh` (RunPod box
  setup), `gen_sweep.py` / `teacher_sweep.sh` (hyperparameter/teacher sweeps across
  pods).
- **`tools/proto/`** — render/flatten prototyping scripts predating (or exploring
  ahead of) the C++ `flatten`/`render` stages (`slim_flatten.py`,
  `render_*.py`, `surf_metrics.py`). Treat as throwaway research scaffolding, not a
  stable API.
- **`tools/shakedown/`** — training-data-plane validation suite run on GPU pods:
  `suite.sh` (6-phase functional suite), `passA.sh`+`drain.py` (feeder throughput),
  `passB.sh`+`ring_dump.py` (determinism + crash-integrity), `passC.sh` (soak),
  `roundD.sh`/`roundF.sh`/`roundGHI.sh` (trainer balance, QAT cycle, corpus/GPU
  scaling). See `tools/shakedown/README.md` for the bug list these runs have caught.

## Inputs / outputs & formats
fenix subcommands: CLI args + config/recipe (TOML); verbosity `-v/-vv/-vvv` +
`--quiet`; outputs land in the project/run dir (manifest + stats JSON + logs +
outputs). Off-binary tooling: mostly reads/writes `.fxvol`/`.fxsurf`/`.fxweights`/
NRRD/PNG and shm training rings; each script's own header comment is the source of
truth for its args.

## Dependencies
fenix subcommands — intra: the stage modules + `core` (arg parser, registry,
Context); third-party via modules. Off-binary tooling — Python (torch, torchao,
TensorRT/onnx tooling per `tools/ml-export/requirements.txt`), pre-approved under
the ML-island exemption; no C++ build dependency either way.

## Invariants & numerics
fenix subcommands are thin — no business logic here; they parse + call the
library, and compile into the single `driver.cpp` TU (unity) or their own
`src/<mod>/<mod>.cpp` TU (split) via the umbrella/module header
(header-only, self-registering). Off-binary tooling has no such constraint — it's
ordinary Python/shell, versioned like any other source but not built by CMake.

## Gotchas / pitfalls
- Keep the **fenix-binary** subcommand set thin; resist accumulating experimental
  one-off binaries there (taberna's tool sprawl was a documented smell). New
  experiments on the C++ side = a subcommand or a test, not a new binary.
- The off-binary `tools/<area>/` scripts are explicitly the release valve for that
  same impulse — sprawl there is expected and fine as long as scripts stay
  self-documented (header comment + README where one exists) and reasonably
  discoverable by area. Don't conflate the two halves of this directory.
- `tools/ml-export/` is a separate venv, never wired into the CMake build — don't
  add its deps to `cmake/deps.cmake`.
- `tools/train/train.py` and friends assume a live `fenix train-feed` shm ring;
  they are not standalone.

## Status & TODO
fenix subcommand roster is live and growing (ingest/preprocess family/segment/
annotate/predictions/winding/flatten/render/postproc/topo/codec/geom/eval/ml/view
all self-register today; several are still stubs pending their module's real
implementation — check `src/<mod>/CLAUDE.md` per module). Off-binary tooling is
active and ahead of the C++ pipeline in places (ink-hunt review, label QC, KD
training loop, shakedown validation suite all in daily use on GPU pods). Open
items: promote `tools/proto/` render/flatten logic into `src/render`/`src/flatten`
once stable; Round E dress-rehearsal (bulk-KD → student export → `predict-surface`
→ eval) per `tools/shakedown/README.md`.
