# ADR 0008 — Optional split (multi-TU) build for fast incremental compiles

**Status:** Accepted (2026-07-01). Evolves the **single-TU unity build** invariant in root `CLAUDE.md` §2.2:
unity stays the **default and canonical** build; the split build is an **opt-in developer accelerator**
(`-DFENIX_SPLIT=ON`) that produces a byte-for-byte equivalent binary. Header-only design is unchanged.

## Context
fenix is header-only with one real translation unit (`apps/driver.cpp` includes `fenix.hpp`). That unity
build is minimal-total-work and great for clean/CI, but it has one compiler process and **any** change
recompiles the whole program. Measured on a 16-core dev box (core build, `-O3`, no ML/GUI): a clean unity
build is **9.7 s single-core**, and every edit pays that full 9.7 s. The pain is the **dev inner loop**, not
clean throughput.

We evaluated the obvious "move impl out of headers into `.cpp` behind `#ifdef UNITY`" split. Rejected: the
codebase has **80+ `template<`** (Volume<T>, the DCT codec, the tracer, samplers…) whose definitions
**must** stay in headers (explicit instantiation over 8 dtypes × N functions is brittle and huge), so that
split would relocate only a minority of (non-template) code for a large, error-prone `#ifdef` cost. And it
is unnecessary: everything is already `inline`, so the same definition in multiple TUs is ODR-legal and the
linker dedups — **multi-TU parallelism needs no header surgery.**

## Decision
Add a CMake `FENIX_SPLIT` option (default OFF). When ON:
- Compile each module as its **own TU** — `src/<mod>/<mod>.cpp` is a one-line `#include "<mod>/<mod>.hpp"`,
  one per module `fenix.hpp` pulls in. These compile in **parallel** and enable **incremental** rebuilds.
- `apps/driver.cpp` under `-DFENIX_SPLIT` includes only what `main()` touches (`core` + `config` +
  `codec/archive`) instead of the whole umbrella, so it stays a light TU and does **not** re-register every
  stage; the module objects' static registrars populate the registry at startup (objects are linked
  directly, so no registrar is dropped).
- A **core.hpp precompiled header** (`FENIX_SPLIT_PCH`, default ON) amortizes the per-TU core+libc++ parse.
- `register_stage` is made **idempotent by name** — a safety net so a stage can never double-register if a
  header is transitively re-included across TUs. In the unity build every stage registers once, so it is a
  no-op there.

The unity build (`FENIX_SPLIT=OFF`) is completely unchanged: one TU, `driver.cpp` includes `fenix.hpp`.

## Consequences
Measured (same box, `-j16`, deps reused):
- **Incremental leaf edit: 1.3 s** (recompile 1 TU + link) vs 9.7 s unity — **~7×**, the real prize. A
  shared-header edit rebuilds exactly its includers (e.g. `io/nrrd.hpp` → 4 TUs); a `core` edit rebuilds
  everything (~8.8 s, unavoidable — core is the base). Dependency tracking is correct (Ninja depfiles).
- **Clean build: 8.8 s** vs 9.7 s — only modest, because the wall is bounded by the **heaviest single TU**
  (`io.cpp` ≈ 5.8 s bundles codec+s3+zarr+nrrd+surface+archive) and by redundant template instantiation
  across TUs (more total CPU, parallelized). **Follow-up lever:** split the heavy umbrellas (io, segment)
  into finer TUs to drop the critical path toward ~2–3 s clean.
- Same binary + behaviour: the split `fenix` registers the identical **35 stages**; tests are already
  per-file TUs and unaffected.
- **Trade-off:** the split does more total CPU work (redundant instantiation) — so unity remains the
  default for clean/CI where total-work matters and there's nothing to rebuild incrementally.

## ccache — the lever for docker/runpod rebuilds
The **first** build must parse libtorch once (frontend-bound: `ml.cpp` is ~60% libtorch header parse, so
`-O` barely moves it and splitting ml further only re-parses torch N times). But a **rebuild** shouldn't
re-pay it. CMake now auto-detects **ccache** (`FENIX_CCACHE`, default ON) as the compiler launcher.
Measured (ML split, 16 cores): cold populate **24 s**, then a rebuild from a **fresh build dir → 0.18 s**
(all 15 TUs cache hits — the 20 s libtorch TU included). A runpod ML unity build's driver TU alone was
~7 min; with ccache + a **persisted `CCACHE_DIR`** it becomes a hit.
- **Operational:** on docker/runpod, **mount `CCACHE_DIR` as a volume** (else a fresh container starts cold),
  and set `CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime,pch_defines` for robust hits.
- **ccache vs PCH:** they conflict (ccache won't cache PCH-using compiles). ccache is the bigger win, so the
  split build uses the core PCH **only when ccache is absent**.
- **Presets:** `cmake --preset split` (core dev) and `cmake --preset ml` (ML dev: libtorch isolated to one
  TU) wrap `FENIX_SPLIT` (+`FENIX_ML`) so it is one command.
- **libtorch firewall (DONE).** `<torch/torch.h>` is now parsed in exactly ONE TU: `src/ml/inference.cpp`.
  The public surface `ml/ml_api.hpp` is torch-free (declares `run_predict_surface`/`run_predict_ink`/`run`
  with torch-free signatures, torch-free stubs when `!FENIX_ML`); `ml/ml.hpp` only registers the stages
  (torch-free); the impl bodies (nets, weights, sliding-window inference) live in `inference.cpp`. Effect:
  the **unity** driver TU under `-DFENIX_ML` dropped **26 s → 9.9 s** (torch removed; unity ML build 26 s →
  ~10 s as driver + inference compile as 2 TUs). In the **split** build `ml.cpp` is now torch-free (0 torch
  header deps) and `inference.cpp` is the sole 20 s torch TU. Combined with ccache, that 20 s parse is only
  paid when `inference.cpp` itself changes — every other edit (incl ml stage registration) is torch-free.
  CMake adds `inference.cpp` explicitly for the unity ML target; the split build globs it.

## Not covered / notes
- **GUI split** isn't wired (gui.hpp needs Qt/VTK and is firewalled behind `FENIX_GUI`; no `src/gui/gui.cpp`
  in the glob). `FENIX_ML` works: `src/ml/ml.cpp` inherits the target's `FENIX_ML` define + torch link.
- Adding a module means adding its `src/<mod>/<mod>.cpp` one-liner (mirrors adding it to `fenix.hpp`); the
  glob is `CONFIGURE_DEPENDS` so CMake picks it up. Keep the unit list in sync with `fenix.hpp`.
