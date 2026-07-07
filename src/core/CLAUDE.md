# core — CLAUDE.md

## Purpose
The substrate every other module builds on. Built **first**. Zero pipeline logic — only
the types, primitives, and services shared everywhere. Read root `CLAUDE.md` +
`docs/conventions.md` before touching this.

## Public API & key types
- `Volume<T>` (`volume.hpp`) — the one 3D array/view. Explicit **ZYX** contiguous
  strides (`stride_x=1, stride_y=nx, stride_z=nx*ny`); dtypes constrained by the
  `VoxelScalar` concept (`u8/u16/u32/s8/s16/s32/f16/f32`; **no f64/u64/s64 dense**).
  `VolumeView<T>` is the non-owning strided view (works for `const T` too, with an
  implicit non-const→const conversion); `Volume<T>` owns storage
  (`make_unique_for_overwrite`, deep-copyable, cheap-movable). `crop`, `at_clamped`,
  `in_bounds`, `flat()` (only when contiguous). **The only** voxel accessor — no
  hand-rolled index macros anywhere in the codebase.
- `types.hpp` — fixed-width scalar aliases (`u8..s64`, `f16` = clang `_Float16`,
  `f32`/`f64`, `usize`/`isize`); `Quantity<Tag,Rep>` strong-typedef template backing
  `Micron`, `KeV`, `Radian`, `WindingNumber`; `Lod` (opaque `u32` enum); `Extent3`/
  `Index3` (ZYX, `s64` fields — index math never overflows at scroll scale) and
  `ChunkCoord` (`chunk_of()`, `chunk_side = 64`).
- `vec.hpp` — `Vec3<T>` (axis-tagged z,y,x; `dot`/`cross`/`norm`/`normalized`) and
  `Mat3<T>` (row-major ZYX, matrix-vector/matrix-matrix `operator*`, `identity()`).
  First-party; **no Eigen**.
- `eig.hpp` — `sym_eig3<T>` (`SymEig3<T>{values, vectors}`): the **one** symmetric-3×3
  eigensolver (cyclic Jacobi, ~50-sweep cap, eigenvalues sorted descending) for the
  whole codebase — structure tensor, Hessian/Frangi, OOF, PCA sheet repair all call
  this. (SVD not yet implemented — see TODO.)
- `error.hpp` — `Errc` enum, `Error{code, message, source_location}`,
  `Expected<T> = std::expected<T, Error>`, `err(...)`; `fenix::version` (date+githash,
  `FENIX_VERSION` build define). Split out of `core.hpp` so leaf headers (e.g.
  `config.hpp`) can depend on `Expected` without the full aggregate.
- `log.hpp` — the real logging system (superseded the old two-line stub): `LogLevel`
  (`trace/debug/info/warn/error/off`, `quiet` = alias of `off`), process-wide
  `log_level()` (seeded from `$FENIX_LOG_LEVEL`), pluggable `LogSink` (`log_sink()`,
  default stderr + `fflush` every record — block-buffered stdout was silently
  clipping logs on timeout/crash), mutex-serialized `log_emit` (parallel_for-safe, no
  interleaving), `FENIX_TRACE/DEBUG/INFO/WARN/ERROR(mod, fmt, ...)` macros (module-
  tagged, gated at both compile time via `FENIX_LOG_COMPILE_MIN` and run time),
  back-compat module-less `log(level, fmt, ...)`, and `ScopeTimer`/
  `FENIX_SCOPE_TIMER(mod, label)` (RAII debug-level timing).
- `config.hpp` — `Config`: minimal first-party TOML reader (no external dep).
  `Config::parse(text)` / `Config::load(path)` → `Expected<Config>`. Comments,
  `[dotted.sections]`, `key = value` (string/int/float/bool/`[a, b, c]` array),
  flattened to `"section.key"` lookups (`has`, `get_string/float/int/bool/array`,
  `entries()`). Backs recipes, annotations, the scrolls registry, stage Params; a
  reflection-driven binder is meant to layer on top later (not yet built).
- `filter.hpp` — `gaussian_blur(VolumeView<f32>, sigma)` (separable, in-place,
  reflect-boundary, parallel over disjoint lines, iterated reflect index so it's
  correct even when kernel radius ≥ line length — ASan-verified fix) and
  `gradient_at(VolumeView<const f32>, z,y,x)` (central difference, clamped).
- `optimize.hpp` — `AdamW`/`AdamWConfig` (decoupled weight decay) over a flat `f32`
  parameter vector, plus a `minimize(params, grad_fn, iters, cfg)` convenience driving
  it with a caller-supplied gradient function. The optimizer engine for the
  diffeomorphic spiral fit (flow + gap-expander + affine + dr) — see
  `src/winding/CLAUDE.md`.
- `surface.hpp` — `Surface`: the in-memory `.fxsurf` — a `(u,v)` grid of ZYX world
  coords + validity (`nu*nv`, row-major v-major/u-fastest), plus optional named
  channels `normal` (per-cell across-sheet unit normal) and `conf` (data-term
  confidence), lazily allocated via `alloc_channels()`. Output of segmentation/
  flattening, input to rendering.
- `pool.hpp` — `WorkerPool`: the first-party PERSISTENT async task pool (N long-lived
  threads, FIFO `submit(fn)`, `stop()` = drop queued + join running — the GUI shutdown
  story: stop pools before the widgets tasks reference die). Distinct from the
  fork-join `parallel_for*` family below; no load-bearing globals — callers own pools.
- `parallel.hpp` — four flavors, all container/cgroup-aware:
  - `cpu_budget()` — effective CPU count = min(cgroup v2/v1 quota, host cores),
    overridable by `FENIX_THREADS`/`OMP_NUM_THREADS`, cached.
  - `init_thread_limits()` — clamps OpenMP's default team size to `cpu_budget()` and
    pins `omp_set_max_active_levels(1)`; call once at startup.
  - `parallel_for(begin, end, body, max_threads=0)` — OpenMP, static schedule by
    default (dynamic if `max_threads>0`, the IO-fan-out throttle path); serial
    fallback with no OpenMP.
  - `parallel_for_dynamic` — `schedule(dynamic,1)` for wildly-uneven coarse work
    items (e.g. the tiled tracer's disjoint tiles).
  - `parallel_for_io(begin, end, threads, body)` — **`std::thread` pool, not
    OpenMP** (OpenMP silently clamps a large `num_threads()` request back to the
    cgroup-derived team size — observed team-of-1 on a 13-CPU-quota box — which
    silently serialized S3 fan-out); atomic work-stealing cursor, intentionally
    oversubscribes for network-blocked workers.
  - `parallel_for_z(dims, body)` — convenience over z-slices.
  - `SerialRegion` (RAII) + thread-local `g_parallel_serial` — the anti-nesting
    switch: wrap each work-unit body in an outer `parallel_for_dynamic` (e.g. per-tile
    tracer) so per-unit kernels (structure tensor, blur, ARAP) run serially instead of
    spawning nested OMP regions (~10× oversubscription measured otherwise).
- `sampling.hpp` — `sample_nearest`, `sample_trilinear` (branch-free fast path when
  the 2×2×2 stencil is fully in-bounds, clamped fallback at borders — the tracer's
  #1 hot loop), `TrilinearStencil`/`trilinear_stencil` (8 clamped flat offsets +
  weights — the scatter primitive for backpropagating into lattice values, e.g. the
  flow velocity field), `sample_trilinear_grad` → `SampleGrad{value, grad}` (analytic
  d/dz,dy,dx — the fit's hot loop). Tricubic/Lanczos not yet implemented.
- `arena.hpp` — `Arena` (aligned bump allocator, `alloc`/`alloc_n<T>` for trivially-
  copyable `T`, `reset()`) + `Arena::Scope` (RAII bump-offset restore for nested
  scratch regions). Trivially-destructible payloads only.
- `rng.hpp` — `Pcg32` (PCG-XSH-RR 64/32, O'Neill 2014): `next_u32/next_u64/next_f32`,
  bias-free `bounded(n)`. Deterministic per-seed/per-stream.
- `hash.hpp` — `hash64(span<const u8>, seed=0)` (wyhash-style wymix, xxh3-class
  quality) + `hash_value<T>(v, seed=0)` for trivially-copyable values.
- `tolerances.hpp` (`fenix::tol`) — named epsilons: `eps`, `dir_eps`, `default_tau`,
  `converge_eps`, `fold_det_floor`. No scattered magic literals elsewhere.
- `core.hpp` — the umbrella aggregate: pulls in all of the above plus `FENIX_ASSERT`
  (active outside `NDEBUG`, `__builtin_trap()` on failure, compiled to nothing in
  release), `Context` (currently just `threads`, `cancelled`, `project_dir` — still a
  stub of the eventual io-handles/logger/pool/arenas/progress bundle), the **stage
  registry** (`Stage{name, help, run}`, `register_stage`/`find_stage`/`registry()`,
  idempotent-by-name so split-build re-registration is a no-op), `stage_unimplemented`,
  and the `FENIX_REGISTER_STAGE(NAME, HELP, FN)` macro modules use to self-register at
  namespace scope.
- `test.hpp` — first-party harness: `TEST(name){...}` auto-registers via static init,
  `REQUIRE` (fatal, returns from the test fn)/`CHECK` (non-fatal, variadic to accept
  braced-initializer commas), `fenix::test::run_all()`; `FENIX_TEST_MAIN` in exactly
  one TU per test binary supplies `main()`.

## Inputs / outputs & formats
None on disk except `Config` (reads `.fxrecipe`/TOML-ish text via `parse`/`load`).
Otherwise provides in-memory types + services only. Owns `docs/conventions.md`'s types.

## Dependencies
Intra: none (leaf). Third-party: libc++ only (`<expected>`, `<print>`, `<format>`,
`<charconv>`); `mimalloc` (allocator) optional; OpenMP (`parallel.hpp`, guarded by
`_OPENMP`, degrades to serial without it); `std::thread` (`parallel_for_io`). No
`std::simd` usage yet despite being planned (see TODO). (libcurl/zlib/blosc2 are
io/codec, not core.)

## Invariants & numerics
ZYX everywhere, f32 primary (f64 only for accumulation-sensitive code — none lives in
core itself yet). Fast-math-safe (validity masks, not NaN sentinels). `Volume`/
`VolumeView` index math is 64-bit (`s64`/`Extent3`/`Index3`). No load-bearing global
constructors — `parallel_for` reads `cpu_budget()` lazily and container thread limits
are applied explicitly via `init_thread_limits()`, not a static ctor. Determinism is
*not* guaranteed (fast-math, OpenMP scheduling) and that's accepted project-wide;
`CHECK`/`REQUIRE` compare with tolerance, never bit-exact.

## Performance notes
This is the hot path of everything. `sample_trilinear`/`sample_trilinear_grad` have an
explicit branch-free fast path for in-bounds stencils (measured hot loop via the
tracer's snap data term — see the `perf(core): trilinear in-bounds fast path` commit).
`gaussian_blur` parallelizes over disjoint lines (race-free in place) with thread-local
scratch, and its interior loop is branch-free/vectorizable (only the two border spans
pay the reflect-index cost). `parallel_for`/`parallel_for_dynamic`/`parallel_for_io`
are all cgroup-quota-aware — a container's `nproc` can lie by 10× relative to its real
CPU quota, and getting this wrong either oversubscribes (nested OMP: measured ~10×
slowdown, guarded by `SerialRegion`) or silently serializes IO fan-out (OpenMP
`num_threads()` clamps to the cgroup team size — fixed by giving `parallel_for_io` its
own raw `std::thread` pool). GPU backend interfaces still TODO (deferred, per root
CLAUDE.md).

## Gotchas / pitfalls
- Never reintroduce per-module `IDX` macros, duplicate eigensolvers, or NaN sentinels —
  those are the exact taberna smells this module exists to kill. `eig.hpp`/`sym_eig3`
  is the one eigensolver; call it, don't write another.
- Don't parallelize IO fan-out with `parallel_for`/OpenMP — use `parallel_for_io`.
  OpenMP silently clamps a requested large team back to the cgroup-derived size with no
  error, which reads as "it ran" while throughput quietly collapses.
- Wrap per-work-unit kernels in `SerialRegion` when the outer loop is already
  `parallel_for`/`parallel_for_dynamic` over independent units — nested OMP regions
  oversubscribe badly and the effect is easy to miss in a profiler that isn't looking
  for it.
- Header-only single-TU: free functions are `inline`/`constexpr`/templates; impl in
  `fenix::<mod>::detail`. `core.hpp` is the umbrella header — new core headers must be
  added to its `#include` list to be visible via `"core/core.hpp"`.
- `Config` is intentionally minimal (no nested tables/inline tables, arrays are flat
  strings) — it's the recipe/annotation substrate, not a general TOML implementation.

## Status & TODO
**Implemented + tested** (release + ASan/UBSan green): `types.hpp`, `vec.hpp`,
`eig.hpp` (`sym_eig3`), `volume.hpp` (`Volume`/`VolumeView`, crop/clamp/const-views),
`tolerances.hpp`, `parallel.hpp` (all four `parallel_for*` variants + `SerialRegion` +
cgroup-aware `cpu_budget`), `rng.hpp` (`Pcg32`), `hash.hpp` (`hash64`/`hash_value`),
`arena.hpp` (bump arena + RAII `Scope`), `sampling.hpp` (nearest/trilinear +
analytic-gradient trilinear + `TrilinearStencil`), `filter.hpp` (`gaussian_blur`,
`gradient_at`), `log.hpp` (full leveled/sink/scope-timer system, in liberal use
across the stack), `config.hpp` (TOML-ish reader), `optimize.hpp` (`AdamW`/
`minimize`), `surface.hpp` (`Surface` + named channels), `error.hpp` (`Error`/
`Expected`/`err`/version), `core.hpp` (assert, `Context` stub, stage registry,
`FENIX_REGISTER_STAGE`), `test.hpp` (`TEST`/`REQUIRE`/`CHECK`). Tests:
`tests/test_core.cpp`, `test_volume.cpp`, `test_substrate.cpp`, `test_filter.cpp`.

**TODO:** SVD (symmetric eigensolver done, SVD still needed for segment); tricubic/
Lanczos samplers; `std::simd` vectorization of hot kernels (currently plain
scalar/OpenMP, no explicit SIMD intrinsics anywhere in core despite the root-level
plan); first-party async thread pool for IO pipelines beyond the ad-hoc pool in
`parallel_for_io` (that one is single-use, not a reusable pool object); the C++26
reflection (de)serializer for config/metadata (currently hand-written `Config`, no
reflection); GPU backend interface shape (open ADR); `Context` is still a 3-field stub,
not the full io-handles/logger/pool/arenas/progress bundle described in the root
CLAUDE.md. Open ADRs: serializer schema-evolution, GPU backend interface shape, f16
support.
