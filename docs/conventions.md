# fenix — Canonical Conventions

The single source of truth for the cross-cutting conventions that prevent whole classes
of bugs. **Encode these in the type system wherever possible** (see `src/core/`); this
doc explains the *why*. Violations are bugs, not style nits.

## Coordinates & axes
- **Order is ZYX everywhere**: z-major, x-fastest. A voxel `(z,y,x)` lives at linear
  index `(z*ny + y)*nx + x`. There is exactly one `Volume<T>` accessor — never hand-roll
  `IDX` macros (taberna had ~15 copies; that is the #1 bug source we are killing).
- **Right-handed** coordinate frame.
- **Vectors are axis-tagged.** A direction/normal is a typed `Vec3` whose components are
  named (z,y,x). taberna stored normals as x,y,z-interleaved over z,y,x voxels — a latent
  bug it had to patch by hand. We forbid bare 3-float arrays for spatial vectors.
- **Dimensions are a typed `Extent3`/`ChunkCoord`, never loose `int nz,ny,nx`.** Index
  math is 64-bit; never multiply three `int`s before widening.

## Units (strong typedefs — no bare scalars for physical quantities)
- `Micron` — physical length / voxel size. `KeV` — scan energy.
- `Voxel` / `ChunkCoord` — integer grid coordinates (chunk = voxel ≫ 6, 64³).
- `WindingNumber` — continuous winding (level sets of the winding field; the spiral
  `shifted_radius / dr_per_winding`). `Radian` — angles (atan2 results, θ).
- `LOD` — resolution level; **0 = full resolution**, higher = coarser.

## Numerics
- **f32 is the primary compute/storage scalar.** Use f64 only where accumulation order
  matters: optimizer state, large reductions, eval metrics. Document each f64 use.
- **Dense voxel dtypes are constrained** to the `VoxelScalar` concept: `u8/u16/u32/
  s8/s16/s32/f16/f32` — **no f64/u64/s64 dense storage** (`Volume<T>` enforces this at
  compile time). `f16` is the clang builtin `_Float16`, used at the codec dtype layer.
- **Never widen u8 CT data to f32 for storage/ingest** — keep it u8 through the pipeline
  until an operation genuinely needs float math; widen transiently, don't store wide.
- **Fast-math is on globally** (`-O3 -ffast-math`, the non-deprecated `-Ofast`).
  Consequences you must respect:
  - Results are **not** bit-reproducible across runs/compilers/ISAs. Tests compare with
    **tolerances** (PSNR/MAE/ULP), never `==` on floats / byte equality.
  - `std::isnan`/`isinf` are unreliable under fast-math. Use explicit **validity masks**,
    not NaN sentinels (the fysics lesson).
- **Tolerances are named constants** in `src/core/` (geometric eps, default τ,
  convergence thresholds, det-J fold floor) — never scattered magic literals.

## Topology / connectivity
- **Default: foreground 6-connected, background 26-connected** (the Betti-Matching-3D
  pairing taberna validated against). Provide the 26/6 dual where an algorithm needs it.
  Every CC/Betti API takes connectivity explicitly; this is just the default.

## Validity
- A surface/grid cell is invalid by an **explicit validity mask/bitset**, not a magic
  coordinate sentinel. (VC used `{-1,-1,-1}`; it leaked through every loop. We don't.)

## Geometry sign conventions
- **Winding increases outward** from the umbilicus. Spiral handedness (CW/ACW) is a
  per-scroll constant carried in metadata, not assumed.
- **recto/verso**: the two faces of a sheet, distinguished by the sign of the directional
  first derivative across sheet thickness (recto = +, verso = −). (Note: the predecessors
  had a documented recto/verso naming swap — define it once here and stick to it.)
- Surface normals point **toward the umbilicus** unless a type says otherwise.

## Storage / files
- **Little-endian only** (all targets are LE) — `static_assert` it.
- Every container: **magic + format version + first-party (xxh3-style) hash**. Readers
  **reject** unknown/old versions cleanly (no migration; regenerate from source).
- Spatial metadata (voxel µm, world origin, axis orientation, scroll id) + provenance
  (recipe, git hash, params hash, input ids) travel **in** every artifact.
- Writes are **atomic** (temp + rename); growable archives use `fallocate` (not
  `ftruncate`); mmap uses a large `MAP_NORESERVE` reservation so offsets never move.

## Error & control flow
- Fallible operations return `std::expected<T, fenix::Error>`. No exceptions
  (`-fno-exceptions`), no RTTI (`-fno-rtti`). No int-return + manual-free cascades.
- `Error` is one rich type: `Errc` (`ok`, `invalid_argument`, `not_found`, `io_error`,
  `fetch_failed`, `decode_error`, `unsupported`, `unimplemented`, `internal`) + `message`
  + `source_location`. Construct via `fenix::err(code, message)`, never a bare
  `std::unexpected{...}` literal.
- **Absent ≠ fetch-failed.** A 404/missing chunk is legitimate air (ZERO). Any other IO
  failure is retried with backoff, then a **hard error** — it must never silently become
  air (the matter-compressor lesson; a transient blip poisoning a TB output is fatal).
  `Errc::fetch_failed` is the distinct code for this; never fold it into `not_found`.

## Assertions
- `FENIX_ASSERT(cond, ...)` — active in debug/sanitizer builds, `__builtin_trap()` on
  failure; compiles to `((void)0)` under `NDEBUG` (release). Use for internal invariants,
  never for validating external/user input (that's `Expected`/`err`).
- C++26 contracts are used alongside it where the toolchain supports them (see root
  `CLAUDE.md` §2.3); `FENIX_ASSERT` remains the portable fallback.

## Logging
- `FENIX_TRACE/DEBUG/INFO/WARN/ERROR(mod, fmt, ...)` — module-tagged (`mod` is a short
  `snake_case` tag naming the emitting module, e.g. `"winding"`, `"codec"`), gated at
  **both** compile time (`trace`/`debug` vanish outside dev builds via
  `FENIX_LOG_COMPILE_MIN`) and run time (process-wide `log_level()`, seeded from
  `$FENIX_LOG_LEVEL`; the driver's `-v/-vv/-vvv`/`--quiet` mutate it). Records are
  mutex-serialized so `parallel_for` output never interleaves, and flushed per-record
  (block-buffered stdout was observed silently clipping logs on timeout/crash — never
  write pipeline logs to unflushed stdout). Levels: `trace < debug < info < warn < error
  < off` (`quiet` is an alias of `off`).

## Concurrency
- **`parallel_for`/`parallel_for_dynamic`** (OpenMP) for CPU-bound data-parallel loops;
  **`parallel_for_io`** (first-party `std::thread` pool, work-stealing) for network/IO
  fan-out — never use OpenMP for IO fan-out: it silently clamps a large `num_threads()`
  request back to the cgroup-derived team size with no error (observed team-of-1 on a
  13-CPU-quota box), which reads as "it ran" while throughput quietly collapses.
- Thread counts are **cgroup-quota-aware** (`cpu_budget()`: min of cgroup v2/v1 quota and
  host core count, overridable by `FENIX_THREADS`/`OMP_NUM_THREADS`) — never trust raw
  `nproc`/`hardware_concurrency` alone inside a container.
- **`SerialRegion`** (RAII, thread-local switch) wraps per-work-unit kernels when the
  outer loop is already parallel (e.g. per-tile tracer) — nested OMP regions
  oversubscribe (~10× measured slowdown) otherwise.
- **No load-bearing global constructors** — thread limits are applied explicitly via
  `init_thread_limits()` at startup, not a static initializer (see root `CLAUDE.md` §2.4).

## Naming (see root `CLAUDE.md` §2.3)
`snake_case` functions/vars/namespaces · `PascalCase` types/concepts ·
`kCamel`/`UPPER_SNAKE` constants · `fenix::<module>` + `fenix::<module>::detail`.
Named tolerance constants live in `fenix::tol` (`src/core/tolerances.hpp`) — e.g. `eps`,
`dir_eps`, `default_tau`, `fold_det_floor` — never a scattered magic literal at a call
site; add new ones there, not inline.
