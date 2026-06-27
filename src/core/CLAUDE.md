# core — CLAUDE.md

## Purpose
The substrate every other module builds on. Built **first**. Zero pipeline logic — only
the types, primitives, and services shared everywhere. Read root `CLAUDE.md` +
`docs/conventions.md` before touching this.

## Public API & key types
- `Volume<T>` — the one 3D array/view. Explicit **ZYX** strides; dtypes
  `u8/u16/u32/s8/s16/s32/f16/f32` (matching the codec; **no f64/u64/s64 dense**); owning
  + non-owning (`VolumeView`) variants; bounds/clamp/reflect/crop helpers. **The only**
  voxel accessor — no hand-rolled index macros anywhere in the codebase.
- `Vec2/3/4`, `Mat3/4`, `Quat` + small dense solvers (closed-form symmetric-3×3 eig, SVD).
  First-party; **no Eigen**. Spatial vectors are axis-tagged (z,y,x).
- Strong units: `Micron`, `KeV`, `Voxel`, `ChunkCoord`, `Extent3`, `WindingNumber`,
  `Radian`, `LOD`.
- `Error` (code + message + source_location/context) and `Expected<T> = std::expected<T,
  Error>`; `FENIX_ASSERT`; named tolerance constants.
- `Logger` (levels, scopes, scoped timers/counters, pluggable sink, over `std::print`),
  `Arena` (per-thread scratch + bump/region), `ThreadPool` + `parallel_for`, `Rng`
  (PCG64), `hash` (xxh3-style), the C++26-reflection (de)serializer (config/metadata).
- Sampling kernels: nearest / trilinear / tricubic(Catmull-Rom) / Lanczos over chunked
  volumes, SIMD + chunk-cache-aware; **trilinear with analytic backward** (the fit's hot
  loop). `std::simd` helpers + NEON/AVX2 fallback.
- The **stage registry** (self-registration) + `Context` (io handles, config, logger,
  pool, arenas, progress/cancel) threaded through every stage.
- The first-party **test harness** (`TEST`/`REQUIRE`/`CHECK`, auto-register) and CLI
  arg/subcommand parser.

## Inputs / outputs & formats
None on disk — provides in-memory types + services. Owns `docs/conventions.md`'s types.

## Dependencies
Intra: none (leaf). Third-party: libc++ only here; `mimalloc` (allocator) optional;
`std::simd`. (libcurl/zlib/blosc2 are io/codec, not core.)

## Invariants & numerics
ZYX, f32 primary (f64 only for accumulation), fast-math-safe (validity masks not NaN).
`Volume<T>` index math is 64-bit. No load-bearing global constructors.

## Performance notes
This is the hot path of everything — sampling kernels, `parallel_for`, arenas, and
`Volume` access must be allocation-free in loops and vectorize under `-Ofast`. GPU
backend interfaces live here (impl deferred).

## Gotchas / pitfalls
- Never reintroduce per-module `IDX` macros, duplicate eigensolvers, or NaN sentinels —
  those are the exact taberna smells this module exists to kill.
- Header-only single-TU: free functions are `inline`/`constexpr`/templates; impl in
  `fenix::core::detail`.

## Status & TODO
**Implemented + tested** (release + ASan/UBSan green): `types.hpp` (scalars, strong units,
Extent3/Index3/ChunkCoord), `vec.hpp` (Vec3/Mat3 + dot/cross/norm), `volume.hpp`
(`Volume<T>` + `VolumeView<T>` incl. const views, crop, clamp), `tolerances.hpp`,
`parallel.hpp` (OpenMP `parallel_for` + serial fallback), `rng.hpp` (PCG32), `hash.hpp`
(wyhash-style `hash64`/`hash_value`), `arena.hpp` (bump arena + RAII Scope), `sampling.hpp`
(nearest/trilinear + analytic-gradient trilinear), `core.hpp` (Error/Expected, Logger,
Context, Stage registry, FENIX_ASSERT, version), `test.hpp` (TEST/REQUIRE/CHECK).
Tests: `tests/test_core.cpp`, `test_volume.cpp`, `test_substrate.cpp`.

**TODO:** symmetric-3×3 eigensolver + SVD (for segment); tricubic/Lanczos samplers;
`std::simd` vectorization of hot kernels; the reflection (de)serializer; first-party async
thread pool; the real C++26 features (currently conservative C++23-ish for portability).
Open ADRs: serializer schema-evolution, GPU backend interface shape, f16 support.
