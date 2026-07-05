# tests — CLAUDE.md

## Purpose
The test, fuzz, and benchmark suites. First-party harness (no third-party test
framework, no Catch2/GTest).

## Public API & key types
- **Unit tests** via `core/test.hpp`: `TEST(name){...}` auto-registers into a global
  case list; `CHECK(...)` (non-fatal, increments a failure counter and continues)
  and `REQUIRE(...)` (fatal, `return`s out of the test on failure). `FENIX_TEST_MAIN`
  defined in exactly one place per binary supplies `main()` → `fenix::test::run_all()`.
  **Per-test-file binaries**: CMake globs `tests/test_*.cpp` and builds each as its
  own executable + `ctest` case (~70 today — `test_core.cpp`, `test_codec_bench.cpp`
  … `test_zarr.cpp`; run `ls tests/test_*.cpp` for the current, fast-moving list,
  spanning core/io/codec/geom/segment/winding/flatten/ml/eval/topo and more). All
  test TUs share one `core.hpp` precompiled header (when ccache is active) to
  amortize the frontend cost of ~70 duplicate parses.
- **Tolerances, not golden/LFS.** Comparisons use PSNR/MAE/ULP-style tolerance
  checks, never bit-exact (the whole stack builds `-ffast-math`, non-deterministic
  across runs/ISA by design). Fixtures today are synthetic generators and small
  in-repo/download-on-demand data — **there is no Git LFS setup in this repo yet**
  (no `.gitattributes` LFS filters); don't assume golden LFS blobs exist until one
  is actually added. Prefer deterministic synthetic generators when adding new
  tests.
- **Fuzzing:** `tests/fuzz_*.cpp` provide `LLVMFuzzerTestOneInput` and no `main`;
  built under `-DFENIX_FUZZ=ON` and linked with `-fsanitize=fuzzer` (every other
  target — the `fenix` binary and every `test_*` binary — gets `fuzzer-no-link`
  only, so sancov instrumentation is everywhere but only the dedicated fuzz
  binaries pull in libFuzzer's own `main`). **Actual harness is Clang's libFuzzer**,
  not AFL++ — current targets: `fuzz_dct_tile_decode`, `fuzz_fxvol_open`,
  `fuzz_lossless_decode`, `fuzz_read_obj`, `fuzz_zarray_json` (parser/codec/
  container entry points). Property/invariant-style checks live inside these
  fuzzers (codec roundtrip ≤ τ, T∘T⁻¹ ≈ id) rather than a separate property
  framework — no such framework exists yet.
- **Benchmarks:** no separate `bench_*` naming convention exists yet; today
  benchmark-style TUs are named `test_*_bench.cpp` / `test_*_perf.cpp`
  (`test_codec_bench.cpp`, `test_rans_perf.cpp`, `test_trace_perf.cpp`) and are
  **standalone programs with their own `main()`**, not `TEST()` cases — e.g.
  `test_codec_bench.cpp` takes a `.nrrd` path + quality levels on argv and reports
  compression ratio, encode/decode throughput, and quality (PSNR/SSIM/MAE/
  percentile abs-error). No JSON baseline storage or CI regression gate is wired up
  yet despite being called for in root CLAUDE.md §5.3 — treat that as aspirational
  until it lands.

## Inputs / outputs & formats
Test fixtures: synthetic generators (primary) + small local files; some tests take
real data paths on argv (see `test_codec_bench.cpp`). No LFS-backed golden corpus
yet. Bench-style tests print human-readable stats to stdout (no JSON output
convention established yet).

## Dependencies
Intra: the modules under test + `core::test` harness. Third-party: libFuzzer via
Clang's `-fsanitize=fuzzer[-no-link]` at build time; modules' own deps otherwise. No
GTest/Catch2/AFL++.

## Invariants & numerics
Tolerances, not equality — never assert bit-exact float results. Tests mirror the
`src/` tree by naming (`test_<module-or-feature>.cpp`), not by directory structure.
Every logic change should ship or extend a test.

## Gotchas / pitfalls
- Don't invoke `-fsanitize=fuzzer` (plain) on anything except `tests/fuzz_*.cpp` —
  it pulls in libFuzzer's own `main` and will duplicate-main-fail the link on the
  `fenix` binary or any `test_*` binary (this exact bug happened once via a
  CMakePresets `"address,fuzzer"` sanitizer combo; see CMakeLists.txt comments).
- Don't assume Git LFS exists for fixtures — check before adding large binary test
  data; ask forrest before introducing an LFS dependency.
- `test_*_bench.cpp` / `test_*_perf.cpp` files are NOT `ctest` unit tests in the
  normal sense — they still get built+registered by the same glob, but they take
  their own argv and are meant to be run by hand for perf investigation, not
  treated as pass/fail CI signal.
- GUI (`view` stage) is smoke-tested only, no automated Qt/VTK test harness.

## Status & TODO
Unit-test harness (`core/test.hpp`) is implemented and in heavy use (~70 per-file
binaries across nearly every module). Fuzzing is implemented for 5 parser/codec/
container entry points under libFuzzer, gated by `FENIX_FUZZ`. Benchmarking exists
informally (`*_bench`/`*_perf` standalone programs) but has no baseline-storage or
CI-regression-gate infrastructure yet — that (and a real golden/LFS fixture policy)
are the two biggest gaps vs. the aspirational description in root CLAUDE.md §5.3.
Open ADRs: fixture download/LFS policy; bench baseline storage layout; whether to
formalize a `bench_*` naming split from `test_*`.
