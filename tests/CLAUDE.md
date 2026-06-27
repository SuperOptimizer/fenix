# tests — CLAUDE.md

## Purpose
The test, fuzz, and benchmark suites. First-party harness (no third-party test framework).

## Public API & key types
- **Unit + golden** tests using the `core` harness: `TEST(name){...}` auto-register,
  `REQUIRE` (fatal) / `CHECK` (non-fatal) with expression capture; shared `main()`.
  **Per-test-file binaries** (each test file → its own tiny executable / TU).
- **Golden** artifacts in **Git LFS**, compared with **tolerances** (PSNR/MAE/ULP) — never
  bit-exact (fast-math). Prefer deterministic synthetic generators where possible.
- **Fuzzing:** `LLVMFuzzerTestOneInput`-style harnesses for every parser/codec/container
  path, run under **AFL++** (afl-clang-fast/lto, persistent mode); CI seed corpus.
  **Property/invariant oracles live in the fuzzers** (codec roundtrip≤τ, T∘T⁻¹≈id, EDT/CC
  properties) — no separate property framework.
- **Benchmarks:** first-party microbench (kernels) + scenario bench (end-to-end); baselines
  as **Git LFS JSON** per runner; CI regression gate on the self-hosted runner.

## Inputs / outputs & formats
Test fixtures: synthetic generators + tiny committed (Git LFS) cubes; download-on-demand
real scroll regions in CI (cached). Bench → JSON stats.

## Dependencies
Intra: the modules under test + `core` harness. Third-party: AFL++/libFuzzer at build time;
modules' own deps.

## Invariants & numerics
Tolerances, not equality. Tests mirror the `src/` tree. Every logic change ships a test.

## Gotchas / pitfalls
Don't commit large binaries outside Git LFS. Don't assert bit-exact float results. GUI is
smoke-tested only.

## Status & TODO
Stub harness + one example test per module + fuzz/bench scaffolding. Open ADRs: fixture
download policy; bench baseline storage layout.
