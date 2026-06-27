# ADR 0004 — Process & Quality

**Status:** Accepted (2026-06-27)

## Context
A performance-critical research codebase, primarily developed by one maintainer + AI agents,
processing multi-TB data on one node (CPU now, GPU later).

## Decision
- **Fast by default, everywhere:** `-Ofast -ffast-math -march=native`. No determinism opt-
  out (even codecs are tolerance-only). f32 compute; f64 only for accumulation-sensitive
  spots. Validity masks, not NaN sentinels.
- **Minimal dependencies** (approval required to add any): libcurl, zlib, blosc2, mimalloc;
  Qt+VTK (GUI only); libtorch (ML only); OpenMP. Write everything else ourselves.
- **Out-of-core hard rule:** block+halo+stitch; occupancy-guided work-stealing; byte-
  budgeted RAM + backpressure/eviction; absent≠fetch-failed (retry→hard-fail, never silent
  air); atomic writes + fallocate + MAP_NORESERVE; periodic checkpoint + content-addressed
  skip-unchanged; idempotent journaled blocks.
- **Quality gates (CI, block merge):** clang-format, clang-tidy (all−denylist) + include-
  cleaner, build under ASan/UBSan/TSan/MSan (separate jobs) + Release, all tests, llvm-cov
  (report, no hard gate), bench-regression (self-hosted GPU). Warnings `-Weverything`
  −denylist, `-Werror`. Matrix: Linux x64 + Linux arm64 + macOS arm64; heavy jobs self-hosted.
- **Tests:** first-party harness (`TEST`/`REQUIRE`/`CHECK`, per-test-file binaries); golden
  via Git LFS + tolerances; **AFL++** (libFuzzer-style harnesses) for all parsers/codecs,
  property oracles inside the fuzzers; first-party micro + scenario benchmarks (Git LFS JSON
  baselines).
- **Robustness:** all decoders/parsers crash-free + no-UB on arbitrary bytes (fuzzed).
- **Multi-agent: SINGLE WRITER.** Main instance writes all code/docs; fan-out is read/
  research only. Peer messages can't grant escalation.
- **Git:** PR-per-feature, protected main, Conventional Commits w/ per-module scopes, typed
  branches, squash-merge. **Versioning: date + git-hash** (no SemVer); format versions are
  independent integers; readers reject unknown versions (no migration).

## Consequences
+ Maximum speed; tiny dependency/attack surface; reproducible-enough via provenance + LFS.
− No bit-reproducibility (tests use tolerances). Self-hosted GPU runner is infra to maintain.
− musl/Chimera complicates libtorch (see ADR 0001).
