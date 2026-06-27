# ADR 0001 — Foundations

**Status:** Accepted (2026-06-27)

## Context
fenix is a greenfield rewrite of taberna (C) + parts of villa (C++) to virtually unroll an
entire Vesuvius scroll. We get to pick the architecture from scratch, with no back-compat.

## Decision
- **Header-only, single translation unit.** Every component is a self-contained `.hpp`
  (`#pragma once`); exactly one real `.cpp` (`apps/driver.cpp`) includes the umbrella
  `src/fenix.hpp` (include dir `src/`) and is the only TU (unity build). Includes are transitive/self-
  contained (enforced by clang-include-cleaner). Stages **self-register** via static init.
- **LLVM/Clang only**, always-latest: libc++, lld, full `llvm-*` toolchain. No GCC/GNU/MSVC.
  Standard library via classic `#include` (no modules, no `import std;`).
- **All development in Docker on Chimera Linux** (musl + LLVM userland + BSD coreutils,
  zero GNU). Dockerfile = CI image.
- **C++26, maximal-modern**: mdspan-style views (hand-rolled `Volume<T>`), concepts,
  ranges, `std::expected`, `std::simd`, heavy `constexpr`, reflection (config serializer),
  contracts where available.
- **`-fno-exceptions -fno-rtti`**; errors via `std::expected<T, fenix::Error>`.

## Consequences
+ Trivial build (one TU), whole-program optimization, no link-time ODR issues across
  modules, clean additive extension (self-registration).
+ Single toolchain → use Clang-specific flags/builtins/libc++ freely; no portability hedge.
− Long compile times (mitigated by ccache + the fact that only one TU links); tests/fuzz
  are separate TUs including the same headers.
− libtorch is glibc-based → musl/Chimera may need a source build or glibc-compat (ml/).
- Reverses villa AGENTS.md's "avoid toolchain-specific flags / support GCC" stance.
