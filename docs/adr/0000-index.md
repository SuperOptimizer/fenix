# Architecture Decision Records — Index

MADR-style records of fenix's load-bearing decisions. The detailed rationale for most
lives in the synthesized predecessor research ([`../research/`](../research/)); these ADRs
state the decision, status, and consequences. New decisions get a new numbered file;
supersede (don't delete) when reversing.

| # | Decision | Status |
|---|---|---|
| [0001](0001-foundations.md) | Foundations: header-only single-TU, Clang-only, Chimera/Docker, C++26 maximal-modern | Accepted |
| [0002](0002-codec-and-container.md) | Codec = CDF 9/7 wavelet only (2D+3D shared core) + lossless; `.fxvol` container | Accepted |
| [0003](0003-unified-unrolling.md) | One unified diffeomorphic unrolling method (not isolated backends) | Accepted |
| [0004](0004-process-and-quality.md) | Fast-by-default numerics, minimal deps, CI/test/fuzz, single-writer agents | Accepted |

Decisions captured but not yet split into their own ADR (see root `CLAUDE.md` §2 +
`docs/conventions.md` + the module `CLAUDE.md` files): error model (`expected`/no-except),
ZYX/f32 + strong types, out-of-core block+halo+stitch, occupancy work-stealing scheduler,
recipe/project formats (`.fxrecipe`/`.fxproj`), geometry toolkit (`geom/`), ML in-tree via
libtorch, eval suite, date+git versioning, format-version-reject (no migration). Promote
any of these to a full ADR when it needs deeper rationale or gets revisited.
