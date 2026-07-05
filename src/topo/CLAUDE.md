# topo — CLAUDE.md

## Purpose
Discrete topology of a binary volume: exact Euler characteristic + Betti numbers
(b0/b1/b2). Feeds `eval`'s TopoScore (currently a Betti-count proxy, not exact
Betti-matching) and is the intended foundation for `postproc` topo surgery. See
`docs/research/research-core.md` (topo/cubical, persist0) for the target design —
cubical persistent homology is **not yet implemented** (see Status below).

## Public API & key types
All in `topo/betti.hpp`, namespace `fenix::topo`:
- `struct Betti { s64 b0, b1, b2, chi; }` — b0 = components (26-conn fg), b1 = tunnels,
  b2 = enclosed cavities, chi = Euler characteristic = b0 - b1 + b2.
- `s64 euler_characteristic(VolumeView<const u8> mask)` — exact cubical-complex cell
  count (V - E + F - C) over voxels treated as closed unit cubes; per-z-plane parallel
  partial sums via `parallel_for`.
- `s64 enclosed_cavities(VolumeView<const u8> mask)` — inverts the mask, runs
  `geom::connected_components(_, Conn::Six)` on the background, counts components that
  never touch the volume border.
- `Betti betti_numbers(VolumeView<const u8> mask)` — b0 and b2 computed directly
  (`geom::connected_components(_, Conn::TwentySix)` for b0, `enclosed_cavities` for b2),
  b1 derived algebraically from chi: `b1 = b0 + b2 - chi`. No independent b1 computation
  or representative-cycle extraction exists.

`topo/topo.hpp` — the `fenix topo` CLI stage — is a **stub**: `run()` returns
`stage_unimplemented("topo")`. It includes `betti.hpp` but does not call it; nothing
in-tree calls `betti_numbers`/`euler_characteristic` yet except tests.
`topo/topo.cpp` is only the split-build TU (`#include "topo/topo.hpp"`), not used in
the default unity build.

## Inputs / outputs & formats
In: `VolumeView<const u8>` binary mask (foreground/background). Out: `Betti{b0,b1,b2,chi}`
(plain struct, no artifact format). No persistence diagrams, no representative cycles —
those are aspirational (see Status).

## Dependencies
Intra: `core` (Volume/VolumeView, parallel_for), `geom::connected_components` (both
Conn::Six for background cavities and Conn::TwentySix for foreground b0 — reused, not
duplicated). Third-party: none.

## Invariants & numerics
Integer/combinatorial, not fast-math-affected — results are deterministic and exact
(no tolerance needed), matching the taberna-validated Betti-Matching-3D oracle
(50/50 binary, 40/40 grayscale) *for b0/b2/chi*. b1 is currently derived, not
independently verified against an oracle. Connected-components label ids are
deterministic (min-index-voxel roots, ascending-root order) post the z-slab-parallel
rewrite — still bit-reproducible despite parallelism.

## Performance notes
Multithreaded (commit 4174c56, 2026-07-01): `euler_characteristic`'s V/E/F/C counts and
`enclosed_cavities`'s background inversion are per-z-plane/per-slab `parallel_for` reductions
with serial merge — all through `core::parallel_for`, clamped to `cpu_budget()` (cgroup
quota), never raw host core count. No GPU path. No out-of-core tiling yet — operates on
a full in-memory `Volume<u8>`; large-volume/windowed screening is unimplemented.

## Gotchas / pitfalls
- **The cubical PH engine described in earlier docs and `docs/research/` does not exist
  in code.** `docs/design/ml-accel-and-distillation.md` explicitly flags this: "TopoScore
  is only the number-based Betti proxy (exact cubical PH is claimed in docs but absent)."
  Don't assume persistence diagrams, dim-0 union-find simplification, or windowed
  dim-1/2 region_betti exist — they're design targets, not code.
- `cc_label`/connected-components lives in `geom`; topo consumes it, don't duplicate.
- b1 is *inferred* from chi, not computed from an independent tunnel-counting method —
  if `euler_characteristic` or b0/b2 have a bug, b1 silently absorbs the error.
- The `topo` CLI stage is a stub; there is no way to invoke this from `fenix run` or the
  CLI today. Only reachable via direct header include (e.g. from tests or `eval`).

## Status & TODO
Implemented: `euler_characteristic`, `enclosed_cavities`, `betti_numbers` (b0/b1/b2/chi),
multithreaded. **Not implemented**: cubical persistent homology (sublevel T-construction,
Z/2 pairs), persistence-based dim-0 simplification, representative cycle extraction,
windowed/tiled `region_betti` for tunnel screening, the `topo` CLI stage (stub), exact
Betti-matching against ground truth. This is the top of the metric-harness gap called out
in `docs/design/ml-accel-and-distillation.md` §0 — `eval`'s official_score needs real
TopoScore, which needs this module built out past Betti-counting.
