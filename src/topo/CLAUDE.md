# topo — CLAUDE.md

## Purpose
Native cubical persistent homology + the connected-components/Betti core that `eval` and
`postproc` build on. See `docs/research/research-core.md` (topo/cubical, persist0).

## Public API & key types
- **Cubical persistent homology** (sublevel T-construction, Z/2): persistence pairs +
  representative cycles. Build toward **exact TopoScore (Betti-matching)**.
- **Fast dim-0** persistence via union-find merge tree + persistence-based simplification
  (`simplify_dim0`: raise low-persistence basins).
- **Betti numbers** (b0/b1/b2, χ) with fg6/bg26 (+ dual) pairing; `cc_label` (shared with
  `geom`); windowed `region_betti` for tunnel screening.

## Inputs / outputs & formats
In: scalar/binary `Volume`. Out: persistence diagrams, Betti numbers, representative
cycles (consumed by `postproc` topo surgery and `eval` TopoScore).

## Dependencies
Intra: `core`, `geom` (cc_label/union-find). Third-party: none.

## Invariants & numerics
Deterministic tie-breaks (by index/gid) — persistence must be reproducible (it's integer/
combinatorial, not fast-math-affected). Matches the Betti-Matching-3D oracle taberna
validated against (50/50 binary, 40/40 grayscale).

## Performance notes
Generic boundary-matrix reduction is O(cells³) worst case → fast dim-0 (near-linear) +
windowed dim-1/2; skip dense windows. Out-of-core via tiling for screening.

## Gotchas / pitfalls
Don't ship the dim-1/2 proxy as if exact (taberna's candid limitation). cc_label lives in
`geom`; topo consumes it (don't duplicate).

## Status & TODO
Full (cubical + dim-0 + Betti). Open ADRs: exact Betti-matching; fast dim-1/2 paths.
