# flatten — CLAUDE.md

## Purpose
Turn a fitted spiral model (`winding::SpiralModel`, a `.fxmodel`) or a dense winding
field into 2D-parameterized `.fxsurf` grids ready for `surf-bake` / `render-layers` /
`view-surf`. Two grid extractors (closed-form per-wrap/unroll, and dense-field
ray-march) plus an optional distortion-minimizing UV refinement (SLIM/ARAP) that
operates on any traced surface, not just flatten's own output.

## Public API & key types
- `flatten.hpp` — the `flatten` stage. `wrap_surface(SpiralModel, wrap, nu, z_lo, z_hi,
  zstep, rmax) -> Surface`: one integer winding level `W`, CLOSED FORM (no ray-march, no
  dense winding volume): `r_ideal = dr_per_winding * (W - winding_offset + theta/2pi)`,
  `r = model.gap.forward(r_ideal)`, `q = {z, r*sin(theta), r*cos(theta)}`,
  `p = model.to_scroll(q)`. Theta sampled half-bin-centred in `(-pi, pi]` to avoid the
  branch-cut seam. `unroll_surface(SpiralModel, w_lo, w_hi, nu_per_wrap, z_lo, z_hi,
  zstep, rmax) -> Surface`: ONE grid over the continuous winding coordinate
  `t in [w_lo, w_hi]` (`wrap = floor(t)`, `theta = 2pi*frac(t) - pi`); `r_ideal` is
  continuous across wrap boundaries by construction — this is THE whole-scroll unroll.
  `run(args, Context&)` is the registered stage entry point.
- `extract_wrap.hpp` — `extract_winding_surface(VolumeView<const f32> winding,
  annotate::Umbilicus, target, nu, r_max) -> Surface`: alternate/legacy path, ray-marches
  outward from the umbilicus per `(theta, z)` and linearly interpolates the radius where
  the dense winding field crosses `target`. Not used by the `flatten` stage's own CLI;
  kept for callers that only have a dense winding volume, not a fitted `SpiralModel`.
- `slim.hpp` — `FlatMesh` (`pos`, `uv`, `tri`, `energy_init`/`energy_final` — area-weighted
  symmetric-Dirichlet, 2.0 = perfectly isometric) and `slim_parameterize(const Surface&,
  iters=20, cg_iters=600) -> FlatMesh`. Builds a triangle mesh directly from the surface's
  valid quads (own inline vertex/triangle builder — no `geom::Mesh` dependency), keeps only
  the largest connected component (union-find), then runs local/global ARAP (Liu et al.
  2008): local step = closest per-triangle rotation, global step = cotangent-Laplacian
  solve via first-party Jacobi-PCG. Initialized from the grid `(u,v)` layout, not from a
  boundary map — there is no ABF++/LSCM stage; SLIM is the only UV solver in the tree.
  Not yet wired into the `flatten` stage or CLI — call it directly as a library function.
- `sparse.hpp` — first-party sparse linear algebra backing SLIM: `Triplets`, `Csr`
  (`from_triplets`, `matvec`), `cg(Csr, b, x, maxit, tol)` (Jacobi-preconditioned CG,
  warm-started, f64 throughout). No Eigen/libigl.

## Inputs / outputs & formats
In: a `.fxmodel` (`winding::SpiralModel`, via `winding::read_fxmodel`) for `flatten.hpp`;
a dense `VolumeView<const f32>` winding field + `annotate::Umbilicus` for
`extract_wrap.hpp`; a `Surface` (any traced/extracted grid) for `slim.hpp`. Out:
per-wrap `.fxsurf` (`<out>_w<W>.fxsurf`) or one unroll `.fxsurf` (`<out>_unroll.fxsurf`),
written via `io::write_fxsurf`. `slim_parameterize` returns an in-memory `FlatMesh`
(pos/uv/tri) — no dedicated OBJ-with-UV writer in this module yet.

CLI: `fenix flatten model=<fxmodel> out=<prefix> [mode=wraps|unroll] [wraps=lo:hi]
[nu=1024] [zstep=4] [z=lo:hi] [rmax=1e9]`. `wraps` default: every `W` whose spiral has
`r_ideal >= 0` somewhere in `[0,2pi)` and `r <= rmax`, up to gap-table extent + 2 (capped
4096 wraps). `z` defaults to the model's umbilicus z-span.

## Dependencies
Intra: `core` (Surface, VolumeView, Vec3f, Context/Expected/Error, logging, stage
registry), `winding` (`SpiralModel`, `read_fxmodel`, gap table, `to_scroll`), `annotate`
(`Umbilicus`, `extract_wrap.hpp` only), `io` (`write_fxsurf`). Third-party: none. No
`geom::Mesh` — SLIM builds/owns its own minimal mesh representation inline.

## Invariants & numerics
- `wrap_surface`/`unroll_surface`: f32 throughout; a cell is invalid when `r_ideal < 0`
  (before the spiral starts at that theta) or `r_canon > rmax`.
- SLIM: f64 internally for the CG solves and energy (accumulation-sensitive per
  conventions), f32 in/out (`Vec3f` positions, `f32` UVs). Degenerate/zero-area triangles
  are dropped defensively (`good()`, area > 1e-7) but real slab quality (no slivers) is
  expected to come from the upstream tracer, not from this module. Cotangent weights are
  clamped to `>= 1e-3` for SPD stability. Tiny Tikhonov `eps=1e-6*I` makes the Laplacian
  SPD for CG (translation is the only real nullspace; the resulting shift is
  energy-neutral) — no large penalty pin, no proximal anchor (both were rejected as
  destabilizing/ill-conditioning).
- No explicit fold-over/injectivity guard is implemented yet in `slim.hpp` — ARAP's local
  rotation step discourages but does not hard-prevent inversions; there is no Jacobian-sign
  check post-solve.

## Performance notes
`wrap_surface`/`unroll_surface` are simple per-cell closed-form evaluations — O(nu*nv),
trivially parallel across `(u,v)` (not yet parallelized in the loop itself). SLIM: each
iteration is O(triangles) for the local step + one CG solve per axis (u, v) capped at
`cg_iters`; CG is Jacobi-preconditioned (essential — unpreconditioned CG on the cotan
system needs far more iterations, and a half-converged solve corrupts the next local
step). Out-of-core: wraps/unroll are extracted independently per model, no stitching
needed (radius is continuous by construction in unroll mode).

## Gotchas / pitfalls
- There is no ABF++/LSCM in this tree — grid UVs come either from the closed-form
  spiral parameterization (`flatten.hpp`) or from SLIM's ARAP solve initialized at the
  grid layout. Don't assume a boundary-fixed conformal solver exists here.
- `extract_wrap.hpp`'s ray-march is a *different, older* extraction path than
  `flatten.hpp`'s closed form — don't conflate the two; the `flatten` stage only uses the
  closed form (`wrap_surface`/`unroll_surface`).
- Theta must stay half-bin-centred off the `±pi` branch cut in the closed-form
  extractors — sampling exactly at `±pi` straddles the wrap boundary the Archimedean
  readout steps on.
- SLIM has no CLI/stage wiring yet — it's a library call only, not reachable from
  `fenix flatten` or any other registered subcommand.
- Cotangent weights, not uniform weights, are required — uniform weights collapse the
  interior of the parameterization.

## Status & TODO
Implemented: `flatten` stage (`flatten.hpp`, `mode=wraps` and `mode=unroll`) — the P6
whole-scroll unroll path, exercised end-to-end (see
`docs/design/winding-pilot-ablation.md`, first unroll artifact: wind10 model →
`flatten mode=unroll` → `render-layers` → `predict-ink`). `extract_wrap.hpp` (dense-field
ray-march extraction) and `slim.hpp` + `sparse.hpp` (ARAP/SLIM UV refinement, first-party
CG) are implemented and unit-testable but not called from the `flatten` stage or any CLI
path.
TODO: wire SLIM as an optional post-pass on extracted wraps (`flatten ... slim=1`); add
an injectivity/fold-over check after the SLIM solve; per-patch (not just per-wrap)
flattening; an OBJ+UV writer for `FlatMesh`. Open ADRs: none yet filed for this module —
per-patch vs per-wrap flattening and whether SLIM becomes the default post-pass are still
informal decisions.
