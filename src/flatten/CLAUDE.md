# flatten — CLAUDE.md

## Purpose
Parameterize a fitted/traced surface into a low-distortion 2D (u,v) map for rendering.
The diffeomorphic fit (`winding`) already gives a near-flat spiral grid; `flatten` refines
per-wrap/per-patch UVs. See `docs/research/villa-vc.md` (ABF/LSCM/SLIM), `villa-ml.md`
(lasagna snap/SLIM).

## Public API & key types
- **ABF++ → LSCM** (angle-based flattening then conformal least-squares) as the primary
  in-core parameterization; **SLIM** (Symmetric-Dirichlet, locally injective) as a higher-
  quality refinement. One unified parameterization API (not two disconnected paths like VC).
- Mesh build from valid surface quads (via `geom::Mesh`), degenerate-triangle cleanup,
  area-preserving scaling, grid-axis alignment, UV upsampling.

## Inputs / outputs & formats
In: `.fxsurf` (QuadSurface grid) or `geom::Mesh`. Out: a UV-bearing `.fxsurf` (distortion-
corrected grid) and/or OBJ with UVs.

## Dependencies
Intra: `core`, `geom`, `io`, `codec`. Third-party: none (first-party sparse solvers in
`core`/`geom`; **no libigl/Ceres/PaStiX**).

## Invariants & numerics
Guard injectivity / no fold-over (Jacobian non-inversion), as in snap_surf. f32 grids,
f64 for the sparse linear solves. Tolerances named in `core`.

## Performance notes
Sparse conformal/SLIM solves per wrap/patch; parallel across patches. Out-of-core: flatten
per wrap independently then stitch.

## Gotchas / pitfalls
Unify ABF/LSCM/SLIM behind one API (VC had them disconnected). Don't fold over (the
Jacobian guard is mandatory). Reuse `geom` mesh ops.

## Status & TODO
Implemented: the `flatten` stage (flatten.hpp) — fitted `.fxmodel` → per-wrap `.fxsurf`
via the CLOSED-FORM wrap equation (r_ideal = dr·(W − offset + θ/2π), r = gap.forward,
p = to_scroll; no ray-march, no dense winding volume; θ half-bin-centred off the ±π
branch cut). `extract_wrap.hpp` (dense-field ray-march variant), `slim.hpp` (ARAP/SLIM
UV refinement + first-party CG/sparse) also live. Wrap surfaces feed surf-bake /
render-layers / view-surf directly.
`mode=unroll` emits THE single unrolled-scroll surface (one grid over the continuous
winding coordinate, nu samples per wrap, radius continuous across wrap boundaries by the
closed form) — feed to render-layers -> predict-ink* for whole-scroll ink.
TODO: run slim on extracted wraps as an optional `flatten` pass; per-patch flattening. Open ADRs: ABF-vs-SLIM default per surface;
per-patch vs per-wrap flattening.
