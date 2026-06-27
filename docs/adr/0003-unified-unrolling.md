# ADR 0003 — Unified Unrolling Method

**Status:** Accepted (2026-06-27)

## Context
Four prior approaches exist: taberna's Eulerian winding field + TV + spiral fit; VC's NLLS
surface tracer; thaumato's winding-angle graph; and Henderson's diffeomorphic spiral fit
(`spiral-v2`). We want ONE cohesive method, not a switch over isolated backends, working
from raw CT and/or surface predictions (ML or classical), out-of-core to whole-scroll scale.

## Decision
- **Backbone = a global compositional diffeomorphism** T: scroll→spiral-space (cylindrical
  SVF/ODE-flow ⊕ per-slice affine `expm` ⊕ radial gap-scale ⊕ umbilicus + `dr_per_winding`).
  Invertible by construction ⇒ no fold-over.
- **Unifying quantity:** spiral-space `shifted_radius = winding·dr` — the same scalar as
  taberna's Eulerian winding field (continuous view). The winding field is a dense data term.
- **Constraints (one vocabulary):** Patches (VC-style first-party NLLS tracer), Tracks
  (skeletonized predictions), PointCollections (annotations), dense fields (sheetness/
  normals/winding-density). Losses = spiral-v2 set (patch/track radius+DT + EM, rel/abs
  winding, lasagna normals+spacing, sym-Dirichlet + bending, shell).
- **Solver:** first-party AdamW (annealed spring, coarse→fine) + L-BFGS/Gauss-Newton late;
  hand-rolled gradients; **no pyro/torchdiffeq/Ceres**.
- **Out-of-core (the gap vs Henderson's in-core ~19h):** resident global flow/gap/affine
  lattices; stream constraint mini-batches in z-tiles; coarse-global warmup. Persist as
  `.fxmodel`; generate per-wrap `.fxsurf` via T⁻¹.

## Consequences
+ Global invertibility (no seam fold-over that block-wise tracers suffer); a small, general
  constraint vocabulary subsumes tracer output + human points + auto tracks + dense fields.
+ Ties taberna's winding field and the spiral fit into one quantity.
− The OOC global fit is novel work (Henderson's is in-core/single-GPU); reconciling block
  winding fields + the global flow at seams is an open problem to validate.
− Skip spiral-v2's accreted heuristics (~150 hyperparameters, autoresearch, redundant losses).
