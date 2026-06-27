# villa `spiral-v2` — Paul Henderson's Diffeomorphic Spiral Fitting

Branch: `origin/spiral-v2` (caf09f6), forked from `origin/main` at 5f5a610. Diffs add **10,813 lines**, almost
entirely under `volume-cartographer/scripts/spiral/`. Related to arXiv 2512.04927 "Diffeomorphic Spiral Fitting" (WACV 2026).

## File inventory (what the branch adds)

| File | Lines | Role |
|---|---|---|
| `scripts/spiral/fit_spiral.py` | 5552 | **The whole algorithm**: model, losses, optimisation, meshing, eval. |
| `scripts/spiral/tifxyz.py` | 334 | Load/save grid quad-mesh surface patches (`Patch` dataclass; `.tifxyz` = per-axis tiff of xyz + mask). |
| `scripts/spiral/point_collection.py` | 319 | Load annotated point sets (PCLs); link points to nearby patches; winding annotations. |
| `scripts/spiral/extract_surface_tracks.py` | 242 | Pre-process: skeletonise surface-prediction zarr into "tracks" (cc3d + kimimaro), store in a dbm. |
| `scripts/spiral/find_inconsistent_windings.py` | 921 | QA tool: detect loop-closure / winding inconsistencies in the patch+PCL graph. |
| `scripts/spiral/connect_overlapping_patches.py` | 689 | Merge overlapping patches into same-winding groups. |
| `scripts/spiral/plot_winding_graph.py` | 1362 | Visualisation/QA. |
| `scripts/spiral/render_ink.py`, `plot_overlap_connections.py` | | Rendering/QA. |
| `scripts/spiral/umbilicus.py`, `geom_utils.py` | | Helpers (umbilicus z→yx curve; `interp1d`). |
| `python/vc/surface_index.cpp`, `volume.cpp` | 781 | nanobind C++ bindings: `SurfacePatchIndex`/`QuadSurface` fast patch lookup + volume sampling. |
| `scripts/spiral/autoresearch.md` | 137 | Protocol for autonomous LLM hyperparameter/loss search against fixed satisfaction metrics. |

Language: **Python + PyTorch**, GPU/CUDA, single-GPU. Deps: `pyro` (Transform abstraction), `torchdiffeq` (ODE
integration), `kornia`, `einops`, `trimesh`, `zarr`, `cc3d`, `kimimaro`, `wandb`, nanobind C++ extension.

---

## 1. The algorithm end-to-end

### Core idea
Learn a **single global invertible map** `T: scroll-space (z,y,x) → spiral-space (z,y',x')` such that the true
papyrus surface, which in the scroll is a deformed spiral, becomes an **ideal Archimedean spiral** in spiral-space:
`r = dr_per_winding · θ/2π`. In spiral-space, "unrolling" is trivial: the winding index is
`shifted_radius / dr_per_winding` where `shifted_radius = ‖yx‖ − θ/2π·dr_per_winding`, and the U coordinate is
`winding·2π + θ` (cumulative angle), V is `z`. Because `T` is a diffeomorphism, it is globally invertible with no
fold-over, and `T⁻¹` reconstructs scroll positions for any spiral grid point (used for meshing and for placing loss
targets).

### The model = a composition of invertible transforms
`SpiralAndTransform.get_slice_to_spiral_transform()` builds
`ComposeTransform([gap_expander, maybe_flip, diffeomorphism, varying_linear, umbilicus]).inv`.
The listed pipeline is the **forward (ideal-spiral → scroll)** direction; `.inv` gives scroll→spiral. Each factor is a
diffeomorphism *by construction*, so the composite is too:

1. **`GapExpandingTransform` (radial gap-scale, the paper's per-winding radial pitch field).** Per `(z, θ, winding)`
   log-scale logits on a coarse lattice → positive scales `exp(·)` → inter-winding distances
   `dr_per_winding · scale`; winding radii are the cumulative sum. A point's radius is remapped by lerping between its
   bracketing windings' transformed radii. Inverse uses `searchsorted` (monotone radial remap → invertible). This lets
   real, non-uniform winding spacing be matched while keeping the ideal-space spiral perfectly uniform. Winding-0 and
   `θ=0` logit are pinned to avoid seam jumps.
2. **`maybe_flip`** — affine `xz`-flip for CW vs ACW handedness (dataset constant `spiral_outward_sense`).
3. **`IntegratedFlowDiffeomorphism` (the ODE flow — the heart of the no-fold guarantee).** A **stationary velocity
   field** `v(zyx)` (default `num_flow_timesteps=1`) sampled from a learned voxel lattice, integrated by **RK4 for 3
   steps** over normalised `[0,1]` flow coordinates. The flow of a smooth ODE is a diffeomorphism; the inverse is
   simply integrating with negative step. This carries the bulk non-rigid deformation. `truncate_at_step` lets you
   apply a partial flow (geodesic toward identity) for coarse-to-fine.
   - Velocity field is **multiresolution**: a coarse lattice (÷6) trilinearly upsampled + a high-res lattice, each with
     its own LR scale. Two implementations:
     - `CartesianFlowField`: `[3, Z, Y, X]` tensor, `grid_sample`d.
     - `CylindricalFlowField`: ragged `(z, r, φ)` lattice with `num_phi[r]=round(2πr)` cells/ring (fine outside, coarse
       inside), local (radial, tangential) basis rotated into yx on the fly, axis ring pinned to 0. More natural &
       memory-frugal for a scroll, but complex.
4. **`VaryingLinearTransform` (per-slice affine).** A `z`-dependent 2×2 `M(z)=expm(L(z))` on `yx`, `L` interpolated
   from per-z logits. `det M = exp(tr L) > 0` ⇒ always invertible & orientation-preserving. Captures per-slice
   shear/scale.
5. **`UmbilicusTransform`.** `z`-dependent yx translation moving the origin onto the umbilicus (scroll centre curve),
   from a smoothed `umbilicus.json`.

Plus one scalar **`dr_per_winding = softplus(logit)`** — the global Archimedean pitch, jointly learned.

### Injectivity / no-fold-over guarantee
Two layers: (a) **structural** — every factor is individually a diffeomorphism (ODE flow; matrix-exp with positive
determinant; strictly-monotone radial remap; translation), so the composition cannot fold; (b) **soft barrier** — a
**symmetric-Dirichlet energy** `tr(G)+tr(G⁻¹)−4` on the induced 2×2 metric of the spiral↔scroll surface map diverges
as any singular value → 0/∞, penalising in-surface collapse/flips. Together they give a *global* injective map, in
contrast to block-wise tracers that can disagree across block seams.

### Objective (weighted sum of losses, AdamW, 30k steps, exp LR decay)
All data losses reduce to two archetypes, evaluated by sampling points and using `T`/`T⁻¹`:
- **Radius losses** — points known to lie on one winding (a patch row/col, a track, a co-winding PCL) should have
  **constant `shifted_radius`**; penalise deviation from the per-track mean (hinge margin `≈0.025·dr`). `within_norm_p`
  > 1 emphasises the worst point; `across` norm `p < 1` is winner-take-all (encourages many fully-satisfied tracks).
- **DT / snap losses** — snap each track's median `shifted_radius` to the nearest **integer** winding, build the target
  point on that ideal winding (same z, θ), map it to scroll-space via `T⁻¹`, and pull the sampled scroll point toward
  it. This is a differentiable "distance-transform" snap to the discrete winding. Turned on late (`loss_start_*_dt`),
  optionally **progressive outward** across windings (anneal cutoff from umbilicus outward).

Data terms and their inputs:
- `patch_radius` / `patch_dt` — verified patches (primary, high weight). `unverified_patch_*` — untrusted patches,
  masked away near trusted geometry, own hyperparameters.
- `rel_winding` / `abs_winding` — PCL annotations: relative winding *difference* between patch pairs, and absolute
  winding number, enforced as integer constraints (handles θ=0 seam crossings explicitly).
- `unattached_pcl_radius/dt` — free ordered point strips.
- `track_radius/track_dt` (very high weight 200/40) with optional **EM mode**: periodically re-assign each
  auto-extracted track to its most-likely integer winding (mode/median, confidence-gated), then fit — robust handling
  of noisy surface-prediction geometry.
- `dense_normals` + `dense_spacing` ("**lasagna**") — dense fields: `nx/ny` predicted scroll-space normals (match the
  spiral radial covector direction via `Jᵀ`), and `grad_mag` = **winding-density** (windings per voxel). Spacing loss
  integrates `grad_mag` along the scroll-space image of a one-winding radial step and forces the line integral to **1**.
  *This is essentially taberna's Eulerian winding-gradient field used as a dense constraint.*
- `shell_outer` / `shell_no_cross` / `shell_z_drift` — outer-boundary tifxyz → polar `(z,θ)→radius` table; pins the
  outermost winding and forbids crossing it.
- Regularisers: `sym_dirichlet` (isometry/anti-fold), `bending` (extrinsic), `umbilicus` (origin pinning).

### Optimisation & phases
AdamW (fused), param groups with separate weight-decay/LR scales for flow vs gap-expander vs linear. Optional **snapping
post-phase**: freeze a target point cloud, cosine-LR sub-passes pulling geometry onto satisfied windings (off by default
now). Resume-from-checkpoint supports widening the z-range.

### Evaluation metrics (the optimisation target for autoresearch)
`satisfied_patches` (≥95% of a patch's quads within `0.5·dr` spiral-radius **and** 4-voxel scan-distance of their target
winding), `satisfied_area`, `satisfied_unattached_pcls`/`_points`, `satisfied_tracks`/`_points`.

---

## 2. Inputs

- **Verified patches** — `tifxyz` grid quad-meshes (proofread VC surface segments): `Patch{zyxs:(H,W,3), scale,
  overlapping_ids, winding}`, stored as per-axis tiff + mask. Loaded, scaled by `downsample_factor`, eroded, z-ROI
  filtered.
- **Unverified ("unproofed") patches** — same format, lower trust.
- **Point collections (PCLs)** — `vc_pointcollections_json_version:1` JSON; each point has `p` (zyx) and optional
  `wind_a` (winding annotation). Split into cross-patch (linked to patches) and unattached strips.
- **Tracks** — dbm of pickled `(N,3)` int32 zyx arrays = skeletons of CC components of the surface-prediction zarr
  (`extract_surface_tracks.py`).
- **Normals zarr** (`las_008_nx/ny.ome.zarr`, uint8) + **grad_mag zarr** (winding density) → lasagna losses.
- **Shell** tifxyz (outer surface).
- **`umbilicus.json`** (z→yx centre curve).
- **Scroll CT zarr** — optional, visualisation only.
- All at `downsample_factor=4`; `voxel_size_um = 2.4·4`; production z-range 7000–16500 (≈2375 slices after ÷4).

## 3. Outputs

- **Checkpoints** (`.ckpt`: model state, optimiser, cfg, z-range).
- **Per-winding meshes** as `tifxyz` (one grid surface per winding, `save_mesh`), built by sampling an ideal spiral
  grid in spiral-space and mapping through `T⁻¹`; plus a **"spliced"** variant that substitutes patch-derived points
  in satisfied regions. Vertices outside the z-ROI marked invalid.
- **Satisfaction JSON** per patch/PCL + overlay PNGs.
- The fitted **spiral parameters**: scalar `dr_per_winding`, flow-field lattices, gap-expander logits, linear logits,
  umbilicus.

## 4. Implementation & out-of-core implications

- **Global, in-core, single-GPU.** `working_set_mode='global'` is the only supported mode; the *entire subvolume* is
  fit jointly under one flow field (`flow_bounds_radius=800`, full z). Patch atlas, tracks, and the lasagna uint8 volume
  are all resident on the GPU. The autoresearch budget is 30k steps ≈ 60 min on a small ROI; the taberna report's
  **~19 h on a 30-winding subvolume, NOT block-wise, is consistent** with this — runtime/memory scale with flow-field
  resolution × volume, and large z-ranges scale per-step sample counts and risk OOM (explicit guards exist).
- **No native out-of-core path.** This is the key gap for fenix. The math is amenable to tiling: the flow params
  themselves are small relative to the CT volume (a lattice), and the constraints (patches/tracks/lasagna) are spatially
  local. Out-of-core could (a) keep the global flow lattice resident but stream constraint data in z-chunks, or
  (b) domain-decompose the flow with overlap and stitch. The cylindrical lattice already reduces radial memory.
- C++ (nanobind) only for fast patch indexing / volume sampling, not the optimisation.

## 5. Key data structures

- `SpiralAndTransform(nn.Module)` — owns all params; emits a pyro `ComposeTransform` per iteration.
- Flow lattices — cartesian `[3,Z,Y,X]` or ragged cylindrical rings (LR+HR).
- `PatchGpuAtlas` — packs all patch grids into GPU tensors for batched row/col track sampling.
- **Spiral space** `(z, y, x)`: Archimedean `r=dr·θ/2π`; `shifted_radius = r − θ/2π·dr = winding·dr`. The whole method
  hinges on `shifted_radius` turning "which winding?" into a smooth scalar.
- `ShellPolarMap` — `(z,θ)→radius` table with confidence, EDT-filled + smoothed.

---

## 6. What fenix should incorporate (durable essence vs cruft)

**Durable algorithmic essence:**
1. **Compositional global diffeomorphism = SVF/ODE-flow ⊕ structured layers.** A stationary velocity field integrated by
   RK4 for the bulk deformation (guarantees a globally invertible, fold-free map — the single biggest advantage over
   block-wise tracers), composed with cheap structured factors: per-slice affine via `expm` (positive det), umbilicus
   shift, and a **radial per-winding gap-scale**. This is the cleanest principled backbone for a unified unroller and
   should be fenix's core representation of the deformation.
2. **Spiral-space `shifted_radius` parameterisation.** Reduces winding assignment to a smooth scalar and makes
   unrolling a trivial coordinate read-off. Ties directly to taberna's Eulerian winding field — the two are the
   continuous and discrete views of the same quantity.
3. **Two loss archetypes + winding-number annotations.** (a) co-winding/equal-`shifted_radius` constraints, (b)
   integer-winding snap (DT). Plus relative/absolute winding-number constraints from sparse annotations. A small,
   general vocabulary that subsumes VC's surface tracer outputs, sparse human points, and auto-traced tracks.
4. **Dense winding-density ("lasagna") spacing loss = line integral of `grad_mag` = 1 per winding.** This is precisely
   taberna's winding-gradient field repurposed as a dense data term — the natural bridge between the Eulerian winding
   field and the diffeomorphic fit. High-value to unify.
5. **Symmetric-Dirichlet isometry energy** (closed-form 2×2 `tr(G)+tr(G⁻¹)`) as the anti-distortion/anti-fold
   regulariser. Standard, cheap, durable.
6. **EM track assignment** for noisy auto-extracted geometry (periodic winner-take-all winding assignment, confidence
   gated). The right way to fold in cheap, abundant, ambiguous surface-prediction skeletons.
7. **`searchsorted`-invertible monotone radial remap** as a pattern for adding data-adaptive invertible layers.

**Implementation cruft (do not port literally):**
- The autoresearch harness, wandb, the ~150-key hyperparameter surface, redundant loss variants (radius `inv` vs
  not, coverage vs hinge), the snapping post-phase, verbose visualisation/QA scripts.
- Dataset-specific shell losses and the cartesian-vs-cylindrical field duplication (pick one; cylindrical is more
  natural for scrolls but ragged-indexing-heavy).
- Per-iteration rebuild of the pyro `ComposeTransform`; the pyro/torchdiffeq dependency itself (a hand-rolled SVF
  integrator is ~30 lines and was already inlined for the stationary fast path).

**The fenix gap to solve:** make this **out-of-core / block-wise with global consistency**. Henderson's method proves a
global diffeomorphic fit is achievable and high-quality but is in-core single-GPU (~19 h, memory-bound). fenix's unified
method should keep the global diffeomorphism's invertibility guarantee while streaming constraints (and possibly the
flow lattice) in z-tiles, reconciling with taberna's block-wise winding field and VC's Ceres tracer at block seams.
