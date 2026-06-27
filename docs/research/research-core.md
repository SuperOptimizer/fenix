# Taberna Core Libraries — Architecture Report

Research for a greenfield C++26 rewrite. Source: `/home/forrest/taberna/src/`.
Read in full: all .c/.h under `segmentation/ topo/ unwrap/ eval/ postproc/ annotate/ io/ common/`
(28 .c + 28 .h). NOTHING MODIFIED.

---

## 0. Cross-cutting conventions (the substrate)

- **Volume layout** (universal): z-major, x-fastest. Voxel `(z,y,x)` at
  `v[(z*ny+y)*nx+x]`. Every module re-defines the `IDX`/`idx`/`V`/`MIDX` macro
  locally; there is no shared accessor. This is the single most repeated idiom in
  the codebase and the most obvious thing to centralize in C++ (a `Volume<T>` view
  with `operator()(z,y,x)` and explicit strides).
- **Scalar types** (`common/types.h`): `u8/s8/u16/.../f32/f64` aliases over
  `<stdint.h>`. `snic.h` *re-declares the identical typedefs inline* (predates the
  shared header) — a smell C++ would kill with one header.
- **Dimensions** passed as loose `int nz, ny, nx` to every function (3 ints, never
  a struct). Sizes computed ad hoc as `(size_t)nz*ny*nx`; a few sites risk `int`
  overflow on the index before the `(size_t)` cast but most are careful.
- **Normal field convention**: `normal` is `3*N` f32 interleaved **(nx,ny,nz)**
  per voxel — i.e. x,y,z order, OPPOSITE to the z,y,x voxel layout. This
  x/y/z-vs-z/y/x mismatch is a live bug magnet: `trace.c::normal_at()` manually
  swaps components (`n[0]=normal[3i+2]` …) to convert. A typed `Vec3` with named
  axes removes this entire class of error.
- **Memory/ownership**: classic C. Out-params are caller-allocated and must be
  pre-zeroed (documented in comments, unchecked). Functions that allocate return
  malloc'd buffers the caller frees, or fill caller out-pointers conditionally
  (`if (node_of_out) *node_of_out = ...; else free(...)`). Error path = manual
  cascade of `free()` then return negative int. No RAII, no cleanup helper; the
  `sheet_tensor.c` 8-buffer alloc/free is the worst offender.
- **Parallelism**: OpenMP `#pragma omp parallel for schedule(static)` on the
  outer `z` loop, pervasively. No SIMD intrinsics anywhere — relies on
  autovectorization. `io/tiff_vol.c` has a `__attribute__((constructor))` that
  forces `omp_set_num_threads` to online core count (works around containers
  reporting affinity=1). `io/mimalloc_tune.c` is another constructor tuning
  mimalloc purge delay. Both are global-init side effects linked into every tool.
- **Determinism**: SNIC, MWS, persistence all advertise determinism (stable
  tie-breaks by index/gid). Important to preserve in the rewrite.

---

## 1. `common/`

`types.h` only — the scalar aliases. No code. In C++26 this becomes the home of
the `Volume`/`Vec3`/`Span3` value types and the shared index math.

---

## 2. `segmentation/` — dense volume → sheets / graph

The largest and most central module. Two parallel philosophies live here:
(A) **dense field → graph → partition** (snic + sheet_tensor + affinity +
partition + stitch), the "experiment E1" signed-graph path; and
(B) **dense field → surface mask** (sheet_tensor/hessian + ridge/ced/trace), the
direct surface-detection path. They share the structure-tensor front end.

### 2.1 `sheet_tensor.{c,h}` — structure-tensor sheet detection (FRONT END)
- **Algorithm**: classical structure tensor. Optional pre-smooth (σ_grad) →
  central-difference gradient → outer product `g gᵀ` (6 unique components
  jxx..jyz) → Gaussian smooth each component at integration scale σ_tensor →
  per-voxel 3×3 symmetric eigendecomposition (cyclic Jacobi, `jacobi3`). Outputs
  **sheetness** `(λ0−λ1)/(λ0+eps)` ∈ [0,1] and **normal** = eigenvector of λ0
  (across-sheet axis, arbitrary sign).
- **Data**: `st_params{sigma_grad,sigma_tensor,eps}`. Allocates **8 full-volume
  f32 buffers** (work, scratch, jxx..jyz). Separable Gaussian (`gaussian_blur`)
  blurs x→y→z through one scratch buffer + a memcpy.
- **API**: `st_default_params()`, `st_sheet_detect(vol,nz,ny,nx,p,sheetness,normal)`.
- **Hot loops**: gradient loop and the per-voxel Jacobi loop (both omp parallel).
  `jacobi3` does 50 sweeps of full 3×3 matrix multiplies (`mat3_mul`) per voxel —
  extremely wasteful: allocates J,Jt,tmp,An,Vn on stack and does 3 matmuls/sweep
  instead of the standard in-place rotation. This is THE per-voxel hotspot and a
  prime rewrite target (closed-form symmetric 3×3 eigensolver, or the in-place
  rotation `hessian.c` already uses).
- **Smell**: two different Jacobi implementations exist (`jacobi3` here, `eig3m`
  in hessian.c, `eig3` in sheet_repair.c) — THREE copies of a symmetric-3×3
  eigensolver with different sort orders (descending |λ| vs ascending). Unify.

### 2.2 `hessian.{c,h}` — multi-scale Frangi plate detector (alt front end)
- **Algorithm**: Hessian eigen-analysis (Frangi vesselness, plate variant). Per
  scale σ: Gaussian blur → second derivatives (γ-normalized, γ=1) → 3×3 Hessian →
  eig (`eig3m`, |λ| ascending) → plate measure `[λ3<0]·exp(−Ra²/2α²)·(1−exp(−S²/2c²))`
  with `Ra=|λ2|/|λ3|`, `S=‖λ‖`. Multi-scale **max** over σ set, keeping the
  best scale's normal. Centers on the sheet ridge (2nd-deriv peak) better than
  the gradient structure tensor.
- **Data**: `hess_params{sigmas,nsigma,alpha,beta_c}`. `gblur` allocates 3 temp
  buffers (a,b,k) per call — re-allocated every scale (smell).
- **API**: `hessian_sheet_detect(...)`.
- Same output contract as sheet_tensor (drop-in alternative).

### 2.3 `ced.{c,h}` — coherence-enhancing diffusion
- **Algorithm**: anisotropic (sheet) diffusion to close porosity at the source.
  Computes normal+sheetness ONCE via `st_sheet_detect`, then explicit
  `u += dt·div(D∇u)` for `iters` steps, with `D = c_perp·I − (c_perp−c_norm)·nnᵀ`,
  `c_perp` tied to sheetness (diffuse strongly where confident). Diffuses ALONG
  sheet, weakly ACROSS.
- **Data**: `ced_params{st,c_norm,dt,iters}`. Allocates normal(3N)+sheet(N)+flux(3N).
  In-place on `vol`. dt ≤ 1/6 for 3D stability (documented, unchecked).
- **API**: `ced_default_params()`, `ced_diffuse(vol,...)`.
- Two omp loops per iter (flux, then divergence). Depends on sheet_tensor.

### 2.4 `ridge.{c,h}` — centerline NMS
- **Algorithm**: non-maximum suppression along the normal to collapse the thick
  sheetness band to a ~1-voxel ridge. Keep voxel iff sheetness≥s_min, ct≥i_min,
  and CT is a local max along ±step·n (trilinear samples). Sub-voxel parabola
  refinement rejects peaks sitting in a neighbor.
- **API**: `ridge_nms(ct,sheet,normal,...,out)` → kept count. Pure function, no
  alloc, **not** OMP-parallelized (serial triple loop) — easy parallel win.
- Has its own `trilin` (duplicated in trace.c).

### 2.5 `snic.{c,h}` — 3D SNIC supervoxels (KEY DATA STRUCTURE)
- **Algorithm**: SNIC (Simple Non-Iterative Clustering, Achanta 2017), 3D port
  from stabia. Single priority-queue (binary max-heap of negated distance) sweep
  from a seed grid (step d_seed, nudged to lowest-gradient voxel in 3×3×3).
  O(N), deterministic, no iteration. Distance = intensity + spatial compactness.
- **CENTRAL DATA STRUCTURE** — `Superpixel`:
  ```c
  typedef struct Superpixel {
    f32 x,y,z,c;                       // centroid + mean value
    u32 n,nlow,nmid,nhig;              // voxel count + value-bucket histogram
    u32 neighs[SUPERPIXEL_MAX_NEIGHS]; // 112 (=56*2) 0-terminated adjacency ids
  } Superpixel;                        // sized to a cacheline multiple
  ```
  This **fixed-capacity adjacency list IS the region-adjacency graph** — the first
  collapse from dense voxels to a sparse graph. Labels are 1-based; slot 0 unused.
  `snic()` returns count of adjacencies that OVERFLOWED the 112 cap (silent
  dropping — a real fragility at scale).
- **API**: `snic_superpixel_count()` (size the array), `snic_superpixel_max_neighs()`,
  `superpixel_add_neighbors()` (idempotent, returns 1 on overflow), `snic(img,
  ...,labels,superpixels)`.
- **Heap**: hand-rolled binary heap with macro `heap_fix_edge`; `heap_alloc`
  over-provisions `img_size*16` nodes (large up-front alloc).
- **Rewrite notes**: the fixed `neighs[112]` is the canonical "designed
  differently in C++" item — should be a small-vector / CSR adjacency. The header
  flags a planned extension: accumulate a per-supervoxel structure-tensor
  6-vector + eigendecompose for per-node sheet ORIENTATION (the spiral-undeform
  work needs orientation on each graph node). The new architecture should bake
  orientation into the node type from day one.

### 2.6 `affinity.{c,h}` — signed region-adjacency graph (KEY DATA STRUCTURE)
- **Algorithm**: builds a SIGNED RAG over foreground voxels (mask≠0). For each
  +x/+y/+z neighbor contact: `across=|n·d|` (edge dir vs sheet normal),
  `w = k_attract·(1−across) − k_repel·(across·sheetness)`. w>0 attract (same
  wrap), w<0 repel (touching different wraps).
- **CENTRAL DATA STRUCTURE**:
  ```c
  typedef struct { u32 a,b; f32 w; } sg_edge;      // signed edge
  typedef struct { sg_edge *edges; int nedges,nnodes; } sgraph;
  ```
  Plus `node_of` (voxel→dense node id or −1) and `voxel_of` (node→voxel) maps.
  Currently **voxel-resolution** (every fg voxel is a node); supervoxel coarsening
  via snic is the documented path to TB scale "behind the same sgraph".
- **API**: `affinity_default_params()`, `affinity_build_voxel(...)`, `sgraph_free()`.
- Edge list grown by `realloc` doubling. Serial build loop (not OMP).
- **Smell**: `nedges/nnodes` are `int` (overflow at scroll scale); the sgraph is
  the data structure that must scale, so it needs 64-bit counts and a CSR/typed
  representation in C++. This is the heart of the "experiment E1" path.

### 2.7 `partition.{c,h}` — Mutex Watershed
- **Algorithm**: Mutex Watershed (Wolf 2018), parameter-free signed-graph
  partition. Sort edges by descending |w|; attractive edge merges two clusters
  unless a mutex forbids; repulsive edge plants a mutex between clusters.
  Repulsion keeps touching wraps separate (the failure unsigned agglomeration
  can't prevent). Union-find (path-halving) + per-root mutex lists.
- **Data**: `ufind{parent,size}`, `mxlist{ids,n,cap}` per root (realloc-doubling).
  `mx_blocked` linear-scans both roots' (possibly stale) mutex lists, resolving
  via find — O(list) per query, the algorithmic soft spot.
- **API**: `mws_partition(g,labels)` → cluster count. qsorts a private copy of
  edges.
- Consumes `sgraph` from affinity.c (only dependency).

### 2.8 `stitch.{c,h}` — cross-brick label stitching
- **Algorithm**: delayed-merge primitive for block-wise processing. Two bricks
  processed independently have unrelated local ids; in the shared halo, match
  labels by majority co-occurrence and union B-labels into A. v0 is pairwise; full
  Lu/Zlateski/Seung recursive-doubling composes these.
- **API**: `stitch_overlap(a_over,b_over,n,na_labels,nb_labels,remap_b)` → total
  global labels.
- **BUG-ish smell**: the "running best per B" majority is NOT a true argmax — it
  only counts the CURRENT consecutive run of a candidate (`cur_a[b]==a` resets on
  any change), so a B-label whose best A-match is interleaved with others can be
  mis-assigned. Comment admits it's "fine for v0 / modest label counts". A real
  contingency table (hash of (a,b)→count) is the correct C++ form.

### 2.9 `trace.{c,h}` — advancing-front surface tracer (the watertight detector)
- **Algorithm**: grows each sheet as a connected surface. From a seed (unvisited
  voxel with sheetness≥seed_thresh, CT≥i_min), an advancing front (BFS over a
  `frontpt` queue) marches in the local tangent plane (8 directions at step) and
  at each step SNAPS back onto the ridge (CT max along ±snap_radius·n). Marching
  over a gap and re-acquiring bridges porosity → watertight (b1=0). A
  normal-consistency gate (`|n·n'|≥normal_cos`) prevents jumping to an adjacent
  wrap. `draw_line` rasterizes the path to keep the surface connected.
- **Data**: `trace_params` (st scales + 7 thresholds). `frontpt{z,y,x,nz,ny,nx}`
  (position + normal). Queue sized to full N (`malloc(N*sizeof(frontpt))`),
  `visited` byte map, plus sheet(N)+normal(3N) from st_sheet_detect.
- **API**: `sheet_trace(...)` → union mask + count; `sheet_trace_lab(...)` →
  per-front s32 label (adjacent wraps stay distinct). The two functions are
  ~95% duplicated (copy-paste with an extra debug-counter block in the _lab
  variant gated on `getenv("TR_DBG")`). Massive dedup target.
- **Smell**: serial seed loop (raster order); `getenv` inside the hot function;
  the union and labeled variants should be one templated routine.

---

## 3. `topo/` — native cubical persistent homology

### 3.1 `cubical.{c,h}` — generic boundary-matrix persistence
- **Algorithm**: cubical persistent homology, SUBLEVEL T-construction (voxels are
  0-cells; a k-cell enters at MAX of its voxels → fg 6-connected, bg 26-connected,
  matched to the Betti-Matching-3D oracle). Z/2 coefficients. v0 = generic
  boundary-matrix reduction (standard PH algorithm: sort cells by filtration,
  reduce columns by XOR until pivots unique).
- **Data**: explicit cell list `cell{val,dim,gid}` (up to 8N cells), global cell-id
  layout (vertex=v; edge=N+a·N+v; square=4N+a·N+v; cube=7N+v). Sparse Z/2 columns
  as sorted `ivec{a,n,cap}` (int vector); `iv_xor` = symmetric difference. `pos[]`
  maps gid→filtration order. `low2col` = pivot row→column.
- **API**: `cubical_persistence(field,...,npairs)` → `pers_pair{dim,birth,death}[]`;
  `cubical_features(...)` → `pers_feat{dim,birth,death,ncells,cells}[]` with a
  representative cycle (the reduced column's cells) per off-diagonal feature.
  `TOPO_INF` = essential-class death.
- **Cost**: O(cells) memory ~8N, reduction worst-case O(cells³). Explicitly
  "correctness-first, small/cropped/octant volumes only." The header notes the
  fast union-find dim-0 and cubical-specific dim 1/2 paths are future work.
- **Consumer**: `postproc/topo_surgery.c` (tunnel localization).

### 3.2 `persist0.{c,h}` — fast dim-0 persistence + simplification
- **Algorithm**: dim-0 PH via union-find merge tree. Sort voxels ascending,
  process in order, union active 6-neighbors, elder rule (older = smaller
  birth/index) records the younger basin's death at the merge saddle. Near-linear
  (sort-dominated). Shared core `dim0_core` does double duty.
- **`simplify_dim0`**: persistence-based topological simplification — every basin
  with persistence (saddle−min) < tau is RAISED to its saddle (removing spurious
  minima/noise at the source). Maintains per-root member linked lists
  (head/tail/next) to raise basins in O(size).
- **API**: `dim0_persistence(...)`, `simplify_dim0(field,...,tau)` (in place).
- Clean, the strongest module in topo/. The merge-tree + member-list pattern is a
  good template for the C++ version.

---

## 4. `unwrap/` — winding field → spiral → deformation

### 4.1 `winding_field.{c,h}` — Eulerian winding-number field (KEY DATA STRUCTURE)
- **THE dense backbone of unrolling.** Scalar field `winding` = continuous wrap
  index per foreground voxel; level sets are sheets; the value is the spiral
  coordinate the Archimedean fit consumes.
- **Algorithm(s)** — the most elaborate single file:
  - `winding_contour_warp`: analytic init that warps the radial coordinate by the
    per-z outer-boundary envelope R(θ) (180 angle bins, fill+smooth) so iso-winding
    contours follow the egg-shaped scroll, `W=(r·Rref/R(θ))/pitch + θ/2π`.
  - `winding_field_solve`: masked relaxation, **red-black Gauss-Seidel IN PLACE**
    (no Jacobi ping-pong — explicitly to avoid an 8 GB second buffer at LOD4).
    Parity coloring makes each color sweep parallel-safe. Supports:
    Dirichlet seeds, warm-start (coarse-to-fine multigrid), a `forcing` term
    (div of target gradient → Poisson gradient-matching so level sets follow real
    sheets), a Tikhonov `anchor_lambda` (preserve angular monodromy / lock tiles
    to a global coordinate), and THREE escalating anisotropy models:
    `aniso` (per-axis diagonal weights), on-the-fly `normal`+`sheetness` building
    `D=I−(1−α)s·nnᵀ`, and full `tensor6` (Dzz,Dyy,Dxx,Dyz,Dxz,Dxy) with lagged
    off-diagonal cross terms scaled by `cross_relax` for stability. Early-exit on
    convergence (maxd<eps).
- **Data**: `wfield_params` — a **12-field god-struct** of optional pointers and
  scalars (forcing, tensor6, normal, sheetness, aniso, anchor_lambda,
  tensor_alpha, cross_relax, …). The function body is a deeply nested branch over
  which optionals are set. This is the #1 "design differently in C++" target:
  the operator (isotropic / diagonal-aniso / on-the-fly-tensor / full-tensor)
  should be a strategy/policy type, not a runtime pointer-soup; the 19-point
  anisotropic stencil deserves to be its own tested unit.
- **API**: `winding_default_params()`, `winding_contour_warp(...)`,
  `winding_field_solve(mask,nz,ny,nx,umb,p,seed_value,seed_mask,winding)`.
- **Dependency**: `annotate/umbilicus.h` (polar init, center).
- **Perf**: the GS sweep is the dense hot loop; cross-term branches add diagonal
  neighbor lookups. `MAT`/`WV` macros re-check bounds+mask per neighbor.

### 4.2 `spiral_fit.{c,h}` — Archimedean spiral least squares
- **Algorithm**: closed-form OLS `r = a + b·θ_total`, θ_total=2π·winding (b =
  pitch/2π). `spiral_fit_lsq(theta,radius,n)` (pure) + `spiral_fit_from_field`
  (collect samples over fg with stride, via umbilicus_polar, then fit).
- **Data**: `spiral_params{a,b,rms,nsamples}`.
- Trivial, clean. Depends on umbilicus.

### 4.3 `deform.{c,h}` — deformation field + invertibility guard
- **Algorithm**: per fg voxel, radial displacement moving radius → ideal spiral
  radius `r_ideal=a+b·2π·winding` (keep θ,z). Disp interleaved (dx,dy,dz).
  `jacobian_fold_fraction`: fraction of interior voxels with `det(I+∇disp) ≤ 0`
  (folds / non-invertibility) — the strongest wrap-merge guard.
- **API**: `deform_build(...)`, `jacobian_fold_fraction(disp,...)`.
- Serial. Depends on umbilicus + spiral_fit. dz always 0 (radial remap only).

### 4.4 `wmetrics.{c,h}` — winding correctness metrics
- `winding_mvf`: Monotonicity Violation Fraction — cast `nrays` rays from
  umbilicus at mid-z, winding should increase outward; fraction of radial steps
  that don't (GT-free). `winding_satisfied_points`: fraction of absolute-winding
  annotations whose sampled field value is within tol.
- Depends on umbilicus + point_collection. Only operates on mid-z slice / absolute
  collections (v0 limitations noted).

---

## 5. `eval/` — metrics & the Kaggle composite score

### 5.1 `topo.{c,h}` — exact discrete topology (FOUNDATION for postproc + score)
- **Algorithm**: `cc_label` (union-find over lower neighbors, 6- or 26-conn,
  compacted 1-based ids). `euler_characteristic` (V−E+F−C over the cubical complex,
  exact cell counting). `betti_numbers` (b0=26-conn fg comps, b2=enclosed 6-conn
  bg cavities not touching border, b1=b0+b2−χ) and `betti_numbers_6` (the
  6-conn-fg / 26-conn-bg pairing the Betti-Matching oracle uses). `region_b1`
  (crop + Betti for windowed tunnel screen).
- **Data**: `topo_conn{TOPO_CONN6,TOPO_CONN26}`, `topo_betti{b0,b1,b2,chi}`.
- The most widely depended-on eval file: postproc, score all use `cc_label`/Betti.

### 5.2 `metrics.{c,h}` — VOI + adapted-Rand + surface dice
- **Algorithm**: Variation of Information split/merge `H(seg|gt)`/`H(gt|seg)` and
  CREMI adapted-Rand precision/recall/are, both permutation-invariant. Uses a
  hand-rolled open-addressing `h64` hash (u64→u64 count, splitmix64 finalizer) for
  the joint and marginal contingency tables. `surface_dice_at_tol` (brute-force
  boundary-voxel agreement within Chebyshev tol). `eval_surface` (ignore-aware
  3-class surface eval → dice + surface dice + Betti).
- **Smell**: open-addressing hash is reimplemented here AND conceptually overlaps
  with the union-find elsewhere; the `(a<<32)|b` joint key caps labels at 2³².

### 5.3 `nsd.{c,h}` — Normalized Surface Dice (DeepMind port)
- **Algorithm**: faithful port of Google `surface_distance` for spacing (1,1,1).
  256-entry marching-cubes surfel-area table (`NSD_AREA`); each voxel-corner gets
  an 8-bit code from its 2×2×2 neighborhood (`code_map`); mixed codes are surface
  points weighted by surfel area; exact squared Euclidean distance transform
  (Felzenszwalb-Huttenlocher 1D `dt1d`, applied along x,y,z = `edt2`); NSD@tol =
  area within tol / total area. Matches the official metric.
- Clean, self-contained, the gold-standard surface metric. `edt2` is serial.

### 5.4 `score.{c,h}` — official Kaggle composite
- **Algorithm**: `Score = 0.30·TopoScore + 0.35·SurfaceDice@2 + 0.35·VOI_score`.
  SurfaceDice via `surface_dice_nsd`. VOI over cc3d(26) instances scored on
  union-fg voxels (`eval_seg`), `voi_score=1/(1+0.3·VOI)`. TopoScore =
  `toposcore_native`: official 2×2×2 octant tiling, weighted Topo-F1 over Betti
  reduction; dim-0 matched count is EXACT (union-find `matched0_crop`, validated
  60/60), dim 1/2 are a CLAMPED PROXY (`rb(pred)+rb(gt)−rb(union)`, clamped to
  [0,min]). Comments are very candid about what's exact vs approximation.
- **Data**: `comp_score{surface_dice,voi,voi_score,topo_score,score,bettis…}`.
- Depends on metrics, topo, nsd. Allocates many full-volume buffers + crops.

---

## 6. `postproc/` — binary surface-mask cleanup (the "Kaggle win condition")

### 6.1 `morph.{c,h}` — classical morphology
- **Algorithm**: `majority_filter` (3×3×3 binary median = 27-nbhd majority,
  threshold 14, iterated → discrete curvature flow), `majority_filter_thresh`
  (general threshold), `remove_small_components` (dust removal via cc_label),
  `fill_holes` (flood bg from border, fill unreached cavities → b2=0),
  `plug_pinholes` (6-enclosed bg voxel), ball morphology
  (`dilate/erode/closing/opening_ball` via Euclidean SE offset list),
  `inplane_close` (normal-aware close: dilate then erode only through in-plane
  26-offsets so sheet centerline reconnects without thickening across),
  `connect_fragments` (set bg voxel fg iff within r it sees ≥2 distinct components
  — VOI stitching without SurfDice cost).
- **Data**: double-buffered `a/b` (alias-safe), ball offset list `off3`.
- Most functions OMP-parallel; `inplane_step`/`plug_pinholes` serial. Depends on
  eval/topo (cc_label, topo_conn). The headline `majority_filter` is the
  competition's biggest single gain.

### 6.2 `sheet_repair.{c,h}` — height-map watertight rebuild
- **Algorithm**: rebuild each connected component as a single-valued height
  surface w(u,v) over its PCA tangent plane. Per component: PCA (mean+cov+`eig3`)
  → normal=smallest-eigvec, tangents=other two → project to (u,v,w) → splat
  heights to a (u,v) grid → morphological closing of `known` cells defines the
  domain (fixes border-connected porosity) → Laplace inpaint interior gaps
  (Gauss-Seidel 300 iters) → rasterize watertight (draw cell + DDA-connect to +u,+v
  neighbors). A height surface over a simply-connected domain is a disk (b1=b2=0).
  `sheet_repair_windowed` runs it on overlapping blocks and ORs (curved sheets
  break the global single-plane assumption).
- **Data**: per-component point gather via offset/cur arrays; per-component grid
  buffers (h,cnt,known,dil,dom,U,V,W). Bails if grid >200M cells.
- **Smell**: THIRD copy of the Jacobi eigensolver (`eig3`); copy of `draw_line`
  (also in trace.c). Serial per component. Depends on eval/topo.

### 6.3 `topo_surgery.{c,h}` — PH-guided tunnel filling
- **Algorithm**: dim-1 topological surgery. Tile into windows; fast `betti_numbers_6`
  screen flags windows with 0<b1≤40; run `cubical_features` on the INVERTED window
  to get representative H1 loops; mark loop edge voxels; dilate-fill (ball radius)
  into the full mask to cap tunnels precisely (vs blunt global dilation that kills
  SurfDice). Repeat `iters`, converge when no window filled. Skips dense windows
  (fg·2>m) where generic reduction is too slow.
- **Data**: reusable sub/inv/loop window buffers. Decodes cubical gid layout to
  recover edge endpoints.
- Depends on eval/topo + topo/cubical. The one consumer of `cubical_features`.

---

## 7. `annotate/` — human annotation inputs

### 7.1 `umbilicus.{c,h}` — scroll center axis (HIGH-LEVERAGE INPUT)
- **Data structure**: `umbilicus{int n; f32 *z,*y,*x;}` — polyline of control
  points sorted ascending by z (struct-of-arrays). Provides the cylindrical
  coordinate system the entire unwrap is built on.
- **Algorithm**: load/save trivial "z y x" text (insertion-sorted by z).
  `umbilicus_estimate`: auto-estimate from a material mask — split z into bands,
  per band centroid then refine by coarse-to-fine local search minimizing angular
  variance of the radial extent (`umb_sym_score`, 72 angles), then 3-point moving
  average across z. `umbilicus_center` (linear interp, clamped),
  `umbilicus_polar` (θ=atan2, radius).
- Consumed by winding_field, spiral_fit, deform, wmetrics. Clean.

### 7.2 `point_collection.{c,h}` — winding annotations + link constraints
- **Data**: `anno_point{z,y,x,wind}`; `point_collection{id[64],absolute,pts,npts,
  cap}`; `link_constraint{a/b xyz, cannot}` (must-link/cannot-link);
  `pc_set{cols,links}`. Own line-based text format (collection/point/cannotlink/
  mustlink, `#` comments).
- `pc_load`/`pc_free`. Cannot-links feed the signed-graph mutex (hard cannot-link).
- Consumed by wmetrics. Classic realloc-doubling dynamic arrays.

---

## 8. `io/` — volume ingest

### 8.1 `mca.{c,h}` — matter-compressor archive reader (the REAL-data path)
- Reads regions of a `.mca` (matter-compressor archive) into u8 z-major volumes.
  `mca_dims`, persistent `mca_handle` (open once: mmap + node-table parse; read
  many regions vs per-read re-open), `mca_read`/`mca_load_region`, `mca_metadata`
  (borrowed JSON blob), `mca_roi_origin` (parse `roi.origin` via vendored json),
  `mca_find_region` (deterministic seeded material-rich box sampling).
- **Deps**: third-party `matter_compressor` (mc_open/mc_archive_*/mc_reader_*) +
  `json`. The only module touching external scroll-data infra. u8 output (CT is
  decompressed to ~1% MAE u8, per sheet_tensor comment).

### 8.2 `nrrd.{c,h}` — NRRD reader/writer
- Minimal NRRD: ASCII header + binary blob. Reads raw/gzip (zlib `inflateInit2`
  +32 auto), u8/u16/f32 little-endian, 3D. `nrrd_read`/`nrrd_read_f32` (cast to
  f32 taberna layout)/`nrrd_write_f32`/`nrrd_write_u8`. Axis order x,y,z
  fastest-first = taberna layout. Used for the Vesuvius feasibility .nrrd cubes.
  Deps: zlib.

### 8.3 `tiff_vol.{c,h}` — multi-page TIFF stacks
- Thin wrapper over vendored `tiff` submodule (`tiff_read_volume`/`tiff_write_volume`,
  LZW+raw multi-IFD). u8 single-channel only (the 320³ surface-detection cubes:
  CT 0..255; label 0=bg/1=surface/2=ignore). PLUS the OMP-thread-count constructor
  (see §0). Deps: tiff.

### 8.4 `mimalloc_tune.c` — allocator tuning constructor (see §0). No header.

---

## 9. End-to-end data flow (segmentation → unwrap)

```
io (mca/nrrd/tiff) ──> f32/u8 CT volume (z-major)
        │
        ▼
annotate/umbilicus ──(auto-estimate or human)──> center axis polyline
        │
        ▼
segmentation/sheet_tensor (or hessian) ──> sheetness[N] + normal[3N]
        │                                         │
        ├─ ced (optional: close porosity)         │
        │                                         ▼
        │                          ┌── PATH B (surface mask) ──┐
        │                          │ ridge_nms ──> thin ridge   │
        │                          │ trace ──> watertight sheets│ (or _lab)
        │                          └──────────────┬────────────┘
        │                                         ▼
        │                       postproc: majority_filter, remove_small,
        │                       fill_holes, sheet_repair, topo_surgery
        │                       (uses eval/topo cc_label + topo/cubical)
        │                                         ▼
        │                          eval/score (NSD + VOI + TopoScore) ── Kaggle metric
        │
        └── PATH A (graph/wrap segmentation, "E1")
            snic ──> Superpixel[] graph  (coarsen; voxel path also exists)
            affinity_build_voxel ──> sgraph (signed edges) [+ cannot-links from
                                                            point_collection]
            partition/mws_partition ──> per-wrap cluster labels
            stitch_overlap ──> global labels across bricks
            eval/metrics (VOI/ARE) ──> segmentation quality
                    │
                    ▼
unwrap/winding_field_solve (mask + umbilicus + optional sheet-normal tensor)
        ──> winding[N]  (THE backbone; level sets = sheets)
        │     (warm-started by winding_contour_warp; seeded by absolute annotations)
        ▼
unwrap/spiral_fit_from_field ──> spiral_params (a,b)
        ▼
unwrap/deform_build ──> displacement[3N];  jacobian_fold_fraction = invertibility guard
        ▼
unwrap/wmetrics (MVF, satisfied-points) ──> unwrap quality (mostly GT-free)
```
Two paths share the structure-tensor front end and the umbilicus. Path B feeds the
surface metric; Path A feeds segmentation + the winding field. The winding field is
the convergence point of the whole pipeline.

---

## 10. Central data structures the new architecture must define well

1. **`Volume<T>` / 3D view** — z-major strides, `(z,y,x)` access, bounds/clamp/
   reflect helpers, sub-box crop (every module hand-rolls this + crop loops).
2. **`Vec3` axis field** — kill the (x,y,z)-interleaved-vs-(z,y,x)-voxel mismatch.
3. **`Superpixel` graph** — replace fixed `neighs[112]` (silent overflow) with a
   small-vector/CSR adjacency; bake in per-node sheet ORIENTATION from the start.
4. **`sgraph` (signed RAG)** — 64-bit edge/node counts; CSR; the structure that
   must scale to TB; unify voxel- and supervoxel-resolution behind one type.
5. **`winding` field + its operator** — the dense backbone; replace the 12-field
   `wfield_params` pointer-soup with a typed diffusion-operator strategy
   (isotropic / diagonal-aniso / tensor) and a separately-tested 19-pt stencil.
6. **Persistence types** (`pers_pair`/`pers_feat` + cubical gid layout) and the
   `topo_betti`/`cc_label` foundation — heavily shared; deserve one clean
   topology core.
7. **Annotation types** (`umbilicus`, `pc_set`) — already clean; mostly port.

---

## 11. Top design issues for the C++26 rewrite

1. **No shared volume/index abstraction.** ~15 local `IDX` macros, repeated
   triple loops, repeated crop-to-subbox code. A `Volume`/`mdspan`-style view with
   compile-time-known rank and explicit strides removes the largest source of
   duplication and indexing bugs.
2. **Normal field axis-order inversion** (x,y,z interleaved over z,y,x voxels) is
   a latent bug, manually patched in trace.c. Typed vectors fix it.
3. **Three+ copies of the symmetric-3×3 eigensolver** (jacobi3 / eig3m / eig3)
   with inconsistent sort orders; multiple copies of `trilin`, `draw_line`,
   separable Gaussian blur, union-find, and dynamic-array doubling. One numerics
   header. `sheet_tensor`'s `jacobi3` (3 matmuls/sweep × 50 sweeps × N voxels) is
   also the per-voxel hotspot — use a closed-form or in-place solver.
4. **`wfield_params` god-struct** (12 optional fields, deeply nested runtime
   branching in the solver). The single biggest "design differently" item: make
   the operator a policy/strategy, extract the stencil.
5. **Fixed-capacity / int-width data structures that silently fail at scale**:
   `Superpixel.neighs[112]` (drops adjacencies, returns overflow count nobody must
   ignore), `sgraph.nedges/nnodes` as `int`, joint-VOI key capping labels at 2³².
   The graph/winding structures must be 64-bit and growable to be the TB-scale
   backbone the docs promise.
6. **Manual error/cleanup cascades** (sheet_tensor's 8-buffer alloc/free, every
   `if(!x){free(...);return -1;}`). RAII / `expected`-style returns.
7. **Duplicated near-identical functions**: `sheet_trace` vs `sheet_trace_lab`
   (~95% copy), the two persistence entry points, the betti6/betti pair. Templates
   / parameters collapse these.
8. **Global constructors as load-bearing config** (OMP thread count in tiff_vol,
   mimalloc tuning) — surprising side effects in a library; make explicit init.
9. **Serial hot loops that should be parallel** (ridge_nms, trace seed loop,
   nsd edt2, sheet_repair per-component, affinity build, deform). The OMP usage is
   inconsistent — some dense loops parallel, comparable ones serial.
10. **Correctness placeholders to finish, not just port**: TopoScore dim-1/2 proxy
    (exact needs Betti matching on cubical.c), VOI_score transform, stitch's
    fake-argmax majority, cubical.c's O(cells³) reduction (needs the fast dim-0/1/2
    paths). The rewrite is a chance to build the exact versions on a real
    persistence/Betti-matching core.
11. **Determinism is a feature** (SNIC, MWS, persistence tie-breaks) — preserve it
    explicitly in any parallelization.
