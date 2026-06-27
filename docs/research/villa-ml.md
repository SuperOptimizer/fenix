# Villa segmentation / ML / surface-support research (for fenix C++26 rewrite)

Scope: `villa/segmentation/`, `villa/lasagna/` (focus `dense_batch_min_cut/`, `snap_surf/`),
`villa/ink-detection/` (skim). Read-only survey. This is the secondary/support cluster:
breadth over depth.

---

## 1. segmentation/

Four subprojects: `models/`, `evaluation/`, `vc_proofreader/`, `model_optimization_framework/`.

### 1a. models/ — sheet/surface segmentation networks

Everything is **3D U-Net family, PyTorch, voxel-grid segmentation** (not mesh). Inputs/outputs
are dense zarr volumes.

- **`models/arch/nnunet/`** — a full vendored copy of `nnUNetv2` (MIC-DKFZ). The production
  segmentation backbone. ResEnc U-Net plans (`nnUNetResEncUNetPlans_48G`). Used for sheet
  segmentation, fiber segmentation, normals regression.
- **`models/multi-task-3d-unet/`** — an in-house, nnUNet-inspired but independent 3D U-Net
  (ResEnc + squeeze-excite, borrows blocks from pytorch-3dunet). Key trait: **shared encoder,
  multiple decoder heads** doing simultaneous segmentation + regression with arbitrary numbers
  of inputs/targets. Only data format is **zarr**, patches shape `(C, Z, Y, X)`, float32 in
  `[0,1]`. Config-driven via a `ConfigManager` reading task YAMLs (`tasks/*.yaml`:
  `sheet_only`, `sheet_normals2`, `uv_normals`, `ink_updated`, etc.).
  - Tasks of note: **normals** task (`tasks/normals/`) regresses per-voxel sheet normals and
    writes face normals / UV volumes — this is the bridge from a voxel segmentation to a
    surface (`obj_to_uv_volume.py`, `write_face_normals_final.py` does normal interpolation /
    z-plane intersection of mesh edges).
  - Losses (`training/losses/losses.py`): SoftDice (+ memory-efficient variant), DC_and_CE,
    DC_and_BCE, DC_and_topk, **DC_SkelREC** (Dice + soft skeleton-recall — for thin sheet/fiber
    continuity), SoftSkeletonRecallLoss, GeneralizedDice, BCEDice, TopK, MaskedCosineLoss
    (for regressing normal vectors / directions), WeightedSmoothL1 (regression heads).
- **`models/batchgeneratorsv2/`** — vendored MIC-DKFZ augmentation library.

Takeaway: segmentation here = supervised **dense 3D semantic segmentation of papyrus sheets +
fibers + normals** via nnUNet/3D-UNet. This is pure ML, **leave external** in the rewrite. What
matters for fenix is the *outputs*: per-voxel sheet probability, fiber maps, and (crucially)
per-voxel **normal / direction fields** that downstream geometry (lasagna) consumes.

### 1b. evaluation/ — segmentation quality scoring (IMPORTANT)

`evaluate.py` is a config-driven harness: load GT label `.tif` + prediction `.tif` (3D via
tifffile), run a list of metric modules named in a YAML, aggregate to a pandas DataFrame, plot
histograms, log to wandb. Each metric module exposes `compute(label, prediction, **hp)`.
Configs: `recto.yml`, `recto-verso.yml`, `fibers-mixed.yml`.

Metrics (`evaluation/metrics/`):
- **dice.py** — binary Dice (torch, GPU), threshold 0.5.
- **dice_multiclass.py** — per-class Dice, with `ignore_index`.
- **centerline_dice.py** — clDice / centerline-(or center-surface)-Dice: skeletonize GT &
  pred (2D per-slice `skimage.skeletonize`, lee/zhang; `surface=True` for 3D), build a
  tolerance radius (EDT) around GT skeleton, measure skeleton-in-tolerance overlap. The key
  metric for **thin-sheet / fiber topology** quality.
- **connected_components.py** — `cc3d` 26-connectivity; abs difference in CC counts per class
  (over/under-segmentation proxy).
- **critical_components.py / critical_components_multiclass.py** — counts of *positive* and
  *negative* critical components (topological errors: splits & merges) via false-negative /
  false-positive masks and component labeling. Topology-aware correctness.
- **skeleton_distance_length.py** — uses **kimimaro** to skeletonize into vertices/edges,
  extracts branch lengths, compares GT vs pred branch-length *distributions* via
  **Wasserstein distance** (histogram). Measures whether the network reproduces the
  statistical structure of fibers/sheets.
- **mean_ap.py** — sklearn average-precision per class (probability-map quality).

No NSD/Hausdorff module present here; the topology metrics (clDice, critical components,
skeleton-length-Wasserstein) are the distinctive ones. **Winding-consistency is NOT scored
here** — that lives in lasagna (winding density/volume losses), not in this eval harness.

### 1c. vc_proofreader/ — human-in-the-loop label QC

A **napari**-based GUI (`main.py`, `image_label_viewer.py`) plus `extract_good_labels.py`.
Purpose: scroll through 3D patches of an image volume + its label volume, let a human
accept/reject patches, and export the accepted ones to a new zarr/tif dataset (with a progress
log). `extract_good_labels.py` is the headless version: scans a zarr in cubic chunks, filters
by nonzero %, labeled %, and `cc3d` connected-component count thresholds, writing only passing
chunks to a sparse output zarr. It's **training-data curation/QC tooling**, not an algorithm —
relevant only as a workflow concept (fenix may want a label-curation step) but nothing to port.

### 1d. model_optimization_framework/

Orchestration over nnUNet: `generate_trainers.py` emits trainer subclass variants for a
hyperparameter grid (lr, weight decay, epochs, oversample %), `run_training.py` /
`run_inference.py` run them across N GPUs, `evaluate_models.py` scores predictions using the
evaluation harness above and picks winners. Pure MLOps/automation. Not for porting.

---

## 2. lasagna/ — surface fitting / tracing core (THE relevant algorithms)

lasagna is the **geometry** half: it turns voxel segmentation/normal/direction fields into an
actual unrolled surface. README `lasagna_3d.md` is the spec. It is a complete 3D rewrite of an
earlier 2D model; the 2D UNet inference is reused, fitting is independent.

### 2a. Core model: differentiable quad-mesh fitting (fit.py + opt_loss_*.py + model.py)

This is the heart, and a strong candidate to reimplement in C++26.

- **Representation**: a scroll region is a quad-mesh tensor `(D, Hm, Wm, 3)` of 3D positions in
  **full-res voxel coords**. D = winding/depth count, H = scroll height, W = arc width along one
  winding. A 2D slice at fixed D is one sheet (quadmesh grid). Stored as a **scale-space
  pyramid** (5 levels, each halves D,H,W) of residuals; reconstruct coarse→fine via trilinear
  upsample + add. Global transform initialized as an **arc** (part of a circle) + winding step.
- **Data terms** (sampled from preprocessed zarr volumes via `grid_sample`):
  - **cos** — periodic layer signal (UNet ch0), oscillates as slice crosses papyrus layers.
  - **grad_mag** — sheet *density* = |∇(fractional winding position)|.
  - **dir0/dir1** — double-angle (180°-symmetric) encoding of in-plane layer normal (UNet
    ch2/3), stored per-axis (x/y/z slicing).
  - **valid** mask, optional **pred_dt** (distance to nearest predicted surface skeleton).
  - 3D fusion uses per-axis weights `w_axis = sqrt(1 - n_axis²)` (in-plane projection); the
    UNet emits projected gradient `gm·sqrt(1-n²)` so `Σgm/Σw` recovers the true 3D gradient.
- **Connection vectors**: between neighbor windings (D±1) the conn vector must hit the next
  winding orthogonally; computed as a ray/bilinear-quad intersection, tracked via a fractional
  `(h_off,w_off)` offset buffer updated each step (not gradient-optimized).
- **Loss library** (`opt_loss_*.py`, each a differentiable term):
  - `dir` (tangent/normal alignment to data), `step` (mesh edge length = mesh_step),
    `gradmag`, `data`, `data_plain`, `pred_dt` — the active set per spec.
  - regularizers: `smooth` (vertex ≈ avg of 4 neighbors), `bend` (penalize >60° folds),
    `flatten`, `cyl` (cylinder/shell prior), `corr` (winding correspondence anchors),
    `station`, `atlas_line`.
  - **winding consistency**: `winding_density` (barrier on sheet packing density) and
    `winding_volume` (auto winding offset/direction, ensures monotone winding numbers). These
    are the *winding-consistency* mechanisms — they live here, not in segmentation/evaluation.
  - `snap_surf` — wrapper around the snap_surf subpackage (see 2c).
- **Optimizer** (`optimizer.py`, `fit.py`): Adam-style PyTorch optimization over the pyramid,
  staged (configs `stages_scalespace.json`, `direct_scalespace.json`, `grow_*.json`): coarse
  level first, then finer; region growing ("grow_left"). Heavy use of custom CUDA `grid_sample`.
- **GPU kernels** (`*.cu`): `grid_sample_3d_u8_kernel.cu` + diff (backward) — trilinear sample
  of a `(C,Z,Y,X)` **uint8** volume at mesh points with offset/inv_scale, fused dequant; sparse
  variants (`sparse_grid_sample_3d_u8*`) for chunked/out-of-core volumes;
  `sparse_prefetch_chunks_kernel.cu` for streaming chunks. `sparse_cache.py` /
  `sparse_tensorstore_cache.py` / `tensorstore_omezarr.py` do out-of-core zarr access. This
  custom uint8 trilinear sampler (forward + analytic backward) is the perf-critical inner loop.

### 2b. dense_batch_min_cut/ — 2D graph extraction preprocessing (C++/OpenCV)

Despite the name, this is **not a classic s-t maxflow/min-cut**; it is a single 10954-line
`dense_batch_preprocess.cpp` doing **2D image → graph** extraction for sheet tracing, built as a
shared lib `libdense_batch_flow.so` (auto-built from `dense_batch_flow.py` via ctypes).
Pipeline per input `.tif` slice (needs OpenCV core/imgproc/ximgproc + libtiff):
1. Inverted threshold (127) → binary foreground islands (dark = papyrus).
2. **Distance transform** through the light domain to nearest dark island (16-bit normalized).
3. **Voronoi** of foreground connected components → component labels + boundaries; boundary
   skeleton (ximgproc thinning), spur-pruned by DT value.
4. **Source-rim ridge** detection: where neighboring source-pixel labels map to distant rim
   points → ridge candidates (separates touching sheets).
5. **Graph extraction**: junction clusters → nodes; traced skeleton paths → edges; edge
   **capacity = min DT along the traced edge** (bottleneck width); path search maximizes the
   minimum DT (widest path), shorter as tie-break. Emits a connectivity report + many debug
   TIFFs (`_graph_capacity`, `_graph_components`, etc.).
6. Optional **dense source-flow** (`--source x,y`): propagate flow from a source point through
   the DT field with backtracking → `_dense_flow.tif`, `_graph_edge_flow.tif`.

So it converts a CT slice into a **width-weighted skeleton graph of sheet boundaries** used to
seed/constrain tracing. The min-cut framing is conceptual (cut sheets apart along narrow
bottlenecks). Reusable idea: DT + Voronoi-boundary skeleton + widest-path graph for separating
touching sheets. The C++/OpenCV code itself is messy and tied to OpenCV/ximgproc — reimplement
the *idea* cleanly, don't port the file.

### 2c. snap_surf/ — multiresolution surface snapping/refinement

A Python subpackage (config/state/tensor/map_pyramid/map_objective/map_global/legacy/
fixture_io). It **snaps/refines a fitted mesh to the volume's true surface** through a
multi-scale (pyramid) optimization driven by `map_objective.py`:
- Operates on a packed (position, normal) field per mesh vertex; samples a surface grid +
  validity at planned support points; computes terms: distance to surface, vector-normal vs
  surface-normal alignment (`w_dist`, `w_vec_normal`, `w_surface_normal`), a **z-lift** term
  (Huber on heading/turn angle, lifts winding in z), smoothness (`w_smooth`), bend (`w_bend`),
  and a **Jacobian** regularizer detecting bad/inverted quads (`jacobian_bad_quad_mask`,
  `inverse_jacobian_bad_quad_mask`) — guards against fold-over.
- Strategy (`SnapSurfMapInitConfig`): seed → candidate growth → fringe growth → global opt
  interleaved, with **repair blocks** for inverted regions, scale levels, region growing. Pure
  optimization-based mesh refinement; outputs OBJ debug meshes (`debug_obj.py`).

Plus many lasagna scripts converting between representations: `tifxyz_io.py` /
`tifxyz_format.md` (the surface format, see §3), `labels_to_winding_volume.py`,
`labels_to_lasagna_normals.py`, `fit2tifxyz.py`, `convert_fit_zarr_to_vc3d_omezarr.py`,
`fitted_to_unet_labels.py`. There is also a 2D `train_unet.py` (the reused UNet: 2D TIFF layer
trainer, masked-MSE on `{0,1,2}` labels) and 3D variants (`train_unet_3d.py`, `eval_unet_3d.py`).

---

## 3. ink-detection/ — the downstream consumer (input contract)

This is the 2023 Grand-Prize-winning ink-detection pipeline (Nader/Farritor/Schilliger).
Models: **TimeSformer** (small, divided space-time attention — the canonical model), plus
Resnet3D-101 and I3D+non-local. Pure ML, **leave external**. What fenix needs is the **input
contract**, i.e. what an unrolled surface must look like.

### Input contract (the load-bearing part)

A "segment" / unrolled surface is a **directory** containing:
- `layers/00.tif, 01.tif, ... NN.tif` — a stack of 2D **flattened surface layers**. Each TIFF
  is the surface resampled at a fixed depth offset from the sheet (the "surface volume": the
  papyrus flattened to a plane, sliced through its thickness). Read as grayscale via
  `cv2.imread(..., 0)`.
- `<segment_id>_mask.png` (or `mask.png`) — 2D validity mask of the flattened region.
- `<segment_id>_inklabels.png` — GT ink (training only; `prepare.py` propagates these in).

Inference (`inference_timesformer.py`): selects a contiguous band of layers
(`start_idx`..`start_idx+in_chans`), default **~26 channels** centered on the surface (config
mentions 65 total available, 26 used; resnet variants use ranges like 18–38). Stacks chosen
layers → `(H, W, C)`, tiles into patches (64×64 stride 32, or 256-stride for i3d), feeds the
spatiotemporal model, outputs a 2D ink-probability map at surface resolution.

So **fenix's unrolling output should be compatible-in-spirit with this "tifxyz + rendered
layer stack"**: produce (a) the parametric surface, and (b) a stack of flattened layer images
through the sheet thickness + a 2D validity mask, all in surface (u,v) coordinates.

### tifxyz_dataset/ (newer, vesuvius pkg, partly stub)

Bridges the modern `tifxyz` surface format to ink training: `patch_finding.py` finds ink-bearing
patches over tifxyz segments + zarr volume; `tifxyz_dataset.py` voxelizes surface labels, builds
normal-offset masks and projected-loss masks, reads volume crops along the surface; `train.py`
is an explicit **stub (do-not-use)**. Confirms the direction of travel: surface stored as
**tifxyz**, ink sampled along surface normals from the raw zarr.

### The surface format: TIFXYZ (`lasagna/tifxyz_format.md`)

The canonical surface interchange format (Volume Cartographer). A **directory**:
- `x.tif, y.tif, z.tif` — single-channel, identical W×H; pixel `(row,col)` = one quad-grid
  vertex, value = that vertex's X/Y/Z in volume voxel coords (prefer float32). The image is a
  regular 2D lattice in (u,v) index space; physical step from `meta.json.scale`.
- `meta.json` (scale, etc.), optional `mask.tif` (channel 0 validity, may be integer-multiple
  higher res), optional `generations.tif`, arbitrary extra per-vertex channel `*.tif`.
- Validity: invalid sentinel `(-1,-1,-1)`; load-time rule invalidates any vertex with `Z<=0`;
  mask values `>=255` keep, `<255` invalidate.

This is essentially a UV→XYZ displacement map. **fenix should read/write TIFXYZ** — it is the
lingua franca tying lasagna fitting, VC, and ink-detection together.

---

## 4. Evaluation metrics catalog (for a fenix eval harness)

Two separate worlds:

**Voxel segmentation (segmentation/evaluation):** Dice (binary + multiclass), clDice /
center-surface Dice (skeleton-in-tolerance), connected-component count diff (cc3d), critical
components (topological split/merge counts, pos & neg), skeleton branch-length **Wasserstein**
distance (kimimaro), mean AP. No NSD/Hausdorff present; topology metrics are the distinctive
emphasis. Config-driven, per-sample → DataFrame → histograms/wandb.

**Surface/winding quality (lasagna, as optimization losses not a scorer):** winding density
barrier, winding volume monotonicity/auto-offset, mesh step regularity, bend (<60°), Jacobian
non-inversion (no fold-over), surface-normal vs data-normal alignment, distance-to-surface.
fenix's eval harness should add explicit **winding-consistency** and **mesh-quality (Jacobian,
self-intersection)** scores — these only exist implicitly as losses today.

NSD (Normalized Surface Dice) is referenced in the team brief but I did **not** find an NSD
implementation in these trees — worth adding fresh in fenix.

---

## 5. Dependencies & what to reimplement in C++26 vs leave external

**Leave external (pure ML, PyTorch/CUDA, train rarely, run via existing weights):**
- nnUNetv2 + multi-task-3d-unet sheet/fiber/normal segmentation. Heavy DKFZ stack
  (batchgenerators, nnunet). fenix consumes their *outputs* (sheet prob, normals, dir fields).
- The 2D layer UNet (cos/grad_mag/dir prediction) — reuse inference; or retrain externally.
- ink-detection (TimeSformer/Resnet3D/I3D) — entirely downstream; fenix only must emit the
  layer-stack + mask + tifxyz it expects.
- vc_proofreader (napari GUI) and model_optimization_framework (MLOps) — not core.

**Reimplement in C++26 (the geometry / numeric core — this is fenix's reason to exist):**
1. **TIFXYZ + OME-zarr I/O** (read/write surfaces and volumes; out-of-core chunk streaming —
   cf. tensorstore_omezarr/sparse_cache). Foundational.
2. **Custom trilinear sampler of uint8 (C,Z,Y,X) volumes at mesh points, forward + analytic
   backward**, with sparse/streaming chunk prefetch (the `grid_sample_3d_u8*` + sparse kernels).
   This is the perf-critical inner loop of fitting; ideal for C++26 + GPU.
3. **Differentiable quad-mesh fitting** (lasagna fit): scale-space pyramid of `(D,H,W,3)` mesh,
   arc init, data terms (cos/grad_mag/dir/data/pred_dt) + regularizers (step/smooth/bend/
   winding density/winding volume), conn-vector bilinear-quad intersection, Adam-style staged
   optimizer. Needs autodiff (hand-written or a small AD layer) — the biggest port.
4. **snap_surf** multiresolution snap/refine: distance + normal-alignment + z-lift + Jacobian
   non-inversion terms, seed/candidate/fringe/repair growth. Port after (3).
5. **dense_batch_min_cut idea** (not the OpenCV file): distance transform + component Voronoi
   boundary skeleton + **widest-path (max-min-DT) graph** for separating touching sheets and
   seeding tracing. Clean reimplementation; depends on a DT + thinning + CC toolkit.
6. **Evaluation harness**: Dice/multiclass-Dice, clDice/center-surface Dice, connected-
   component & critical-component (topology) counts, skeleton branch-length Wasserstein, plus
   *new* NSD and explicit winding-consistency + mesh-Jacobian/self-intersection scores.

**External libs to replace with C++ equivalents:** OpenCV/ximgproc (DT, thinning, contours),
cc3d (connected components), kimimaro (skeletonization), skimage.skeletonize, scipy EDT,
tifffile/zarr/tensorstore, PyTorch autograd + custom CUDA. C++26 candidates: a tensor/AD layer
or libtorch interop for fitting, a DT/skeleton/CC geometry toolkit, TIFF + zarr/tensorstore C++.
