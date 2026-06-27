# ThaumatoAnakalyptor (TA) — Architecture & Algorithm Report

Source: `/home/forrest/villa/thaumato-anakalyptor/`
Author: Julian Schilliger (part of Vesuvius Challenge 2023 Grand Prize submission, with Youssef Nader & Luke Farritor).
Purpose for fenix: TA is the most directly relevant prior art — an end-to-end automatic pipeline that virtually unrolls an *entire* scroll from a raw CT volume. This report documents what it does, how, and what a clean C++26 rewrite ("fenix") should keep, change, or drop.

---

## 0. One-paragraph essence

TA does **not** trace sheets in 2D slices like classic Volume Cartographer. Instead it (1) detects papyrus *surface points* in 3D using gradient analysis of voxel brightness, producing an oriented surface **point cloud** for the whole scroll; (2) chops that point cloud into subvolumes and runs a **3D instance segmentation network (Mask3D)** to cut the cloud into many small **sheet patches**; (3) builds a **graph** whose nodes are patches and whose edges encode a *relative winding-angle offset* `k` (≈0 for "same sheet, adjacent patch", ≈±360° for "one wrap away") plus a `certainty`; (4) solves a global **winding-angle assignment** problem on that graph (each node gets a continuous winding angle `f_star`), which is exactly the "unrolling" — it linearizes the spiral; (5) turns the winding-angle-labeled point cloud into a **mesh**, (6) **flattens** the mesh to a 2D UV plane, and (7) **textures/renders** by sampling the original volume along the surface (and ± depth layers) to produce the flattened layered image stack that ink detection runs on.

The intellectual core is stage 3–4: **representing unrolling as a graph optimization over winding angles.**

---

## 1. Full pipeline, stage by stage

The CLI/GUI driver order (README "Execute the Pipeline" + `GUI/MainWindow.py`):

### Stage A — Volume preprocessing / grid cells
- Scroll scan is a stack of 2D TIFs (or OME-Zarr). TA wants **8µm** resolution grid cells. `generate_half_sized_grid.py` downsamples 4µm scans to 8µm. Volume is stored as a 3D grid of cell blocks (cubic chunks, e.g. 500³).
- **Umbilicus**: a human/GUI-provided polyline (`umbilicus.txt`) tracing the scroll's central axis through z. Coordinate convention is quirky (`uc = y+500, z+500, x+500`). The umbilicus is the reference for *every* "radial / outward" direction in the pipeline and for converting Cartesian positions to winding angles. `GUI/UmbilicusWindow.py` is the annotation tool.

### Stage B — Surface point cloud generation  (`grid_to_pointcloud.py` + `surface_detection.py`)
Per volume block (with the umbilicus giving a radial `reference_vector` for that block):
1. **Blur**: uniform 3×3×3 box blur (`uniform_blur3d`).
2. **3D Sobel** gradient → vector field `(Gx,Gy,Gz)` of brightness gradient (`sobel_filter_3d`, chunked for memory).
3. **Vector convolution** (`vector_convolution`): in sliding 20³ windows, find the "mean indiscriminative" local sheet-normal direction. Because a sheet normal `v` and `-v` are equivalent, it can't just average; it samples ~200 random candidate unit vectors and picks the one minimizing summed *undirected* angular distance (angle taken over [0, π/2]). This is a cheap robust normal estimate that ignores sign.
4. **Orient** all normals to agree with the umbilicus radial `reference_vector` (`adjust_vectors_to_global_direction`), then trilinearly upsample back to full res.
5. **Directional 1st derivative** = project Sobel gradient onto the oriented normal; **2nd derivative** = Sobel of that, again projected. A sheet seen edge-on is a *bell curve* of intensity across its thickness, so its center is where **2nd derivative ≈ 0 and |1st derivative| is large**.
6. Two surfaces emerge: **recto** (1st deriv > +thr) and **verso** (1st deriv < −thr) — the front and back faces of each papyrus sheet. (README note: recto/verso names are accidentally swapped.)
- **Output**: per-block PLY point clouds with `position (x,y,z)`, `normal (nx,ny,nz)` (unit, oriented), optional RGB. Recto/verso saved separately. Runs on GPU (PyTorch), DDP multi-GPU + DataLoader workers.

### Stage C — Instance segmentation into patches  (`pointcloud_to_instances.py` + `mask3d/`)
- Load 200³ PLY cells, split into overlapping **50³ subvolumes** (stride 25 → 50% overlap).
- Run **Mask3D** (a 3D transformer-style instance segmentation net; Minkowski/`Res16UNet18B` sparse-conv backbone, PyTorch Lightning, checkpoint `last-epoch.ckpt`). For each subvolume it outputs `pred_masks (N_points × N_instances)`, `pred_scores`, `pred_classes`. Class==1 = papyrus.
- Each instance = one **sheet patch**: a locally near-planar cluster of surface points. Post-processing filters by score, distance to umbilicus, and fits a sheet/plane (stores coefficients).
- **Output**: per-subvolume archives (`.7z`/tar) each holding `surface_N.ply` + `metadata_surface_N.json` (score, distance, centroid, fitted coefficients, winding angle estimate).

### Stage D — Graph construction  (`instances_to_graph.py`)
- **Node** = one patch, keyed `(block_x, block_y, block_z, patch_id)`; carries centroid, z, and an umbilicus-relative `winding_angle` (the `f_init` seed), optional ground-truth angle.
- **Edges** connect patches that *overlap in space* (within a block via Mask3D, and across adjacent blocks via the 50% subvolume overlap). Each edge stores:
  - `k` = relative winding-angle offset between the two patches. ≈0 → same sheet continuing; ≈±360 → the neighbor is the next/previous wrap of the spiral; `same_block` flag marks edges that jump windings inside one subvolume (these have |k| ≈ 360).
  - `certainty` = confidence the edge is correct (from overlap quality / patch agreement).
- **Output**: a `ScrollGraph` (pickle `.pkl`) and a compact **`graph.bin`** for the C++/GPU solver.

### Stage E — Winding-angle graph solve  (`graph_problem/` — the core)
- The C++ (`graph_problem`) / CUDA (`graph_problem_gpu`) solver reads `graph.bin`, assigns every node a continuous winding angle `f_star`, writes `output_graph.bin`. `instances_to_graph.py --create_graph` merges the solution back into the pickle (`..._BP_solved.pkl`). **Details in §2.**

### Stage F — Mesh generation  (`graph_to_mesh.py`)
- With each point now carrying a global winding angle, TA **unrolls**: position is parameterized as `umbilicus(z) + t · radial_direction` where the winding angle gives the wrap index. `optimize_adjacent_cpp` (a C++ binding) smooths the per-point radial `t` so adjacent windings don't overlap and the sheet is consistent.
- Builds an ordered point set → triangulated **mesh** (`.obj`), with normals oriented toward the umbilicus. Helpers: `meshing_utils.cpp` removes self-intersecting/degenerate triangles (SAT collision test on a KD-tree) and keeps the largest connected component.

### Stage G — Flattening / UV  (`mesh_to_uv.py`, `slim_uv.py`, `mesh_to_uv` SLIM)
- Mesh is flattened to 2D. Primary method is a custom **cylindrical + BFS angle-accumulation** unwrap: BFS over vertices accumulating Δangle about the umbilicus → `u`; distance-from-umbilicus → `v`; then orthographic refinement (raycast occlusion test, heatmap "undeformation" to reduce distortion). `slim_uv.py` provides SLIM (Scalable Locally Injective Maps) as an alternative/refinement. Output `.obj` with UV.

### Stage H — Texturing / rendering  (`mesh_to_surface.py`, `large_mesh_to_surface.py`, `rendering_utils/`)
- For each UV pixel inside a triangle (barycentric point-in-poly), compute the 3D surface point and sample the **original scroll volume** there, plus a stack of **±N layers along the surface normal** (e.g. ±32). Produces the flattened **layered TIF stack** (one image per depth) + composite. GPU batched (PyTorch Lightning DDP), async disk I/O. Also emits a **PPM** (per-pixel position+normal map, Volume-Cartographer-compatible) via `obj_to_ppm.py`/`ppm_to_layers.py`.
- The resulting flattened surface stack is what the separate **ink-detection** model (TimeSformer, Nader's repo) consumes.

---

## 2. The core: winding-angle graph optimization

This is the heart and the part fenix most needs to get right. Two solver generations exist.

### Data model (`node_structs.h`, `main.cpp`)
- **Node**: `z`, `f_init` (seed winding angle), `f_tilde` (current iterate), `f_star` (final), `gt`/`gt_f_star` (optional ground truth), `deleted`, `fixed`, plus (GPU version) many extra fields: `winding_nr`, `sides[18]`, `confidence`, `connectivity`, `happiness`, `closeness` — belief-propagation-style auxiliary states.
- **Edge**: `target_node`, `certainty`, `certainty_factor/factored` (adaptive reweighting), `k` (target winding offset), `same_block`, `fixed`, `gt_edge`.
- `graph.bin` layout: `uint32 num_nodes`; per node `{float z, float f_init, bool gt, float gt_f_star}`; then per node `{uint32 id, uint32 num_edges, edges[{uint32 target, float certainty, float k, bool same_block}]}`. Output `.bin`: `num_nodes` then per node `{float f_star, bool deleted}`.

### The objective
Find winding angles `f` minimizing a certainty-weighted disagreement with the edge offsets:
`minimize  Σ_edges certainty · penalty( (f_target − f_source) − k )`.
A trivial chain (DFS/MST propagation) would solve a tree, but the graph has **cycles** (the spiral closes on itself thousands of times) and **noisy/wrong edges**, so a robust global relaxation is needed.

### Solver, generation 1 — iterative "spring" relaxation (`main.cpp`, CPU/OpenMP)
- **`update_nodes`** (the kernel): for each node,
  `f_star_i = ( Σ_j certainty_j · (f_tilde_j − spring_constant · k_j) + o · f_tilde_i ) / ( Σ_j certainty_j + o )`,
  then `f_tilde ← f_star`. This is a weighted Jacobi/Gauss-Seidel step toward satisfying every spring `f_j − f_i = spring·k`. `o` is a self-anchoring term (inertia toward current value).
- **`solve`** anneals `spring_constant` over `steps` (e.g. start high ~6 "warmup", decay to 1.0) across `num_iterations` (~2000), so the field first stretches to the right global scale then settles — like simulated-annealing on the spring stiffness.
- **`update_weights`** adaptively down-weights edges whose realized offset disagrees with the consensus: `certainty_factor` moves toward `exp(−x·error)` (1 winding off → ~0.4). **`remove_invalid_edges`** prunes grossly violated edges between rounds. This is the robust/outlier-rejection mechanism — wrong patch links get suppressed.
- **`calculate_scale`** uses `--estimated_windings` (e.g. 60) to rescale so `(max f − min f) ≈ 360·windings`.
- **`assign_winding_angles`** finalizes via **Prim MST** (`prim_mst_assign_f_star`): build a max-confidence / min-k-disagreement spanning tree from the relaxed `f_tilde`, then propagate exact angles down the tree, snapping each node to the **closest valid winding angle** (`closest_valid_winding_angle`: nearest `f_init + 360·n`). MST gives a clean, cycle-free final labeling consistent with the relaxed solution.
- Other machinery: `find_largest_connected_component` (drop floating debris), `solve_fold`/`update_fold_nodes` (handle sheet "folds"), `auto_winding_direction`/`invert_winding_direction_graph` (pick spiral handedness automatically), histogram/video diagnostics of the winding-angle distribution.

### Solver, generation 2 — GPU belief-propagation-ish (`solve_gpu.cu`, ~5.7k lines)
- `update_nodes_kernel` is the same spring update plus a richer message-passing layer: per-node `sides[18]` accumulate directional evidence, `wnr_side` carries winding-number-side certainty, `connectivity`/`confidence`/`happiness` gate edge weights. Same-block edges get boosted (×7.5–10), other-block scaled by `other_block_factor`. It clamps interactions to within ±3·360° of the current estimate (locality) and uses adaptive per-edge certainty factors `1/(1+k_err²/2)`. Effectively a GPU annealed graph solver with BP-style auxiliary states for robustness at full-scroll scale. `main_gpu.cpp` drives it; `main_py.cpp` exposes a Python binding.
- There is also an older/alternative **PGMax belief propagation** path (`belief_propagation/bp_pgmax.py`, `BP_node_deactivation/`) and an **even older random-walk stitcher** (see §3).

### Earlier paradigm — random-walk stitching (`sheet_generation/solver.cpp`, `instances_to_sheets.py`)
Before the graph solver, TA assigned windings by **random walks** over the patch graph: start at a seed patch with winding 0, repeatedly step to overlapping patches accumulating discrete winding increments `k`, detect valid **loop closures** (returned to a visited (volume,patch,k) state consistently), reject walks with spatial overlaps or too-short loops, and aggregate many walks (frontier-weighted sampling to explore unassigned regions) into a consistent sheet. This is the "Random Walk" the README still describes; the C++ `graph_problem` spring/BP solver superseded it for the Grand Prize.

---

## 3. C++ / CUDA accelerated kernels

Two independent native subsystems plus the Mask3D ops.

### (a) `graph_problem/` — the winding solver (most important)
- `main.cpp` (CPU, OpenMP), `solve_gpu.cu` + `solve_gpu.h` + `main_gpu.cpp` (CUDA), `main_py.cpp` (pybind), `node_structs.{h,cpp}` (graph + binary IO), `argparse.hpp`. Builds the `graph_problem` / `graph_problem_gpu` executables. Accelerates the global winding-angle relaxation/BP over millions of nodes.

### (b) `sheet_generation/` — pybind11 point-cloud & mesh ops
- `pointcloud_processing.cpp` → loads PLY blocks from tar/7z (`happly.h`, libarchive), KD-tree NN/radius search (`nanoflann.h`), dedup by winding angle, subsample, overlap/normal alignment. Exposes `load_pointclouds`, `upsample_pointclouds`.
- `solver.cpp` → the random-walk sheet stitcher (`solve_random_walk`), multithreaded with frontier-weighted sampling; uses Eigen.
- `meshing_utils.cpp` → triangle self-intersection (SAT on KD-tree), winding-angle-aware filtering, `cluster_triangles` (connected components, keep largest). Exposes `compute_intersections_and_winding_angles`, `cluster_triangles`.
- Vendored: `hdbscan/` (clustering), `nanoflann.h` (KD-tree), `happly.h` (PLY IO).

### (c) `mask3d/` — sparse-conv / point ops CUDA (third-party)
- `third_party/pointnet2/_ext_src/` (ball_query, group_points, interpolate, sampling) and `utils/pointops2/` (knnquery, grouping, sampling, aggregation, attention, relative position encoding). These are standard point-cloud network ops backing the Mask3D inference. fenix would *replace* these with whatever ML stack it picks, not port them.

---

## 4. Key data structures & intermediate formats

| Stage | Format | Contents |
|---|---|---|
| Volume | TIF stack / OME-Zarr, 8µm grid cells (e.g. 500³) | raw CT brightness |
| Umbilicus | `umbilicus.txt` | polyline of scroll axis vs z |
| Surface cloud | per-block `.ply` (recto/verso) | pos, oriented normal, optional RGB |
| Patches | `.7z`/tar of `surface_N.ply` + `metadata_surface_N.json` | per-instance points, score, dist, centroid, sheet coeffs, winding-angle seed |
| Graph | `ScrollGraph` pickle `.pkl` + `graph.bin` | nodes (patches) + edges (k, certainty, same_block) |
| Solved graph | `output_graph.bin`, `..._BP_solved.pkl` | per-node `f_star` winding angle, deleted flag |
| Mesh | `.obj` (+ `.vcps` ordered point set) | unrolled vertices, triangles, normals, UV |
| UV | `.obj` triangle_uvs; PPM | 2D flattening |
| Render | layered `.tif` stack, composite, zarr/memmap; `.ppm` | flattened surface + depth layers |

The two binary contracts a rewrite must respect/replace: **patch metadata JSON** and **`graph.bin`** (node/edge layout in §2).

---

## 5. ML models & how inference is wired

- **Surface detection (Stage B)**: *no learned model* — pure classical 3D signal processing (blur → Sobel → robust normal vote → derivative thresholds) on GPU via PyTorch tensors. (This is a strength: deterministic, retrainable-free.)
- **Instance segmentation (Stage C)**: **Mask3D** — a 3D instance-segmentation transformer over sparse voxels (Minkowski-style `Res16UNet18B` backbone), configured by Hydra, run under PyTorch Lightning, checkpoint `mask3d/saved/train/last-epoch.ckpt`. Inference path: PLY → preprocess (coords+RGB+dummy labels, STPLS3D-style normalization) → `Mask3DInference.prepare_item` → model → `{pred_masks, pred_scores, pred_classes}` → filter class==1. Training data + scripts under `mask3d/` and `generate_*_dataset.py`.
- **Ink detection**: out of scope of this repo (separate TimeSformer model), consumes the flattened layered TIFs.

---

## 6. Dependencies

- **Python/ML**: PyTorch, PyTorch Lightning, Hydra/OmegaConf, MinkowskiEngine (sparse conv), Mask3D + pointnet2/pointops2 custom CUDA ext.
- **Geometry/IO**: Open3D, numpy, scipy, scikit-learn, tifffile, zarr, opencv (`cv2`), PyQt5 (GUI), 7z/libarchive.
- **C++**: OpenCV (graph_problem video/IO), OpenMP, CUDA, Eigen, pybind11, yaml-cpp, libarchive; vendored nanoflann, happly, hdbscan; `argparse.hpp`.
- **Runtime**: RTX 4090 / ≥24 GB VRAM, **196 GB RAM + 250 GB swap**, ≥32 threads, NVMe; ~2× scroll size scratch space; precompute "expected to take a few days". Docker (`DockerfileThaumato`, nvidia-docker).

---

## 7. Algorithmic essence for fenix + what was painful

### What fenix should reimplement (the durable ideas)
1. **3D surface-point extraction by gradient/derivative analysis** — the bell-curve-across-thickness model (blur → 3D Sobel → sign-agnostic normal estimate oriented by umbilicus → 1st/2nd-derivative thresholds for recto/verso). Classical, deterministic, GPU-friendly; a natural fit for C++26 + a compute backend. No training needed.
2. **The winding-angle graph formulation** — patches/nodes with edges carrying a relative offset `k ∈ {0, ±360, …}` and a certainty; **unrolling = assigning each node a global continuous winding angle.** This is the central abstraction worth keeping verbatim.
3. **A robust global solver** for that objective: annealed weighted relaxation (`f_i ← Σ w(f_j − spring·k)/Σw`) + adaptive edge reweighting + invalid-edge pruning + final MST snap to valid 360° multiples. Cleanly expressible as a C++26 sparse-graph numeric solver (and the part most worth making fast/parallel/GPU).
4. **Winding-angle → mesh → UV → layered render** chain, with self-intersection cleanup and depth-layer sampling of the original volume.

### What was painful / slow in the Python+CUDA hybrid (avoid in fenix)
- **Three+ overlapping solver generations** (random-walk `solver.cpp`, PGMax BP, spring/BP `graph_problem`) and dead/duplicated files (`instances_to_sheets (strange solve change…).py`, "Remove unused/old files" TODO). Lots of cruft, unclear source of truth.
- **Process-per-stage with fragile file handoffs**: `.ply` → `.7z` → `.pkl` → `.bin` → `.obj` → `.tif`, with hardcoded paths, manual `--continue_from`/`--recompute` step bookkeeping, recto/verso naming bug. Huge intermediate storage (~2× scroll) and serialization overhead.
- **Heavy heterogeneous deps**: PyTorch+Lightning+Hydra+MinkowskiEngine+Open3D+OpenCV+pybind11+CUDA all in one Docker; brittle to build (must hand-compile C++ if image not rebuilt). A single C++26 binary with one compute backend removes most of this.
- **Resource hunger**: days of precompute, 196 GB RAM + 250 GB swap — largely Python memory blowup and redundant point-cloud copies across the Python/C++ boundary. Native unified memory + streaming would cut this.
- **Mask3D dependency** for patching is the one genuinely learned component; it needs training data and a fragile sparse-conv stack. fenix should decide early whether to keep an ML patcher or replace instance segmentation with a classical/region-growing clustering of the surface cloud (cheaper, no training, but possibly less robust at sheet touchings).
- **GPU solver complexity**: `solve_gpu.cu` ballooned to ~5.7k lines with many ad-hoc heuristic fields (`happiness`, `sides`, `closeness`). The math underneath is a simple weighted relaxation; a fenix rewrite can recover a clean, well-specified energy minimization instead of accreted heuristics.
