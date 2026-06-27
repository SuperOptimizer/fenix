# taberna — tools & orchestration research report

Scope: `/home/forrest/taberna/tools/` (~55 C drivers), `/home/forrest/taberna/scripts/`
(Python + shell orchestration), `/home/forrest/taberna/gui/` (Qt6+VTK viewer). Goal: reconstruct
the actual end-to-end workflow that turns a raw scroll CT volume into an unrolled surface, and
separate the current pipeline from abandoned experiments. Read-only; no files modified.

---

## 0. Big picture — TWO codebases share one repo

taberna is really **two distinct pipelines** that happen to share `src/` libraries and the `.mca`
I/O layer:

1. **The WINDING / UNROLL pipeline (the real product, current).** Operates on `.mca` pyramid
   archives of a full scroll. Core solver `sheet_sep3d` → coarse-prior tiling (`tile_winding_mr.py`)
   → TV regularize (`wind_tv`) → flatten (`unroll_wind`). Produces an unrolled papyrus image
   (`.pgm`) or a VC3D `tifxyz` surface. This is what "virtually unroll a scroll" means here.

2. **The Kaggle "Surface Detection" pipeline (a side competition, mostly experimental).** Operates
   on small TIFF cubes with ground-truth label TIFFs. Predicts a binary papyrus-surface mask, scored
   by TopoScore / SurfaceDice / VOI. This is the `surface_*`, `run_surface`, `sd_measure`, `ridge_*`,
   `eval_seg`, `topo_native`, `nsd_native`, `test_*` cluster. Many redundant tuning iterations.

A third, older NRRD-based "classical unwrap" path (`run_pipeline.c`, `unroll.c`) predates the `.mca`
winding pipeline and is now a reference/legacy implementation.

### Library / dependency layering (top `CMakeLists.txt`)
- `third-party/libs3` — anonymous S3 fetch.
- `third-party/matter-compressor` — the `.mca` codec; provides the `matter_compressor` target
  (`mc_open`, `mc_archive_open_dims`, `mc_archive_read_region`, …). This is the archive reader the
  whole pipeline + GUI sit on. (Format spec: `docs/mca-format-v2.md`.)
- `third-party/fysics` — physics kernels **+ the `mca_export` tool that BUILDS the `.mca` archives**
  from the open-data zarr; also exposes the `fysics` lib (noise est, guided denoise, MUSICA, aircut
  valley detection) used by a few tools (`sheet_sep3d`, `surface_prep`, `surf_pred`, `trace_mca`,
  `sd_measure`, `sheet_sep3d`).
- `src/` first-party libs: `taberna_io` (mca/nrrd/tiff), `taberna_annotate` (umbilicus, points),
  `taberna_segmentation` (sheet_tensor, snic, affinity, partition, ridge, trace, ced, hessian,
  stitch), `taberna_eval` (metrics, nsd, score, topo), `taberna_topo` (cubical, persist0),
  `taberna_postproc` (morph, sheet_repair, topo_surgery), `taberna_unwrap` (winding_field,
  spiral_fit, deform, wmetrics).

---

## 1. The CANONICAL end-to-end pipeline (current / working)

### Stage A — archive ingest (upstream of `tools/`, in `third-party/fysics`)
Vesuvius open-data S3 zarr volume → `fysics` `mca_export` → a **`.mca` LOD pyramid**
(`data/exports/pherc0332_L0.mca … _L5.mca`, also `PHercParis4_L1/L2`). `.mca` = matter-compressor
archive v2: mmappable, DCT-16³ + dead-zone quant + CABAC, 8 independent LODs, global quality `q`.
This is the single input to everything downstream. Scripts that support ingest:
`crawl_opendata.py` (build S3 manifest), `tune_region.py` (download a 1024³ region as raw zarr),
`l5_occupancy.py` / `s3_occupancy.py` (size estimates).

### Stage B — winding solve (the heart): `tools/sheet_sep3d`
`sheet_sep3d ARC OUTBASE lod z0 y0 x0 dz dy dx [minseg pitch LAM WEQ ztie recto LPRI radw barclose
zmed SP umbref wdrescue robsig usesheet slicecv cxoff [priorvol priorlod GLAM]]`
- Reads an `.mca` sub-volume. Per-slice 2D sheet labels (structure-tensor sheetness via
  `st_sheet_detect` + radial valley-count step edges) → one **global IRLS least-squares winding
  field** with a weak radius prior; auto-calibrates `pitch` (set `pitch=0`) to the sheet-crossing
  count so the field's absolute scale is data-pinned.
- Output: `OUTBASE_vol.f32` = winding field. **`_vol.f32` format is the central currency:** int32
  header `{dz,dy,dx,z0,y0,x0}` then `dz*dy*dx` float32, NaN off-material. Winding ≈ wraps from
  umbilicus (`r/pitch + θ/2π`).
- `priorvol/priorlod/GLAM` let a tile inherit a coarse global field as a per-voxel anchor — the key
  to tiling (below).

### Stage C — tiled multi-resolution winding: `scripts/tile_winding_mr.py` (WORKING tiler)
`tile_winding_mr.py ARC FINELOD Z0 Y0 X0 DZ TS NY NX STEP COARSELOD OUTDIR [NZ STEPZ]`
- **The validated tiling strategy.** (1) Solve ONE coarse global winding over the whole region at
  `COARSELOD` (one connected, internally consistent crop). (2) Run each fine `DZ×TS×TS` tile with
  that coarse field as `priorvol` + `LPRI≈8` + `GLAM≈0.3` so all tiles share an absolute scale.
  (3) Merge = **trivial direct average over overlaps** (no per-tile integer offset, no BFS).
- Prints seam-agreement diagnostics. Output: `OUTDIR/merged.f32` (`_vol`-style header).
- **`tile_winding.py` is the NEGATIVE diagnostic** that proved independent-tile stitching FAILS
  (adjacent crops agree only ~6% within 0.25 wrap; each crop assigns different integer offsets).
  Kept only as documentation of why the shared-global-reference fix exists. **Abandoned as a path.**

### Stage D — winding regularization: `tools/wind_tv` (current best cleaner)
`wind_tv ARC OUT lod z0 y0 x0 dz dy dx priorvol priorlod [lambda=0.3] [niter] [zc] [gmin] [pitch]
[order_mu]`
- Weighted-TV-L2 (ROF) regularization of the merged winding via Chambolle–Pock primal-dual, edge
  weight = sheetness. Flattens each sheet to one winding level (kills the along-sheet `floor(W)`
  flip) while letting `u` jump ~1 across gaps. **First method to beat plain `floor(W)`**: along-sheet
  flip 3.35%→2.14% (−36%); downstream the TV-cleaned field unrolls **+73% sharper sheet edges,
  +33% contrast** (validated, `docs/touching-sheets-plan.md` Phase 3a). Output `OUT_tv.f32`.

### Stage E — flatten / unroll: `tools/unroll_wind` (current unroller)
`unroll_wind ARC lod WINDING.f32 OUT.pgm [SAMP=160] [CY CX | "auto"]`
- Reads the (TV-cleaned) winding `_vol.f32` + the archive CT. **radial mode** collapses azimuth
  (sheet-vs-z quality reslice); **spiral mode** spreads wraps across azimuth (`w + atan2/2π`) for a
  text-unroll. `auto` finds the umbilicus center. Optional **FINE mode** samples a finer archive
  over a thin z-band for ink-scale resolution the winding LOD doesn't carry. Output: `OUT.pgm`.

### One-command driver: `scripts/unroll_pipeline.py`
`unroll_pipeline.py ARC FINELOD Z0 Y0 X0 DZ TS NY NX STEP COARSELOD OUTDIR [NZ STEPZ] [--samp N]
[--fine FINEARC SCALE ZA ZN] [--no-tv] [--tv-lambda L]`
Chains the validated pieces: **[1] `tile_winding_mr.py` → merged.f32 → [1.5] `wind_tv` →
merged_tv.f32 (on by default) → [2] `unroll_wind` → unroll.pgm**. This IS the canonical pipeline
entry point.

### Alternate output target — VC3D surface: `tools/wind_tifxyz`
`wind_tifxyz ARC lod WINDING.f32 OUTDIR [samp=160]` — exports the winding field as a `tifxyz`
QuadSurface (`{x,y,z,mask}.tif` float32 + `meta.json`), the Vesuvius Challenge surface-mesh format,
by scattering material voxels into (z, render-coord) bins. `tools/tifxyz` reads/renders such
surfaces back against an archive (interop with VC3D-grown segments).

---

## 2. Tool clusters (all ~55)

### Cluster 1 — Winding / unroll core (CURRENT product)
| tool | role | status |
|---|---|---|
| `sheet_sep3d` (1098 LOC) | central 3D winding solver, `.mca`→`_vol.f32` | **CURRENT — core** |
| `wind_tv` | TV winding regularizer (Phase 3a) | **CURRENT — best cleaner** |
| `unroll_wind` | flatten winding → unrolled `.pgm` | **CURRENT — unroller** |
| `wind_tifxyz` | winding → VC3D `tifxyz` surface export | **CURRENT** |
| `wind_label` | `floor(W)` instance-label volume `_lab.i32` + despeckle | CURRENT (robust interim, fallback to wind_tv) |
| `wind_couple` | z-couple per-slice 2D winding fields (3D SOR, Tukey) | CURRENT (z-coherence) |
| `sheet_sep` (833 LOC) | 2D delamination-graph winding (recto/verso) prototype | working prototype, superseded by sheet_sep3d |
| `wind_poisson` (542 LOC) | independent Poisson/Laplace winding | DIAGNOSTIC cross-validation |
| `wind_diag` | render winding bands on a slice (TIF overlays) | DIAGNOSTIC |
| `wind_cut` | Boykov–Kolmogorov/Ishikawa ordered-label min-cut (Phase 3b) | EXPERIMENTAL / banked (verified maxflow, low-yield) |
| `polar_sheet` | polar (angle×radius) reslice for optical-flow tests | DIAGNOSTIC |

### Cluster 2 — Unroll family (older iterations of the unroller)
| tool | role | status |
|---|---|---|
| `unroll` (NRRD-era, drives winding_field/spiral_fit/deform) | single-region end-to-end legacy unroll | legacy reference |
| `unroll_mr` | multi-res native-region unroll (coarse geom + native content) | working mid-level |
| `unroll_full` | seamless whole-scroll unroll, coarse global + native tiling | most advanced of THIS family |
| `progressive_unroll` | coarse→fine pyramid warm-start, wrap-count stability check | DIAGNOSTIC |
| `air_trace` | recto/verso air-boundary tracing → winding (4 PPMs) | DIAGNOSTIC |
*Note: these predate / parallel the `unroll_wind` + `tile_winding_mr` path, which is the one wired
into `unroll_pipeline.py`. `unroll_wind` is the current unroller; this family is earlier work.*

### Cluster 3 — Archive / format I/O utilities
| tool | role |
|---|---|
| `mca_slice` | extract one z-slice of `.mca` → TIFF (inspection) |
| `extract_cube` | crop a subcube of a NRRD (test-data utility) |
| `tifxyz` | info/render a VC3D `tifxyz` surface against an archive |

### Cluster 4 — Touching-sheets / supervoxel segmentation (research, `docs/touching-sheets-plan.md`)
| tool | role | status |
|---|---|---|
| `svaff_seg` (231 LOC) | SNIC supervoxels + signed affinity + Mutex-Watershed + **winding-gate** | reusable infra; NOT the wrap separator (MWS abandoned as vehicle) |
| `snic_view` | visualize 3D SNIC supervoxels | DIAGNOSTIC |
| `affinity_seg` | voxel-level signed-affinity + MWS render | DIAGNOSTIC |
| `sheetscale` | multiscale-σ argmax fused-wrap detector / touch fraction | CURRENT instrument (weak; thickness↔fusion confound) |
| `trace_mca` | advancing-front sheet tracer → `_lab.i32` + PPM | working (alt to ridge) |
| `trace_real` | ridge vs trace benchmark on real data (Betti b0/b1/b2) | benchmark |
Findings log (in the plan doc): winding-as-label works; **the wrap instance label is just
`floor(regularized winding)`**; MWS/supervoxel detour added fragmentation for no benefit; the
residual leaked-touch defect is only ~1% (`leak_metric.py`), so the heavy ordered-cut is low-yield.

### Cluster 5 — Kaggle Surface Detection (TIFF in, mask out; mostly experimental)
Current/used: `surface_pipeline` (full: CED→advancing-front trace→PH tunnel surgery→dust→score,
"everything wired in"), `surface_predict` (structure-tensor + ridge-NMS + postproc → pred TIFF),
`surf_pred` (smooth 0–1 probability field via CED, `.mca` in), `surface_trace` (tracer-only),
`surface_ced` (CED variant), `run_surface` (no-ML baseline), `surface_sweep` (param grid),
`sd_measure` (comprehensive metric harness + preprocessing test), `sheet_detect` (NRRD→sheetness+
normals utility).
Abandoned tuning/variant dead-ends: `surface_prep` (denoise/MUSICA), `surface_clean` (closing),
`surface_hess` (Hessian/Frangi), `surface_multi` (ST×Hessian fusion), `surface_opt`,
`surface_optimize`, `surface_repair` (PCA height-map), `surface_connect` (fragment merge),
`ridge_clean`, `ridge_connect`, `st_sweep` (ridge/structure-tensor sweeps).

### Cluster 6 — Evaluation / topology / tests
- Metric drivers: `eval_seg` (VOI/Rand/SurfaceDice on NRRD), `topo_native` (TopoScore),
  `nsd_native` (surface Dice), `pers_dump` (persistence diagram), `run_e1` (synthetic
  concentric-shell validation of signed partition), `show_inconsistencies` (winding holonomy at
  annotated points).
- Unit tests (exit 0 = pass): `test_topo`, `test_repair`, `test_features`, `test_persist0`.

### Cluster 7 — Legacy NRRD classical unwrap
`run_pipeline` (NRRD vol + umbilicus.txt → sheet detect → signed affinity → MWS → winding_field →
spiral_fit → deform; reports MVF, spiral rms, Jacobian fold fraction). The original architecture,
now superseded by the `.mca` winding pipeline; good as the documented reference of the intended
math (winding_field_solve, spiral_fit_from_field, deform_build, jacobian_fold_fraction).

---

## 3. Python orchestration & scoring scripts

| script | purpose | deps |
|---|---|---|
| `unroll_pipeline.py` | **canonical end-to-end driver** (tile → wind_tv → unroll) | stdlib, subprocess |
| `tile_winding_mr.py` | **working multi-res tiler** (coarse prior + GLAM, direct-avg merge) | numpy |
| `tile_winding.py` | negative diagnostic (independent stitching fails) | numpy |
| `official_score.py` | authoritative Kaggle scorer via `topometrics` (Betti-Matching-3D + Google surface_distance + cc3d) | numpy, PIL, topometrics |
| `robustness_bench.py` | multi-location `sheet_sep3d` robustness table (scale gate, edge, backsw, climb, z-coherence) | stdlib |
| `param_sweep.py` | one-param-at-a-time sensitivity sweep of `sheet_sep3d` | stdlib |
| `leak_metric.py` | quantify leaked/fused touches in a winding field (defect 2 = ~1% tail) | numpy, scipy |
| `overlap_consistency.py` | ground-truth-free crop-overlap winding agreement | numpy |
| `render_merged.py` | render merged.f32 (mid-z spiral + z-y reslice) | numpy, PIL |
| `render_compare.py` | 3-way center-slice comparison (raw/conservative/aggressive) PNGs | numpy, PIL |
| `crawl_opendata.py` | build S3 manifest of all scrolls/volumes | stdlib (urllib) |
| `tune_region.py` | download chunk-aligned 1024³ region as raw zarr | stdlib |
| `l5_occupancy.py` / `s3_occupancy.py` | scroll occupancy/size estimates from zarr | numpy / stdlib |

Build/orchestration shell: `build_all.sh` (full clang-23/lld + libc++ + llvm-libc overlay stack;
builds Qt 6.10.2 from source → minimal VTK → taberna C pipeline + GUI), `resume_taberna.sh`
(rebuild only taberna against pre-built Qt/VTK), `apply_patches.sh` (replay submodule patches),
`build_qt.sh`/`build_vtk.sh`/`resume_vtk_taberna.sh`.

---

## 4. The GUI (`gui/`, Qt6 + VTK 9, ~577 LOC, opt-in `-DTABERNA_GUI=ON`)
A 2×2 MDI viewer for `.mca` archives, modeled on VC3D's 4-pane layout:
- `main.cpp` / `main_window.{h,cpp}` — `QMdiArea` with 3 ortho slice panes (axial/coronal/sagittal)
  + 1 VTK volume-render pane; File ▸ Open .mca; auto-picks slice LOD (≤2048 max dim) and volume LOD
  (≤256³ for GPU upload).
- `volume_source.{h,cpp}` — **the ONLY matter-compressor coupling** (deliberately isolated so the
  mc rewrite touches one file). `mc_open` for header, `mc_archive_open_dims` +
  `mc_archive_read_region` for dense u8 regions/slices.
- `slice_view.{h,cpp}` — `QGraphicsView` CPU-blitting a slice QImage, wheel-scrolls slices.
- `volume_view.{h,cpp}` — `QVTKOpenGLNativeWidget` + `vtkSmartVolumeMapper` over `vtkImageData`;
  window/level, colormaps (gray/bone/hot/viridis), blend modes (composite/MIP/MinIP/avg/iso),
  gradient opacity, crop, GPU-capability report.
It is a **visualization/inspection tool**, decoupled from the C pipeline (the core pipeline has no
Qt/VTK dependency). It does NOT participate in producing the unroll; it only views `.mca`.

---

## 5. File formats flowing between stages
- **zarr (open-data S3)** — raw source CT, chunked uint8/uint16. Input to `fysics mca_export`.
- **`.mca` (matter-compressor archive v2)** — the working volume store: compressed LOD pyramid,
  mmappable, global `q`. `docs/mca-format-v2.md`. THE input to every winding/unroll/surface tool.
  Side files: `.mca.calib`, `.mca.progress`.
- **`_vol.f32` / `merged.f32` / `_tv.f32`** — winding field. int32 `{dz,dy,dx,z0,y0,x0}` header +
  `dz*dy*dx` float32 (NaN off-material). **The central inter-stage currency** of the winding
  pipeline (produced by sheet_sep3d/tile_winding_mr/wind_tv, consumed by unroll_wind/wind_tifxyz/
  wind_label and the leak/overlap/render scripts).
- **`_lab.i32`** — wrap-index instance-label volume (same header, int32) from `wind_label`/`trace_mca`.
- **`.pgm` / `.ppm`** — unrolled papyrus image (`unroll_wind`) and diagnostic renders.
- **`.tif` / TIFF stacks** — `mca_slice`/`unroll`/`unroll_*` outputs; Kaggle surface pipeline I/O
  (CT image + 0/1/2 label + predicted mask); `wind_diag` overlays.
- **`tifxyz` directory** — VC3D QuadSurface: `{x,y,z,mask}.tif` float32 + `meta.json` (produced by
  `wind_tifxyz`, read by `tifxyz`). The interop surface format with Vesuvius Challenge tooling.
- **NRRD** — legacy `run_pipeline`/`unroll`/`sheet_detect`/`extract_cube`/`eval_seg` volumes.
- **`umbilicus.txt`, points** — annotation inputs (`taberna_annotate`).

---

## 6. What is CURRENT vs ABANDONED (one-line verdict)
- **CURRENT product pipeline:** `fysics mca_export` → `sheet_sep3d` → `tile_winding_mr.py` →
  `wind_tv` → `unroll_wind` (driver: `unroll_pipeline.py`); optional `wind_tifxyz` export;
  `wind_label`/`wind_couple` support; GUI for inspection.
- **Diagnostics kept (not product):** `wind_poisson`, `wind_diag`, `polar_sheet`, `snic_view`,
  `affinity_seg`, `progressive_unroll`, `air_trace`, `robustness_bench.py`, `param_sweep.py`,
  `leak_metric.py`, `overlap_consistency.py`, `render_*.py`, all `test_*`.
- **Banked / experimental:** `wind_cut` (verified maxflow, awaiting a high-fusion region),
  `svaff_seg` (reusable supervoxel infra; MWS rejected as separator).
- **Abandoned negatives:** `tile_winding.py` (independent stitching), and the Kaggle surface
  tuning swarm (`surface_prep/clean/hess/multi/opt/optimize/repair/connect`, `ridge_clean/connect`,
  `st_sweep`).
- **Legacy reference:** `run_pipeline`, `unroll` (NRRD-era classical unwrap), `sheet_sep`.
