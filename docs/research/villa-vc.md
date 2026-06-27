# Volume Cartographer (VC3D) — Architecture Report

Target: `/home/forrest/villa/volume-cartographer/` ("VC3D" fork maintained by the
Vesuvius Challenge team; descended from EduceLab VC -> Schilliger -> Allgaier ->
Schilling/Johnson). ~325 .cpp + ~215 .hpp. Purpose: virtually unwrap papyrus
sheets out of scroll CT volumes — segment a sheet, flatten it to 2D, render a
texture image from the volume, which then feeds ink detection.

This fork is **specialized for Herculaneum** and has already shed much of the
original VC's generality (the original was a general toolkit built on ITK/VTK;
this fork has **no ITK and no VTK at all** — see Dependencies).

---

## 1. Module / library structure & CMake target graph

Top-level dirs: `core/` (the library), `apps/` (≈45 CLI tools + the `VC3D` Qt GUI),
`libs/` (vendored third-party), `utils/` (low-level I/O helpers), `docs/`,
`python/`, `scripts/`.

CMake (top `CMakeLists.txt`, C++ standard set high; `BUILD_SHARED_LIBS ON`):

**Vendored libs (`libs/`, mostly header-only or small):**
- `OpenABF/` — header-only ABF++ / LSCM angle-based flattening.
- `flatboi/` — standalone SLIM flattening tool (libigl-based), optional (`VC_BUILD_FLATBOI`).
- `libigl_changes/` — patched libigl (SLIM solver, optional PaStiX sparse backend).
- `c3d/` — custom lossy chunk codec ("C3D", ~50x compression of CT chunks).
- `cc3d/`, `edt/`, `djikstra3d/` — connected components, Euclidean distance transform, 3D Dijkstra (small algorithm headers).
- `ECL-MaxFlow/` — optional CUDA max-flow (for Lasagna), off by default.

**Library targets (all in `core/`):**
- `vc_core` — the big monolith. ~40 source files: data types (`Volume`,
  `VolumePkg`, `Segmentation`, `QuadSurface`, `PlaneSurface`, `Surface`),
  zarr/tiff I/O (`Zarr`, `Tiff`, `Slicing`, `render/*`), sampling/compositing,
  point collections, geometry, surface metrics, thinning, umbilicus. Links:
  OpenCV (core/imgproc/imgcodecs/calib3d/video), nlohmann_json, OpenMP, TIFF,
  OpenABF, CURL, vendored `utils` + `utils_c3d_codec`, Blosc (private).
- `vc_flattening` — `ABFFlattening.cpp` only; links `vc_core` + OpenABF.
- `vc_tracer` — segmentation/growth engine (`GrowPatch`, `GrowSurface`,
  `growth_strategies/*`, `Tracer`); links `vc_core` + `vc_flattening` + Ceres,
  uses vendored `edt`.
- `vc_inpaint` — `InpaintSurface.cpp`; links Ceres.
- `vc_lasagna` — fiber/winding "lasagna" line-graph + max-flow cut solver
  (`lasagna/*`); links Ceres, optional CUDA/AMGX.
- `vc_atlas` — fiber-intersection "atlas" winding model (`Atlas.cpp` 183KB,
  `AtlasConstraints`, `FiberIntersections`); links `vc_core`, `vc_lasagna`, Ceres.

**App targets (`apps/`):** ~45 `vc_*` CLI tools, each a thin `main()` over the
libs, plus `VC3D` (the Qt6 GUI, ~80 files under `apps/VC3D/`). Notable CLIs:
`vc_grow_seg_from_seed`, `vc_grow_seg_from_segments` (segmentation),
`vc_flatten`, `vc_render_tifxyz`, `vc_render_video`, `vc_obj2tifxyz`/`tifxyz2obj`,
`vc_tifxyz_winding`, `vc_calc_surface_metrics`, `vc_gen_normalgrids`/`vc_ngrids`
(normal-field generation), `vc_merge_tifxyz`/`merge_patch` (surface merging),
`vc_volpkg_convert`, `vc_zarr_to_tiff`/`zarr_recompress`. The `apps/diffusion/`
tree (`vc_diffuse_winding`) does winding-number diffusion.

Target graph (libs): `utils` -> `vc_core` -> {`vc_flattening`, `vc_inpaint`,
`vc_lasagna`} -> {`vc_tracer`, `vc_atlas`} -> apps. Ceres is required by
everything segmentation/atlas/lasagna-related but **not** by `vc_core` itself.

---

## 2. Central data structures & on-disk formats

### `.volpkg` volume package
A directory ending in `.volpkg` with a top `config.json` (project: name,
version, lists of volume / segment / normalgrid *entries*, each an
`{location, tags[]}` where location may be a relative path or a remote URL).
Subdirs: `volumes/`, `paths/` (segmentations — the default "segments"
directory; configurable), `atlases/`, `backups/`. `VolumePkg` (vc_core) is the
in-memory handle: lazy maps of loaded `Volume`/`Segmentation`/`QuadSurface`,
autosave support, remote-cache root. **Note:** old VC used a different `.volpkg`
layout (`config.json` + `volumes/` + per-segment `pointset.vcps`); this fork
replaced point-set segmentations with tifxyz surfaces.

### Volume (zarr)
A volume is an **OME-Zarr-like pyramid** directory + `meta.json`. Storage order
**ZYX**; dtype `uint8` or `uint16` (papyrus volumes are converted to uint8).
`meta.json`: `{type:"vol", uuid, name, width(X), height(Y), slices(Z),
voxelsize, min, max, format:"zarr"}`. On disk: `.zgroup`, `.zattrs`, and scale
arrays named `0`,`1`,`2`,... each a zarr array (`.zarray` + chunk files, chunk
default 64³, compressor blosc). Also supports remote HTTP/`s3://` stores
(resolved to HTTPS, AWS SigV4 auth) and a custom `.c3d` lossy chunk codec.
`Volume` (vc_core) gives typed region read/write (`readZYX`/`readXYZ`), raw
chunk I/O, `sample()` with Nearest/Trilinear/Tricubic/Lanczos, pyramid
reduction policies (Mean/Max/BinaryOr), virtual-downsample fallback for missing
levels. **Surface-prediction volumes** are stored as ordinary zarr volumes too
(uint8, thresholded distance to predicted sheet) and listed as extra volumes.

### Surface representation — `QuadSurface` / **tifxyz** (THE central format)
The scroll-sheet segmentation. In memory: a dense `cv::Mat_<cv::Vec3f> points`
— a **regular 2D grid where each cell stores a 3D world (voxel) coordinate**;
invalid cells are `{-1,-1,-1}`. Plus `cv::Vec2f scale` mapping grid indices to
nominal surface coordinates (e.g. scale `0.0078125` = 1/128, i.e. the grid is
128× denser than nominal). Three coordinate systems: nominal/world (3D voxels),
internal-relative ("pointer", center = origin), internal-absolute (grid index,
upper-left = origin).

On disk (a "**tifxyz**" directory): `x.tif`, `y.tif`, `z.tif` — three
single-channel float32 tiled TIFFs holding the X/Y/Z coordinate bands of the
grid — plus `meta.json` `{type:"seg", format:"tifxyz", uuid, scale[2],
bbox[[x,y,z],[x,y,z]]}`. Optional extra **channels** are extra TIFFs in the same
dir: `mask.tif` (uint8 validity, 255=valid; can be multi-layer), plus channels
like `winding.tif`, normal-fit quality, generation info, UV. `QuadSurface`
supports load (`load_quad_from_tifxyz`), save with atomic rename, snapshots into
`backups/`, range-based iteration (`validPoints()`, `validQuads()` with
structured bindings), set ops (`surface_diff`/`union`/`intersection`),
`SurfacePatchIndex` spatial index. `PlaneSurface` is the other `Surface`
subclass (infinite cutting plane for slice viewers).

`Segmentation` (vc_core) is a thin wrapper: a path + `meta.json` + lazily-loaded
`QuadSurface`. So a "segmentation" on disk *is* a tifxyz directory under
`paths/`.

### Other formats
- **OBJ meshes** — interchange with external tools; `vc_obj2tifxyz` rasterizes a
  UV-bearing OBJ into a tifxyz grid, `vc_tifxyz2obj` does the reverse.
- **Normal grids** — zarr volumes storing per-voxel 3D unit vectors as uint8
  triplets (x/y/z arrays, 128 = neutral/none), produced by `vc_gen_normalgrids`
  / `vc_ngrids`; consumed as a data term in tracing.
- **Render output** — multi-page / per-Z-layer tiled TIFFs (e.g. `000.tif`,
  `001.tif`, ... one per normal-offset layer), 8/16-bit, DEFLATE/LZW, DPI from
  voxel size. This layered texture stack is the ink-detection input.
- **Point collections** (`VCCollection`, `PointCollections`) — JSON sets of
  annotated 3D points with winding labels, used for correction points and
  surface-metric ground truth.
- **Atlas** — under `atlases/<name>/`: a copied base `shell_*.tifxyz`, `links.json`
  (typed winding links between fibers), V4 metadata, V3 fiber mappings. Encodes
  inter-winding structure (`winding = floor((atlasU - zero_winding_column)/period)`).

---

## 3. Segmentation algorithms (the "optical flow"/tracing/growth pipeline)

Lives in `vc_tracer` (`GrowPatch.cpp` 215KB, `GrowSurface.cpp` 166KB,
`growth_strategies/`, `Tracer.hpp`, `CostFunctions.hpp`, `SurfTrackerData`).

Despite the historical name "optical flow segmentation," the actual method is
**nonlinear-least-squares surface tracing/growing solved with Ceres**:

- **Pipeline:** seed (`vc_grow_seg_from_seed`, seeding logic places points along
  thresholded high-confidence regions) -> `tracer()` bootstraps a small
  `QuadSurface` grid -> iterative **generational growth**: each generation adds
  candidate grid corners in allowed directions, initializes them by **affine
  extrapolation** from existing neighbors, attaches loss terms, and runs a global
  or windowed Ceres solve; greedy corner-adding alternates with optimization, with
  heuristics to accept/skip/roll-back. `vc_grow_seg_from_segments` extends/merges
  existing surfaces.
- **Variables:** 3D positions of grid corners (`cv::Mat_<cv::Vec3d> dpoints`),
  with a parallel `state` mask (LOC_VALID / COORD_VALID / PROCESSING / FAIL).
  `SurfTrackerData` maps grid locations to parametric coords and tracks Ceres
  residual-block IDs for incremental re-solves.
- **Loss terms (Ceres auto-diff,** `SPARSE_NORMAL_CHOLESKY`, CPU, OpenMP**):**
  - *StraightLoss* — 3-point curvature/smoothness (collinearity).
  - *DistLoss* — preserve metric grid spacing (edge length ≈ `unit`).
  - *SignedDistanceToSurface / SpaceLineLoss(Acc)* — **data term**: pulls corners
    toward the thresholded **surface-prediction zarr** via a cached trilinear 3D
    interpolator (`CachedChunked3dInterpolator<uint8_t>`); samples along edges,
    not just endpoints, to escape local minima.
  - *NormalDirection / Normal3DLineLoss* — align surface normals with the
    **normal-grid** fields (fiber direction priors).
  - *ParamMetricLoss2D, DistLoss2D, StraightLoss2D* — keep the 2D
    parameterization non-degenerate.
  - *CorrectionLoss* — soft pull through user **correction points**.
  Weights are JSON-configurable (defaults e.g. NORMAL 10, STRAIGHT 0.2, DIST 1).
  Losses are **region-aware/conditional**: generated only where points are valid,
  marked via a `loss_status` bitmask to avoid duplicates
  (`emptytrace_create_missing_centered_losses`, `add_missing_losses`).
- **Growth control:** `GrowthConfig` (which neighbors, required neighbor count),
  `CandidateOrdering` (rank candidates by valid-neighbor count + support depth),
  `ComponentPruning`. Directions are relative to the flattened 2D grid (up/down/
  left/right/all). Multi-scale coarse-to-fine via `growth_scale_level`.
- **CPU vs CUDA:** CPU is the production path; CUDA-sparse Ceres paths exist but
  are disabled ("CPU always faster"). Optional `NeuralTracerConnection` streams
  predictions from an external model over a socket every N generations.

---

## 4. Flattening / UV / unwrapping

Two **alternative** parameterizations:
- **ABF++ then LSCM** (in-core, primary): `core/src/ABFFlattening.cpp` (96KB) via
  header-only `OpenABF`. `abfFlattenInternal` builds a triangle mesh from valid
  quads, cleans it (degenerate-triangle filtering by edge-length percentiles,
  boundary "pinch" splitting, optional downsample 1/2/4/8×), runs
  `OpenABF::ABFPlusPlus<double>::Compute()` (minimize angular distortion), then
  `AngleBasedLSCM<double>::Compute()` (Eigen sparse solve for conformal UVs);
  falls back to LSCM-only if ABF fails. Post: area-preserving scaling
  (√(area3D/area2D)), grid-axis alignment, UV upsampling. Output either fills the
  `uv` channel or rasterizes a new distortion-corrected tifxyz grid. Invoked by
  `vc_flatten` and inside `vc_render_tifxyz`.
- **SLIM** (`libs/flatboi`, optional): libigl `igl::slim_precompute`/`slim_solve`
  with Symmetric-Dirichlet (or Conformal/ARAP) energy, iterative, NaN-guarded,
  optional PaStiX sparse backend. Decoupled standalone tool — higher quality,
  slower; not wired into the core API.

Pipeline: tifxyz `QuadSurface` (or UV-bearing OBJ) -> triangle mesh -> 2D UV ->
flat grid -> tifxyz/OBJ out. `vc_obj2tifxyz`/`vc_tifxyz2obj`/`vc_obj_uv_lift`/
`vc_straighten` are the conversion/refinement tools. Solvers: Eigen (LSCM), libigl
+ optional PaStiX (SLIM). No Ceres in flattening.

---

## 5. Rendering (surface -> flattened texture)

`Render.cpp` + `render/ChunkedPlaneSampler` + `Slicing.cpp` + `Compositing.cpp`,
driven by `vc_render_tifxyz` (and `vc_render_video`, `vc_project_tifxyz`):

1. From a (flattened) `QuadSurface`, `surf->gen()` builds a
   `cv::Mat_<cv::Vec3f> coords` (a 3D world coord per output pixel) plus
   per-pixel normals and a validity mask.
2. **Layering:** for each layer offset along the surface normal
   (`base + normal * zStep*i`, range `zStart..zEnd`), `readCompositeFastImpl`
   trilinearly samples the volume into a `LayerStack`.
3. **Compositing** (`Compositing.cpp`): per-pixel reduce across layers — mean,
   max, min, alpha, beer_lambert, dvr, first_hit_iso, gradient_mag,
   gamma_weighted, etc., with pre/post transfer-function LUTs, optional
   normal/gradient lighting, CLAHE, raking-light shading.
4. **Output:** one tiled TIFF per layer (8/16-bit), via `TiffWriter` (incremental
   tiled writes, parallel-friendly), DPI from voxel size; optional colormaps.

**Chunked zarr read + cache** (the performance core): `ZarrChunkFetcher` reads
chunks from disk/HTTP (Range requests) decoding blosc/zstd/lz4/`.c3d`; a
persistent LRU `ChunkCache` (decoded-byte budget, priority threadpool, distance-
to-viewport prefetch) sits over per-thread direct-mapped `LocalChunkCache`/
`ChunkSampler` (16384 slots, last-chunk fast path, `std::fma` trilinear,
Catmull-Rom tricubic). `ChunkedTensor`/`Array3D` are the in-memory chunked array
types.

Data flow overall: **volume (zarr) -> [surface-prediction zarr] -> segmentation
(tracer -> tifxyz QuadSurface) -> flatten (ABF/SLIM -> flat tifxyz) -> render
(layered TIFF texture) -> ink detection**. Side branches: normal-grid generation
feeds the tracer; winding/atlas/lasagna tools resolve inter-sheet winding order;
surface metrics evaluate against point-collection ground truth.

---

## 6. Dependency footprint (critical for fenix's minimal-deps goal)

`find_package` in root CMake: **Qt6** (GUI only), **Ceres** (segmentation/atlas/
lasagna/inpaint optimization), **Eigen3**, **OpenCV** (core/imgproc/imgcodecs/
calib3d/video/ximgproc/flann/videoio — used *pervasively* as the array/image
type `cv::Mat`/`cv::Vec3f`, not just for vision), **nlohmann_json**, **CURL**
(remote zarr + S3), **TIFF** (libtiff, all raster I/O), **ZLIB**, **Boost**
(program_options only), **Blosc** + **zstd** + **lz4** (chunk compression),
**CGAL** (only `apps/src/alpha_wrap.cpp` — alpha-wrapping for ignore-label masks),
**OpenMP**. Optional: CUDA + AMGX (lasagna max-flow), PaStiX (SLIM), libbacktrace.

Vendored (not system deps): OpenABF, libigl (patched), c3d codec, cc3d, edt,
djikstra3d, ECL-MaxFlow.

**Key findings for a minimal rewrite:**
- **No ITK, no VTK** — already dropped vs original VC. Don't reintroduce.
- **No z5 / no xtensor / no spdlog** — this fork wrote its **own zarr reader**
  (`utils/src/zarr.cpp`, supports zarr v2/v3, sharding, blosc/zstd/lz4) and its
  own logging. Only stray comment references to z5 remain. Custom `Array3D`/
  `ChunkedTensor` replace xtensor.
- **OpenCV is the deepest tendril** — `cv::Mat_`/`cv::Vec` are the universal data
  types across surfaces, sampling, rendering. Removing OpenCV is the single
  biggest rewrite decision (needs a replacement small-vector + 2D image type).
- **Ceres + Eigen** are load-bearing for segmentation and flattening — hard to
  drop unless reimplementing the NLLS solver. Boost is trivially removable
  (program_options only). CGAL is one tool. CURL/TIFF/blosc/zstd/lz4 are the
  irreducible I/O set for zarr + TIFF.
- The `docs/api/*.md` are **partly stale** (reference `z5::Dataset`, a
  `util/ChunkCache.hpp` with `xt::xarray` that no longer exists — the live one is
  `render/ChunkCache.hpp`). Trust code over docs.

---

## 7. Major data flow (summary)

```
CT scan -> uint8 OME-zarr Volume  ──(neural net, external)──> surface-prediction zarr
                                                  │
                          normal-grid zarr <── vc_gen_normalgrids
                                                  │
seed points ─> vc_grow_seg_from_seed/segments (Ceres tracer) ─> tifxyz QuadSurface (paths/)
                                                  │
                       vc_flatten (ABF++/LSCM)  or  flatboi (SLIM) ─> flat tifxyz
                                                  │
                       vc_render_tifxyz (layered sampling + compositing) ─> multi-layer TIFF
                                                  │
                                          ink-detection model
   side: vc_tifxyz_winding / atlas / lasagna / diffuse_winding  -> inter-sheet winding order
         vc_calc_surface_metrics  -> QA against point-collection GT
```

---

## 8. Pain points / architectural debt to fix in a C++26 rewrite

1. **God-files.** `GrowPatch.cpp` (215KB), `GrowSurface.cpp` (166KB), `Atlas.cpp`
   (183KB), `SurfacePatchIndex.cpp` (128KB), `ABFFlattening.cpp` (96KB),
   `QuadSurface.cpp` (94KB), `Volume.cpp` (75KB). Single-TU monoliths mixing
   data, solver setup, heuristics, and I/O. Split into small composable units.
2. **OpenCV as the core type system.** `cv::Mat_<cv::Vec3f>` everywhere couples
   geometry to a heavy vision lib. A clean `Grid2D<Vec3f>` / `Image<T>` plus a
   small linalg type would remove the biggest dependency and clarify ownership
   (`rawPointsPtr()` returns raw owning pointers; `QuadSurface(cv::Mat*)` "takes
   ownership" — manual lifetime management ripe for RAII).
3. **Invalid-sentinel `{-1,-1,-1}`** instead of an explicit validity mask/optional
   leaks through all loops; an explicit bitset/`std::mdspan`-backed validity layer
   would be safer.
4. **Heuristic-laden tracer.** Accept/skip/rollback thresholds, perturbations,
   experimental flags (rollout, cell_reopt), multiple overlapping loss types — the
   logic is empirical and entangled with Ceres problem construction. A clean
   separation of (a) surface model, (b) cost terms, (c) growth policy, (d) solver
   driver would help; consider a typed loss registry.
5. **Two disconnected flatteners** (in-core ABF vs external SLIM) with no common
   interface; unify behind one parameterization API.
6. **Stale/contradictory docs and dead references** (z5, xtensor, two ChunkCaches)
   — generated confusion. Single source of truth.
7. **Format sprawl & versioning.** tifxyz (3 TIFFs + channels), zarr meta variants
   (`meta.json` vs `metadata.json` vs synthesized), atlas V1→V4 with hard
   rejection of old versions, OBJ interchange, point-collection JSON. fenix is
   designing NEW incompatible formats — opportunity to define ONE coherent,
   versioned, self-describing surface container (coords + validity + named
   channels + scale + bbox + provenance) and ONE volume descriptor, rather than
   directory-of-TIFFs + sidecar JSON.
8. **Manual chunk-cache layering** (3 tiers, string-keyed composite dispatch,
   blocking HTTP with no timeout, hardcoded trilinear in multi-slice reads
   ignoring requested interpolation). A single typed chunk cache with async I/O
   and pluggable interpolation/compositing (enum-dispatched) would be cleaner and
   faster.
9. **Pervasive raw `new`/owning pointers, `abort()` for unimplemented paths,
   thread-local escape hatches** (`skipShapeCheck`) — replace with RAII, smart
   pointers/`std::expected`, and explicit error types in C++26.
10. **Coordinate-order foot-guns** — ZYX storage vs XYZ UI vs grid indices;
    encode the order in the type system (strong typedefs) rather than convention +
    comments.
