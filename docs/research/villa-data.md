# Villa Monorepo — Data Access, Formats & Conventions (research for fenix C++26 rewrite)

Scope: read-only research of `/home/forrest/villa/`. Focus: scroll data access model, on-disk/cloud
formats, the `vesuvius-c` and `vesuvius` (Python) APIs, the unrolling-pipeline data formats, foundation
preprocessing, and monorepo conventions worth adopting.

---

## 1. Scroll data access model & canonical formats

### 1.1 The big picture
A *scroll* is a single CT-scanned Herculaneum papyrus. The canonical primary data is a 3D volume of
voxel intensities. Scroll 1 at full res is **14376 × 7888 × 8096 voxels** (Z, Y, X), dtype **uint8**
(the `vesuvius-c` header documents this explicitly). Data is huge (terabytes), so nobody downloads it
whole — both libraries stream sub-volumes/chunks on demand from a public HTTP server and cache locally.

Index order convention everywhere: **Z, Y, X** (Z = slice number, increasing Z = next `.tif` slice;
Y = down within a slice; X = right within a slice). fenix should adopt the same ordering globally.

### 1.2 On-disk / cloud format: OME-Zarr v2 multiresolution pyramids
- Primary storage is **Zarr v2** arrays, packaged as **OME-Zarr** (multiscale) groups.
- A volume is a zarr *group* containing resolution levels named `0`, `1`, `2`, … where `0` is full
  resolution and each subsequent level is downsampled (typically 2×). Level `0` is what `vesuvius-c`
  points at directly (`.../54keV_7.91um_Scroll1A.zarr/0/`).
- Each level is a zarr *array* with a `.zarray` (array metadata) and, at the group root, `.zattrs`
  (OME multiscales metadata: axes, datasets, coordinateTransformations/scales).
- `.zarray` fields actually parsed (see C `zarr_metadata`): `shape[3]`, `chunks[3]`, `compressor`
  (`{id, cname, clevel, shuffle, blocksize}` — Blosc), `dtype` (e.g. `|u1` = uint8, `|u2` = uint16),
  `fill_value`, `order` (`C`/`F`), `zarr_format` (2), `dimension_separator` (`.` or `/`).
- **Chunk/shard layout**: each chunk is a separate file. Path is `<array>/<z>/<y>/<x>` when
  `dimension_separator == '/'`, else `<array>/<z>.<y>.<x>`. Chunks are Blosc-compressed blocks of the
  chunk-shaped sub-array. **Missing chunk file = all-`fill_value` (usually all-zero)**; zarr omits
  empty chunks, so a 404/missing file is *not* an error — readers must treat it as zeros. Typical
  chunk size is 128³ or similar cube.

### 1.3 S3 / HTTP bucket structure
- Public data server: **`https://dl.ash2txt.org`** (also mirrored on S3; Python supports `s3://`
  with `anon=True` unsigned access and auto-detects AWS EC2 to prefer `local`/S3).
- Canonical path scheme (from `scrolls.yaml`):
  `full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/`
  - `PHercParis4.volpkg` — the **.volpkg** (volume package), the EduceLab/VC container for a scroll
    and its derived data (volumes, segmentations, metadata).
  - `volumes_zarr_standardized/` (or `volumes_zarr/`) — the zarr volumes within the volpkg.
  - Filename encodes acquisition params: `<energy>keV_<resolution>um_Scroll<N><variant>.zarr`
    (e.g. `54keV_7.91um_Scroll1A`). Energy in keV, resolution in µm/voxel.
- Segments live under e.g. `other/dev/scrolls/1/segments/54keV_7.91um/<timestamp>.zarr/` with a
  sibling `<timestamp>_inklabels.png`.
- Scroll/volume identity is a 4-tuple in practice: **(scroll_id, energy, resolution[, variant])**;
  segments add a **segment_id = a timestamp** (e.g. `20230827161847`).

### 1.4 scrolls.yaml registry (the config that maps IDs → URLs)
Nested map: `scroll_id → energy → resolution → { volume: <url>, segments: { seg_id: <url> } }`.
Scrolls 1–5 registered with canonical energy/resolution (mostly 54keV/7.91µm; Scroll3 53keV;
Scroll5 53keV). The Python `Volume` resolves canonical energy/resolution when not given.

---

## 2. vesuvius-c — single-header C library

Single file `vesuvius-c.h` (~5100 lines). Pattern: define `VESUVIUS_IMPL` in exactly one `.c` then
`#include`. Header-only, STB-style. C library; CMake auto-discovers deps.

### 2.1 Dependencies
- **libcurl** — fetching chunks over HTTP (single + multi/async handles).
- **c-blosc2** — decompress/compress zarr Blosc chunks.
- **json-c** — parse `.zarray` / metadata JSON.
- **ffmpeg** (optional) — chunk → video.
- system math lib. Targets Ubuntu + macOS (incl. Apple Silicon homebrew paths).

### 2.2 Core types
- `chunk { int dims[3]; float data[]; }` — 3D sub-volume, **float32** in memory (always converted
  from uint8 on read), 16-byte aligned, flexible array member, Z/Y/X row-major.
- `slice { int dims[2]; float data[]; }` — 2D.
- `volume { char cache_dir[1024]; char url[1024]; zarr_metadata metadata; }` — wraps one zarr array;
  holds the parsed `.zarray` and where to fetch/cache.
- `zarr_metadata`, `zarr_compressor_settings` — as in §1.2.
- `mesh { f32* vertices; s32* indices; f32* normals; u8* colors; s32 vertex_count, index_count; }` —
  triangle-only, 3-component normals, RGB u8 colors.
- `nrrd`, `ppm`, `TiffImage`/`DirectoryInfo`, `rgb`, `histogram`, plus async state structs
  (`MultiDownloadState`, `ChunkLoadState`).

### 2.3 API surface (all `vs_` prefix; `vs__` = private static)
- **Volume / zarr access**: `vs_vol_new(cache_dir, url)` (downloads/parses `.zarray`),
  `vs_vol_get_chunk(vol, vol_start[3], chunk_dims[3])` (the workhorse), async
  `vs_vol_get_chunk_start/poll`, `vs_vol_free`. Low-level zarr:
  `vs_zarr_parse_metadata/parse_zarray`, `vs_zarr_read_chunk`, `vs_zarr_fetch_block`,
  `vs_zarr_decompress_chunk`, `vs_zarr_write_chunk`, `vs_zarr_compress_chunk`.
- **curl**: `vs_download`, `vs_download_start/poll` (async multi).
- **chunk/slice ops**: new/free/get/set, `vs_chunk_graft` (copy sub-region), `vs_slice_extract`,
  pooling (`vs_maxpool/avgpool/sumpool`), morphology (`vs_erode/dilate`), `vs_threshold`, `vs_mask`,
  `vs_normalize_chunk`, `vs_transpose`, `vs_histogram_equalize`, `vs_unsharp_mask_3d`, `vs__convolve3d`,
  connected components (`vs_connected_components_3d`, `vs_flood_fill`, `vs_remove_small_components`,
  `vs_count_labels`), `vs_chunk_min/max`.
- **mesh / geometry**: `vs_march_cubes` (isosurface), `vs_mesh_new/free/translate/scale/get_bounds`,
  `vs_chamfer_distance`, `vs_colorize`.
- **I/O formats**: OBJ (`vs_read_obj/vs_write_obj`), PLY (`vs_ply_read/write`), NRRD (`vs_nrrd_read`),
  TIFF (full reader/writer + tag printing, `vs_tiff_to_chunk/slice`), PPM, BMP (`vs_bmp_write`),
  VCPS (`vs_vcps_read/write` — VC point-set), video (`vs_chunks_to_video`, `vs_write_ppm_frame`).
- **histogram/stats**: `vs_histogram_new`, `vs_slice/chunk_histogram`, `vs_calculate_histogram_stats`.

### 2.4 How reads work & caching
`vs_vol_get_chunk` requires `vol_start` and `chunk_dims` to be **multiples of the zarr chunk size**
(it asserts alignment; arbitrary offsets are a TODO). It computes the covered chunk grid range, and
for each chunk: builds the cache path (`cache_dir/z/y/x`); if it exists on disk → read+decompress;
else download from `url/z/y/x`, write to cache, decompress. Missing downloads are skipped (treated as
zeros). Decompressed uint8 is widened to float32 and `vs_chunk_graft`'d into the output chunk at the
right offset. **Caching = mirror the zarr directory tree on local disk**; persists between runs to
avoid re-downloads. uint16 (`|u2`) is parsed but decompression is currently asserted-out
(uint8-only in practice).

### 2.5 Conventions in the header (worth copying)
Documented at top: ownership rules (a `_new` that takes a pointer to fill a struct field *takes
ownership* and frees it in `_free`; char* path/url args are NOT subsumed); Z/Y/X index order; return
codes (`0` = success for non-pointer-returning fns, `NULL` = failure for pointer-returning);
caller frees returned pointers, using the type's `_free` when one exists. Fixed-width typedefs
`u8..s64, f32, f64`. Clear "Public APIs" vs "Private APIs" banner sections.

---

## 3. vesuvius — Python library

`pip install vesuvius`; `vesuvius.accept_terms --yes` gates first use. Python ≥3.10,<3.14. Split deps:
a **minimal core** (`volume-only`: numpy, pillow, pynrrd, requests, pyyaml, fsspec, numcodecs,
**zarr 2.x**, s3fs, cachetools) and a heavy **`models`** extra (torch, lightning, monai, timm,
nnunetv2, numba, batchgenerators). Note: pinned to **zarr v2** (numcodecs <0.16). Read-only library;
does not modify remote data.

### 3.1 The `Volume` abstraction (`data/volume.py`, ~1260 lines)
`Volume(type, scroll_id, energy, resolution, segment_id, format='zarr', normalization_scheme,
return_as_type, return_as_tensor, domain, path, download_only, anon, ...)`.
- **Three construction modes**: (a) direct `type='zarr', path=...` (local/http/s3); (b)
  `type='scroll'|'segment'` resolved through `scrolls.yaml` by (scroll, energy, resolution[, seg]);
  (c) shorthand strings (`"Scroll1"`, a bare timestamp → segment, an int → scroll).
- Opens via `open_zarr` (fsspec-backed; s3 with `anon`). Handles zarr **Array** or **Group** (OME
  multiscales) — for a group it exposes resolution levels; level `0` = full res.
- `load_ome_metadata()` reads `.zattrs` (tries store attrs, then `/`, `/0`, then explicit `.zattrs`
  files). `get_url_from_yaml()` does the registry lookup. `grab_canonical_energy/resolution`.
- **`__getitem__`** is NumPy-style `(z, y, x)` with an optional 4th index = **resolution level**
  (subvolume_idx), default 0. Slices supported → returns a sub-volume ndarray (or torch tensor).
- **Normalization schemes** applied on read: `none`, `instance_zscore`, `global_zscore`
  (needs mean/std), `instance_minmax`, `percentile_minmax` (p1/p99), `ct` (needs intensity_props
  mean/std/percentile_00_5/percentile_99_5). `return_as_type` casts dtype; `return_as_tensor` → torch.
- **Segments**: `download_inklabel()` fetches `<segid>_inklabels.png` sibling → `self.inklabel`
  (grayscale ndarray, Y×X). `download_only` mode fetches just the ink label.

### 3.2 `Cube` abstraction
`Cube(scroll_id, energy, resolution, z, y, x, ...)` — accesses **volumetric instance-segmentation
annotated cubes** (the `Cube` notebook): a small labeled sub-volume + its label volume, fetched by
its (z,y,x) origin via yaml lookup, with optional caching/normalization. Used for 3D instance
segmentation training data.

### 3.3 Chunk indexing / occupancy (`zarr_chunk_index.py`, `chunks_filter.py`)
Builds an **occupancy bitmap** of which zarr chunks physically exist (lists S3/local chunk keys,
caches a serialized bitmap sidecar keyed by a `.zarray` signature). Used to skip empty regions and to
generate valid patch positions: `generate_exact_chunk_positions`,
`generate_bounded_sliding_window_positions`, `filter_positions_by_chunk_overlap`,
`compute_touched_chunks`, `compute_patch_non_empty_mask`. This is the data-loader's way of only
sampling non-empty volume regions — directly relevant to efficient training/inference tiling.

### 3.4 tifxyz — surface (segment) representation (`tifxyz/`, has API.md)
A **tifxyz surface** = a directory representing a flattened papyrus sheet as a 2D parametric grid of
3D coordinates:
```
segment_name/
  x.tif, y.tif, z.tif   # per-grid-cell volume coordinates, float32 (the (u,v)->(x,y,z) map)
  meta.json             # scale, bbox, uuid
  mask.tif              # optional validity mask
  *_label.{tif,png,jpg} # optional grayscale labels (e.g. ink)
```
`Tifxyz` class + `read_tifxyz/write_tifxyz/list_tifxyz/load_folder`. Two **resolution modes**:
`"stored"` (direct array access, fast) and `"full"` (Catmull-Rom interpolated upsampling to full
volume resolution for sub-pixel sampling). Properties: `valid_quad_mask`, `valid_quad_indices`,
`valid_vertex_mask`, `quad_area`, `quad_centers`, `bbox`, `uuid`. Auto-discovers sibling labels;
`load_label(index|filename|suffix)`. `hierarchical_tiling.py` builds coarse→fine tile hierarchies
over a segment for processing. This is the central "surface" format an unrolling pipeline produces and
that rendering consumes.

### 3.5 Utilities & entrypoints (from README)
Console scripts: `vesuvius.predict` (nnUNet-v2 inference on zarr, distributable via
`--num_parts/--part_id`), `vesuvius.blend_logits` (Gaussian blending), `vesuvius.finalize_outputs`
(softmax/argmax → uint8), `vesuvius.compute_st` (structure tensors + eigen-decomposition on big
zarr — fiber orientation), `vesuvius.voxelize_obj` (.obj mesh → voxel `.tif` stack / 3D labels),
`vesuvius.render_obj`/`mesh_to_surface` (render a mesh's surface-volume layers), `vesuvius.flatten_obj`
(`slim_uv` UV flattening), `vesuvius.refine_labels` (Frangi-based fiber/surface label refinement),
`vesuvius.train`, `vesuvius.napari_trainer` (interactive), `vesuvius.proofreader`.
Module layout under `src/vesuvius/`: `data/` (Volume, Cube, VCDataset, chunk indexing), `tifxyz/`,
`image_proc/` (mesh, intensity, geometry, blur, features), `models/` (nnUNet-derived training/
inference, augmentation, datasets, configuration), `structure_tensor/`, `rendering/`,
`neural_tracing/`, `napari_trainer/`, `utils/`, `scripts/`, `install/configs/scrolls.yaml`.

---

## 4. Formats an unrolling pipeline must read/write

| Concept | Format | Notes |
|---|---|---|
| **Volume** (raw CT) | OME-Zarr v2 group, levels `0..N`, uint8 (some uint16), Blosc chunks, Z/Y/X | streamed by chunk; missing chunk = fill_value |
| **.volpkg** | directory container | scroll + volumes + segmentations + metadata (EduceLab/VC) |
| **Segment / surface** | **tifxyz** dir (`x/y/z.tif` float32 grid + `meta.json` + `mask.tif`) | the (u,v)→(x,y,z) flattening; stored vs full resolution |
| **Mesh** | `.obj` (+ `.mtl`), `.ply`, EduceLab `.vcps` point sets | triangle meshes; UV-flattened via slim_uv |
| **Masks** | `.tif` (binary/grayscale) | validity masks, region masks |
| **Ink labels** | `<segid>_inklabels.png` (grayscale, Y×X) or `*_label.{tif,png,jpg}` aligned to surface grid | the prediction target |
| **Instance labels** | `Cube` annotated sub-volumes (zarr/tif), voxelized `.obj` | 3D instance segmentation |
| **Slice stacks** | numbered `.tif` (`1000.tif`, `1001.tif` …) | legacy/raw representation; increasing Z |
| **Structure tensors** | zarr arrays (ST components + eigenvalues/vectors) | fiber orientation fields |
| **Skeleton annotations** | WebKnossos `.nml` (zip) | via foundation/obj2nml |
| **Other scalar volumes** | NRRD | VC interop |

Rendering flow: surface (tifxyz or .obj) + volume → sample volume along the surface → "surface volume"
= a stack of layers around the sheet (the flattened image used for ink detection).

---

## 5. foundation — preprocessing (raw scans → aligned scroll volume), high level

- **scanning/** — acquisition side (little README content present).
- **scroll-builder/** — a server-side **dependency-graph build system** (`builder.py` + `builder.yaml`)
  that regenerates derived data (e.g. render surface volumes from `.obj` via Thaumato's
  `mesh_to_surface`) whenever inputs or scripts change. Regex-matches volpkg directory structure,
  runs each step in Docker with read-only/read-write mounts. It's a Make-like orchestrator over the
  `.volpkg`, not the math itself.
- **volume-registration/** — `find_transform.py`: aligns a **moving** source (another zarr volume, or
  a Neuroglancer-loadable VTK mesh) to a **fixed** zarr volume. Interactive Neuroglancer session:
  coarse manual rotate/translate/flip, then ≥3–4 landmark point pairs auto-fit a transform; optional
  (discouraged) SimpleITK auto-registration on low-res levels. Inputs/outputs are zarr + JSON
  transforms (`--initial-transform`, `--output-transform`), voxel size in µm. This is how multiple
  scans/energies of the same scroll are brought into a common coordinate frame; masked zarrs
  (`*_masked.zarr`) are used.
- **obj2nml/** — converts `.obj` triangle meshes → WebKnossos `.nml` skeleton annotations (zip).
- Others: `shift-optimizer`, `cloud-image`, `comparison-website`, `kaggle-visualizer`, `scrollcase`,
  `datasets/` (fibers-dataset, register-meshes-to-fibers, volumetric-instance-labels).

High-level flow: **raw scan(s) → (registration/masking into a common frame) → standardized OME-Zarr
volume in a `.volpkg` → segmentation (VC3D surface tracer / Thaumato) producing surfaces (tifxyz/obj)
→ rendering to surface volumes → ink detection.** scroll-builder ties the derived-data steps together.

---

## 6. Monorepo conventions worth adopting (from AGENTS.md / README / docs)

**AGENTS.md** is the standout artifact — a layered agent-governance doc. Principles to mirror in
fenix's AGENTS.md/CLAUDE.md:
1. **Read-only by default / no side effects on discovery.** Never run install/bootstrap/build scripts
   unprompted; report the exact minimal command and ask. (Agent-mode env gates:
   `AGENTS_AGENT_MODE=1`, `AGENTS_ALLOW_INSTALL=1`.)
2. **Scope first, then act.** Smallest change that solves the task; per-subproject playbooks; don't run
   one subproject's tooling in another.
3. **Don't guess build systems** — look for local `README/docs/scripts/CMakeLists/pyproject` first;
   prefer existing scripts.
4. **Correctness & reproducibility by default**: preserve behavior/outputs; avoid nondeterminism
   (iteration order, FP accumulation order, unseeded shuffles); never silently relax numerics
   (no `-ffast-math`/`-Ofast`/epsilon fudging) unless explicitly allowed.
5. **Performance work must be measured**: baseline, profiler, before/after table with command, dataset,
   build type, iteration counts, p50/p95.
6. **Portability is hard-required**: Ubuntu + macOS, amd64 + arm64; gate SIMD/intrinsics with safe
   fallbacks; avoid OS-specific code without guards.
7. **Reviewable diffs**: small/focused; two-step refactor (mechanical no-behavior-change, then
   functional/perf with measurements).
8. **Tests not optional**: run subproject tests/smoke before claiming success; add a regression test
   if none covers the change.
9. **Structured final report**: what changed (+ rationale), how to build/run (exact commands), how
   verified (tests + dataset), perf numbers + methodology, risks/limitations.
10. A **maintenance habit**: docs like `docs/atlas_implementation.md` say "update this file whenever
    the code/behavior changes" — keep design docs adjacent and self-maintaining.

Other adoptable specifics: prefer **RelWithDebInfo** for profiling / **Release** for final numbers;
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`; treat a representative `.volpkg` as the canonical perf workload.

---

## Implications for fenix (C++26 rewrite)

- A first-class **OME-Zarr v2 reader** is the foundation: parse `.zarray`/`.zattrs`, Blosc2
  decompression, multiresolution levels, `dim_separator` both forms, missing-chunk-as-fill,
  chunk-aligned sub-volume reads, local-mirror disk cache, HTTP(curl) + S3 + local backends. Native
  uint8 and **uint16** (vesuvius-c punted on uint16 — fenix should support it). Async/parallel chunk
  fetch.
- Settle on **Z/Y/X** ordering and fixed-width types globally, matching vesuvius-c.
- Core in-memory types analogous to `chunk`/`slice` (consider keeping native dtype + lazy float
  conversion rather than always widening to f32 like vesuvius-c does).
- Surface format = **tifxyz** (x/y/z float32 grids + mask + meta.json + aligned labels) with
  stored/full resolution + interpolated sampling; mesh I/O OBJ/PLY/VCPS; NRRD/TIFF support.
- Reuse the **scrolls.yaml** (scroll→energy→resolution→volume/segments) registry idea and the
  **chunk-occupancy bitmap** for efficient non-empty tiling.
- Adopt the AGENTS.md governance model wholesale for fenix.
