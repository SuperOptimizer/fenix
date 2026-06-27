# taberna — Build, Config, Data Formats & Algorithmic Roadmap (research for C++26 rewrite)

Source: `/home/forrest/taberna` (read-only research; nothing modified). taberna is a
"villa-free" classical (no-ML) pipeline that mirrors, preprocesses, compresses, and then
**virtually unwraps** Vesuvius scroll CT volumes. The repo is currently almost entirely
C; the rewrite target is greenfield C++26 with new (non-compatible) file formats.

---

## 1. Build configuration & toolchain

### 1.1 Top-level CMake (`/home/forrest/taberna/CMakeLists.txt`)
- `cmake_minimum_required(3.20)`, `project(taberna LANGUAGES C VERSION 0.1.0)`. The **core
  pipeline is C-only**; C++ is enabled lazily *only* when `-DTABERNA_GUI=ON` (`enable_language(CXX)`,
  `CMAKE_CXX_STANDARD 17`, AUTOMOC) for the Qt6/VTK viewer.
- Default `CMAKE_BUILD_TYPE=Release` if unset.
- **Key options (all documented inline with rationale):**
  - `TABERNA_FRAME_POINTER` (OFF): `-fno-omit-frame-pointer`, applied **globally before** the
    third-party subdirs so the matter-compressor decode hotpath unwinds for `perf record --call-graph fp`.
    "Negligible cost on aarch64."
  - `TABERNA_THINLTO` (OFF): cached ThinLTO (`-flto=thin` + `lld --thinlto-cache-dir`) applied
    **only to taberna's own targets**, after the third-party subdirs. Clang-only (warns otherwise).
    DTLTO deliberately not offered (on one machine == threaded ThinLTO + cache).
  - `TABERNA_FAT_LTO` (ON): emit fat LTO objects (bitcode + machine code) so `.o/.a` link with non-LTO consumers / plain ar/nm.
  - `TABERNA_FAST` (OFF): "ricing" — `-Ofast -ffast-math -funroll-loops -fno-math-errno -march=native
    -fno-semantic-interposition`, **taberna targets only**, applied after third-party subdirs so it
    **never reaches matter-compressor** (fast-math would reorder the DCT and change decoded bytes).
    Justified: the FP-heavy winding solve + flatten loops are ~57% of runtime.
  - `TABERNA_MIMALLOC` (ON): links mimalloc (NOT LD_PRELOAD) via `-Wl,--no-as-needed <lib> -Wl,--as-needed`
    so its malloc/free win ahead of libc. Carried by an INTERFACE lib `taberna_malloc` that
    `taberna_io` links (every tool links io) + a tuning constructor in `io/mimalloc_tune.c`. Big win
    on the multithreaded MC decode path (many small concurrent allocs).
- **ccache** auto-detected and forced as `CMAKE_C/CXX_COMPILER_LAUNCHER` (cached, set before subdirs
  so submodules + GUI inherit it).
- **OpenMP** found at top level, linked per-target (per-voxel hot loops: structure tensor, CED, NSD,
  morphology, persistent homology).
- Build order is deliberate (see §2): libs3 → matter-compressor → fysics → tiff → (LTO/FAST/mimalloc
  applied here) → src/ libs → tools → optional gui.

### 1.2 Presets (`CMakePresets.json`, v3, Ninja)
Hidden `base` preset: Ninja + `cmake/clang-lld.toolchain.cmake` + ccache + PIC + mimalloc. Four
concrete presets, each its own build dir:
| Preset | Build dir | Key cache vars |
|---|---|---|
| `release` (canonical) | `build-llvm` | Release, -O3, mimalloc |
| `profiling` | `build-prof` | Release + `TABERNA_FRAME_POINTER=ON` |
| `fast` | `build-fast` | `TABERNA_FAST=ON` + `TABERNA_THINLTO=ON` |
| `debug` | `build-debug` | Debug `-O0 -g3` + ASan+UBSan (`-fno-sanitize-recover=all`), **mimalloc OFF** (interposer fights ASan) |

### 1.3 Toolchain file (`cmake/clang-lld.toolchain.cmake`)
"Maximal-LLVM / minimal-GNU" stack. Prefers versioned LLVM via `TABERNA_LLVM_SUFFIX` (default `-23`,
i.e. **clang-23 / llvm-23**), falling back to unsuffixed names.
- Compiler: clang/clang++. Linker: `CMAKE_LINKER_TYPE LLD` + pinned `ld.lld-23` via `--ld-path`.
- All binutils set to the `llvm-*` equivalents (ar, ranlib, nm, objcopy, objdump, readelf, strip, addr2line).
- Runtime: `--rtlib=compiler-rt` everywhere (static builtins per-object → coexists with gcc-built apt VTK/Qt).
- `TABERNA_LIBCXX` (opt-in): full libc++ + libc++abi + LLVM libunwind instead of libstdc++/libgcc_s.
  Only coherent if **every** C++ lib in the link is libc++ → Qt & VTK must also be built `-DTABERNA_LIBCXX=ON`.
  Because libc++ headers are leaner, it force-includes the common transitively-relied-on headers
  (cstdlib, cstring, cstdint, vector, string, memory, algorithm, …) into every C++ TU rather than
  patching dozens of VTK/Qt headers.
- `TABERNA_LLVMLIBC` (experimental): apt `libllvmlibc-23-dev` static overlay over glibc. Its objects
  use local-exec TLS, so it can be linked **only into final executables**, never shared libs → Qt/VTK
  build normally and only taberna's own exes get the overlay.

### 1.4 Build scripts (`scripts/`)
- **`build_all.sh`** — the full from-source maximal-LLVM stack: clang-23 + lld-23 + llvm binutils +
  compiler-rt + libc++/libc++abi/libunwind + the llvm-libc overlay. Sequence: (0) `apply_patches.sh`
  → (1) **early smoke test** that libc++ + overlay actually link+run (fail in seconds, not hours) →
  (2) Qt 6.10.2 from official source tarballs (only qtbase + qtshadertools + qtdeclarative) →
  (3) minimal VTK (only the QML-viewer modules) → (4) taberna C pipeline + GUI. `LLVM_FLAGS` carries
  the toolchain file, libc++, llvm-libc overlay, ccache.
- **`build_qt.sh`** — minimal Qt6 (`v6.10.2`, 3 modules) from git, libc++ stack, → `build-qt/install`.
- **`build_vtk.sh`** — minimal VTK from the vendored submodule: disable ALL modules, enable only
  `GUISupportQtQuick` (QQuickVTKItem, VTK-in-QML), `GUISupportQt`, `RenderingVolumeOpenGL2` (GPU
  volume ray-cast), `RenderingOpenGL2`, `RenderingUI`, `InteractionStyle`; dependency closure pulls
  CommonCore/RenderingCore. Shared libs, Qt6. → `build-vtk/install`.
- **`apply_patches.sh`** — idempotent replay of `patches/<sub>/*.patch` into `third-party/<sub>`
  (reverse-check = already applied → skip; forward-check → apply; else FAIL loudly = upstream moved).
- Resume helpers (`resume_taberna.sh`, `resume_vtk_taberna.sh`), plus many Python experiment drivers
  (`tile_winding_mr.py`, `unroll_pipeline.py`, `robustness_bench.py`, `leak_metric.py`,
  `official_score.py`, `param_sweep.py`, etc. — the experiment harness lives in Python, calling the C tools).

### 1.5 Patches (`patches/`)
Submodules are pinned to pristine upstream commits, so local edits live as replayable patches. **One
patch today:** `patches/vtk/scn-libcxx-wrap_iter-base.patch` — VTK's vendored `vtkscn` calls
`std::__wrap_iter<char*>::base()`, removed in LLVM ≥18 libc++; switched to `operator->()`
(`std::__to_address`). Triggers only on `-DTABERNA_LIBCXX=ON`; not reported upstream (libstdc++ builds
unaffected). Note: `matter-compressor`, `libs3`, `fysics`, `tiff` are first-party SuperOptimizer
submodules (no patches needed — edited at source); only the third-party VTK needs patching.

---

## 2. Target dependency graph

Submodules (`.gitmodules`): `libs3`, `matter-compressor`, `fysics`, `tiff` are first-party
(`git@github.com:SuperOptimizer/*`); `vtk` is upstream Kitware (shallow). (`third-party/json`, `png`,
`jpg` also present; json.c is compiled into `taberna_io`.)

```
CURL, zstd, ZLIB, Threads, OpenMP, libm  (system)
        │
   libs3 (s3) ──────────────┐  ← the ONE shared S3 client (leaf), built FIRST
        │                   │
        ▼                   ▼
 matter-compressor      fysics (fysics, fysics-process, mca_export)
 (matter_compressor)    consumes s3 + matter_compressor
        │                   │
        └────────┬──────────┘
                 ▼
            taberna_io  (+ tiff, ZLIB, json.c, taberna_malloc/mimalloc)
                 │
   ┌─────────────┼──────────────┬───────────┬──────────┐
   ▼             ▼              ▼           ▼          ▼
taberna_      taberna_      taberna_     taberna_   taberna_
segmentation  annotate      eval         topo       postproc(→eval,topo)
   │             │              │           │          │
   └─────────────┴──────► taberna_unwrap (→annotate) ◄─┘
                                 │
                                 ▼
                          tools/*  (~60 experiment-driver exes)
                                 │
                          gui/ (taberna_viewer: Qt6+VTK, →matter_compressor) [opt-in]
```

**The "ONE shared libs3" constraint (central to the build order):** both matter-compressor and fysics
each vendor their own libs3 copy. If each linked its own, the final link would have duplicate `s3_*`
symbols. The top-level CMake builds `libs3` first (`EXCLUDE_FROM_ALL`), then forces
`MC_LIBS3_TARGET=s3` so matter-compressor links the shared target instead of `tools/vendor/libs3/libs3.c`,
and fysics consumes the same `s3` target. matter-compressor's CMake has the `MC_LIBS3_TARGET` cache
hook precisely for this; fysics's CMake errors out (`needs the s3 target`) unless a parent provides it.

**matter-compressor byte-determinism constraint:** MC is the codec whose archives must round-trip
predictably. Its CMake compiles `matter_compressor` **target-scoped** at `-O3 -march=native -ffast-math
-ffp-contract=fast` *regardless* of parent flags (the f32 DCT wants fast-math + FMA to vectorize), but
the top-level CMake is careful to apply `TABERNA_THINLTO` and `TABERNA_FAST` only *after* the MC subdir
so they never touch it. The MC source comments are explicit: fast-math already makes archives
**not byte-identical across ISA/opt-level**, but encode+decode in the *same build* agree exactly (the
tau/max-error corrections are computed against that build's own reconstruction, and the export pipeline
encodes+decodes with one build). MC also gates ThinLTO behind PGO (`MC_THINLTO` alone = measured ~15%
slower decode from link-time inliner mispredicting the branch-dense range-coder; only +5% *with* profiles).
mca-format-v2.md restates this as "matter-compressor's byte-determinism forbids LTO-without-PGO."

**fysics** (`third-party/fysics`): CT preprocessing kernels (FFT, Paganin phase retrieval, dering,
downscale, guided filter, MUSICA, z-drift, noise, register, phasecorr, zarr_io, pipeline) + two CLIs:
`fysics-process` (zarr→zarr preprocess) and **`mca_export`** (fused single-pass preprocess → `.mca`
archive, reads local or `s3://`). Compiled `-Ofast -ffast-math -funroll-loops -march=native`. Links
`s3` + `matter_compressor`. Several taberna tools (`sd_measure`, `surface_prep`, `surf_pred`,
`trace_mca`, `sheet_sep3d`) link fysics directly.

**src libs** (all static, link OpenMP + libm as available):
- `taberna_io` — `nrrd.c tiff_vol.c mca.c mimalloc_tune.c json.c`; PUBLIC-links `tiff`,
  `matter_compressor`, ZLIB, `taberna_malloc`. The ingest hub every tool depends on.
- `taberna_segmentation` — `snic sheet_tensor ridge ced trace hessian affinity partition stitch`.
- `taberna_annotate` — `umbilicus point_collection`.
- `taberna_eval` — `metrics topo score nsd` (VOI, adapted-Rand, topology, Kaggle composite).
- `taberna_topo` — `cubical persist0` (native cubical persistent homology).
- `taberna_postproc` — `morph sheet_repair topo_surgery`; depends on eval + topo.
- `taberna_unwrap` — `winding_field spiral_fit deform wmetrics`; depends on annotate.
- `tools/` — ~60 executables (the experiment phase). Notable: `mca_slice`, `extract_cube`,
  `wind_poisson`, `wind_tv`, `wind_cut`, `wind_label`, `unroll_wind`, `unroll_*`, `sheet_sep3d`,
  `svaff_seg`, `sheetscale`, `surface_*` (Kaggle pipeline), `tifxyz`/`wind_tifxyz` (VC3D interop),
  `pers_dump`, test exes.
- `gui/taberna_viewer` — Qt6 (Widgets, OpenGLWidgets) + VTK9 (CommonCore, CommonDataModel,
  Rendering{Core,Volume,VolumeOpenGL2,OpenGL2}, GUISupportQt, InteractionStyle) 2×2 MPR + GPU volume
  render; links `matter_compressor`; `vtk_module_autoinit`.

---

## 3. Data / file formats

### 3.1 `.mca` — matter-compressor archive v2 (`docs/mca-format-v2.md`, src/io/mca.c)
The primary working volume format. v2 (design locked 2026-06-20) replaced v1's static-fixed-slot
layout (logical size == *uncompressed* volume, hit the ext4 16 TiB / 48-bit mmap walls) with a
**compact dense heap**: logical = physical = compressed bytes, no sparse-file dependency.
- **Goals:** compact/dense, streamable (lock-free random-position shard writes by uncoordinated
  workers, *and* single-writer full export — same format), variable-rate noise-floor quantization,
  uniform 16-bit addressing in all three axes.
- **Coordinate frame:** uniform 2^16 cube. Per-axis 16 bits → Morton key 3×16 + 3 LOD bits = 51-bit
  u64. **Conditional rotation**: only when the longest axis > 2^16, rotate the umbilicus (scroll axis)
  onto the (1,1,1) body diagonal so the empty cylinder corners let all 3 axes fit 2^16 (PHercParis4
  74272→~58k). Volumes that already fit stay axis-aligned (PHerc0332 = 15761×15761×33592, z<65536, no
  rotation). Header stores the **3×3 rotation matrix + f64 origin** mapping archive→world so coords
  stay recoverable and the unwrap pipeline knows the umbilicus direction. Rotation stored in the
  128 KB user-metadata carve-out, not a new header field.
- **Hierarchy (per LOD, LODs fully independent):** DCT block = 16³ voxels (transform+quant unit, own
  air mask) → shard = 16³ blocks = 256³ voxels = 4096 blocks (network-IO / fetch / write-atomicity
  unit, a Z-order rope of blocks) → shard grid up to 256 shards/axis at LOD0 → 8 independent LODs.
- **File layout per LOD region (dense, no holes):** `[header][shard occupancy: 2 bits/shard][shard
  offset table: {u64 offset, u32 len}/shard at a computed Morton slot][shard data heap: packed REAL
  shards, append-grown]`. Index regions sized to actual shard dims, allocated outright; heap grows by append.
- **Access model:** runtime **mmaps the whole archive**; reads a shard's `{offset,len}` from the
  computed slot, decodes the contiguous region directly from the live mapping (no read()/gather). A
  shard is the unit of network IO (one HTTP range request). v2 is compact (~21 GB dense for PHerc0332)
  so whole-file mmap is trivial (vs v1's 9.75 TB mapping that hit the 48-bit VA ceiling).
- **Occupancy** is a 3-state streaming manifest: `DONT_KNOW` (ask remote), `ALL_ZERO` (air, skip),
  `REAL` (has data). **Index** is actually a 3-level dense node tree (root→inner→shard, flat 4096-u64
  child-offset arrays indexed by chunk-coord nibble), crash-safe, in-place updatable, covers 2^20 voxels/axis.
- **Shard body (REAL shard = contiguous chunk-blob):**
  `[f32 q][u64 xxh64][u16 fmaplen][fracmap][bitmask 512B = 4096 bits present/air][u16 lengths ×
  npresent][block payloads]`. Block offsets are implicit prefix-sums of lengths; locate block k by
  rank(bitmask,k). xxh64 over (fracmap|bitmask|lengths|payloads) = verify-on-decode. Overhead ~1–3% at q≈32.
- **Block codec (16³ DCT, unchanged from mc):** mask-aware, context-coded air mask, **harmonic air-fill
  (red-black SOR)** before integer DCT-16 + dead-zone quant + CABAC; air force-zeroed on decode.
- **Quantization:** one **global q** in the header; `tau = 2q` (archive size ∝ tau). Default q≈32/tau=64
  (~60×, sub-noise-floor); q=16 (43×) or q=8 (27×, edge-safe) for more fidelity. ~21 GB full archive
  for PHerc0332 (5× smaller than v1).
- **io/mca.c** access: `mca_dims`/`mca_open` mmap the file and call `mc_open`/`mc_reader_dims`/
  `mc_archive_open_dims`; `mca_metadata` returns the JSON user-metadata blob; `mca_roi_origin` parses
  `roi.origin` from that JSON (so .mca region exports carry their volume origin). `mca_read(h, lod,
  z0,y0,x0, dz,dy,dx)` returns a decoded **u8 cube** (z-major).

### 3.2 NRRD (`src/io/nrrd.{c,h}`)
Hand-rolled ~30-line-spirit reader for public ground-truth: text header terminated by a blank line
(`\n\n`) + raw or gzip-compressed payload. Parses `type` (uint8/uint16/…), sizes, encoding. Used for
`instance-labels-harmonized.zip` (Scroll 1, 256³ cubes, per-voxel sheet-**instance** labels) and the
GP cubes — the feasibility/validation data. No zarr/S3 reader needed for the planned work.

### 3.3 TIFF volume stacks (`third-party/tiff`, `src/io/tiff_vol.{c,h}`)
Vendored minimal dependency-free TIFF reader/writer + a custom 2D near-lossless codec. taberna added
**TIFF-LZW decode** + `tiff_read_volume` (multi-IFD z-stack). Used to ingest the **Kaggle Surface
Detection** data: `train_images/<id>.tif` + `train_labels/<id>.tif`, each a **320³ u8 multi-page TIFF,
LZW-compressed** (Compression=5, one strip/page, little-endian). Label values: 0 background, 1 surface,
2 ignore/not-evaluated (~71% of volume; only ~29% scored). `src/io/tiff_vol.c` → `tiff_load_u8` gives
a z-major u8 volume.

### 3.4 Winding-field `.f32` (internal)
The unwrap backbone format. A flat little-endian file: **header = 6× int32 `{dz, dy, dx, z0, y0, x0}`
(24 bytes)** followed by `dz*dy*dx` float32 winding values (a 256³+ region's per-voxel winding
coordinate on a given LOD grid). Produced by `tile_winding_mr.py` (→ `merged.f32`), regularized by
`wind_tv` (→ `*_tv.f32`), labeled by `wind_label` (`_lab.i32` instance-label volume) / `wind_cut`,
consumed by `unroll_wind`, `wind_tifxyz`, etc. Origin fields make tiles self-locating for stitching.

### 3.5 tifxyz / VC3D QuadSurface (interop only — `tools/tifxyz.c`, `tools/wind_tifxyz.c`)
The interop format with VC3D surface predictions. A segment is a **directory of single-page float32
TIFFs `x.tif`/`y.tif`/`z.tif`**: each a (u,v) grid whose pixels hold the **volume (z,y,x) coordinate**
of that flattened surface point (value −1 = invalid), plus optional `mask.tif` (u8) + `meta.json`.
`tifxyz render` samples a `.mca` at each (z,y,x) to render the flattened papyrus; `wind_tifxyz` exports
taberna's winding field *as* a tifxyz QuadSurface (u = spiral render-coord `rc = w + atan2(y−cy,x−cx)/2π`,
v = z, MEAN-accumulated source coords). This is the "plug ours into VC3D / plug VC3D into ours" bridge.

### 3.6 Annotation formats (`src/annotate`, unwrapping-plan §3)
taberna's own simple formats (concept borrowed from prior art, not schema): `umbilicus.txt` = one
`z y x` per line; **point collections** = plain JSON `{collection_id, winding_is_absolute:bool,
points:[{id, zyx, wind}]}` (absolute collections calibrate, relative collections link) + explicit
`must_link`/`cannot_link` pair lists for the partition.

Other outputs: PGM/PPM renders (unroll output, overlays), `_lab.i32` instance-label volumes.

---

## 4. Algorithmic roadmap (the design docs)

### 4.1 Goal & strategy (`unwrapping-plan.md`)
Take a deformed CT volume of a spiral-wound scroll and **undeform it into an ideal Archimedean spiral**,
recording the spiral params **+ an invertible deformation field** (det-J > 0 guarded), so the sheet can
be unrolled for ink detection. **Hard constraints: no ML, no matter-compressor dependency, no villa
app dependency; every stage runs per-256³-brick + halo and stitches** (volume is tens of TB, never
resident). villa / Paul Henderson's `spiral-v2` are **idea sources & benchmarks only**, never port targets.

**Per-brick pipeline:** [1] sheet detection (CED pre-filter → structure tensor → sheet normal +
coherence; OOF/Descoteaux sheetness scalar; phase-symmetry channel) → [2] signed affinity graph
(optional SNIC supervoxels; + attractive in-plane, − repulsive across-sheet edges; human must/cannot-link
as hard edges) → [3] partition (GASP avg-linkage + cannot-link, preferred over MWS) → [4] Eulerian
winding-number field (integrate structure-tensor azimuthal orientation via Poisson/heat-method,
Dirichlet seeds) → [5] global stitch (Lu/Zlateski/Seung delayed-merge for chunking-invariant labels;
accumulate generalized winding across bricks; loop-closure holonomy check) → [6] Archimedean fit
(`r = a + b·θ` LSQ + invertible deformation field, det-J > 0) → [7] unroll output (per-winding meshes;
SLIM per-patch final flatten).

**Human annotation** (~1 week, targeted+reusable): umbilicus polyline (highest leverage), absolute-winding
seeds (Dirichlet), relative-winding links at breaks/touches (must/cannot-link), correction points,
damage masks. Pipeline auto-flags *where* to annotate via loop-closure holonomy, merge hotspots, det-J<0 folds.

**Verification metrics (tiered):** Tier-0 GT-free inner loop (det-J>0 % foldings, self-intersection,
radial winding monotonicity, loop-closure holonomy); Tier-1 asymmetric VOI (VOI_merge ≫ VOI_split,
a cross-wrap merge is catastrophic) + ARAND; Tier-2 SurfaceDice@τ + TopoScore (Betti); Tier-3 WJF +
Paul's `satisfied_*`; Tier-4 symmetric Dirichlet / quasi-conformal distortion; Tier-5 TRE on held-out
landmarks. **Experiments E1–E6** gated, E1 (signed-affinity feasibility, de-risks the whole bet) first.
**Compute budget:** working volume is LOCAL (matter-compressor on NVMe Gen5), no remote-streaming
problem; design for **~1% MAE / ~2.5 graylevels** lossy-compression error on the u8 cubes.

### 4.2 Segmentation SOTA (`segmentation-sota.md`)
Two dominating facts: (1) **the published method that does taberna's job is Paul Henderson's
Diffeomorphic Spiral Fitting (WACV 2026, arXiv 2512.04927)** — a global invertible diffeomorphism
(per-slice affine + ODE flow + radial gap-scale) fit to surface/fiber inputs; its geometric core is
classical, only the per-voxel inputs are nnU-Net (the part taberna replaces with the structure tensor).
Caveat: global ~19 h on a 30-winding subvolume, NOT block-wise → out-of-core re-engineering is the new
work. (2) **Connectomics has built taberna's front-end at TB/PB scale** with a classical core (only
affinity prediction is learned): oversegment → RAG → greedy agglomeration → chunking-invariant
block stitching. **The lever is the affinity/sheetness field, not the clustering algorithm.** Adopt:
two-channel sheet detection (structure tensor for normals + OOF/Descoteaux for the scalar — OOF
resolves the ~1-voxel inter-wrap gap the Hessian merges); **signed-graph partitioning (MWS/GASP)**
because adjacent wraps touch but must not merge (repulsive edges prevent collapse); the connectomics
block-wise RAG + Volara two-level + Lu/Zlateski/Seung delayed-merge stack; Eulerian field + winding-number
+ closed-form spiral fit over a global mesh. Open problems: structure-tensor affinity quality unproven
(E1's bet), sub-resolution inter-wrap gap, chunking-invariant stitching for signed partitioners
unsolved, single-object-wound-N-times topology is novel.

### 4.3 Surface detection / Kaggle (`surface-detection-kaggle.md`)
Vesuvius "Surface Detection" Kaggle (Dec 2025–Feb 2026, $100K). **The model is commoditized
(everyone trained the same nnU-Net ResEnc-UNet); the leaderboard was won in classical post-processing
— exactly taberna's territory.** Post-proc toolkit (ranked by score): iterated 3×3×3 binary median
(27-neighborhood majority vote = discrete curvature flow, best value/line), dust removal (drop small
CCs, biggest single jump), small-hole plug via a 256-entry 2×2×2 LUT, binary closing + fill_holes,
PCA/height-map sheet repair (= local unwrap, appears 3× independently), Euler-number / Betti topology
repair. **Touching/adhering sheets is the named open problem** (everyone relied on the model or
multi-threshold erosion). **taberna's status:** vendored TIFF-LZW + `tiff_read_volume`; native exact
VOI (bit-exact vs official), reducible TopoScore without persistent homology (6-conn FG / 26-conn BG,
2×2×2 tiling), a from-scratch cubical persistent-homology engine (`src/topo/cubical.c`, validated
50/50 binary + 40/40 grayscale vs Betti-Matching-3D). Classical detector arc: raw sheetness 0.27 →
ridge NMS along structure-tensor normal **SurfaceDice 0.68** → first official composite **0.4406**
(SurfaceDice 0.612, VOI 0.647, **TopoScore 0.000** — 6304 tunnels kill it). **The wall is the
detector** (structure-tensor ridge plateaus at ~0.68 SurfD, ceilinged by label-convention placement
offset, not tunable knobs); PCA height-map repair failed on real curved porous ridges. Official metric:
`0.30·TopoScore + 0.35·SurfaceDice@2 + 0.35·VOI_score`. taberna's differentiator thesis: prevent
merges at the segmentation level (signed affinity + MWS) rather than repairing them post-hoc.

### 4.4 Touching sheets (`touching-sheets-plan.md`)
The one place the working winding pipeline still merges two wraps (undercount + winding leak), because
the local signal `across·sheetness` is identical for a within-sheet radial contact and an inter-wrap
touch. **Governing principle (from 3 production systems — Thaumato, FASP/VC3D, Henderson): attach each
surface element a winding number and gate merges on Δwinding** (within-sheet Δw≈0, inter-wrap touch
Δw≈±1 — same local geometry, opposite global gap). Phased by risk×yield. **Key implemented findings
(extensive negative-result log, all 2026-06-23):**
- Phase 0a `sheetscale.c`: multiscale-σ argmax detector is weak (touch fraction flat ~0.30), but lone
  sheets peak at σ≤0.7 so the default σ_tensor≈1.5 over-smooths.
- Phase 1 (pivotal): **winding-as-label works; MWS is the wrong vehicle** — supervoxel+MWS fragments,
  but a direct voxel-level `floor(winding)` band map gives clean concentric per-wrap rings. The
  instance label is just `floor(regularized winding)`.
- Phase 3 `wind_label.c`: ships `floor(W)` instance-label volume + in-plane mode-filter despeckle.
  Radial rays are the **wrong column geometry** on a deformed scroll (W non-monotone along a straight ray).
- **Phase 3a `wind_tv.c` — first method to beat floor(W):** global weighted-TV regularizer
  (Chambolle–Pock primal-dual, `λ/2·‖u−W‖² + ∫ g·|∇u|`, g=sheetness). Removes the pervasive
  along-sheet flip globally (no local filter can — both halves of a flipped sheet are locally
  self-consistent). Along-sheet flip 3.35% → 2.14% (−36%); **+73% sharper, +33% higher-contrast
  unroll** (validated on the real product). Default λ=0.3. This is the working win.
- Phase 3b (negatives): soft min-slope ordering term can't split a gapless fused touch (a discrete
  cut-placement decision); the hard ordered-label min-cut (`wind_cut.c`: verified Dinic maxflow +
  nested level-set, all 7 selftests pass) reproduces floor(U) with min-sheetness boundaries but can't
  *invent* a missing wrap without the LOGISMOS min-separation ∞-arc constraint. **Leak quantification
  (`leak_metric.py`): defect-2 (leaked touches) is a ~1% long tail, NOT dominant** → the heavy LOGISMOS
  build is low-yield on accessible mid-scroll data; `wind_cut`'s maxflow+scaffold are banked for a
  deep high-fusion core. **Phase 3a (`wind_tv`) is the high-value touch work, done & validated.**
- Phase 5 north star: Henderson diffeomorphic Archimedean spiral fit (injective map → fused wraps
  cannot merge), after Phases 1–3 prove the order prior at small scale.

---

## 5. Notes for the C++26 rewrite
- C-only core today; only the GUI needs C++17 (Qt/VTK). The rewrite makes everything C++26.
- Toolchain is firmly clang-23/lld-23/libc++/compiler-rt with an optional llvm-libc overlay — the
  C++26 features assume a very recent clang.
- The two non-negotiable build invariants worth preserving: **ONE shared libs3** (no duplicate
  symbols) and **matter-compressor isolated from LTO/fast-math globals** (byte-determinism). These are
  CMake-ordering constraints, not code constraints.
- New file formats with no back/forward compat are wanted, but the *content* each existing format
  carries is the spec: .mca (compact dense Morton-shard archive, mmap+range-fetch, global-q DCT,
  3-state occupancy manifest, rotation matrix in metadata), winding `.f32` (origin-tagged per-voxel
  winding region), tifxyz (u,v→volume-coord QuadSurface for VC3D interop), NRRD/TIFF for GT ingest,
  simple umbilicus+point-collection annotation JSON.
- The algorithmic roadmap is the heart: classical no-ML block-wise pipeline, structure-tensor
  front-end, signed-graph segmentation, Eulerian winding field, weighted-TV winding regularization
  (`wind_tv` is the proven core), Archimedean/diffeomorphic spiral fit as the end-state. The
  touching-sheets log is a rich record of what's been tried and ruled out — preserve those negative
  results so the rewrite doesn't repeat them.
```
