# fenix adversarial code review — master index (2026-07-02)

Full-repo adversarial review of `/Users/forrest/fenix`, conducted by **15 adversarial
reviewers**: 11 per-module reviewers plus 4 cross-cutting sweeps (concurrency,
out-of-core/OOM, error-discipline, performance). Every high/critical finding was
**independently adversarially verified** (a second agent attempted to refute each one
against the actual source). This file is the entry point for the implementation team
that will fix everything; per-unit detail files are listed below.

## Stats

| Metric | Value |
|---|---|
| Reviewers | 15 |
| Raw findings | 153 |
| After dedup | 153 |
| High/critical findings | 43 |
| Independently verified → confirmed | 43 |
| Refuted | 0 |
| Unverified medium/low findings | 110 |

## Confirmed high/critical findings

Sorted critical-first, then by unit. Unit links go to the per-unit detail files.

| Severity | Unit | Location | Title |
|---|---|---|---|
| critical | [codec](./codec.md) | `src/codec/rans.hpp:72` | RansModel::from_counts normalization can underflow the largest freq → out-of-bounds write of the slot table (encode side, natural data) |
| critical | [codec](./codec.md) | `src/codec/rans.hpp:40` | RansModel::from_freqs never validates that deserialized freqs sum to rans_scale → out-of-bounds slot write on corrupt files |
| critical | [codec](./codec.md) | `src/codec/dct_block.hpp:426` | decode_tile_dct trusts nsigs / context map / category symbols → OOB heap writes and shift UB on corrupt tile payloads |
| critical | [build-ci](./build-ci.md) | `.github/workflows/ci.yml:17` | CI is 100% red: actions/checkout@v4 cannot run inside the musl Chimera container — every gate is dead |
| critical | [sweep-oom-ooc](./sweep-oom-ooc.md) | `src/segment/trace_stream.hpp:30` | OOC tracer tile fetch silently turns a hard fetch failure into air |
| critical | [sweep-errors](./sweep-errors.md) | `src/segment/trace_stream.hpp:30` | stream_tile_u8 swallows a hard fetch failure and returns an UNINITIALIZED tile as data |
| high | [core](./core.md) | `src/core/filter.hpp:29` | gaussian_blur reads out of bounds when the kernel radius reaches the line length (single-application reflect) |
| high | [codec](./codec.md) | `src/codec/lossless.hpp:83` | lossless_decode trusts n, B, enc_size and the freq table → OOB heap writes and unbounded reads/allocations on corrupt input |
| high | [codec](./codec.md) | `src/codec/entropy.hpp:47` | get_var reads past the buffer and has shift UB; decode_plane header reads and subspan are unbounded |
| high | [codec](./codec.md) | `src/codec/archive.hpp:247` | read_chunk_as blob bounds check overflows on a corrupt slot length → SEGV (page-table leaves are not CRC-protected) |
| high | [io](./io.md) | `src/io/s3.hpp:121` | Thread-local CURL easy handle leaks once per ephemeral fetch thread — unbounded over a scroll export |
| high | [io](./io.md) | `src/io/zarr.hpp:31` | Local fetch_object conflates ANY open failure with absent — transient IO error silently becomes air |
| high | [io](./io.md) | `src/io/zarr.hpp:179` | Short/truncated chunk blob is silently treated as absent → fill-value air |
| high | [ml](./ml.md) | `src/ml/infer.hpp:160` | ckpt_load pollutes the accumulator on a short/corrupt checkpoint, then reports "no resume" |
| high | [ml](./ml.md) | `src/ml/infer.hpp:98` | Checkpoint header does not identify model/weights/input/norm — default-on resume silently mixes different runs |
| high | [ml](./ml.md) | `src/ml/infer.hpp:365` | Torch exceptions (CUDA OOM) escape the predict loop; batched path unwinds past a joinable std::thread producer → std::terminate |
| high | [segment](./segment.md) | `src/segment/trace_stream.hpp:30` | Streamed tracer turns a zarr fetch failure into an all-zero (air) tile |
| high | [winding](./winding.md) | `src/winding/spiral_model.hpp:55` | Fit losses compare the branch-cut-discontinuous model winding to cut-free integer wrap labels — irreducible ±1 seam along θ = π |
| high | [geom-flatten-render](./geom-flatten-render.md) | `src/geom/marching.hpp:66` | Marching-tetrahedra quad case triangulates across the wrong diagonal — holes + double-cover in every 2-2 sign split |
| high | [geom-flatten-render](./geom-flatten-render.md) | `src/render/surface_render.hpp:31` | render_surface computes tangents from invalid neighbours' zero coords — garbage normals at every valid-region border |
| high | [geom-flatten-render](./geom-flatten-render.md) | `src/geom/mesh.hpp:57` | read_obj aborts the process on malformed files (std::stoi under -fno-exceptions), reads uninitialized floats, and never validates face indices |
| high | [pre-pred-anno-fs](./pre-pred-anno-fs.md) | `src/preprocess/preproc_cli.hpp:65` | CLI numeric parsing uses throwing std::stoi/stof/stod under -fno-exceptions — malformed args abort the process |
| high | [pre-pred-anno-fs](./pre-pred-anno-fs.md) | `src/fs/fxfs.cpp:102` | fxfs fx_read samples per BYTE through the locked block cache; the 'current block' caching its own comment claims does not exist |
| high | [apps-tools-tests](./apps-tools-tests.md) | `src/io/slice.hpp:137` | CLI/stage arg parsing and OBJ import use throwing std::sto* under -fno-exceptions — malformed input aborts the process |
| high | [apps-tools-tests](./apps-tools-tests.md) | `tests/test_zarr.cpp:33` | The 'transient fetch error must NEVER become air' invariant has zero test coverage — and the zarr reader currently violates it for truncated chunks |
| high | [apps-tools-tests](./apps-tools-tests.md) | `CMakePresets.json:67` | The fuzz preset is unbuildable and zero fuzz harnesses exist for the implemented untrusted-input parsers |
| high | [build-ci](./build-ci.md) | `Dockerfile:15` | OpenMP runtime is not installed in the Chimera Docker image or CI — canonical builds are silently serial and the tsan gate cannot catch OpenMP races |
| high | [build-ci](./build-ci.md) | `CMakePresets.json:69` | fuzz preset cannot link: -fsanitize=fuzzer applied to every target injects libFuzzer's main alongside driver.cpp's and every test's main |
| high | [sweep-concurrency](./sweep-concurrency.md) | `src/io/s3.hpp:121` | Per-call thread spawning in parallel_for_io leaks one CURL easy handle (with its 256 KiB buffer) per spawned thread — unbounded over a whole-scroll export |
| high | [sweep-concurrency](./sweep-concurrency.md) | `src/codec/archive.hpp:402` | VolumeArchive::block16 lazily initializes the block cache without synchronization — data race / use-after-free on an API documented as thread-safe |
| high | [sweep-oom-ooc](./sweep-oom-ooc.md) | `src/codec/archive.hpp:86` | .fxvol mmap reservation hard-caps any archive at 1 TiB — below plausible whole-scroll compressed size |
| high | [sweep-oom-ooc](./sweep-oom-ooc.md) | `src/io/io.hpp:389` | Eight CLI stage entry points decode whole archives to dense f32 (read_volume()), 4× RAM on u8-native data |
| high | [sweep-errors](./sweep-errors.md) | `src/io/zarr.hpp:31` | Local fetch_object treats EVERY open failure as absent=air (EMFILE/EACCES/EIO become fill) |
| high | [sweep-errors](./sweep-errors.md) | `src/io/zarr.hpp:179` | Short/truncated zarr chunk blob is silently treated as absent (fill) instead of an error |
| high | [sweep-errors](./sweep-errors.md) | `src/io/zarr.hpp:296` | copy_zarr_region_local never checks chunk write success — ENOSPC yields a silently truncated zarr store |
| high | [sweep-errors](./sweep-errors.md) | `src/geom/mesh.hpp:57` | read_obj uses std::stoi on untrusted file tokens — malformed OBJ aborts under -fno-exceptions; unchecked >> pushes uninitialized vertices |
| high | [sweep-perf](./sweep-perf.md) | `src/ml/infer.hpp:113` | Default-on inference checkpoint blocks the GPU pipeline with a serial O(voxels) scan + full-accumulator write |
| medium | [codec](./codec.md) | `src/codec/archive.hpp:402` | block16 lazily creates the block cache on a const path with no synchronization → data race / use-after-free |
| medium | [segment](./segment.md) | `src/segment/grow.hpp:378` | CT-valley guards sample the CT view with prediction-space coords, ignoring DataField::ct_ds |
| medium | [winding](./winding.md) | `src/winding/flow.hpp:82` | `flow_point_backward` overflows its fixed `traj[33]` stack buffer for `flow_steps > 32` |
| medium | [eval-topo-postproc](./eval-topo-postproc.md) | `src/eval/metrics.hpp:72` | VOI split and merge components are swapped (both `voi` and `voi_union`) |
| medium | [build-ci](./build-ci.md) | `.github/workflows/ci.yml:20` | Split build (FENIX_SPLIT) is never CI-verified, and its src/*/*.cpp glob makes a missing module TU a silent stage dropout |
| medium | [sweep-oom-ooc](./sweep-oom-ooc.md) | `src/eval/eval.hpp:29` | fenix eval is in-core at ~30+ bytes/voxel: dense-f32 loads + dual EDT + 8 B/voxel union-find parent |

Note: some findings appear under both a module reviewer and a cross-cutting sweep
(e.g. `trace_stream.hpp:30`, `zarr.hpp:31/179`, `s3.hpp:121`, `archive.hpp:402`,
`mesh.hpp:57`). These are the same underlying defects independently confirmed twice —
fix once, close all linked entries.

## Per-unit detail files

Each file contains the full write-ups for the confirmed findings above, plus that
unit's **unverified medium/low findings and any refuted findings at the bottom**.

- [core](./core.md)
- [codec](./codec.md)
- [io](./io.md)
- [ml](./ml.md)
- [segment](./segment.md)
- [winding](./winding.md)
- [eval-topo-postproc](./eval-topo-postproc.md)
- [geom-flatten-render](./geom-flatten-render.md)
- [pre-pred-anno-fs](./pre-pred-anno-fs.md)
- [apps-tools-tests](./apps-tools-tests.md)
- [build-ci](./build-ci.md)
- [sweep-concurrency](./sweep-concurrency.md)
- [sweep-oom-ooc](./sweep-oom-ooc.md)
- [sweep-errors](./sweep-errors.md)
- [sweep-perf](./sweep-perf.md)

## Suggested implementation order

### Wave 1 — Correctness & data corruption (silent wrong results, memory unsafety, crashes)

The invariant-breakers and OOB writes. Everything here can corrupt scientific output or
crash on real/corrupt data.

1. **Silent air / fetch-error swallowing** (the project's hardest rule):
   `src/segment/trace_stream.hpp:30`, `src/io/zarr.hpp:31`, `src/io/zarr.hpp:179`,
   `src/io/zarr.hpp:296`.
2. **Codec memory safety** (encode-side underflow first — it corrupts on *natural* data):
   `src/codec/rans.hpp:72`, `src/codec/rans.hpp:40`, `src/codec/dct_block.hpp:426`,
   `src/codec/lossless.hpp:83`, `src/codec/entropy.hpp:47`, `src/codec/archive.hpp:247`.
3. **Core/geom numerics & buffer bugs:** `src/core/filter.hpp:29`,
   `src/winding/flow.hpp:82`, `src/geom/marching.hpp:66`,
   `src/render/surface_render.hpp:31`, `src/segment/grow.hpp:378`,
   `src/eval/metrics.hpp:72`, `src/winding/spiral_model.hpp:55`.
4. **ML checkpoint integrity:** `src/ml/infer.hpp:160`, `src/ml/infer.hpp:98`.
5. **Abort-on-malformed-input (`std::sto*` under `-fno-exceptions`):**
   `src/geom/mesh.hpp:57`, `src/preprocess/preproc_cli.hpp:65`, `src/io/slice.hpp:137`.

### Wave 2 — Concurrency & resource safety

- `src/codec/archive.hpp:402` — block-cache lazy init race (fix once for both entries).
- `src/io/s3.hpp:121` — per-thread CURL handle leak (fix once for both entries).
- `src/ml/infer.hpp:365` — torch exception unwinding past a joinable producer thread → terminate.

### Wave 3 — Out-of-core & performance

- `src/io/io.hpp:389` — dense-f32 whole-archive decodes in eight stage entry points.
- `src/codec/archive.hpp:86` — 1 TiB mmap reservation cap.
- `src/ml/infer.hpp:113` — checkpoint scan blocking the GPU pipeline.
- `src/fs/fxfs.cpp:102` — per-byte locked cache lookups in fx_read.
- `src/eval/eval.hpp:29` — in-core eval memory footprint.

### Wave 4 — Build/CI & test hygiene

- `.github/workflows/ci.yml:17` — restore CI (checkout in musl container). *Do this
  early in parallel with Wave 1 so subsequent fixes land under green gates.*
- `Dockerfile:15` — install the OpenMP runtime (build + tsan coverage).
- `CMakePresets.json:67` / `CMakePresets.json:69` — fix the fuzz preset link model and
  add fuzz harnesses for the untrusted-input parsers (codec/zarr/OBJ — directly guards
  the Wave 1 fixes).
- `tests/test_zarr.cpp:33` — add regression coverage for the never-become-air invariant.
- `.github/workflows/ci.yml:20` — CI-verify FENIX_SPLIT and de-glob the module TU list.

After each wave: run the relevant presets (asan/ubsan for Wave 1, tsan for Wave 2,
bench for Wave 3) and mark the finding's outcome in its per-unit file.
