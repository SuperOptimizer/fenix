I have read all 15 files. Here is the compact digest of every medium/low finding marked "unverified (medium/low)", grouped by unit file.

---

# Medium/Low Findings Digest

## apps-tools-tests.md (apps/driver.cpp, tools/, tests/)
- apps/driver.cpp:103-117 — Recipe runner rejects multi-line TOML arrays and mis-splits comma-containing stage args (bug)
- apps/driver.cpp:37-142 — Driver dispatch, `info`, and recipe runner have zero test coverage (hygiene)
- src/core/test.hpp:32-43 — Test harness passes with zero registered cases; failure counter not thread-safe (design)
- tests/test_substrate.cpp:16 — Tautological assertion (`x || true`); seed sensitivity not actually tested (hygiene)
- apps/driver.cpp:78-85 — `fenix info` walks every chunk serially through coverage() — minutes on full-scroll (performance)
- apps/driver.cpp:60 — `--threads` documented but parsed nowhere; Context.threads is dead (hygiene)
- tools/proto/ — 8 unregistered one-off Python scripts (documented "tool sprawl" smell) (hygiene)

## build-ci.md (CMake, presets, Dockerfile, CI, bootstrap)
- .github/workflows/ci.yml:55 — include-cleaner (IWYU) CI gate is `|| true`; documented-blocking check can never fail (hygiene)
- Dockerfile:40-42 — env-var ccache launcher bypasses CMakeLists sloppiness injection; PCH compiles never hit persisted cache (performance)
- CMakeLists.txt:118 — `-march=native` objects + ccache dir across heterogeneous hosts can reuse wrong-ISA objects (correctness)
- cmake/clang-toolchain.cmake:13 — CMAKE_LINKER_TYPE needs CMake ≥3.29 but min is 3.28; silently ignored, wrong linker (hygiene)
- CMakePresets.json:72-75 — profiling/relwithdebinfo presets omit fast-math/-march=native; profiles measure different codegen (performance)
- install-runpod-ubuntu2404.sh:51-59 — `set -e` aborts silently in symlink loop; the apt line preventing it is `|| true`-masked (bug)
- bootstrap.sh:22 — `docker run -it` used unconditionally; fails with no TTY (hygiene)

## codec.md (src/codec/)
- src/codec/lossless.hpp:71 — Untrusted superblock/stream sizes drive unchecked allocations → abort under -fno-exceptions (correctness)
- src/codec/archive.hpp:245 — ABSENT (NOT_SURE) chunks silently decode as fill; conflated with ZERO on every read path (design)
- src/codec/archive.hpp:506 — gather_box_f32 edge-clamp path rescans whole output box once per covering block (performance)
- src/codec/archive.hpp:127 — open() always opens O_RDWR and mmaps PROT_WRITE even for read-only use (resource-safety)
- src/codec/dct_block.hpp:117 — quant_one_block heap-allocates two vectors per 16³ block in encode hot loop (performance)
- src/codec/archive.hpp:453 — sample_f32/block16 accept negative coords and index with negative modulo → OOB read (bug)

## core.md (src/core/)
- src/core/vec.hpp:36-39 — cross(a,b) computes b×a; frame declared right-handed but cross is left-handed (correctness)
- src/core/eig.hpp:30,33 — sym_eig3 convergence test is absolute 1e-20; never fires in f32, so every eigensolve runs all 50 sweeps (performance)
- src/core/volume.hpp:89-91 — Volume/Arena allocate via throwing operator new under -fno-exceptions; OOM terminates; dims unvalidated (resource-safety)
- src/core/arena.hpp:28-29 — alloc_n size-overflow and alloc offset-overflow allow too-small block handed out (bug)
- src/core/rng.hpp:39 — Pcg32::bounded(0) is modulo-by-zero UB (bug)
- src/core/config.hpp:64-67 — Config::get_int round-trips through f64; out-of-range cast is UB, large s64 lose precision (correctness)
- src/core/core.hpp:61 — Context::cancelled is a plain bool; cross-thread cancellation is a data race (concurrency)
- src/core/test.hpp:27-30 — Test-harness failure counter not atomic; CHECK inside parallel_for can lose increments → false PASS (hygiene)
- src/core/core.hpp:40-51 — FENIX_ASSERT defined after leaf headers; Volume/Arena/sampling can't use it, no debug bounds checking (design)

## eval-topo-postproc.md (src/eval/, src/topo/, src/postproc/)
- src/eval/eval.hpp:172 — `eval-set --baseline` advertised as CI regression gate but silently ignored (hygiene)
- src/postproc/morph.hpp:20 vs src/geom/morphology.hpp:14 — Duplicate `majority_filter` with divergent border semantics (design)
- src/postproc/morph.hpp:100-128 — ball_dilate/erode/close/open are O(n·r³) brute force; geom's EDT gives O(n) (performance)
- src/eval/metrics.hpp:17,31,53 — Serial full-volume loops in metric hot paths (dice/iou/voi, jacobian_fold_fraction, threshold/peak) (performance)
- src/eval/deformation.hpp:14-17 — jacobian_fold_fraction never checks dz/dy/dx dims match (hygiene)
- src/eval/eval.hpp:70-92 — score_pair holds both full f32 volumes alive through entire scoring pass (performance)

## geom-flatten-render.md (src/geom/, src/flatten/, src/render/)
- src/geom/edt.hpp:19-44 — EDT in f32 not exact beyond ~4096-voxel distances; violates "EDT is exact" invariant (correctness)
- src/render/render.hpp:35 — render stage widens u8 archive to whole in-core f32 volume and writes f32 output (performance)
- src/flatten/slim.hpp:151-152 — slim.hpp claims SLIM/local injectivity but implements plain ARAP with no fold-over guard (correctness)
- src/render/unroll.hpp:34,37-45 — unroll accumulates mean CT in f32 (count saturates at 2^24); also fully serial triple loop (correctness)
- src/geom/marching.hpp:57 — marching_tetrahedra vertex indices truncate to s32 with no overflow check (bug)
- src/geom/CLAUDE.md:5-6 — claims toolkit "Implemented in full"; most of it does not exist (hygiene)
- src/render/render.hpp:22,25-31 — render CLI usage/description/includes contradict what stage does; arg parse errors ignored (hygiene)
- src/geom/mesh.hpp:52-59 — read_obj silently keeps only first 3 indices of polygon faces (correctness)

## io.md (src/io/)
- src/io/zarr.hpp:290-297 — copy_zarr_region_local chunk writes are unchecked and non-atomic (resource-safety)
- src/io/zarr.hpp:69-89 — Zarr dtype validation missing; 8-byte and big-endian dtypes silently decode as garbage (correctness)
- src/io/zarr.hpp:92-111 — fill_value never parsed from .zarray; nonzero-fill zarrs read missing chunks as 0 (correctness)
- src/io/slice.hpp:137,141-185 — slice/video CLI std::stoll/stof/stoi throw → abort under -fno-exceptions; slice index unbounded → OOB read (bug)
- src/io/nrrd.hpp:50,70-71 — NRRD header sizes unvalidated; negative/overflowing `sizes` drive allocation/read with bogus count (bug)
- src/io/surface.hpp:65 — .fxsurf reader: nu*nv can overflow s64 before the bounds check (correctness)
- src/io/jpeg.hpp:122-132 — fdct8x8 lazily inits function-local static matrix with non-atomic flag; data race if two threads encode JPEGs (concurrency)
- src/io/slice.hpp:210-233 — video_cmd: unchecked fwrite to ffmpeg pipe; early ffmpeg death → SIGPIPE kills process; out-path unescaped into shell (bug)

## ml.md (src/ml/)
- src/ml/weights.hpp:70-81 — .fxweights parser: unbounded record walk + u64 overflow in blob range check + shape unvalidated → OOB/SIGBUS (correctness) [skipped, out of pass scope]
- src/ml/infer.hpp:365-371 — No NaN/Inf guard on fp16 forward output; one overflowed patch poisons overlap neighborhood; f32→u8 cast of NaN is UB (correctness)
- src/ml/inference.cpp:180-184 — CLI parsing: std::stoi on positional slots crashes on keyword args; ckpt_every=0 → division by zero (bug)
- src/ml/augment_cli.hpp:44-47 — `fenix augment` writes f32-coded .fxvol; violates u8-native/never-f32-on-disk rule (hygiene)
- src/ml/infer.hpp:113-136 — Default-on checkpointing: serial O(voxels) max-scan + quantize + multi-GB write on GPU consumer thread every 128 tiles (performance)
- src/ml/infer.hpp:596-602 — predict_surface_rots without angle 0 leaves rotation-clipped corners as silent zeros (correctness)
- src/ml/infer.hpp:98-108 — CkptHeader fwritten whole with two uninitialized 4-byte padding holes (hygiene)

## pre-pred-anno-fs.md (src/preprocess/, src/predictions/, src/annotate/, src/fs/)
- src/fs/fxfs.cpp:111-112,136,141 — fxfs silently serves round-clamped u8 for non-u8 archives (and meta.toml lies about it) (correctness)
- src/annotate/umbilicus.hpp:24-26 — Umbilicus::center is a linear scan invoked per voxel via polar() → accidental O(N·Z) in winding hot path (performance)
- src/preprocess/dering.hpp:84-104 — dering pass 1 traverses volume plane-strided (z innermost); both passes single-threaded (performance)
- src/preprocess/deconv.hpp:76 — Wiener deconv with reg<=0 produces 0/0 NaN that fast-math writes out as silent garbage (correctness)
- src/preprocess/guided.hpp:34-38 — guided box_mean is O(N·r) per axis, not O(N) He-Sun-Tang promises; also serial (performance)
- src/predictions/field.hpp:26-33 — predictions::normalize sorts external fields containing NaN (UB); reads flat()[0] of possibly-empty field (correctness)
- src/annotate/umbilicus.hpp:26 — Umbilicus::center divides by zero on duplicate z control points (bug)

## segment.md (src/segment/)
- src/segment/structure_tensor.hpp:40-41 — structure_tensor_sheetness halo can undershoot true stencil radius → reflect-boundary contamination at tile seams (correctness)
- src/segment/trace_surface.hpp:84-86,209-210 — trace-surface/render-sheet CLI stof/stoi/stoll throw under -fno-exceptions; unchecked sscanf reads uninitialized floats (crash)
- src/segment/structure_tensor.hpp:124,138 — ct_sheetness_coarse reads OOB when a dimension < ds; compute_normal_field produces zero-extent volume (bug)
- src/segment/affinity.hpp:25-26,41-44 — build_signed_affinity stores voxel linear indices as s32; overflows above 2³¹ voxels (correctness)
- src/segment/hessian.hpp:37-68 — hessian_sheet per-voxel eigen pass fully serial; materializes 4 whole-volume side buffers per scale (performance)
- src/segment/tracer.hpp:89-99 — trace_patch finite-difference gradient is O(N²) per iteration; f32 step also loses perturbation on large coords (performance)
- src/segment/patch_graph.hpp:137,144-146 — make_patch winding-angle stats break across the atan2 ±π branch cut (correctness)
- src/segment/grow.hpp:854-857 — binkey negative bin coordinates alias across axis fields of packed key near volume origin (bug)
- src/segment/partition.hpp:34-59 — mws_partition mutex lists never deduplicated; blocked() re-finds every stale entry → quadratic (performance)

## sweep-concurrency.md (cross-cutting)
- src/ml/infer.hpp:330-381 — ML pipelined inference: producer thread has no shutdown path; libtorch exception → std::terminate, no stop flag to join early (concurrency)
- src/ml/infer.hpp:263,298,330 — Pipelined inference overlaps two full-width OpenMP teams; 2× CPU oversubscription (performance)
- src/segment/trace_stream.hpp:27-38 — stream_tile_u8 silently converts hard fetch failure into all-zero tile (correctness) [duplicate of segment high finding]
- src/fs/fxfs.cpp:96-113 — fxfs fx_read takes sharded-cache lock per byte; the "current block cached" claim not implemented (performance) [duplicate]

## sweep-errors.md (repo-wide error handling)
- src/io/zarr.hpp:69-89,103 — Zarr dtype never validated; big-endian (`>`) and 8-byte dtypes decode to per-voxel garbage (silent-corruption)
- src/segment/trace_surface.hpp:83-84,209-210 (+ io/slice, preproc_cli, dering, augment_cli; silent-default variant io.hpp:28-32, eval.hpp:57, render.hpp:27) — std::sto* CLI parsing across five modules aborts on malformed args; from_chars sites silently default (no-exceptions)
- src/winding/stitch_stream.hpp:173-186 — windings.txt written without stream-error check; ENOSPC truncates winding assignment silently (silent-corruption)
- src/codec/archive.hpp:531-538 — VolumeArchive::commit() ignores msync failure; "durable checkpoint" contract silently breaks on EIO (resource-safety)
- src/io/nrrd.hpp:70-71 — read_nrrd allocates from unvalidated header dims; malformed header aborts (bad_alloc) or overflows (fuzz-surface)
- src/io/jpeg.hpp:268-272 — write_jpeg never checks fwrite/fclose; truncated image reported as success (resource-safety)
- src/core/volume.hpp:89-98 — Volume<T> construction cannot report allocation failure; CLI-sized giant extents abort (design)

## sweep-oom-ooc.md (memory-scale & out-of-core)
- src/codec/archive.hpp:598 — write_level_ buffers entire level's compressed payloads in RAM before serial commit (performance)
- src/ml/infer.hpp:552 — Rotation/ensemble TTA holds up to four full-volume f32 buffers concurrently (performance)
- src/io/io.hpp:177 — export-scroll air-skip fallback can dense-load near-full-res pyramid level as occupancy map (performance)
- src/geom/connected_components.hpp:30,132 — connected_components allocates 12 B/voxel dense aux arrays; block+halo claim unimplemented for CC/EDT/morphology (design)
- src/codec/archive.hpp:353,360-366 (+ io.hpp:448-455) — finalize() holds O(all-real-chunks) pending vector and walks dense chunk grid; fxinfo repeats dense walk (performance)
- src/winding/patch_field.hpp:235,194-204 — build_eulerian_winding_field: serial full-grid re-gauge reduction inside every GS iteration; per-cell KdTree rasterization (performance)
- src/predictions/field.hpp:26 — predictions::normalize Percentile copies whole field and fully sorts it (hygiene)

## sweep-perf.md (cross-cutting performance)
- src/io/io.hpp:344-355 — export-scroll: all DCT encoding single-threaded on consumer, between two parallel stages (performance)
- src/eval/score.hpp:115-121 — official_score computes dominant 26-conn CC twice per mask; union-find uses 8 bytes/voxel where 4 suffice (performance)
- src/eval/eval.hpp:29-36 (+ io.hpp:389) — eval and transcode decode u8 archives to dense f32 just to threshold/re-encode (performance)
- src/codec/dct_block.hpp:117-122 — quant_one_block heap-allocates two vectors per 16³ block in encode hot loop (performance) [duplicate of codec finding]
- src/codec/archive.hpp:45-58 — crc32c is bytewise software table on every chunk write AND read (performance)
- src/io/zarr.hpp:31-33,181-196,74-89 — local zarr chunk read is istreambuf byte-copy; chunk scatter does per-voxel dtype dispatch and bounds tests (performance)
- src/ml/infer.hpp:357-368 — Inference H2D/D2H uses pageable memory and synchronous copies (no pinned staging) (performance)
- src/codec/archive.hpp:401-444 — block16 cache has no in-flight miss dedup; concurrent misses decode same 64³ tile up to 64× (performance)
- src/ml/infer.hpp:276-277,76 — pct_bounds forces full patch copy per tile on ink-norm path (performance)

## winding.md (src/winding/)
- src/winding/patch_field.hpp:389 — fill_surface_from_field Laplacian warm-start is O(hole²); ~10¹⁰ ops on max_hole-sized hole (performance)
- src/annotate/umbilicus.hpp:24-26 — Umbilicus::center linear scan; O(#slices) inside fit's innermost loop (performance)
- src/winding/winding_field.hpp:32 — winding_init chirality disagrees with SpiralModel::winding_at and documented shifted_radius convention (correctness)
- src/winding/diffeo_fit.hpp:71-77 — winding_backward silently assumes identity gap-expander while forward applies gap.inverse (correctness)
- src/winding/stitch_stream.hpp:173-185,303-315 — stitch_streamed never verifies windings.txt write succeeded; missing fragments silently assigned gauge-min winding (resource-safety)
- src/core/optimize.hpp:47 — AdamW keeps optimizer state in f32, violating stated f64-optimizer-state policy (hygiene)
- src/winding/diffeo_fit.hpp:117 — fit_loss ignores r_min while gradient path enforces it; reported loss and optimized objective differ (correctness)
- src/winding/patch_field.hpp:345-363 — fill_surface_from_field does grid index math in int; overflows for surfaces beyond ~2³¹ cells (bug)
- src/winding/winding.hpp:6-12 — module umbrella omits half the module (diffeo_fit, patch_field, cosegment, stitch_stream, fit_bridge never reach unity TU) (hygiene)

---

## Themes

The medium/low findings cluster into a small number of recurring defect classes:

**Untrusted-input / robustness gaps (largest class).** Missing validation on every external-parse surface: zarr dtype/endianness/fill_value, NRRD/`.fxsurf`/`.fxweights` header sizes with s64/u64 overflow before bounds checks, and OBJ/config numeric fields. These repeatedly combine with the `-fno-exceptions` build to turn `std::sto*` throws and `vector`/`operator new` allocation failures into process aborts rather than typed `Expected` errors — a cross-cutting refactor (a shared `from_chars`-based `parse<T>` helper, nothrow allocation) would close dozens of them at once.

**Silent data corruption / error-swallowing.** The project's cardinal "absent ≠ fetch-failed, never silent air" rule is violated at several seams (local `fetch_object`, short chunk blobs, `stream_tile_u8`), and unchecked writes (`copy_zarr_region_local`, `windings.txt`, `write_jpeg`, `commit()` msync) report success on ENOSPC/EIO. NaN handling under `-ffast-math` (predictions sort, fp16 forward, Wiener reg=0) is a related silent-garbage sub-theme.

**Performance — serial-where-siblings-are-parallel, and redundant work.** Many hot paths are single-threaded triple loops (dering, hessian, eval metrics, unroll, checkpoint scan) while neighboring code uses `parallel_for_z`; algorithmic O(n·r³)/O(hole²)/O(N²)/O(N·Z) loops where O(n)/O(log n)/analytic forms exist (ball morphology, Laplacian fill, FD tracer gradient, umbilicus linear scan, eigensolver sweep count); and pervasive u8→f32 whole-volume widening and per-block heap churn that inflate both time and RAM.

**Concurrency latent hazards.** Non-atomic flags/counters (Context::cancelled, test failures, JPEG static init, ML producer with no stop path) that are safe only by current single-threaded usage.

**Numeric-width and index-overflow bugs.** s32 linear indices / int grid math that overflow above 2³¹ voxels (affinity, marching, patch_field, config get_int), against a 2¹⁸-per-axis design envelope.

**Hygiene / documentation drift.** Stale or false CLAUDE.md and code comments (geom "implemented in full", fxfs "block cached", render NRRD usage), dead documented flags (`--threads`, `--baseline` as `|| true`/ignored), incomplete module umbrellas, tautological tests, and tool sprawl.