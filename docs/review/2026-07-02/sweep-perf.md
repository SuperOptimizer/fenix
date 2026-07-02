# sweep-perf — cross-cutting performance review (all of src/)

Scope: performance across module seams on the real hot paths — (1) ingest/transcode (zarr fetch → decode →
DCT encode), (2) inference (archive decode → tensor prep → H2D/D2H → accumulator), (3) eval metrics on
1024³+, (4) codec encode/decode SIMD coverage.

**Overall assessment.** The hot paths are in genuinely good shape: the DCT-16 panel SIMD, tile-parallel
encode in `write_level_`, the decode-once-to-dense-u8 inference input, the factorized Gaussian weight
profiles, the prep/GPU pipelining, and the multithreaded eval primitives are all real, measured wins, and
the module docs are honest about what remains. The findings below are the seams that are still
single-threaded or redundant where the surrounding stages are parallel: the default-on inference
checkpoint stalls the GPU pipeline for the full serialize+write; `export-scroll` funnels all DCT encoding
through one consumer core; `official_score` computes the dominant CC labeling twice per mask with an
8-byte/voxel union-find; and several read paths still widen u8 to f32 or do per-voxel dispatch work that
the rest of the codebase already learned to avoid.

## [high/performance] Default-on inference checkpoint blocks the GPU pipeline with a serial O(voxels) scan + full-accumulator write

**Verdict:** CONFIRMED — All cited facts verified. src/ml/infer.hpp:113-136: ckpt_save does a serial max-scan over nvox floats (line 115), serial u16 quantize (124-130), and synchronous fwrite+fflush of nvox*2 bytes. It is invoked synchronously from the batched consumer loop at infer.hpp:377 (via save_ckpt, lines 254-257) with the full accumulator (prob.data(), d.count()); the consumer is the sole GPU-submitting thread and has already synchronized via .to(kCPU) at line 368, and the producer blocks once the 2-slot ring (line 322) fills, so the GPU is idle for the entire save. Default-on confirmed at src/ml/inference.cpp:205-208 (ckpt_path = outpath + ".ckpt" unless ckpt=off); ckpt_every=128 default at infer.hpp:71. At 2048³/patch=256/overlap=0.5 that is 3375 tiles → 27 saves × (two serial 34 GB passes + 17 GB write). No doc, ADR, or commit message (67f60d5, 33ec18b) documents this cost as measured-and-accepted; the inline "save cadence, not per-tile" comment amortizes but does not bound the stall, and total checkpoint cost grows superlinearly with volume size. The scenario is reachable on the default path; only correction is that 10-25 min is the pessimistic end (~5-15 min is more typical at 2048³ on NVMe) — same order of magnitude.

**Fix notes:** Fix direction is sound with corrections: (1) only the fwrite can be backgrounded — the snapshot quantize must finish before the consumer resumes scattering (accumulator mutates), so run the max-scan and quantize inline but parallelized (parallel_for), then hand the u16 buffer to the writer thread; (2) the preallocated u16 snapshot buffer adds +2 bytes/voxel (17 GB at 2048³) — allocate lazily on first save and account for it in the RAM byte budget per the out-of-core rule; (3) keep temp+rename atomicity and capture header n_done together with the snapshot; (4) at successful completion (infer.hpp:385) join/cancel any in-flight writer before std::remove, otherwise the writer can recreate the .ckpt after deletion; (5) "skip if previous write in flight" is safe since the older renamed checkpoint remains valid. Simpler than scaling ckpt_every with tile count: a wall-clock cadence (save at most every N minutes) directly bounds checkpoint overhead as a fraction of runtime. Also apply the same treatment (or at least the parallel scan/quantize) to the serial B==1 path's save at infer.hpp:428.

**Location:** src/ml/infer.hpp:113-136 (ckpt_save), src/ml/infer.hpp:377 (call site), src/ml/inference.cpp:207 (default `ckpt_path = outpath + ".ckpt"` — resumable by DEFAULT)

**Evidence:**
```cpp
inline void ckpt_save(const std::string& path, CkptHeader h, const float* acc, s64 nvox) {
    float mx = 0.0f;
    for (s64 i = 0; i < nvox; ++i) mx = std::max(mx, acc[static_cast<usize>(i)]);   // serial max scan
    ...
    for (s64 i = 0; i < nvox && ok; ) { ... buf[j] = static_cast<u16>(...); ok = std::fwrite(...); }
```
called from the batched consumer loop:
```cpp
if (ckpt_on && (n_tiles % opt.ckpt_every < nb)) save_ckpt(done);
```

**Failure scenario:** checkpointing is on by default (`run_predict` sets `ckpt_path` unless `ckpt=off`).
`ckpt_save` runs synchronously on the consumer thread that also drives the GPU forward; the prep ring has
only 2 slots, so the producer fills them and stalls too — the GPU sits idle for the whole checkpoint. The
max-scan and u16 quantize are single-threaded loops over the full accumulator, plus an fwrite of
nvox·2 bytes. At 2048³ that is a serial pass over 8.6 G floats plus a 17 GB file write, ~26 times per run
(3375 tiles / ckpt_every=128) — roughly 10-25 minutes of dead GPU time added to every default whole-volume
predict, growing with volume size exactly where inference is already slowest.

**Suggested fix:** (a) parallelize the max-scan and the quantize with `parallel_for` (they are trivially
data-parallel); (b) move the fwrite off the consumer: snapshot into a preallocated u16 buffer (parallel
quantize), then write on a detached checkpoint thread while the consumer keeps feeding the GPU (one
in-flight checkpoint at a time, skip if the previous one is still writing); (c) scale `ckpt_every` with
tile count so the checkpoint overhead is a bounded fraction of the run.

**Outcome:** fixed per the fix-notes, with the noted corrections applied: (1) `ckpt_max` and
`ckpt_quantize` (src/ml/infer.hpp) parallelize the max-scan and quantize with `parallel_for`,
running INLINE on the consumer thread (not backgrounded) since the accumulator is being mutated
again immediately after — only the `fwrite` (in `ckpt_write`) is handed to a background thread via
the new `CkptWriter` struct; (2) `CkptWriter::buf` is allocated lazily on first `save()` call
(`if (buf.size() != nvox) buf.assign(...)`), not preallocated at construction, so the +2 bytes/
voxel snapshot cost is only paid once checkpointing actually triggers; (3) `ckpt_write` keeps the
temp+rename atomicity and writes `n_done` as part of the same header struct captured with the
snapshot (no separate write); (4) both success-completion sites (`predict_surface_filled`'s
batched-path and serial-path returns) now call `ckpt_writer.join()` before `std::remove`, so an
in-flight background write can't recreate `.ckpt` after the success-path delete; (5) `save()`
returns immediately (no-op) if a previous write is still `busy`, matching "skip if previous write
in flight" — the next cadence tick catches up. Did NOT implement "scale ckpt_every with tile
count" or a wall-clock cadence bound — the background-write approach already removes the GPU-stall
problem at the root (the consumer only pays for the parallelized scan+quantize, not the multi-GB
fwrite), so a cadence heuristic seemed like added complexity without added benefit; happy to add
one if profiling on the actual box shows the scan+quantize itself is still a meaningful fraction
of wall time. Also applied the same treatment to the serial (B==1) path's `save_ckpt` call, per
the fix-note's closing instruction. Not build- or perf-verified — the GPU box
(root@162.43.172.181:13280) refused the SSH connection during this session; this is a manual
correctness review only, not a measured before/after. A follow-up session should build on the box
and re-measure the checkpoint-save wall-clock cost at a realistic volume size.

## [medium/performance] export-scroll: all DCT encoding is single-threaded on the consumer, between two parallel stages

**Verdict:** unverified (medium/low)

**Location:** src/io/io.hpp:344-355 (consumer tile loop → `a.write_chunk`), contrast src/codec/archive.hpp:587-626 (`write_level_` two-phase parallel encode)

**Evidence:**
```cpp
// consumer thread (the only one calling into the archive):
for (s64 tz = 0; ...)
    for (s64 ty = 0; ...)
        for (s64 tx = 0; ...) {
            for (s64 z = 0; z < T; ++z) ... block[...] = rv(std::min(...), std::min(...), std::min(...));
            if (auto w = a.write_chunk(0, tc, block); !w) ...   // encode_tile_dct runs HERE, serially
```
`write_chunk_` calls `encode_tile_dct<T>` inline — the full two-pass quant+cluster+RDOQ+rANS encode — one
tile at a time on the single consumer thread. The N producer threads only fetch.

**Failure scenario:** the pipeline is tuned for an ~8 MiB/s WAN link (per the comment), but the tool also
takes local zarr roots and in-region/high-bandwidth sources. There, throughput caps at one core's encode
rate: the module doc's ~2.5-3.2 GB/s encode figure is tile-parallel; a single core is roughly 1/nᵗʰ of
that (~150-300 MB/s), so a local-disk export of a multi-TB level runs an order of magnitude slower than
the machine allows while every other core idles. The per-voxel gather with three `std::min` per voxel adds
to the same serial budget. Note `ingest`/`transcode` already solved this exact problem with
`write_level_`'s parallel-encode/serial-commit split — export-scroll bypasses it.

**Suggested fix:** split the consumer into (1) a parallel encode stage — for each fetched region, encode
its 64³ tiles with `parallel_for` into `(ChunkCoord, payload, zero)` buffers (exactly the `write_level_`
phase-1 pattern; `commit_encoded_` is already factored out for this) — and (2) the serial
`commit_encoded_` + `commit()` writer. The archive stays single-writer; only the encode fans out.

## [medium/performance] official_score computes the dominant 26-conn CC twice per mask; union-find uses 8 bytes/voxel where 4 suffice

**Verdict:** unverified (medium/low)

**Location:** src/eval/score.hpp:115-121 (official_score), src/eval/score.hpp:34-35 (voi_union CC), src/eval/score.hpp:99-100 (betti_numbers → CC again), src/geom/connected_components.hpp:31 (`std::vector<s64> parent`)

**Evidence:**
```cpp
inline Score official_score(...) {
    s.surface_dice = nsd(pred, gt, tau);
    s.voi_score = eval::voi_score(voi_union(pred, gt).total());   // CC(pred,26) + CC(gt,26)
    s.topo_score = topo_score(pred, gt);                          // betti_numbers → CC(pred,26) + CC(gt,26) AGAIN
```
and in `connected_components`:
```cpp
std::vector<s64> parent(static_cast<usize>(n));   // 8 B/voxel
...
std::unique_ptr<s32[]> root_label(new s32[static_cast<usize>(n)]);
```

**Failure scenario:** CC is the acknowledged dominant cost of an eval (design doc: 1024³ eval = 23.3 s
after the multithreading fix). `official_score` labels each mask's foreground 26-connected twice — once in
`voi_union`, once in `betti_numbers` (b0) — so ~2 of the 4-6 CC invocations per pair are pure recompute,
directly inflating every `eval`/`eval-set` run (the distillation loop runs these constantly). Separately,
the union-find `parent` array is s64: 8.6 GiB transient per 1024³ mask (and 68 GiB at 2048³, i.e.
infeasible) on an algorithm that is memory-bandwidth-bound — for any volume under 2³² voxels a u32 index
halves both footprint and traffic.

**Suggested fix:** compute `CcResult` for pred and gt once in `official_score` and pass the labelings into
`voi_union` and a `betti_numbers` overload that accepts a precomputed b0/labels. Template the union-find
index type on `n < 2^32` (u32 fast path, s64 fallback) — labels are already s32.

## [medium/performance] eval and transcode decode u8 archives to dense f32 just to threshold / re-encode

**Verdict:** unverified (medium/low)

**Location:** src/eval/eval.hpp:29-36 (load_f32 → `read_volume()`), src/io/io.hpp:389 (transcode_vol → `read_volume(0)`)

**Evidence:**
```cpp
inline Expected<Volume<f32>> load_f32(const std::string& path) {
    if (...".fxvol") { auto a = codec::VolumeArchive::open(path); ...
        return a->read_volume();   // f32 widen — even when src_dtype is u8
```

**Failure scenario:** every eval pair loads pred+gt as dense f32 — 8.6 GiB at 1024³ (vs 2.1 GiB native
u8), 68 GiB at 2048³ — then immediately thresholds to u8 masks. The archive already has the native path
(`read_volume_as<u8>`, used by inference.cpp) and the decode-to-u8 path is also faster (smaller D2 stores,
smaller cache footprint). Same pattern in `transcode`: a 2048³ u8 archive is widened to a 34 GiB f32
volume before re-encoding (and before the scale255 narrowing). This is also the one place left on a hot
path that violates the project's no-u8-widening rule. The `threshold`/`peak` helpers are additionally
serial single-threaded loops over the full volume (3 passes at 1024³+).

**Suggested fix:** in `load_f32`/`score_pair`, branch on `src_dtype()==u8` and threshold directly from the
u8 volume (threshold value scales trivially); parallelize `threshold`/`peak` with `parallel_for`. In
`transcode_vol`, read `read_volume_as<u8>` when the source is u8 and re-encode natively.

## [medium/performance] quant_one_block heap-allocates two vectors per 16³ block in the encode hot loop

**Verdict:** unverified (medium/low)

**Location:** src/codec/dct_block.hpp:117-122

**Evidence:**
```cpp
inline std::pair<s32, u32> quant_one_block(...) {
    std::vector<T> blk(static_cast<usize>(V));          // alloc #1, per block
    for (...) blk[...] = src[...];                       // gather in T
    std::vector<f32> c = to_f32<T>(blk);                 // alloc #2 + a full extra pass
```

**Failure scenario:** this runs once per 16³ block on the encode path the module benchmarks in GB/s: a
2048³ export encodes 2.1 M blocks → ~4.2 M heap allocations plus a redundant gather-then-widen double pass
over every voxel (the data is touched twice before the DCT even starts). Under the tile-parallel
`write_level_` all cores hammer the allocator simultaneously. Encode already regressed to ~2.5-3.2 GB/s
after clustering/RDOQ; this is avoidable overhead on top.

**Suggested fix:** widen directly in the gather loop into a caller-provided/stack `f32 buf[4096]`
(`alignas(64)`), eliminating both vectors and the second pass; `coef`/`lev` are already caller-owned, so
the function becomes allocation-free.

## [low/performance] crc32c is a bytewise software table on every chunk write AND read

**Verdict:** unverified (medium/low)

**Location:** src/codec/archive.hpp:45-58 (crc32c), used at src/codec/archive.hpp:250-251 (read_chunk_as) and 634 (commit_encoded_)

**Evidence:**
```cpp
inline u32 crc32c(const u8* p, usize n, u32 crc = 0) {  // Castagnoli, software table
    ...
    for (usize i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xffu] ^ (crc >> 8);
```

**Failure scenario:** a byte-at-a-time table CRC runs ~0.3-0.8 GB/s/core. It is computed over every
compressed payload on write and again on every `read_chunk_as` — including the decode-once inference input
path and every eval load. At low q (ingest default q=2, ratio only a few ×) the payload bytes are a large
fraction of the decoded bytes, so CRC becomes a visible tax on a decode path measured at ~4.5 GB/s.
The project compiles with `-march=native` on x86-64 and arm64, where hardware CRC32C
(`_mm_crc32_u64` / `__crc32cd`) runs ~10-20 GB/s.

**Suggested fix:** use the hardware CRC32C instruction when available (compile-time `__SSE4_2__` /
`__ARM_FEATURE_CRC32`, both true under `-march=native` on the target boxes), fall back to slicing-by-8.
Same polynomial, no format change.

## [low/performance] Local zarr chunk read is istreambuf-iterator byte-copy; chunk scatter does per-voxel dtype dispatch and bounds tests

**Verdict:** unverified (medium/low)

**Location:** src/io/zarr.hpp:31-33 (fetch_object local path), src/io/zarr.hpp:181-196 (scatter loop), src/io/zarr.hpp:74-89 (cast_dtype re-parses the dtype string per voxel)

**Evidence:**
```cpp
std::vector<u8> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
```
and per voxel:
```cpp
if (gx < origin.x || gx >= origin.x + extent.x || gx >= m.shape.x) continue;
...
v = detail::cast_dtype<T>(data + static_cast<usize>(off) * esz, m.dtype);  // string-dispatch per voxel
```

**Failure scenario:** local zarr is the input to `ingest-zarr` and to `export-scroll`'s occupancy map, and
the same scatter runs for every remote chunk. The istreambuf read has no size reserve and copies through
the streambuf a character at a time (typically a few hundred MB/s); the scatter loop then tests bounds and
re-inspects `m.dtype` (string indexing + size decode) for every voxel of every chunk instead of
intersecting the chunk box with the region once and, for the ubiquitous u8 case, memcpy'ing contiguous
x-runs. For a local-disk whole-level ingest this per-voxel work is the CPU bottleneck of the fetch stage.

**Suggested fix:** local path: `stat`/`std::filesystem::file_size` + one `fread` into a pre-sized vector.
Scatter: compute the clipped `[lz0,lz1)×[ly0,ly1)×[lx0,lx1)` intersection outside the loops; specialize
`T==u8 && dtype=='|u1'` to row memcpy, and hoist the dtype kind/size to two locals for the generic case.

## [low/performance] Inference H2D/D2H uses pageable memory and synchronous copies (no pinned staging)

**Verdict:** unverified (medium/low)

**Location:** src/ml/infer.hpp:357-368 (batched path: `xhost.to(dev)`, `pr.contiguous().to(torch::kCPU)`)

**Evidence:**
```cpp
auto xhost = opt.half ? blob.to(torch::kFloat16) : blob.clone();  // pageable CPU tensor
...
auto xin = xhost.to(dev);                                          // sync pageable H2D
...
auto surf = pr.contiguous().to(torch::kCPU).to(torch::kFloat32)...  // sync D2H
```

**Failure scenario:** each batch moves ~100 MB up (fp16, b=3, 256³) and ~100 MB down. Pageable transfers
run ~5-6 GB/s and serialize with compute; pinned transfers run near link speed (~25 GB/s PCIe4) and can
overlap the forward via `non_blocking=true` on a side stream. That is ~35-40 ms of exposed transfer per
~600 ms batch (~6%) that the existing producer/consumer pipeline cannot hide because the copy happens on
the consumer thread between forwards. The half-precision cast already lands in a fresh CPU tensor, so
making that tensor pinned is free.

**Suggested fix:** allocate the post-cast staging tensor with `.pin_memory()` (or
`torch::TensorOptions().pinned_memory(true)`) and use `to(dev, /*non_blocking=*/true)`; likewise a pinned
D2H destination + `copy_(..., true)` with one `cuda::synchronize()` before the scatter. Measure — the
memory notes show this box responds to transfer-shape changes (batch=3/pipelining did; channels_last
didn't).

## [low/performance] block16 cache has no in-flight miss dedup — concurrent misses decode the same 64³ tile up to 64×

**Verdict:** unverified (medium/low)

**Location:** src/codec/archive.hpp:401-444 (block16 miss path), src/codec/block_cache.hpp:47-59 (put no-ops if present)

**Evidence:**
```cpp
if (auto r = block_cache_->get(key)) return r;
... auto tile = read_chunk_as<u8>(lod, tc);   // full tile decode — no per-tile in-flight guard
... block_cache_->put(k, cr);                  // silently drops if another thread won
```

**Failure scenario:** the streaming consumers (`fs` bridge serving reads, `gather_box_f32` under parallel
patch gather) fan out over 16³ blocks; the 64 blocks of one tile share a single expensive decode
(rANS + 64 IDCTs + deblock). When several threads miss different blocks of the same tile simultaneously —
the common case for a box straddling a tile — each runs the full tile decode and 64 `put`s, all but one
discarded. Worst case is 64 redundant tile decodes for one tile of demand, turning the "decode amortizes
across the tile" design win into its opposite exactly under load.

**Suggested fix:** per-tile in-flight registry (map<tile_key, shared_future/CV> guarded by the shard
mutex, or a small striped `std::mutex` array indexed by tile Morton key): first misser decodes, others
wait and read from cache.

## [low/performance] pct_bounds forces a full patch copy per tile on the ink-norm path

**Verdict:** unverified (medium/low)

**Location:** src/ml/infer.hpp:276-277 (caller), src/ml/infer.hpp:76 (signature)

**Evidence:**
```cpp
std::vector<float> tmp(out, out + PN);          // 67 MB alloc+copy per 256³ patch
float lo, hi; detail::pct_bounds(tmp, 0.5, 99.5, lo, hi);
```

**Failure scenario:** `pct_bounds` only reads its input, but takes `const std::vector<float>&`, so every
ink-inference tile allocates and copies a 67 MB buffer (256³) before the two histogram passes — an extra
~200 MB of memory traffic per tile on the prep stage that the pipeline is trying to keep under the GPU
forward time.

**Suggested fix:** change `pct_bounds` to take `std::span<const float>` (or `const float*, usize`) and
pass `{out, PN}` directly; delete `tmp`.
