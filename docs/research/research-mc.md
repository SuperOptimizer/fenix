# matter-compressor — research report (for a C++26 greenfield rewrite)

Target: `/home/forrest/taberna/third-party/matter-compressor/` — a first-party
(author: SuperOptimizer) C23 library: a lossy block codec + `.mca` on-disk
archive + in-RAM decoded-block cache + S3/HTTP volume-streaming layer, plus a
surface sampler/renderer, all for Vesuvius-Challenge micro-CT scroll data (dense
`u8` scalar volumes, 100 TB+ scale). Consumed by interactive viewers (VC3D) and
ML dataloaders.

This report covers the parts in scope for the rewrite: **codec, .mca format,
volume/streaming, public API, performance, test-locked invariants, and rewrite
recommendations.** The rewrite gets a brand-new, non-compatible file format.

A recurring finding worth stating up front: **the public header
`matter_compressor.h` and several in-file banner comments are STALE.** They
describe a per-voxel air mask, harmonic SOR air-fill, "self-contained mask in
payload", an integer Q14 DCT, a "CLOCK/NRU cache", per-shard mutexes, and a
"c3d/c3g" block codec. **All of those were removed.** Trust the `.c` bodies, not
the header prose. The rewrite should start from the actual behavior documented
below.

Source layout (in-scope files):
- `src/matter_compressor.{h,c}` — public single-header API (1176 lines) + 2-line .c stub.
- `src/mc_codec.{c,h}`, `src/mc_codec_float.h` — the block codec.
- `src/mc_archive.{c,h}` — the `.mca` writer/reader.
- `src/mc_stream.c` — flat + streaming `mc_reader`.
- `src/mc_cache.{c,h}` — in-RAM decoded-block cache.
- `src/mc_volume.c`, `src/mc_zarr.c` — remote streaming/transcode volume.
- (out of scope for the rewrite's storage core but present: `mc_render.c`,
  `mc_segment.c`, `mc_solve.c`, `mc_trace.c`, `mc_surface.c` — sampler/renderer
  and geometry tools.)

================================================================================
## 1. THE COMPRESSION CODEC
================================================================================

### Algorithm: 16³ block, separable 3D DCT-II, dead-zone quant, CABAC

Fixed block = **16³ voxels** (`MC_BLK=16`, `N3=4096`). One archive chunk =
**256³ = 16³ blocks** (`MC_CHUNK=256`). The codec is a transform coder, NOT a
wavelet or learned codec. Pipeline per block (`mc_enc_block`, mc_codec.c:629):

1. **DC removal.** `dc = round(sum/N)` over ALL 4096 voxels (no air concept).
   `blk[i] = vox[i] - dc` as f32. (NEON path vectorizes the sum/count and the
   subtract; scalar path is branchless for auto-vectorization.)
2. **Forward 3D DCT-16** (`mc_dctf3_fwd`, mc_codec_float.h:147). Separable: three
   passes of `mc_dctf1d_fwd` (1D DCT-II, even/odd partial-butterfly, length-16
   only) with a cache-blocked axis rotate (`mc_rotf`, 8×8 tile) between passes.
   Orthonormal cosine matrix `g_mc_cmf` built once in `mc_dctf_init` from doubles
   then stored as f32; inverse is the transpose (DCT-III). The inverse is
   **sparse-aware**: it skips zero coefficients (post-dequant lines are mostly a
   few low-frequency nonzeros), and `dec_block_coefs_ext` reports the nonzero
   per-axis extent so a constant (DC-only) block short-circuits to a `memset`.
3. **Dead-zone quantization.** `step[i] = quality * (1 + L1freq)^MC_HF_EXP`,
   `L1freq = cz+cy+cx`, `MC_HF_EXP=0.65` (higher frequencies quantized coarser).
   Dead zone `MC_DZ_FRAC=0.80` of a step. Encode uses a **reciprocal-step table**
   (`rstep_tab`) so quant is a multiply, not a divide; fused branchless quant+clamp
   to int16 levels. Dequant recon = `sign(l)*step*(|l| - 1 + 0.80 + 0.40)`
   (`mc_deqf_one`).
4. **Entropy coding (CABAC-style binary range coder).** One range-coded stream per
   block, one flush (~5 bytes). Header bins first (all context-coded with trained
   priors): a reserved flag bit (was "mixed", now always 0), a has-corrections
   bit, then the 8-bit DC. Then the coefficients (`enc_block_coefs`): coded in
   ascending-L1-frequency **scan order** (lazily-built per-size scan table
   `g_scanS`), with an adaptive **EOB** (last-significant position, bit-length +
   bypass suffix), per-position **significance** contexts keyed on `(band,
   recent-significance-density)` (8 bands × 4 density buckets), and an
   **adaptive-unary + Exp-Golomb magnitude** ladder (12 contexts) + a bypass sign.
   Then sparse max-error corrections if present.

The range coder (`mc_rangecoder.h` section in mc_codec.c): classic carry-less
range coder, `RC_TOP=1<<24`, 12-bit probabilities, shift-by-4 adaptation
(`p += (4096-p)>>4` on a 0 bit). Bypass bits, batched bypass (`enc_bypass_n`),
order-0 Exp-Golomb. All integer — this half is fully bit-deterministic.

### The training step (mc_train) and trained context priors

Per-block contexts reset every block, so without priors every adaptive bin starts
at p=0.5 and wastes the first ~32 bins. The codec ships **baked corpus-trained
priors** (`RC_PLO`/`RC_PHI`, mc_codec.c:121-140): two endpoint tables (trained at
q=1 and q=12 on PHercParis4 2.4 µm fysics-masked data) of `u16[8 classes][32
slots]`, where the 8 classes are SIG/MAG/EOB/MASK/MASKU/MASKA/FLAG/DC (the three
MASK classes are now dead — air mask removed — but slots remain). `2048` = an
untrained/neutral slot. At a given q, `rc_prior_build_into` interpolates each slot
in **log2(q)** between the two tables into the per-ctx `pri[8][32]`. The decoder
knows q (stored per chunk), so priors cost zero side information.

`tools/mc_train` (built with `-DMC_TRAIN`): runs the encoder with `RC_TRAIN`
macros that count `(class, slot, bit==0)` occurrences, then emits new prior
tables. Output can be pasted into the source OR stored **per-volume** in the
archive (`mc_archive_set_priors`/`mc_codec_set_priors`), in which case every open
decodes with those instead of the baked tables. Priors are **process-global**
(set once before encode/decode threads start; a generation counter forces per-ctx
rebuild) — this matches the "one quality dial per process" design.

### Max-error bound (near-lossless mode)

`mc_codec_ctx_set_max_error(tau)`: after transform coding, the encoder locally
reconstructs the block (one extra inverse DCT), finds every voxel with
`|orig-decoded| > tau`, and codes sparse corrections so `|err| ≤ tau` on every
voxel. **NOTE the integration subtlety**: the actual mc_codec.c path codes
corrections as **INTEGER deltas** (`cdel`, `enc_eg` Exp-Golomb of gap/sign/
magnitude, mc_codec.c:721-729), NOT the f32-residual blob described in
`mc_codec_float.h` (`mc_maxerr_build`). The float-blob helpers in
mc_codec_float.h are an isolated, unit-testable reference that is **not wired
into the archive path**. The decoder applies corrections after clamp+dc
(mc_codec.c:770-782), hardened: `ncorr` and positions clamped so a corrupt stream
can never write out of `dst`.

The **preset ladder** (`g_presets`, mc_codec.c:504): 8 calibrated (q, tau) pairs
where level L guarantees `|err| ≤ 2^L` per material voxel (air bit-exact) at the
ratio-optimal quality. Archival = (q0.5, tau1) ≈ 2.9× (beats zstd-19 lossless
1.96×; there is deliberately no lossless mode); preview = (q64, tau128) ≈ 93×.

### What makes it byte-deterministic — and the IMPORTANT CAVEAT

The README claims "NEON and AVX2 builds produce byte-identical archives" and
"never add -ffast-math (breaks cross-ISA bitstream identity)". The mechanism:
- The **entropy coder is pure integer** → bit-exact everywhere.
- The **DCT/quant are f32**, but compiled with **strict IEEE** (no `-ffast-math`,
  no FMA contraction, fixed evaluation order) → bit-reproducible across ISAs
  *as long as that build discipline holds*.

**The test suite tells a softer story than the README.** `mc_det_archive.c`'s
own banner states cross-platform CI asserts **metric equivalence** (size within
tolerance, PSNR/MAE to a few digits, **max-error exact**), **NOT bit identity**:
"encoder float paths are allowed to differ in last-bit rounding across compilers/
ISAs as long as the delivered quality is identical." And `mc_codec_float.h`'s own
header says f32+fast-math is NOT bit-reproducible and the max-error pass is
"best-effort" against *this build's* reconstruction (a different-ISA decoder may
breach tau silently).

So the real determinism contract is two-tier:
- **Within one build/ISA: bit-identical**, and the test suite hard-asserts this
  for parallel-vs-serial encode (`mc_archive_decode_chunk` memcmp), par-vs-serial
  render/sample, full-vs-partial-fetch streaming decode, flat-vs-streaming reader
  values, frozen-cache-vs-dense sampling, and the seeded box sampler.
- **Across builds/ISAs: metric-equivalent**, not guaranteed bit-identical, unless
  you pin `-ffp-contract=off` + identical evaluation order (which the project does
  for the codec but does not test as a hard cross-ISA byte equality).

This is THE central design tension. The hard constraint "DCT/codec must not be
reordered by fast-math or LTO" is real (it protects the within-build determinism
and the cross-ISA *metric* equivalence), but the rewrite should decide
deliberately whether it wants **true cross-ISA byte identity** (→ make the DCT
integer/fixed-point again, or specify a strict evaluation order + forbid FMA as
an enforced, tested contract) or is content with **metric equivalence** (→ the
current f32 approach is fine and faster).

### Structural-experiment findings (experiments/STRUCTURAL_FINDINGS.md)

Five "obvious" codec improvements were tried on real 1024³ q4 data and ALL
rejected: (1) inter-scale/LOD residual coding is −15…−22% (worse — DCT+DC removal
already captures intra-block low-freq; prediction injects HF, 1.39× AC energy);
(2) cross-block directional intra prediction ~0% (short correlation length); (3)
variable/octree block size 0% (16³ already optimal — no smooth regions for 32³);
(4) shape-adaptive/mask-aware transform N/A (no masking in the pipeline anymore);
(5) persistent cross-block context +0.1% (trained priors already start at
stationary values). Unified root cause: the data is dense, high-frequency,
short-correlation-length texture with no air structure. **Remaining real gains
(untested): RDOQ/trellis quant (encoder-only, decoder-compatible, ~5-15%); better
energy compaction (learned/KLT/directional transform or a 3D-wavelet core);
corpus-trained q-matrix.** These are the only avenues worth codec R&D in the
rewrite.

================================================================================
## 2. THE .mca ARCHIVE / CONTAINER FORMAT
================================================================================

Format version `MC_VERSION = 7` (v7 added the per-chunk material-fraction map).
An **appendable, crash-safe, mmap-backed** archive: one handle both appends and
decodes (no writer/reader split), reopens across process runs and keeps
appending, and is a fully valid decodable archive after *every* appended chunk.

### File layout

```
[0,    256)     header (256B, MC_HDR)
[256,  128KB)   user metadata region (free-form, zero-padded), MC_META_CAP
[128KB, EOF)    archive data: index node/shard tables + chunk blobs,
                BOTH appended at EOF in append order
```

**Header fields** (offsets `MCH_*`): u32 magic `"MCC\0"`; u32 version; u32
NX/NY/NZ (x fastest); **u64[8] per-LOD root-node offset** (0 = empty LOD); u64
total length (= append cursor/EOF, authoritative only at close); u64 metaoff/
metacap/metalen; f32 build quality; u32 block codec (forced 0=CABAC); u64
optional prior-blob offset.

### Hierarchy: 8 independent LODs, each a dense 3-level node tree

8 LODs (LOD0 = full res), each its own tree, each **independently fetchable AND
independently decodable — a HARD constraint (no cross-LOD dependency)**. Per LOD,
a chunk coordinate (voxel>>8, ≤12 bits/axis) decomposes into three 4-bit nibbles
indexing **3 dense levels**: root node → inner node → shard. Each node/shard is a
flat **`MC_GRID3=4096` (16³) array of u64 child offsets** = 32 KB, directly
indexed by the chunk-coord nibble, **slot 0 = absent**. No bitmap, no
rank-packing → every node is **updatable in place**, which is what makes appends
crash-safe. `mc_resolve_chunk` (mc_archive.h:133) is three bounds-checked array
reads. Covers 2^20 voxels/axis.

A shard slot value can be: `0` (absent / never written), `MC_SLOT_ZERO=1`
(VISITED, decodes to all-air — lets prefetch skip re-fetching air), or a real
chunk-blob offset (always ≥ MC_HDR).

### Chunk blob (v7)

```
[f32 q][u64 xxh64][u16 fmaplen][fmap bytes]
[512B block-bitmap][present-block u16 lens][self-contained block payloads]
```
`MC_BLOB_HDR=14`. `q` = the chunk's own quality (per-chunk q is supported for
rate control / ROI). `xxh64` covers fmap+bitmap+lens+payloads (seed
`"mcchunk"`). `fmap` = RC-coded per-block material fractions (4096 nibbles, 0..15
≈ 0..100%) for rejection-free ML sampling. The 512B bitmap is 4096 bits (one per
16³ block); present-block lengths are u16; **per-block payload offsets are
IMPLICIT (prefix sum of present lengths)**. So: one ranged GET fetches a whole
chunk; bitmap+lens+one payload fetches a single block. ZERO blocks cost 1 bitmap
bit; all-air chunk → no blob at all (slot stays absent/zero).

The O(n²) prefix-sum-per-block lookup is avoided by a **thread-local per-chunk
index** (`mc_chunk_index`, mc_archive.h:176) built in one O(4096) pass, keyed on
`(arc, chunk_off, content-hash)` — the hash tag is mandatory because chunk_off is
reused across archives/re-appends (same tree position → same EOF offset).

### Append path, commit-word ordering, crash safety

- **Large virtual reservation** mmap'd once (`MC_RESERVE` default 10 TiB,
  `MAP_NORESERVE`; `-DMC_RESERVE` shrinks it for sanitizers). `base` never moves
  → all offsets stay valid lock-free as the file grows.
- **Growth** (`w_ensure`): rounded by `MC_GROW_STEP=1GiB`, and uses
  **`fallocate` not `ftruncate`** to allocate real blocks — a sparse-mmap write
  defers allocation to page-fault time where ENOSPC raises an *uncatchable
  SIGBUS* (an observed crash). Serialized by `grow_mu` (growth only; decode never
  touches it).
- **Allocation**: `w_alloc` = `atomic_fetch_add(&cursor)` (lock-free bump);
  `w_alloc_aligned` = CAS loop for 8-aligned index nodes (the u64 child slots are
  read as `_Atomic uint64_t` → ldapr/stlr → **bus-fault on misaligned addresses**
  on aarch64; this is a tested invariant, `mc_node_align_test.c`).
- **The commit word** (`w_install_blob`, mc_archive.c:458, under `write_mu`):
  payload bytes are `memcpy`'d to EOF FIRST, then the chunk offset is published
  into the shard slot with an **atomic release store**:
  ```c
  atomic_store_explicit((_Atomic uint64_t*)(base+slot), off, memory_order_release);
  ```
  Any concurrent reader that acquire-loads a non-zero slot is guaranteed to see a
  fully-written blob. The file is valid after every such publish. `gen` is then
  bumped (`atomic_fetch_add release`) as a change-epoch other layers poll;
  `MCH_TOTLEN` is updated with a relaxed (approximate) store, made authoritative
  only at `mc_archive_close` (which also `msync`s and `ftruncate`s back to the
  exact used size).

### Concurrency model

- **Lock-free**: all reads/decodes, the cursor bump, the coverage memo, and all
  node-child loads (acquire, paired with the publish release).
- **Locked**: `grow_mu` (file growth), `write_mu` (index mutation + slot publish
  — serialized because the lock-free node-create had a concurrent-first-touch
  race that could orphan a subtree; `mc_archive_concurrent_test.c` asserts no
  lost nodes under 16-thread racing appends).
- **Coverage memo `cov`** (mc_archive.c:258): an open-addressed, atomic-slot,
  power-of-two (`MC_COV_CAP=1<<20`) resident-region set so frozen render reads get
  O(1) PRESENT/ZERO/ABSENT without a per-block tree walk (the tree walk was a
  measured 49 ms render cost). Key packs `lod<<36 | cz/cy/cx`; state in the top 2
  bits. Never resizes. Backfilled on miss from the tree.

### Read / decode / region API

- `mc_archive_chunk_offset` → blob offset (tree walk, file_len-bounded).
- `mc_archive_decode_block` → one 16³ block; uses a **per-thread codec ctx in a
  pthread_key** (with a free-on-exit destructor — a bare `_Thread_local` leaks
  ~400 KB per render-worker join, and a per-call alloc re-runs the 4096-powf step
  table). Bounds the block byte range against the live `cursor` (SIGBUS-safe on
  corrupt length tables).
- `mc_archive_decode_chunk` → whole 256³, internal worker pool over 4096
  independent blocks (atomic `next` counter, one ctx per worker, auto threads cap
  16). Bit-identical to serial.
- `mc_archive_read_region` → arbitrary axis-aligned box into a **strided** output
  (pass torch/numpy strides for zero-copy tensor fill); only touched blocks
  decoded, in parallel; out-of-volume + absent reads as 0.
- `mc_archive_block_blob` → pointer into live mmap to a block's raw compressed
  bytes (no decode) for GPU upload paths.

### ML sampling

- `mc_archive_block_present` (bitmap bit) and `mc_archive_block_fraction`
  (fraction-map nibble /15, decoded once per chunk per thread) — no voxel decode.
- `mc_archive_sample_boxes` — **deterministic seeded** box sampler (xorshift64),
  draws boxes uniformly, accepts only those whose mean block fraction ≥
  `min_frac` (material-rich rejection via the fraction map). Same seed → same
  boxes regardless of thread count/machine (tested).
- `mc_archive_read_regions` — batch multi-crop fill, workers process whole crops
  in parallel; the ML dataloader primitive.

### Append variants (public)

`mc_archive_append_chunk_raw` / `_raw_q` (per-chunk q) / `_ctx` (caller ctx, the
efficient tight-loop path with tau) / `_par` (internal parallel encode, byte-
identical to serial — striped so workers write disjoint bitmap bytes, then
stitched + hashed) / `_target` (rate targeting: a 1/16-block sample encode + a
single power-law `bytes∝q^-0.75` correction, ~6% overhead, lands within ~10-20%)
/ `_compressed` (verbatim blob copy, the .mca→.mca fast path). Plus
`mc_archive_reserve_index` (pre-create the index path so an exporter can cluster
the whole index at the front for one-GET streaming startup — what `tools/mc_export`
exploits, repacking Morton-ordered with coarse LODs clustered up front).

### Integrity

`mc_chunk_compute_hash` recomputes xxh64; `mc_verify_archive` walks all 8 LOD
trees, bounds-checks every offset, validates each blob with `mc_blob_struct_ok`,
and compares stored vs recomputed hash, returning the corrupt-chunk count.
(FINDINGS.md: mc_verify itself once crashed on corrupt input — now hardened.)

================================================================================
## 3. THE VOLUME / STREAMING API
================================================================================

Two byte-source tiers and two volume entry points.

### Readers (mc_stream.c)

`mc_reader` reads an already-built archive either **flat** (`mc_open`, mmap'd
buffer, bounded by `len`) or **streaming** (`mc_open_streaming`, pulls byte ranges
through an `mc_read_fn` callback on demand). Streaming caches: a FIFO of the last
512 node tables (16 MB — resolving a chunk is 3 dependent 32 KB reads), the
current chunk blob window (`cbuf`), and (partial mode) a per-chunk header cache.

**Partial-fetch mode** (`mc_reader_set_partial_fetch`): a block decode fetches
only the chunk's bitmap+lens (≤8.7 KB, cached) + that block's own payload (~<100 B)
instead of the whole chunk blob — far lower cold random-access cost over S3.
Tested **bit-identical** to full-fetch, with >4× fewer bytes for sparse access.

`mc_chunk_offset_chk` distinguishes "resolved-absent (air)" from "node-table read
FAILED (network)" via an `err` flag — a streaming caller that maps offset 0 to
permanent air MUST use it or a transient error poisons the region as ZERO forever.

### mc_zarr — source-format reader (mc_zarr.c)

Standalone zarr reader behind a byte-source callback. **Important: c3d/v3-sharded
support was REMOVED** despite the docs/header describing it. `mc_zarr_open` now
*rejects* v3-sharded zarrs; only **v2-flat raw/blosc** (blosc shuffle
unsupported) survives, and inner edge must be 128 or 256 (sub ∈ {1,2}, since the
transcode sub-chunk arrays are sized `[8]=2³`). The v2 reader fetches+decodes a
chunk to dense u8 before returning. (The design doc `docs/mc_volume_design.md`
describing vendoring c3d is **out of date**.)

### mc_volume — remote volume as a local .mca

`mc_volume_open(url, cache_dir, cache_bytes, quality)`: URL with `://` → remote
(s3/https through libs3), else local filesystem zarr. **LOD discovery** = probe
`<root>/0`, `/1`, … by directory number (NGFF-multiscales-by-convention, NOT by
parsing a multiscales metadata block) until a gap. **S3 creds**: anonymous-first,
signed-retry (open-data buckets reject wrongly-signed requests). The local
`<name>.mca` in `cache_dir` is the **persistent transcode cache** for ALL LODs;
`cache_bytes` is the resident decoded-block RAM budget (the .mca itself grows
**unbounded** — no disk eviction).

`mc_volume_open_streaming`: when the URL is already a built `.mca`, chunks are
copied **verbatim** (`mc_archive_append_chunk_compressed`) — no decode/re-encode.
`mc_mca_probe` sniffs a remote .mca header (dims/nlods/q) cheaply.

**Transcode pipeline** (zarr path): a cold region → batched ranged GET(s) of the
source chunk(s) → assemble dense 256³ (sub-chunk blit for inner edge 128) →
`mc_archive_append_chunk_raw` re-encodes into the local .mca → coverage flips
PRESENT. Three thread teams: **download** (default 8, each drains a LIFO-stack
share into one `s3_get_batch` of 32-way concurrent GETs), **decode/transcode**
(default nproc/2 — memory-bandwidth bound; each worker owns persistent 16 MB
dense/tile scratch, 64-aligned), and a **fill pool** (archive→RAM-cache decode,
shard-partitioned, single-owner-per-shard, no locks).

**Block serving**: `mc_volume_try_block` (async/lock-free render path: ABSENT →
record a *region-granularity* miss + return 0 zeros; ZERO → zeros + 1; PRESENT →
cache copy + 1). `mc_volume_get_block` (blocking CLI: `ensure_region` spin-waits
up to ~100 s polling coverage, then `fill_pool_wait` before reading to avoid
racing the lock-free fill workers).

**Single-flight** (all under `v->mu`): an `inflight[]` region set (popped, being
fetched, not yet appended — the actual dedup; cited ~15× re-decode without it), an
open-addressing `rs_set` for O(1) push-dedup against the LIFO request stack, and
the request stack itself. "Done" is determined purely by archive coverage
flipping PRESENT/ZERO — there is no explicit completion signal.

**Streaming throughput** (`mc_stream_fetch_batch`): the standout engineering —
resolves all blob offsets, **sorts + coalesces near-adjacent blobs into a few
large sequential GETs** (gap ≤ 512 KB, run ≤ 64 MB), with an EMA-sized over-read
margin for the trailing blob + a single follow-up GET for the occasional tail.
Turns the per-chunk 3-4-round-trip ~2 MB/s ceiling into bulk sequential reads.

**Decode-queue backpressure** is **byte-budgeted** (`staging_bytes`, default 2 GB)
not slot-counted — downloads run far ahead of the CPU-bound decode pool to keep
the network saturated.

**Render clock (freeze/thaw)**: `mc_volume_freeze` makes the resident cache
immutable for a frame (lock-free reads); `mc_volume_thaw` reopens writes, drains
the frame's misses (PRESENT → fill pool, overlapping the render; ABSENT →
`req_push`), bumps the pin epoch, and collates the pipeline-depth stats snapshot.
`mc_volume_render_gen` / `mc_volume_region_gen` (direct-mapped, keyless, 64 K
slots) let a viewer skip provably-identical frames per-viewport.

### Caching (mc_cache.c)

In-RAM **decoded-block** cache: an mmap arena of **4 KB slots** (one 16³ block =
4096 B = one page), **64-way sharded** (top hash bits select shard, low bits the
intra-shard open-addressing bucket; per-shard arena slice). Default eviction is
**S3-FIFO** (SOSP'23: small/probation ring + main ring + ghost fingerprint ring;
ghost-hit → admit straight to main; scan-resistant for slice sweeps); CLOCK
optional. Per-shard parallel arrays (`slot_key`, `slot_ref`/freq, `slot_inmain`,
`slot_epoch`); 20-bit coords + 3-bit LOD packed into a guarded key (so 0 = empty).

**No per-shard locks** (the header's "one mutex each" is stale). Safety is by
**phase discipline**: frozen phase = unlimited lock-free readers, zero writers
(arena is an immutable snapshot; acquire/release on the `frozen` flag pairs all
fill-phase writes); thaw phase = writers partitioned single-owner-per-shard.
**Pins**: `cache_touch` stamps `slot_epoch=epoch`; eviction treats epoch-matching
slots as hot; `thaw` does `epoch++` to silently un-pin everything. A `force`
budget guarantees a victim even if all slots are pinned.

API: `mc_cache_new_archive`, `mc_cache_get` (zero-copy hit; frozen miss → NULL,
record miss; unfrozen miss → decode-in-place + insert), `mc_cache_get_copy`,
`mc_cache_update` (blocking batch fill, shard-bucketed, join = barrier),
`mc_cache_best_lod` (finest resident ancestor LOD — render-now-refine-later),
`mc_cache_resolve` (update + pointer table + epoch-pin), async tickets (cancel at
chunk-group boundary, not mid-decode), `mc_cache_miss_mark`/`misses_drain`
(lock-free dedup miss queue, 64 K slots), `mc_cache_resize` (live, discards
resident blocks). The archive binding memoizes the chunk offset thread-locally
keyed on `a->gen` so a growing archive invalidates stale offsets without locking.

================================================================================
## 4. PUBLIC API SURFACE & KEY DATA STRUCTURES
================================================================================

Single header `matter_compressor.h`, four logical modules + sampler/renderer +
zarr/s3/volume. Opaque handles: `mc_codec_ctx`, `mc_archive`, `mc_reader`,
`mc_cache`, `mc_cache_ticket`, `mc_sampler`, `mc_lod_sampler`, `mc_zarr`, `mc_s3`,
`mc_volume`. Key value structs: `mc_buf` (growable byte sink), `mc_build_opts`,
`mc_box {z0,y0,x0}`, `mc_block_id {lod,bz,by,bx}`, `mc_cache_stats`,
`mc_cover {ABSENT,PRESENT,ZERO}`, `mc_sample_src` (the pluggable block source:
`block()` returning a stable or `tmp`-filled 4096-byte pointer, an optional cheap
`resident()` probe, `owns_ptr`, and an optional `dense` direct-addressing path),
`mc_sample_lods` (8 levels), `mc_plane`, `mc_quad`, `mc_render_params` (filter +
one of ~10 composite modes incl. SHADED/INK/DEPTH/PERCENTILE/STDDEV with a large
PBR parameter block), `mc_volume_config`, `mc_volume_stats`,
`mc_volume_level_meta`, `mc_zarr_range`. Enums: `mc_preset` (8 levels),
`mc_cache_policy`, `mc_filter`, `mc_comp`.

Ownership: explicit, every `*_new`/`_open` has a matching `_free`/`_close`. The
sampler/renderer is a substantial second surface (plane/quad/point-grid render,
LOD-matched rendering, 3D resampling for surface volumes and oriented ML crops,
colormaps, DoG/SSAO post). For the storage-core rewrite the sampler can be
treated as a client of `mc_sample_src` and kept orthogonal.

================================================================================
## 5. PERFORMANCE
================================================================================

- **Determinism vs speed**: never `-ffast-math` (breaks cross-ISA equivalence,
  zero speedup — hot paths are integer); LTO only with PGO profiles (ThinLTO+PGO
  ~+5-7% decode; LTO without profiles ~15% slower).
- **SIMD**: NEON always on ARM; AVX2 at x86-64-v3 baseline; AVX-512 opt-in only
  (compile-tested). SVE skipped (fixed 8-lane i32 problem already saturates a NEON
  q-reg/AVX2 ymm). NEON paths in the codec hot loops (sum/count, dc-subtract);
  sampler/renderer ships NEON + SSE4.1 4-wide + AVX2 8-wide kernels.
- **Multithreading**: blocks are fully independent → embarrassingly parallel.
  Per-op pthread pools with atomic work-stealing counters (not a persistent pool).
  Whole-chunk helpers ~640 MB/s encode, ~1.5-2 GB/s decode per process; single
  cold block ~9 µs / ~0.007-0.023 ms.
- **Allocation patterns**: the recurring theme is **per-thread/per-worker
  persistent scratch** to avoid hot-path allocation and TLS dynamic lookups: the
  ~30 former `_Thread_local` codec globals were folded into one heap
  `mc_codec_ctx` passed by pointer (the dynamic-TLS `__tls_get_addr` calls were
  ~14% of decode CPU and forced range-coder state to spill to the stack); decode
  uses a pthread-key ctx; volume decoders own 16 MB dense/tile buffers (per-call
  16 MB allocs across N threads drove the kernel into direct page reclaim);
  cache arena is `MAP_NORESERVE` lazy.
- **Decode hotpath**: resolve chunk (memoized) → per-chunk index (thread-local,
  O(4096) once) → range-decode coefficients with EOB + extent tracking →
  sparse-aware inverse DCT (skips zero coeffs, DC-only short-circuit) → dequant →
  clamp+dc → optional corrections. Allocation-free; coder state in registers.

================================================================================
## 6. INVARIANTS / CONTRACTS FROM THE TEST SUITE
================================================================================

- **Determinism (the precise contract)**: NOT byte-identical across platforms —
  only metric-equivalent (size tolerance, PSNR/MAE tolerance, **max-error
  EXACT**). Within a platform, **bit-identical** for: parallel-vs-serial encode
  (`decode_chunk` memcmp==0), parallel-vs-serial and single-vs-multi-thread
  render/sample, full-vs-partial-fetch streaming decode, flat-vs-streaming reader
  values, frozen-cache-vs-dense sampling, and the seeded box sampler (same seed →
  identical boxes).
- **Lossy bounds**: with tau, `|err| ≤ tau` per material voxel; preset level L ⇒
  tau = 2^L (`mc_preset_tau(l)==1<<l`), ascending quality; **fully-air blocks
  decode to exact 0** (bitmap skip); air-at-material-boundary bounded only by
  RMSE/tau (the per-voxel air mask was removed — air is no longer forced to 0
  except whole-air blocks).
- **Per-chunk quality isolation**: per-chunk q independent; lower q ⇒ strictly
  lower error; no cross-contamination on interleaved decode.
- **All-air chunk → no blob** (slot stays absent, decodes to zero).
- **Structural**: index nodes are 8-aligned (bus-fault invariant); concurrent
  appends never orphan a subtree (reachable by tree walk, not just the memo);
  reopen preserves chunks + metadata byte-exactly; metadata length round-trips
  exactly; metadata over capacity is rejected without corrupting the file; reopen
  with mismatched dims → NULL and leaves the file intact.
- **Robustness (hard contract)**: NO crash/OOB/UB on ANY input (ASan+UBSan) for
  the read surface (`mc_open`, `mc_metadata`, `mc_chunk_offset`, `mc_decode_block`
  both readers + partial, `mc_verify_archive`); wrong values are acceptable, a
  SEGV is a FAIL. `mc_open` is the untrusted-bytes gate (rejects `len<MC_HDR` /
  bad magic). NaN/Inf/OOR per-chunk q is clamped (NaN→uint16 cast is UB). All 5
  historical fuzz OOB sites fixed (FINDINGS.md).
- **Cache**: read-through correctness (`get_copy` == direct decode) across
  roomy/tiny/cold-churn/frozen states; frozen multi-thread reads = zero
  mismatches; eviction without corruption; S3-FIFO hit-rate ≥ CLOCK on scan+hot;
  stale-until-`invalidate_chunk` after re-append; frozen `update` refused;
  resolve→freeze→get returns the SAME pointer (zero-copy).
- **Volume/streaming**: re-get of a resident block adds 0 net bytes; coverage
  PRESENT/ZERO/ABSENT respected; far-corner reads as air; `region_gen` monotonic;
  partial-fetch bit-identical to full with >4× fewer bytes; S3 test SKIPs cleanly
  on network error.
- **Zarr**: inner edge accepted only ∈ {128,256}; absent key == air, not error.

================================================================================
## 7. WHAT A C++26 REWRITE SHOULD REDESIGN
================================================================================

The new format is non-compatible, so this is about *what the format must support*
and *what was awkward*.

### Keep (these are load-bearing and well-validated)
- **16³ DCT blocks / 256³ chunks / 8 independent LODs.** The structural
  experiments prove 16³ is optimal for this data and that LOD/octree/prediction
  variations don't pay. Keep LOD independence as a hard guarantee.
- **Self-contained per-block payloads** (one block decodable from bitmap + lens +
  its own bytes) — this is what enables partial-fetch, GPU upload, and per-block
  cache fills. Non-negotiable for the format.
- **Per-block bitmap + implicit (prefix-sum) offsets + per-chunk fraction map.**
  Cheap random access and rejection-free ML sampling fall out of this.
- **Append-at-EOF + dense in-place index + release-store commit word + fallocate
  growth.** The crash-safety story is simple and correct; reproduce it.
- **Coverage states ABSENT/PRESENT/ZERO** and the `_chk` absent-vs-fetch-failed
  distinction — both are load-bearing for streaming correctness.
- **Trained context priors, per-volume overridable, stored in the archive.**
- **The decoded-block cache phase model (freeze/thaw, epoch pins, S3-FIFO).**

### Redesign / fix
1. **Resolve the determinism contract explicitly.** Decide and *enforce in tests*
   one of: (a) true cross-ISA byte identity → integer/fixed-point DCT or a
   strictly-specified f32 evaluation order with FMA forbidden and a cross-ISA
   golden-bytes test; or (b) metric equivalence only → document it loudly and
   drop the README's byte-identity claim. C++26 makes (a) easier (`<cstdfloat>`,
   contract assertions, `constexpr` tables, explicit `std::fma` control). The
   integer max-error corrections already assume same-build reconstruction —
   formalize that boundary.
2. **Stop letting the public docs drift from the code.** The header, the design
   doc, and several banners describe removed features (air mask, harmonic fill,
   c3d/c3g, integer DCT, per-shard mutexes, CLOCK default). A rewrite should
   generate or test docs against reality. The dead prior classes
   (MASK/MASKU/MASKA) and the reserved "mixed" flag bit are vestigial — drop them.
3. **First-class per-axis dims, not "cubic with padding afterthought."** The C API
   bolts `nx/ny/nz` on next to `dim`; the chunk-coord space is 12 bits/axis (2^20
   voxels/axis) — make the new format's coordinate width and dim handling explicit
   and uniform.
4. **Make the .mca cache's unbounded disk growth a real policy.** `mc_volume`'s
   local transcode .mca never evicts on disk. The new format/volume should support
   a bounded on-disk cache (the index is already region-addressable; add a free
   list / generation-based reclaim).
5. **Per-chunk q is in the blob but priors are process-global.** This couples
   "one quality dial per process" with per-chunk q in an awkward way. Consider
   per-archive (not process-global) priors and a cleaner per-chunk-q story so two
   archives can be open at once with different priors.
6. **The alignment footgun (atomic u64 node slots must be 8-aligned).** In C++26,
   model index nodes as a proper aligned type (`alignas`, `std::atomic_ref` over
   correctly-aligned storage) so the constraint is in the type system, not a CAS
   `w_alloc_aligned` discipline + a separate alignment test.
7. **Worker model**: replace per-op pthread spawn/join with a real reusable
   executor (C++26 `std::execution`/senders or a thread pool). The atomic
   work-stealing counter pattern is fine; the spawn-per-call is wasteful. Keep the
   "one codec context per worker, passed by pointer, no TLS" rule — it was a 14%
   win; in C++ this is just a per-worker object, no `__tls_get_addr`.
8. **Single-flight + coverage as the source of truth** worked but is spread across
   three ad-hoc structures (`inflight[]`, `rs_set`, the LIFO stack) with a
   documented `rkey(0,0,0,0)==0` aliasing footgun and a 100 s busy-wait in the CLI
   path. Redesign as one coherent async region-fetch coordinator (futures/
   coroutines) with condvars instead of spin.
9. **Codec ratio gains worth pursuing** (from STRUCTURAL_FINDINGS): RDOQ/trellis
   quant (encoder-only, decoder-compatible), corpus-trained q-matrix, and possibly
   a directional/learned transform or 3D-wavelet core — these are the only
   non-rejected avenues. The format should leave a codec-version field so the
   block codec can evolve without a container break.
10. **Source-format layer (mc_zarr) is vestigial/contradictory** (c3d removed,
    v3-sharded rejected, only v2-flat raw/blosc survive, docs describe the
    opposite). For the rewrite, define exactly which source formats the volume
    ingests and keep the transcode layer cleanly separable from the storage core.

### Format must support (summary checklist for the new container)
- Random access to a single 16³ block over a high-latency source with ≤ ~2 small
  ranged reads (bitmap+lens cached, then payload).
- One-GET whole-chunk fetch and one-GET front-clustered index (exporter path).
- Verbatim chunk copy between archives (no re-encode).
- Append-while-readable, crash-safe after every chunk, reopen-and-continue.
- Per-chunk quality + per-volume priors + integrity hash, all self-describing.
- 8 independently decodable LODs; per-block material fraction for sampling.
- Coverage tri-state (absent/present/air) queryable without decode.
- A codec-version field and room for q-matrix / alternate-transform evolution.
