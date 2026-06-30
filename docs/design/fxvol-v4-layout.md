# `.fxvol` v4 — container layout & access design

> Design note backing [ADR 0006](../adr/0006-fxvol-v4-container.md). Synthesized 2026-06-30 from a
> 3-agent SOTA survey (chunked/sharded array formats; crash-safe append + S3 CAS; cloud-optimized LOD
> layout) on top of the matter-compressor `.mca` lineage ([`../research/research-mc.md`](../research/research-mc.md)),
> which already prototyped per-LOD root offsets, sentinel-as-coverage slots, implicit per-block prefix-sum
> offsets, Morton coarse-first export, `fallocate`+release-store commit, and absent-vs-fetch-failed.

## 1. Constraints (locked, from forrest 2026-06-30)
1. **One file per volume** — all chunks + every LOD + the index packed into one `.fxvol`.
2. **Dual access** — streaming sequential (prefix = coarse preview) AND random seek, from that one file.
3. **Network atom = the 64³ chunk** (= one DCT tile = 4³ of the 16³ blocks). A client always range-GETs a
   whole 64³ chunk. The 64³ tile is also the **atomic decode unit** (its blocks share rANS tables) and the
   **cache unit**. 16³-block / voxel = the addressable **view** granularity.
4. **Largest volume = 2¹⁸/axis** ⇒ 2¹² = 4096 chunks/axis ⇒ **2³⁶ ≈ 6.9×10¹⁰ chunks** at LOD 0.
5. **8 GB RAM ceiling** ⇒ resident state must be O(working set), never O(chunks).
6. **Fully greenfield** — no compat with villa/ScrollPrize/VC; interop by writing new adapters there.
7. **LOD-only quality scaling** — explicit octave pyramid; NO in-stream progressive/SNR layers (the client
   chooses which LOD's chunks to fetch). Local access = **mmap the whole file as an array**.

## 2. Two byte-orderings of one format
A `.fxvol` is one file with two states; `fxvol finalize` converts LIVE → SEALED.

- **LIVE** (ingestion / working store): blobs and page-table nodes appended at EOF in production order;
  `mmap` (`MAP_NORESERVE`, huge fixed reservation); `fallocate` growth; crash-safe commit (§5). Local
  random access via the in-file radix page-table. Append-while-readable.
- **SEALED** (distribution): repacked **coarse-first** (LOD_top → LOD0) with the index **front-loaded**
  (COG discipline — first ~16 KB read parses the whole layout); a truncated `Range:` GET returns a
  complete preview at the coarsest octaves. Still `mmap`-able locally. A `cloud_optimized` superblock flag
  is cleared by any in-place edit that breaks coarse-first ordering.

```
 SEALED layout (offsets grow downward; coarse data first):
 ┌────────────────────────────────────────────────────────────────┐
 │ superblock A (4 KiB)  ┐ double-buffered, crc32c'd commit record  │
 │ superblock B (4 KiB)  ┘ {magic"FXVL",ver=4,committed_eof,         │
 │                          index_root_off,commit_seq, dtype,dims    │
 │                          ZYX, voxel µm, world origin, axes,       │
 │                          provenance(recipe+git+params+inputs),    │
 │                          OME-NGFF multiscales mirror, flags}      │
 ├────────────────────────────────────────────────────────────────┤
 │ master directory: u64 lod_index_root[13]; per-LOD {grid extent,  │
 │                   index byte-range, q/hf/dz params}              ← "self-describing leader"
 ├────────────────────────────────────────────────────────────────┤
 │ LOD12 (coarsest) tiles + LOD12 radix page-table   ── preview ──┐ │
 │ LOD11 ...                                                       │ │  a Range:0..N GET
 │  ...                                                            │ │  = leader + all
 │ LOD1 tiles + page-table                                         │ │  octaves coarser
 │ LOD0 (full-res) tiles + LOD0 page-table   (tail)          ◄─────┘ │  than the target
 └────────────────────────────────────────────────────────────────┘
   each tile blob: [u32 leader=len][DCT tile payload][crc32c trailer]
```

## 3. Chunk key & the 3-level radix page-table
- **Compressed-Morton Z-order** over the (z,y,x) **chunk** grid. At 2¹⁸/axis the chunk grid is 2¹²/axis ⇒
  **36-bit key**, fits a u64 (28 spare bits for LOD/flags if ever packed). Use BMI2 `pdep`/`pext` (3D masks
  `0x9249…`,`0x2492…`,`0x4924…`) — one instruction/axis, branchless, SIMD-friendly. "Compressed" = skip
  bit *i* of an axis once `2^i ≥ grid[axis]`, so anisotropic grids pack densely. NB: the current
  `archive.hpp` key `z<<40|y<<20|x` is *not* an interleaved Morton code (no locality) — replace it.
- **3-level fixed-stride radix, split 12 + 12 + 12** (per the decision):
  ```
  key[35:24] → L0 node (4096 × u64 child-offset)   ← tiny, resident (32 KiB)
  key[23:12] → L1 node (4096 × u64 child-offset)   ← demand-paged
  key[11:0]  → leaf slot (4096 × slot)             ← demand-paged
  ```
  Nodes live in the (sparse) file and are `mmap`'d. Empty subtrees = an ABSENT child-offset at the level
  they go empty (a whole empty super-region collapses to one L0/L1 entry, not 2¹²–2²⁴ slots). Lookup =
  3 bit-slices + 3 pointer-follows, no search. **On-disk = O(populated)** (file holes); **RAM = O(touched
  pages)**. A dense 2³⁶ index would be 512 GiB — never materialized.
- **Leaf slot = sentinel-as-coverage** (no side bitmap): `offset==2⁶⁴-1` ⇒ **ABSENT** (unknown / not
  fetched); `len==0` ⇒ **ZERO** (proven all-air, decode to fill, skip re-fetch); else ⇒ **REAL** (blob at
  offset). **fetch-failed is NOT a slot state** — it is a runtime `Expected<…,Error>` (retry+backoff →
  hard fail), never written as ABSENT/ZERO.
- **Leaf index encoding** = Neuroglancer **minishard**: a `[3,n]` u64le table (delta-coded chunk IDs;
  delta-coded offsets; sizes) + a small fixed shard-index of `(start,end)` per minishard. **`identity`
  hash, not Murmur** — Murmur load-balances a CDN but scatters spatial neighbours, poison for halo reads.
  `preshift_bits` clusters a tile + its 26 neighbours into one minishard ⇒ ~1 page-fault per blocked access.

Sizing at the 2¹⁸ envelope (sparse scroll, so actual ≪ these):

| level | fan | entries (dense) | dense size | resident |
|---|---|---|---|---|
| L0 | key[35:24] | 4096 | 32 KiB | always (tiny) |
| L1 | key[23:12] | 4096/node | — | touched nodes only |
| leaf | key[11:0] | 4096/leaf | — | touched leaves only |

## 4. LOD pyramid (explicit, no progressive)
- **13 octaves** 2¹⁸→2⁶ (top = whole volume in one 64³ chunk). Each octave is its **own DCT sub-archive +
  own radix page-table**, reached via `lod_index_root[k]` in the master directory. Independently
  addressable (fetch at the needed resolution directly) and independently best-quality coded (its own
  clustered rANS tables — the reason the tile-DCT beat the wavelet; embedded subbands would re-couple them).
- **Downsample recipe:** downsample the **whole level globally THEN re-tile** (never per-tile — avoids
  aliasing and tile-seam error). Prefilter = 2×2×2 box / local-mean (cheap) or Gaussian (record
  `type` in the multiscales mirror); **labels/masks would use mode/nearest** (Tier-5 concern, not now).
  Then edge-replicate partial tiles (already in `write_volume`) → DCT-16 encode → write octave k+1.
- **Quality scaling is the client's.** The downloader fetches whichever LOD's chunks it wants; the codec
  embeds no SNR-progressive layers. (The coarse octaves already *are* the thumbnails/previews.)

## 5. Crash-safe append (LIVE) — the two orderings, kept separate
Two independent problems the old notes conflated:
- **(A) intra-process reader visibility** — release/acquire on the 8-byte-aligned atomic slot; a reader
  that acquire-loads a REAL slot is guaranteed to see the fully-written blob. (Pure memory model.)
- **(B) cross-crash durability** — the OS writes dirty mmap pages back in arbitrary order, so a
  release-store alone is NOT crash-safe; needs an explicit barrier + a torn-write-proof commit record.

**Append one tile** (lock-free except growth): `off = cursor.fetch_add(len)` (bump allocator) → `fallocate`
if growth needed (under a growth-only mutex; **never `ftruncate`** — lazy alloc + full disk = uncatchable
`SIGBUS`, fatal under `-fno-exceptions`) → write `[u32 len][payload][crc32c]` at `base+off` → release-store
`encode(off)` into the leaf slot (node *creation* serialized under a small mutex; the blob write + slot
store stay lock-free).

**Commit** (batched at checkpoints, not per-tile): `msync(MS_SYNC)` the blob + page-table region (**data
durable first**) → build the inactive superblock with `commit_seq+1`, `committed_eof=cursor`, fresh crc32c
→ `msync` that one superblock page (**then the pointer**, HDF5-SWMR flush-dependency). **Open** validates
both superblocks, adopts the higher `commit_seq` that passes crc; everything past its `committed_eof` is a
crashed half-append and is ignored. The A/B double-buffer + monotonic `commit_seq` makes the commit pointer
torn-write- and ABA-proof; per-blob crc32c catches a torn slot pointing at garbage.

## 6. S3 multi-writer (distribution / cooperative ingest)
Pattern shared by Icechunk / Lance / Delta / Iceberg / TileDB: **everything immutable & content-addressed
except one tiny mutable ref, swapped by an atomic conditional write.**
- `chunks/{hash}`, `manifests/{id}`, `snapshots/{id}` (parent-linked), `transactions/{id}` — immutable,
  written `PUT If-None-Match:*` (idempotent retry / dedup).
- one mutable `repo` ref — commit loop: GET ref (ETag E) → stage immutables → `PUT repo If-Match:E`
  (200=done; 412=conflict→re-GET, rebase if disjoint chunk coords else app-conflict; 409=retry). First
  create = `If-None-Match:*`. Carry a monotonic `commit_seq` inside the ref body to defeat ETag ABA.
- **Prerequisite: SigV4 signing + PUT in `src/io/s3.hpp`** (today anonymous-GET only — conditional writes
  require SigV4). Fallback for stores without conditional-PUT: put-if-absent on zero-padded version files
  (`refs/<branch>/{NN}.json`).

## 7. Decoded-16³-chunk cache (the 16³ / voxel view)
- **Unit:** a decoded **16³ chunk** (one DCT block; f32 = 16 KiB), keyed by the 16³-block-grid Morton.
  The 64³ tile is still the atomic DECODE unit (shared rANS tables), so on a miss the containing 64³ tile
  is decoded once and ALL 64 of its 16³ chunks are inserted — the decode amortizes across the tile while
  eviction stays 16³-fine. (`block16.hpp` → `BlockCache`; `archive.block16()`/`voxel()`.)
- **Policy: sharded SIEVE** (NSDI'24) — on hit, set a visited bit (no list move ⇒ low contention,
  scan-resistant, hit-ratio > LRU); a single "hand" advances only on eviction. Partition into ~2–4×
  thread-count shards, each its own SIEVE + lock. (CLOCK/NRU is the simpler fallback; W-TinyLFU a future
  upgrade if hit-rate ever dominates; **avoid ARC — patented**.)
- **Refcount-pin** tiles in active use so eviction can't free a tile mid-decode/read. **Byte-budgeted**
  (configurable, e.g. 1–4 GiB), evict until under budget. **Fetch-error-safe**: ABSENT/ZERO may be cached;
  a transient IO/S3 error is an `Expected` error, never memoized as a valid (air) tile.
- **Amortization:** 16³-block/voxel access has strong spatial locality (neighbours share a tile) ⇒ ~1 tile
  decode (~0.25 ms at ~4–4.6 GB/s) per ~64 block accesses ⇒ ~4 µs/block amortized.

## 8. Access flows
- **Local random (mmap-as-array):** `(z,y,x) voxel → (lod,chunk) → morton → radix lookup → slot`. REAL →
  cache get-or-decode(tile) → index the 16³ block / voxel. ABSENT/ZERO → fill. Page-faults pull only
  touched index nodes + tile bytes.
- **Streaming sequential / range-GET (SEALED):** `Range: 0..N` → leader + all octaves coarser than the
  target → decode for a preview; widen N for finer octaves.
- **Network (64³ atom):** resolve slot (minishard index, ≤3 range-GETs, ~1 amortized with cached indices)
  → range-GET the one 64³ blob → decode → cache.

## 9. Implementation plan (replaces `src/codec/archive.hpp`)
Phased, each step measured/tested (fuzz the parser — no-UB-on-any-bytes is a hard rule):
1. ✅ **DONE (2026-06-30)** — **Morton key + 3-level radix page-table** over mmap (`MAP_NORESERVE`,
   `posix_fallocate`), sentinel-as-coverage slots, bump allocator. Single-LOD, LIVE only. In `archive.hpp`
   (replaced the flat-tail-index first cut, same public API); tests `test_archive` + `test_fxvol`
   (multi-chunk Morton, sparse tri-state, write/read volume, garbage-open robustness), release + ASan-clean.
2. ✅ **DONE (2026-06-30)** — **Crash-safe commit**: double-buffered crc32c superblock (two alternating
   slots, monotonic seq), per-blob crc32c, data-before-pointer `msync` (data then superblock), open adopts
   the highest-seq valid slot; `commit()` checkpoint + `close()`. `test_fxvol` covers checkpoint persistence,
   double-buffer fallback recovery (corrupt latest slot → recover prior), blob-crc detection; release + ASan.
   NB Phase 2 versions {committed_eof, root}, not the page table (mutated in place) — full COW is later.
3. ✅ **DONE (2026-06-30)** — **Decoded-16³-chunk cache** (`block_cache.hpp`: sharded SIEVE, shared_ptr
   pinning, byte budget); `archive.block16()`/`voxel()` decode the 64³ tile once and cache its 64 chunks.
   `test_fxvol` covers tile-mate hit amortization, byte-budget eviction, and voxel-view exactness vs
   read_volume; release + ASan-clean.
4. ✅ **DONE (2026-06-30)** — **LOD pyramid**: per-LOD radix roots in the superblock (`lod_root[20]`),
   global 2³-box downsample→retile per octave (down to a single 64³ chunk), LOD-parameterized
   write_chunk/read_chunk/coverage/read_volume/block16/voxel + dims_at/chunk_extent. `write_volume` builds
   the whole pyramid. `test_fxvol`: per-level dims/coverage + each octave ≈ box-downsample (PSNR); release + ASan.
5. ✅ **DONE (2026-06-30, core)** — **`finalize(dst)`**: LIVE → SEALED repack, COARSE-first across octaves
   (coarsest LOD at the front → truncated GET = preview), Morton order within a LOD, compressed blobs copied
   VERBATIM (no re-encode → no extra loss), ZERO/ABSENT preserved. `lod_root_offset()` accessor. `test_fxvol`:
   round-trips byte-identical at every LOD, and finalize flips fine-first → coarse-first; release + ASan.
   (Still optional: a front-loaded minishard index + `cloud_optimized` flag for minimal range-GET round-trips.)
6. **S3 path**: SigV4 PUT in `s3.hpp`; content-addressed objects + `If-Match` CAS commit; minishard range-GET.
7. GPU two-phase decode (deferred; format already supports it).

## 10. Sources
fenix: [`research-mc.md`](../research/research-mc.md) (the `.mca` working reference), ADR 0002 (container
half), ADR 0005 (codec), `src/codec/archive.hpp` (the first cut being replaced), root CLAUDE.md §2.4/§2.6.
External (from the 3-agent survey): Zarr v3 sharding ZEP-2 (fixed-stride `(offset,nbytes)` index,
`2⁶⁴-1` empty sentinel); Neuroglancer precomputed sharded (two-level minishard, compressed Morton,
identity vs murmur hash); TensorStore OCDBT (paged versioned B+tree, manifest CAS); HDF5 1.10 chunk indexes
+ SWMR flush-dependency + paged aggregation; Icechunk v2.1 (immutable + `repo` conditional-PUT CAS); AWS S3
conditional writes (If-None-Match/If-Match, SigV4); Lance/Delta/Iceberg version-file CAS; COG / OGC 21-026
(IFDs-before-data leader, coarse-first); OME-NGFF multiscales; SIEVE (NSDI 2024); Moon et al. Hilbert
clustering (TKDE 2001); Blosc2 NDim (b2nd sub-chunk). URLs in the agent reports archived with this session.
