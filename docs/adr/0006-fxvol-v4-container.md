# ADR 0006 — `.fxvol` v4 container layout (single-file, mmap'd radix page-table, LOD pyramid)

**Status:** Accepted (2026-06-30). **Amends the container half of [ADR 0002](0002-codec-and-container.md)**
(which remains in force for the codec substrate); supersedes the first-cut `fstream`+tail-index in
`src/codec/archive.hpp`. Synthesized from a 3-agent SOTA survey (chunked/sharded formats · crash-safe
append + S3 CAS · cloud-optimized LOD layout) on top of the matter-compressor `.mca` lineage
([`../research/research-mc.md`](../research/research-mc.md)), which already prototyped most of this.

## Context
A scroll volume is **one logical thing** and must be **one file** (`.fxvol`) holding every chunk, every
LOD level, and the index — not a directory-of-chunks (Zarr) nor one-file-per-shard (Neuroglancer). The
largest volume we will ever store is **2¹⁸ per axis** ⇒ at the 64³ chunk (2⁶/axis) that is **2¹² = 4096
chunks/axis = 2³⁶ ≈ 6.9×10¹⁰ chunks** at LOD 0. A flat in-RAM index (today's `archive.hpp`
`unordered_map`) needs ~3.4 TB of RAM at that envelope — dead on arrival against the **8 GB ceiling**.
The format must serve **both** local random access (`mmap` the whole file as an array, address 16³ blocks
and voxels through it) **and** streaming/range-GET (a truncated read yields a coarse preview). It is
**fully greenfield** — no backward/forward compat with villa / ScrollPrize / volume-cartographer data or
wire formats; interop is by writing **new adapters** in those tools, not by conforming.

## Decision
- **One file per volume, two byte-orderings of the same format.**
  - **LIVE** (ingestion/working): `mmap` (`MAP_NORESERVE`, huge fixed reservation so offsets never
    move); blobs + page-table nodes **appended at EOF**; `fallocate` growth (never `ftruncate`);
    crash-safe commit (below). Local random access via the in-file page table = "mmap as array."
  - **SEALED** (distribution): an `fxvol finalize` pass repacks the same file **coarse-first**
    (LOD_top → LOD0) with the index **front-loaded** (COG discipline: one ~16 KB GET parses the whole
    structure) ⇒ a truncated `Range:` GET returns a complete coarse preview. Still `mmap`-able locally.
- **Index = a 3-level fixed-stride radix page-table on a 36-bit compressed-Morton chunk key**, per LOD.
  Key split **12 + 12 + 12** (L0/L1/leaf, 4096 entries each); nodes live in the (sparse) file and are
  `mmap`'d ⇒ on-disk = O(populated), resident RAM = O(touched pages), **independent of volume size**.
  Lookup is bit-slice + pointer-follow (no search). Leaf index format = **Neuroglancer minishard**
  (delta-coded `[3,n]`), **identity hash** (NOT Murmur — preserve halo locality), `preshift_bits` to
  cluster neighbours. **Morton, not Hilbert** (hot-path BMI2 `pdep`/`pext`; composes with LOD = 3-bit
  shift). **Sentinel-as-coverage tri-state in the slot** (ABSENT / ZERO-air / REAL-offset), no side
  bitmap. **fetch-failed is never a stored state** — it is a runtime `Expected` error (retry→hard-fail),
  never memoized as air (root CLAUDE.md §2.4 hard rule).
- **LOD = an explicit pyramid only; no in-stream progressive/SNR layers.** 13 octaves (2¹⁸→2⁶), each its
  own DCT sub-archive + its own radix page-table (independently addressable + best-quality coded);
  global antialiased downsample **before** retile (seam-free); edge-replicated partial tiles. Quality
  scaling is the **client's** job (fetch chunks from whichever LOD it wants) — the codec does not embed
  progressive quality. (+1/7 ≈ +14% storage, accepted.)
- **Crash-safe commit (LIVE):** a **double-buffered, crc32c'd superblock** holding
  `{committed_eof, index_root_off, commit_seq}`; on open, adopt the higher `commit_seq` that passes crc;
  everything past `committed_eof` is treated as nonexistent. Commit ordering is **data-before-pointer**
  (`msync` blobs + page-table pages, *then* flush the superblock — HDF5-SWMR flush-dependency), batched
  at checkpoints. Per-blob crc32c; every offset/len bounds-checked against `committed_eof` before deref.
  Intra-process reader visibility via release/acquire on 8-byte-aligned slots.
- **S3 is READ-ONLY** (forrest, 2026-06-30): a `.fxvol` is written locally (LIVE mmap), `finalize()`d to
  SEALED, uploaded out-of-band, then served by **anonymous byte-range GET** (`io/s3.hpp`, extensible from
  libs3 for range/batched GET). The SEALED coarse-first ordering means a truncated range-GET already yields
  a preview. **No S3 write path / SigV4 / conditional-PUT / multi-writer CAS** (an earlier Icechunk-style
  draft is dropped).
- **Access granularity:** the **64³ chunk is the atomic decode + network + cache unit** (its 64 blocks
  share clustered rANS tables + sequential streams ⇒ a 16³ block is not independently decodable, by
  design — that sharing is the ratio win). **16³-block / voxel addressing is a view** served by a
  **decoded-tile cache: sharded SIEVE, refcount-pinned, byte-budgeted** (resolves the open io/ cache-
  eviction ADR). Amortizes to ~1 tile decode per ~64 block accesses given spatial locality.
- **Interop without conformance:** mirror OME-NGFF `multiscales` fields (axes ZYX, per-level `scale`,
  voxel µm) in the self-describing leader so adapters are easy; the container itself is `.fxvol`.

## Consequences
+ RAM is bounded by the **working set, not the volume** (paged radix table + byte-budgeted tile cache) —
  the 2¹⁸³ / 8 GB constraint is satisfiable. One file = trivial to move/serve; mmap-as-array locally;
  one range-GET = a coarse preview remotely; crash-safe append-while-readable; lock-free concurrent
  readers; read-only anonymous S3 serving (no write/auth complexity). LOD-only keeps the codec simple.
− **Two byte-orderings** (live vs sealed) ⇒ a `finalize` pass; the append form is not itself
  range-GET-optimal (accepted — don't make one layout do both).
− Coarse LODs cost +14% storage. − A 16³
  block always costs a 64³ tile decode (cheap + cached, but a 64× compute factor for a true random
  single-block miss). − No reproducible cross-ISA bytes (inherited, accepted).

## References
ADR 0002 (container half amended here), ADR 0005 (the codec this stores). Research:
[`research-mc.md`](../research/research-mc.md) (the `.mca` lineage — the working reference for most of
this), the 3-agent survey captured in [`../design/fxvol-v4-layout.md`](../design/fxvol-v4-layout.md)
(Zarr ZEP-2 sharding, Neuroglancer minishard, TensorStore OCDBT, HDF5 SWMR, Icechunk CAS, COG, SIEVE).
