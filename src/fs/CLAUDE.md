# fs — CLAUDE.md

## Purpose
`fxfs` — a **FUSE** filesystem that exposes a `.fxvol` archive as a flat, **uncompressed** ZYX array, so any
tool can `mmap`/read it like a `numpy.memmap` without doing its own decode/cache/recompress. Decode happens
on demand (read-fault → decode the covering 64³ tile via the archive's decoded-16³ SIEVE cache); the kernel
page cache caches decoded pages. **The strategic version of "treat the archive as a raw array."**
See [ADR 0007](../../docs/adr/0007-fxvol-fuse-transparent-array.md).

## Public API & key types
- `fenix-fxfs <archive.fxvol> <mountpoint> [fuse opts]` — a standalone binary (has `main()` + `fuse_main`),
  **not** part of the header-only single-TU build. Firewalled behind `-DFENIX_FS=ON` (links `libfuse3`).
- Mount presents: **`volume.raw`** (regular file, size `Z·Y·X·elemsize`, C-order x-fastest ZYX voxels) and
  **`meta.toml`** (dims, dtype, q, lods). `read`/`mmap` of `volume.raw` → `archive.block16()` per covering
  16³ block (cached) → bytes.

## Inputs / outputs & formats
In: a `.fxvol` **v5** archive (read-only, `codec::VolumeArchive::open`; v5 carries a `src_dtype` tag in the
superblock — see [ADR 0006](../../docs/adr/0006-fxvol-v4-container.md) / the v4→v5 upgrade in `codec`). Out:
a FUSE mount; `volume.raw` (raw voxels) + `meta.toml`. No new on-disk format.

## Dependencies
Intra: `codec` (`VolumeArchive`, `BlockCache`, `archive.hpp`), `core`. Third-party: **libfuse3** (pkg-config
`fuse3`) — the only consumer of this dep, firewalled here (§2.5; approved by forrest 2026-06-30).

## Invariants & numerics
**Read-only in v1** (writes → `-EACCES`). A decode/fetch error is **`-EIO`, never silent air** (root §2.4).
Element type is read from the archive's `src_dtype()` (u8 for these scrolls — native-dtype v5 archives store
u8 bytes directly, no f32 intermediate). `fx_read`'s fast path (`src_dtype() == DType::u8`) `memcpy`s native
bytes straight out of `block16`; a non-u8 archive falls back to `sample_f32` (widen-per-voxel, round+clamp to
u8) — fxfs still only *serves* u8 out of `volume.raw` regardless of source dtype, hardcoded (see Gotchas).
Lossy-writeback (when writable mounts land) is the load-bearing caveat in ADR 0007: recompress-on-flush means
write-then-read is not identity → a dirty-tile uncompressed overlay holds writes verbatim until an explicit
seal.

## Performance notes
`fx_read` walks the requested byte span in whole block-local x-runs: for each run's starting voxel it fetches
its covering decoded 16³ block once (`block16`, SIEVE-cached) and `memcpy`s the contiguous bytes that block
covers along x (up to 16, clamped by the block/row/request boundary) — one `block16` call per ≤16 bytes
instead of per byte. The 64³ tile decode amortizes across its 64 constituent blocks via the byte-budgeted
sharded SIEVE cache (`reserve_cache`, default 1 GiB here, 16 shards). Resident RAM is bounded by the cache,
never the volume. FUSE is multithreaded (`block16` is const + the cache is sharded/locked). Future:
block-aligned gather/`memcpy` across multiple blocks per call; larger `max_read`; writable mounts + overlay.

## Gotchas / pitfalls
- libfuse must be installed to build (`pkg-config fuse3`); the `-DFENIX_FS=ON` configure fails loudly if not.
- Unmount with `fusermount3 -u <mountpoint>` (not `umount`, unless root).
- fxfs hardcodes `st.esz = 1` / u8 output regardless of `src_dtype()` — a non-u8 archive still decodes
  correctly (via `sample_f32`) but `raw_size`/`meta.toml` assume 1-byte elements; a `--dtype` option (or
  reading `esz` from `src_dtype()`) is the generalization for u16/f32 sources.
- Single global `State* g` — one archive per process (matches "mount one `.fxvol`"); not reentrant for
  multi-archive mounts.

## Status & TODO
**Implemented:** read-only mount (`getattr`/`readdir`/`open`/`read`), `volume.raw` + `meta.toml`, decode via
the cache, u8-native fast path + generic `sample_f32` fallback for non-u8 archives (post v5 native-dtype
archive). **TODO:** writable mounts (dirty-tile overlay + recompress-on-flush + `seal`), `--dtype`/derive
`esz` from `src_dtype()` instead of hardcoding u8 output, per-LOD `volume_lN.raw`, block-aligned gather
across block boundaries for big reads, a mount smoke-test in CI (needs `/dev/fuse`). Open: whether to also
offer a `userfaultfd` mmap layer for read-mostly inference (ADR 0007 alternative).
