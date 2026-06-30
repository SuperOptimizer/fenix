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
In: a `.fxvol` (read-only, `codec::VolumeArchive::open`). Out: a FUSE mount; `volume.raw` (raw voxels) +
`meta.toml`. No new on-disk format.

## Dependencies
Intra: `codec` (VolumeArchive, BlockCache), `core`. Third-party: **libfuse3** (pkg-config `fuse3`) — the
only consumer of this dep, firewalled here (§2.5; approved by forrest 2026-06-30).

## Invariants & numerics
**Read-only in v1** (writes → `-EACCES`). A decode/fetch error is **`-EIO`, never silent air** (root §2.4).
Element type is carried in (u8 for these scrolls; the archive store is dtype-agnostic). f32 decode → round
+ clamp to the element type. Lossy-writeback (when writable mounts land) is the load-bearing caveat in
ADR 0007: recompress-on-flush means write-then-read is not identity → a dirty-tile uncompressed overlay
holds writes verbatim until an explicit seal.

## Performance notes
A `read` walks the requested voxel span, caching the "current" 16³ block so a contiguous x-run costs one
`block16` call; the 64³ tile decode amortizes across its 64 blocks via the byte-budgeted sharded SIEVE
cache (default 1 GiB here). Resident RAM is bounded by the cache, never the volume. FUSE is multithreaded
(`block16` is const + the cache is sharded/locked + `read_chunk` reads the mmap concurrently). Future:
block-aligned gather/`memcpy` of full x-runs; larger `max_read`; writable mounts + overlay.

## Gotchas / pitfalls
- libfuse must be installed to build (`pkg-config fuse3`); the `-DFENIX_FS=ON` configure fails loudly if not.
- Unmount with `fusermount3 -u <mountpoint>` (not `umount`, unless root).
- The archive carries no dtype tag (dtype-agnostic store) — the mount assumes u8; a `--dtype` option is the
  generalization for u16/f32 sources.

## Status & TODO
**Implemented:** read-only mount (`getattr`/`readdir`/`open`/`read`), `volume.raw` + `meta.toml`, decode via
the cache. **TODO:** writable mounts (dirty-tile overlay + recompress-on-flush + `seal`), `--dtype`/u16,
per-LOD `volume_lN.raw`, block-aligned gather for big reads, a mount smoke-test in CI (needs `/dev/fuse`).
Open: whether to also offer a `userfaultfd` mmap layer for read-mostly inference (ADR 0007 alternative).
