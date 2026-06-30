# ADR 0007 — `fxfs`: a FUSE filesystem exposing a `.fxvol` as a transparent uncompressed array

**Status:** Accepted (2026-06-30). Builds on [ADR 0006](0006-fxvol-v4-container.md) (the `.fxvol` container)
and [ADR 0005](0005-retire-wavelet-dct-tile-codec.md) (the DCT-16 tile codec it decodes). Introduces a new
**optional** component (`src/fs/`) and a new GUI-tier-style firewalled dependency (**libfuse**).

## Context
A `.fxvol` is a compressed, 64³-tiled, out-of-core volume. Today every consumer (pipeline stages, tools,
analysis scripts) must go through the archive API and do its own block addressing, decode, caching, and —
for writers — re-encode. forrest wants the archive usable as if it were **one flat, uncompressed ZYX
array**: `mmap` the whole thing, index voxels directly, and let the system transparently decode the
covering 64³ tile on read and recompress on write, with caching handled for you. This makes the archive a
drop-in `numpy.memmap`-style backing store for inference, visualization, and ad-hoc analysis without every
tool re-implementing the codec dance.

The volumes are huge (2¹⁸/axis envelope; PHerc Paris 3 = 68417×42403×42403 ≈ 112 TiB uncompressed at u8),
so the "uncompressed array" is **virtual** — only touched tiles are ever decoded/resident. The element type
is the source's native dtype (u8 for these CT scrolls — the codec is dtype-agnostic in storage; see
[ADR 0006](0006-fxvol-v4-container.md) and the u8-native IO path).

## Decision

### Userspace (FUSE), not an in-kernel module
Expose the archive through a **FUSE** filesystem (`fxfs`), not a custom kernel module. Rationale:
- **Float codec in the kernel is hostile.** The codec is a float DCT-16 + rANS; kernel code can't freely
  use the FPU/SIMD (`kernel_fpu_begin/end` around every transform) and has no libc/`std::`. Porting
  `dct_block.hpp` into the kernel is a rewrite into a far worse environment.
- **Safety/maintenance.** A codec bug in-kernel is a panic, not a `SIGSEGV`; out-of-tree modules track
  kernel-API churn forever. The codec's "no UB on any bytes" robustness contract would become "no oops."
- **Dev model.** We build in Docker on Chimera/musl/LLVM, header-only userspace. FUSE runs fine on musl;
  a kernel module needs host kernel headers + privileged containers.
- **Caching is free.** The kernel **page cache** caches decoded pages for us; `mmap` (shared, writable)
  works through FUSE's `read`/`write`/`readpage`/`writepage` paths.

**userfaultfd** is the noted alternative for a pure read-mostly "it's just an array" `mmap` with no
filesystem and no dependency; it is kept as a future option for read-heavy inference. A real **kernel
module is explicitly deferred** — only revisited if FUSE throughput proves insufficient, and even then
userfaultfd is the next rung before a module.

### Layered design
1. **Decoded-tile cache + `read_block`/`write_block(64³)` API** in `codec/archive.hpp` — this is
   [ADR 0006](0006-fxvol-v4-container.md) **Phase 3** (sharded **SIEVE**, refcount-pinned, byte-budgeted),
   and is the shared foundation under FUSE, userfaultfd, *or* direct library use. The pipeline (which uses
   the `Volume` API) benefits from this layer with no FUSE at all. **Built and tested first.**
2. **`fxfs` daemon** (`src/fs/`, a new build target + `libfuse`) mounts a `.fxvol` and presents:
   - `…/volume.raw` — the flat uncompressed ZYX array, size = `Z·Y·X·sizeof(dtype)`, C-order (x fastest,
     matching project ZYX). Reads/`mmap` fault in → decode the covering 64³ tile(s) via the cache.
   - `…/meta.toml` — voxel µm, world origin, scroll id, dtype, dims, LOD scales (mirrors the leader).
   - (later) per-LOD `volume_lN.raw` for multiscale access.

### Read/write semantics (the load-bearing policy)
- **Default mount is read-only.** This is the safe, lossless, common case (inference/visualization/analysis).
- **Writable mounts are opt-in and writeback is LOSSY.** A dirty tile is recompressed at quantization `q`
  on flush, so write-then-read through the mapping is **not identity** (`X′ ≈ X`). Treating it as a plain
  uncompressed array silently introduces codec loss — surfaced explicitly, not hidden.
- **Dirty-tile overlay.** Writes land in an uncompressed, byte-budgeted overlay of whole 64³ tiles kept
  verbatim until an explicit **flush/seal** (or eviction) recompresses them. Reads of dirty tiles hit the
  overlay (exact), so within a session writes are lossless until sealed.
- **Write amplification is whole-tile.** The DCT tile is atomic — touching one voxel dirties and
  recompresses its entire 64³ tile. `fxfs` batches dirty pages → whole-tile recompress on flush.
- **Page→tile.** A 4 KiB page fault decodes the whole containing 64³ tile (256 KiB at u8); the flat offset
  → (z,y,x) → Morton chunk key → blob mapping is computed both directions.

### Bounds
- The decoded-tile cache and the dirty overlay are **byte-budgeted** (configurable, default a few GiB) —
  resident RAM stays bounded by the working set, never the volume (consistent with the 8 GB out-of-core
  ceiling). Decoded tiles are 64× the compressed size, so the budget governs eviction (SIEVE).
- `--threads` controls decode concurrency; absent/fetch-failed tiles follow the codec's tri-state +
  `Expected` rules (a fetch error is an `EIO`, never silent air).

## Consequences
+ Any tool can `mmap` a `.fxvol` and treat it as an uncompressed ZYX array; the kernel page cache does the
  caching; no per-tool codec plumbing. No kernel-panic risk; fits the Docker/musl/header-only model. The
  Phase-3 cache also accelerates the in-process pipeline independently of FUSE.
− **Lossy writeback** (default-RO mitigates; writable mounts must document it). − **Write amplification** to
  whole 64³ tiles. − FUSE adds context-switch/IPC overhead vs a raw local `mmap` (acceptable; userfaultfd
  or a kernel module are the escape hatches if it ever bites). − Single-host (no network FS). − New
  dependency **libfuse** (§2.5) — firewalled to `src/fs/`, off by default (`-DFENIX_FS=ON`), like the GUI.

## References
[ADR 0006](0006-fxvol-v4-container.md) (the container + the decoded-tile cache this promotes to a public
API), [ADR 0005](0005-retire-wavelet-dct-tile-codec.md) (the codec decoded on fault). External: FUSE
low-level API + `mmap`/`writepage` semantics; Linux `userfaultfd(2)` (the deferred read-mostly alternative);
the SIEVE eviction policy (ADR 0006). Dtype-agnostic storage: `codec/dct_block.hpp` (no dtype in the
bitstream — `T` is widen-in/clamp-out only).
