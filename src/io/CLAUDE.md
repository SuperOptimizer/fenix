# io — CLAUDE.md

## Purpose
All data ingest/egress: source formats, the S3/HTTP client, the codec-archive IO, the
local transcode cache, and the data registry. The boundary between fenix and the world.
See `docs/research/villa-data.md`, `research-deps.md`, `research-fysics.md` (zarr/S3).

## Public API & key types
- **OME-Zarr reader** — v2 **and** v3 + sharded; raw + blosc2/zstd/lz4 chunks; missing
  chunk = fill (air); chunk-aligned region reads; ZYX. (We write the zarr logic; blosc2
  is the only compression dep.) **Read-only** — fenix never writes zarr.
- **Image/volume formats (first-party):** TIFF (multi-page volume stacks, tiled, 8/16-bit),
  PNG, JPEG (read+write), NRRD (utility). No libtiff/libpng/libjpeg.
- **S3/HTTP client** — a **C++ rewrite of libs3** over **libcurl**: anonymous + AWS
  SigV4, byte-range + batched/coalesced GETs, retry/backoff, low-speed watchdog.
- **Resource locator** — one URI type auto-detecting local path / `s3://` / `http(s)://`.
- **Codec-archive IO** — open/append/read `.fxvol` (delegates to `codec`); the streaming
  reader (flat + partial-fetch); **transcode cache** (cold zarr → `.fxvol` working store,
  bounded LRU eviction, configurable size).
- **TOML data registry** — `(scroll, energy, resolution[, segment]) → URLs/paths`
  (scrolls.yaml successor), resolved via the shared config reader.
- **Importers** (one-time): VC `.volpkg`/tifxyz + villa OME-zarr → our containers (to
  bootstrap from proofread patches/annotations). Import-once, then fully native.

## Inputs / outputs & formats
In: OME-zarr, `.fxvol`, TIFF/PNG/JPEG/NRRD, VC formats (import). Out: `.fxvol` (via codec),
PNG/JPEG/TIFF, the transcode cache.

## Dependencies
Intra: `core`, `codec`. Third-party: **libcurl, zlib, blosc2** (+ mimalloc). No others.

## Invariants & numerics
**Absent ≠ fetch-failed** (404 → air; other → retry then hard-fail — never silent air).
Atomic write-temp-rename; `fallocate`; `MAP_NORESERVE` mmap. LE-only; magic+version+hash;
reject unknown versions. Out-of-core: occupancy-guided, byte-budgeted backpressure.

## Performance notes
Streaming throughput = sort+coalesce near-adjacent blob GETs into few large sequential
reads; download pool → bounded queue → decode/compute pool. Local chunks mmap'd
(halo faults only touched pages).

## Gotchas / pitfalls
- uint16/uint32 paths must actually work (vesuvius-c punted on uint16 — we don't).
- The transcode cache must be **bounded** (mc_volume grew unbounded — a noted flaw).
- Keep the S3 hard-error-vs-404 distinction typed (`Expected`), not an out-param int.

## Status & TODO
**Implemented:** `surface.hpp` (`.fxsurf` r/w — hand-rolled LE binary of `core::Surface`: header +
coord/valid/normal/conf arrays; magic+version, version-rejecting; atomic write-temp-rename; the sink for
the out-of-core streaming tracer's per-tile fragments); `nrrd.hpp` (raw NRRD r/w); `zarr.hpp` (OME-Zarr v2 **raw** reader — `.zarray`
parse, chunk-aligned region reads, missing chunk = air, ZYX, uint8/16/32 + f32; fetches local
**or** remote chunks via `fetch_object`, parallelized with `parallel_for`); `s3.hpp` (libcurl
S3/HTTP GET — anonymous, `s3://`→https virtual-hosted, thread-local reused handle, low-speed
stall watchdog, exponential-backoff retry, **404→absent vs hard-fail distinct**; a fresh rewrite
of SuperOptimizer/libs3's design). Subcommands: `ingest` (NRRD→.fxvol), **`ingest-zarr`** (pull a
zarr region from local/`s3://`/`http(s)://` → .fxvol/.nrrd), **`export`** (.fxvol LOD level → NRRD),
**`finalize`** (.fxvol → SEALED coarse-first), **`fxinfo`** (dims/LODs/coverage/size/ratio), **`compare`**
(PSNR/MAE/max-abs between two NRRDs). Validated byte-exact against direct chunk fetch; pulled a 1024³
PHerc Paris 4 slab from S3 in ~60 s (729 chunks). End-to-end archive roundtrip verified on real CT:
`ingest crop512.nrrd q8 → finalize → export LOD0 → compare` = **37.51 dB, == the codec's q8 PSNR** (the
archive adds zero loss — verbatim blob storage + verbatim finalize copy); `close()` ftruncates the
fallocate'd tail so a 512³ q8 archive is ~2.8 MiB (not 64 MiB), all 4 LODs export.
**TODO:** blosc2/zstd chunk decompression (raw-only today); zarr v3 + sharded; SigV4 auth (write
+ private buckets); byte-range/coalesced batch GET (libs3 `s3_get_batch`); TIFF/PNG/JPEG; the
transcode cache + data registry. **Cache eviction settled — sharded SIEVE + refcount-pin + byte budget
([ADR 0006](../../docs/adr/0006-fxvol-v4-container.md)); applies to the in-RAM decoded-tile cache and is
the default for the disk transcode cache.** The `.fxvol` container layout (single-file, mmap'd 3-level
radix page-table, crash-safe append + S3 CAS) is now specified in ADR 0006 +
[`docs/design/fxvol-v4-layout.md`](../../docs/design/fxvol-v4-layout.md). Open ADRs: zarr v3 shard format
(ingest side); importer coverage.
