# io — CLAUDE.md

## Purpose
All data ingest/egress: source formats, the S3/HTTP client, the codec-archive IO, the
on-demand training cache, and quicklook/review tooling. The boundary between fenix and
the world. See `docs/research/villa-data.md`, `research-deps.md`, `research-fysics.md`
(zarr/S3), `docs/design/ink-hunt.md` (import-obj/project review pipeline).

## Public API & key types
- **OME-Zarr reader** (`zarr.hpp`) — v2 (`.zarray`) **and** v3 (`zarr.json`), including v3
  **sharding_indexed** (index_location=end, optional crc32c trailer); raw + gzip + blosc2
  chunk codecs (blosc needs `FENIX_HAVE_BLOSC2`; raw-zstd unsupported — typed reject).
  Missing chunk = fill (zarr's own "omitted = air" semantics); a present-but-wrong-size
  blob is a **hard error**, never silently demoted to air. `fetch_object()` is the one
  local/remote-unified fetch primitive (local via `::fopen`/errno, remote via `http_get`).
  `read_zarr_region<T>()` (T=u8 native or f32) does chunk-aligned parallel region reads
  (`parallel_for_io`, fan-out capped — `FENIX_ZARR_FETCH_THREADS`, default 24 remote /
  unbounded local). `copy_zarr_region_local()` raw-copies a chunk-aligned sub-box to a
  standalone local zarr v2 group (atomic per-chunk write-temp-rename), preserving dtype.
  **Read-only** — fenix never writes remote/production zarr.
- **Image/volume formats (first-party, no libtiff/libpng/libjpeg):** `tiff.hpp` (classic
  LE + BigTIFF reader, strips/tiles, uncompressed **or LZW** w/ horizontal or
  floating-point predictor, f32/u8/u16 grayscale); `png.hpp` (reader only: 8-bit
  gray/RGB/palette/gray+alpha/RGBA + 1/2/4-bit gray + 16-bit gray downshifted;
  non-interlaced only, Adam7 typed-rejected; zlib is the only dep); `jpeg.hpp` (first-party
  **baseline encoder**, `Image` struct — gray or RGB, no chroma subsampling — used by
  `slice`/`video`/`project`; no JPEG reader). **NRRD support was removed project-wide** —
  `.fxvol` (the DCT codec archive) is the only volume container fenix reads or writes; the
  bench/tracer loaders read `.fxvol` (or a local `.zarr` level-0 region) directly.
- **S3/HTTP client** (`s3.hpp`) — libcurl GET, anonymous + **AWS SigV4** (private
  buckets — `sigv4.hpp`, first-party SHA-256/HMAC, no OpenSSL), thread-local
  `ThreadHandle` (RAII-wrapped easy handle: cleans up at thread exit — `parallel_for_io`
  spawns short-lived threads per fetch call, so a bare handle would leak), a process-wide
  `CURLSH` share pool (DNS + TLS session cache; connection cache deliberately **not**
  shared — cross-thread sharing SEGV'd libcurl 8.5 at high fan-out), low-speed stall
  watchdog (not a hard transfer timeout), exponential backoff+jitter, HTTP/2 multiplex.
  **404 → absent(air); everything else → retry then hard Error** — never silent air.
- **`sigv4.hpp`** — pure function of (method,host,path,query,timestamp,creds); creds
  from `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN`; region parsed
  from the virtual-hosted host.
- **`scan_meta.hpp`** — parses the bm18/nabu `metadata.json` sidecar (voxel µm, energy,
  Paganin delta/beta, unsharp params, f32 export window) via targeted string scan (no
  JSON DOM), local or remote.
- **`cached_volume.hpp`** — `CachedVolume`: on-demand training chunk cache, a `.fxvol`
  archive lazily filled from a (local/remote) zarr. `gather_box_f32`/`gather_box_u8` fetch
  any ABSENT 64³ chunk on first touch (fetch runs lock-free; only the coverage
  claim+append is serialized — an in-flight set lets peer fetches of different regions run
  concurrently), then every later query is served locally. One writer process per cache
  file (exclusive `flock` on a `.lock` sidecar — two processes appending the same archive
  corrupts coverage). Source-identity binding (`.src` sidecar) rejects re-pointing an
  existing cache at a different zarr. Offline mode: an unreachable source is fine if the
  cache already fully covers what's asked (dims come from the archive). Bounded: RAM via
  `reserve_cache`, disk via `disk_budget` (drop+recreate on overflow, hot set refills).
  Retries a failed fill by re-claiming (transient S3 error must not kill a multi-hour run).
- **`import_obj.hpp`** — `import-obj`: VC segment OBJ (uv-textured triangle mesh) →
  `.fxsurf`, barycentric-rasterized over a regular uv grid; optional affine or VC
  `transform.json` (XYZ→ZYX remap, `p_new = post_scale*(M*(p_old*pre_scale))`) to
  register cross-scan segments. The ink-hunt pipeline's entry point.
- **`project.hpp`** — `project`: z-projection (max/mean) of an `.fxvol` prob volume to
  JPEG — the ink-hunt review image.
- **`slice.hpp`** — `slice` (any-axis 2D slice → JPEG, optional red/tinted overlay of a
  second volume) and `video` (H.264 .mp4 scrub via an external `ffmpeg` process — NVENC
  probed, else libx264; fenix links no video library).
- **`surface.hpp`** — `.fxsurf` v3 r/w: coords go through `codec/tile2d`'s per-64²-tile
  affine+tangent-frame coordinate front-end (height field + 2 near-zero remainders),
  DCT-64² + dead-zone quant + rANS, encode-time-**verified** ≤ `coord_tau` (default 1/4
  voxel) 3D max error per valid cell with raw-tile fallback; normals/conf ride the same
  front-end at their own tolerances; validity rANS'd. Magic+version, atomic
  write-temp-rename.
- **`tifxyz.hpp`** — `import-tifxyz`: VC x/y/z.tif + meta.json scale → `.fxsurf`
  (XYZ→ZYX, −1 = invalid). The dir may be **local or an http(s)/s3 URL** (open-data
  bucket segment dirs) — files go through `fetch_object`, one code path.
- **`cache.hpp`** — the local artifact cache (fetch remote data ONCE, recompress to
  fenix-native, serve from disk): `default_cache_dir()` (`$FENIX_CACHE` >
  `$XDG_CACHE_HOME/fenix` > `~/.cache/fenix`), `cache_key()` (readable tail + full-URL
  hash — same-tail sources can't collide), `cached_surface()` (remote/local tifxyz →
  cached `.fxsurf`, `.src` identity sidecar), and **`CachedPyramid`** — a
  `codec::VolumeSource` over an OME-zarr **multiscale** root: one `CachedVolume`
  (lazily-filled `.fxvol`) per pyramid level under `<cache>/vol/<key>_l<k>.fxvol`, so
  the viewer engine streams chunks on first view and hits disk after. Level discovery
  probes `<root>/<k>/.zarray` with an existing-cache fallback (offline mode);
  levels are validated to actually 2× halve. Implements the `codec::VolumeSource`
  best-effort API: `chunk_state`/`block16_local`/`gather_box_f32_local` never touch the
  network; `schedule_chunk` queues a background fill on an internal 4-thread
  `WorkerPool` at 2-aligned 2×2×2 chunk-GROUP granularity (=128³ = one open-data zarr
  object, so neighbours don't re-download it 8×), deduped in-flight;
  `ready_generation()` bumps per landed group (the viewer's redraw edge-trigger).
  `reserve_cache` gives EVERY level half the budget (a cap, not an allocation — a
  depth-halving split starved coarse levels into per-frame re-decode).

## Inputs / outputs & formats
In: OME-Zarr v2/v3(+sharded), `.fxvol`, TIFF (classic+BigTIFF)/PNG, VC tifxyz/OBJ+
transform.json (import). Out: `.fxvol` (via codec), JPEG/MP4 (quicklook), `.fxsurf`.
No zarr writer (except the local raw-copy sub-box helper) and no PNG writer.

## Dependencies
Intra: `core`, `codec`, `geom` (import-obj mesh reading). Third-party: **libcurl, zlib**
(+ **blosc2** gated behind `FENIX_HAVE_BLOSC2`) + mimalloc. `sigv4.hpp` is dependency-free
(hand-rolled SHA-256/HMAC). `video_cmd` shells out to an external `ffmpeg` binary at
runtime (not linked). No libtiff/libpng/libjpeg/libcurl-alternatives.

## Invariants & numerics
**Absent ≠ fetch-failed**: 404/ENOENT/ENOTDIR → air; every other errno/HTTP status →
retry+backoff then hard `Error` — never silent air. A present-but-wrong-size chunk blob
is ALSO a hard error (not a demotion to air) — this was a real bug (see Gotchas) and is
now enforced with strict size equality. Atomic write-temp-rename for `.fxsurf` and for
`copy_zarr_region_local`'s per-chunk writes. LE-only; magic+version; reject unknown
versions. `-fno-exceptions`-safe parsing: CLI/JSON numeric parsing goes through
`std::from_chars` (never `sto*`, which throws). Out-of-core: `export-scroll` is
occupancy-guided (coarse-pyramid air-skip) + resumable (coverage tri-state is the
checkpoint) + byte-budgeted (bounded prefetch queue).

## Performance notes
- **S3 fetch:** `parallel_for_io` sizes fetch fan-out directly (does NOT clamp to
  `cpu_budget()` — network-bound work needs more concurrency than the box has cores);
  remote fan-out capped (`FENIX_ZARR_FETCH_THREADS`, default 24) to avoid self-congestion
  on a CPU-quota-limited box, local reads stay unbounded (no endpoint contention).
- **`export-scroll`:** producer/consumer prefetch pipeline (N producers fetch regions →
  bounded queue → single consumer DCT-encodes/writes/commits) overlays network wait with
  compute; default 8 producers tuned for ~100 ms-RTT WAN (`FENIX_EXPORT_PREFETCH`
  overrides). Commits are **batched** (`commit_every=64` default) — a commit COW-copies
  the touched page-table path, so per-region commits stranded ~45% of committed bytes as
  orphaned index nodes (measured); batching shares one COW across N adjacent regions.
  `KMP_BLOCKTIME=0` + `OMP_WAIT_POLICY=passive` set before first OMP use so idle workers
  sleep instead of spinning through ~100 ms network waits.
- **`CachedVolume`:** fills run OUTSIDE the lock (only the coverage-claim scan/append is
  serialized) — holding the exclusive lock across the fetch collapsed throughput to
  ~0.3 draws/s under 8 concurrent feeders; unlocked-fetch fixed that.
- **libcurl:** thread-local handle reuse (connection/TLS) + process-wide `CURLSH`
  DNS/TLS-session cache + HTTP/2 multiplexing; `Content-Length`-based body reservation
  avoids doubling reallocs on multi-MB chunks.
- Local zarr chunks: plain `fopen`/`fread` (not mmap) today; `fetch_object`'s local path
  matches the remote path's absent-vs-hard-fail semantics via errno.

## Gotchas / pitfalls
- **A leaked libcurl easy handle per ephemeral fetch thread was a real bug** (found +
  fixed 2026-07-02, see `docs/review/2026-07-02/io.md`): `parallel_for_io` spawns
  short-lived `std::thread`s per call, so a bare `thread_local CURL*` leaked at every
  thread exit — unbounded over a multi-hour export. Fixed via the `ThreadHandle` RAII
  wrapper in `s3.hpp`. Don't regress this if `thread_handle()` is touched again.
  Do NOT share `CURL_LOCK_DATA_CONNECT` across threads via `CURLSH` (SEGV'd libcurl 8.5
  under real fan-out, reproduced 2026-07-02) — DNS/TLS-session sharing only.
  Symmetric bug fixed same day: local `fetch_object` used to conflate ANY open failure
  (EACCES/EIO/EMFILE) with ENOENT/"absent"; now errno-authoritative. And a
  present-but-short chunk blob used to silently demote to fill/air; now a hard error.
- uint16/uint32 zarr paths must actually work (vesuvius-c punted on uint16 — we don't).
- These CT scrolls are u8 (`|u1`) — **never** widen a whole volume to f32 in RAM
  end-to-end (ingest/export/transcode all special-case `dtype_size==1` to stay
  u8-native; the DCT codec widens one 64³ tile at a time internally regardless).
- One writer process per `CachedVolume` cache file — enforced by `flock`, not just
  convention; two feeders sharing a cache cratered throughput with silent corruption risk.
- `CachedVolume`'s source-identity `.src` sidecar exists because re-pointing a cache at
  a different zarr silently serves wrong-volume voxels otherwise.
- Keep the S3/zarr hard-error-vs-absent distinction typed (`Expected`), never an
  out-param bool/int.
- `ffmpeg` for `video` is an external runtime dependency (not linked) — `video_cmd`
  fails loudly with `Errc::io_error` if the binary can't launch; SIGPIPE is
  scope-ignored around the frame-write loop so a dead ffmpeg surfaces as a clean Error
  instead of killing the process.

## Status & TODO
**Implemented (CLI subcommands, self-registered, see `io.hpp`):**
`ingest-zarr` (zarr region, local/s3/http → .fxvol or raw `.zarr` sub-box, u8-native),
`export-scroll` (whole zarr level → .fxvol, out-of-core + resumable + air-skip prefetch
pipeline), `transcode` (.fxvol → .fxvol at new DCT q, u8-native or `scale255` for [0,1]
prediction fields), `finalize` (.fxvol → sealed coarse-first), `fxinfo` (.fxvol OR
.fxsurf inspector, `--json`), `compare` (PSNR/MAE/max-abs between two .fxvol volumes), `fxupgrade`
(.fxvol v4→v5 in-place superblock patch, adds source-dtype tag), `import-obj` (VC OBJ +
optional affine/transform.json → .fxsurf), `import-tifxyz` (VC tifxyz → .fxsurf),
`project` (.fxvol z-projection → JPEG), `slice`/`video` (any-axis quicklook JPEG/MP4,
optional overlay). All formats: zarr v2+v3(+sharded, blosc2/gzip/raw), TIFF (classic+
BigTIFF, uncompressed+LZW), PNG (read), JPEG (write only), `.fxsurf` v3,
S3 anonymous+SigV4. `CachedVolume` on-demand training cache is live and used by the ML
feeder pipeline (see `src/ml/CLAUDE.md`). The 2026-07-02 io review
(`docs/review/2026-07-02/io.md`) found and fixed the CURL-handle-leak, local-fetch-
absent-conflation, and short-blob-silent-air bugs listed above; remaining lower-severity
items from that review (untrusted-input hardening on archive/zarr sizes / fxsurf dims overflow)
should be checked against current code before assuming fixed.
**TODO / not yet implemented:** zarr **writer** (fenix remains read-only for zarr; the
only zarr "write" path is the local raw-copy sub-box helper, which is a byte-verbatim
copy, not an encoder); PNG **writer**; JPEG **reader**; TIFF writer; a general TOML data
registry (`(scroll,energy,resolution[,segment]) → URLs`, the scrolls.yaml successor) —
not present in `src/io/` today, config lookups are ad hoc per-tool; SigV4 **PUT**
(CAS uploads — GET-only today); raw-zstd zarr v3 codec (typed-rejected, blosc/gzip/raw
only). The `.fxvol` container layout is specified in
[ADR 0006](../../docs/adr/0006-fxvol-v4-container.md) +
[`docs/design/fxvol-v4-layout.md`](../../docs/design/fxvol-v4-layout.md). Open ADRs:
zarr v3 shard format refinements; importer coverage beyond OBJ/tifxyz.
