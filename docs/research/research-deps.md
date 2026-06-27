# taberna dependency / IO subsystem research

Scope: the vendored third-party libs that taberna uses for network IO and
codec/parsing, plus the `src/io` wrappers that turn them into volume ingest.
Goal of the eventual C++26 rewrite: an EXTREMELY MINIMAL third-party surface.
All findings are read-only; nothing was modified.

Layout note: the dependency graph is
`libs3` (leaf) -> `matter-compressor` + `fysics` (both link the *shared* `s3`
target) ; `tiff` + `json` + `zlib` + `matter-compressor` -> `taberna_io`
(`src/io`). `jpg`/`png` are vendored but currently **not linked by anything**
in taberna (only their own standalone tests). VTK + Qt6 are GUI-only, opt-in.

---

## 1. libs3 — first-party minimal S3 client (`third-party/libs3/`)

### Purpose
A small, self-contained replacement for the S3 subset of the AWS C++ SDK.
~2700 lines of C23 in one `libs3.c`, public API in `libs3.h` (~580 lines, very
well documented). Only hard dependency is **libcurl >= 7.75** (needs curl's
built-in `CURLOPT_AWS_SIGV4`). JSON/XML responses are parsed by tiny internal
"scrapers" (`json_string_field`, `xml_tag`) rather than a real parser. curl
does not leak into consumers' TUs.

### Public API surface (by area)
- **URL parsing**: `s3_url_parse/free/to_https/is_s3`. Handles
  `s3://`, `S3://`, region-qualified `s3+us-west-2://`. Renders virtual-hosted
  `https://bucket.s3.region.amazonaws.com/key`.
- **Credentials**: `s3_credentials_from_env`, `s3_credentials_load(profile)`
  with a 6-step resolution chain (explicit profile via
  `aws configure export-credentials`; EC2 IMDSv2 queried directly + cached
  in-process w/ refresh-before-expiry; SSO profiles from `~/.aws/config`;
  default export-credentials; `~/.aws/credentials`+`config` INI; env vars).
  Optional per-request `s3_cred_provider_fn` for STS rotation on multi-hour jobs.
- **Client**: `s3_client_new(s3_config*)` / `s3_client_free`. Config covers
  static creds, cred provider, region, bearer/basic auth (for non-S3 HTTP),
  non-AWS endpoint override (MinIO/localstack, path-style), timeouts, retries,
  redirects, user-agent, and `coalesce_gap` for range merging.
- **Object ops**: `s3_get`, `s3_get_to_file` (streams to FILE*, constant mem),
  `s3_get_range`, `s3_get_range_into` (zero-alloc into caller buffer),
  `s3_get_parallel` (one big object split into concurrent ranges),
  `s3_head`, `s3_put`, `s3_put_file`, `s3_put_if_match` (optimistic-concurrency
  via ETag/412), `s3_delete`, `s3_copy` (server-side x-amz-copy-source),
  `s3_get_conditional` (If-None-Match / 304 revalidation).
- **Batched ranged GET** (the hot path for chunked zarr/mca rendering):
  `s3_get_batch` — array of `s3_range_req {url, offset, length, dst}`, runs
  `max_concurrency` transfers via `curl_multi`, coalesces adjacent ranges into
  one transfer and splits them back transparently. Optional `dst` buffer gives
  a zero-copy/zero-alloc path.
- **Async batch** (frame-budgeted IO for render loops): `s3_batch_submit` /
  `_poll(budget_ms)` / `_ready` / `_take` / `_cancel` / `_wait` / `_free`,
  plus `s3_prewarm` to pre-open N TCP+TLS connections on the calling thread.
  Async batches are single-thread-affine (borrow the thread's connection pool)
  and are NOT coalesced (so members can be cancelled independently).
- **Multipart upload**: `s3_multipart_create` -> `_upload_part`/`_upload_part_file`
  or `_upload_parts_parallel` -> `_complete`/`_abort`. For writing multi-GB
  recompressed volumes.
- **ListObjectsV2**: `s3_list` / `s3_list_ex` (max_keys, start_after) /
  `s3_list_all` (auto-paginating callback). Returns objects + CommonPrefixes.
- **Process-wide fast abort**: `s3_global_abort/reset/is_aborted` — flips an
  atomic so every in-flight transfer returns sub-ms on shutdown.

### S3 operations / auth / threading
- **Auth** (`apply_auth`): when AWS creds present, uses curl's `CURLOPT_AWS_SIGV4`
  (`aws:amz:<region>:s3`) + `CURLOPT_USERPWD` for SigV4 signing, adds
  `x-amz-security-token` for STS. Else bearer token, else basic auth, else
  anonymous/unsigned (the path used for the **public open-data buckets** —
  matter-compressor's first client attempt is anonymous, see below).
- **Threading model**: an `s3_client` is shareable across threads. Each thread
  gets its own curl easy handle (`thread_handle()`, pthread TLS w/ destructor)
  and its own batch connection pool (`thread_batch()`), so concurrent calls
  never contend on a handle. A process-global semaphore (`conn_cap`) bounds
  total concurrent connections. `CURLOPT_NOSIGNAL` is set (required for
  thread-safe timeouts). Credential cache + abort flag are process-wide and
  atomic. `do_request` retries (default 3) on 5xx/401/403/network with
  exponential backoff + jitter, using a **low-speed watchdog** (1KB/s for
  transfer_timeout_s) rather than a hard total-time cap so slow big downloads
  finish instead of restarting forever.
- **How scroll data is fetched**: taberna never calls libs3 from `src/io`
  directly. The consumers are the submodules:
  - `matter-compressor/src/mc_volume.c` + `mc_zarr.c`: opens an `s3_client`
    (anonymous for public buckets), reads zarr chunks and `.mca` shards with
    `s3_get` / `s3_get_range` and, on the hot path, `s3_get_batch` with 16–32
    way concurrency. This is how remote Vesuvius zarr volumes and `.mca`
    archives are streamed.
  - `fysics/zarr_io.c` + `fysics/tools/mca_export.c`: same pattern — a global
    `s3_client`, `s3_get` for single chunks, `s3_get_batch` for batched chunk
    pulls; `mca_export` is the tool that downloads a zarr scroll and writes the
    `.mca` archive.
  - `matter-compressor/tools/mc_fetch.c`: standalone fetch utility.

### Assessment
This is a serious, well-engineered, load-bearing component — far more than a
toy. It is the single most valuable vendored lib. The async/batch/coalescing
machinery is exactly tuned to this pipeline and has no drop-in std/3rd-party
equivalent short of pulling in the full aws-sdk-cpp (huge) or writing curl_multi
glue from scratch.

---

## 2. tiff — vendored minimal TIFF + custom codec (`third-party/tiff/`)

Two distinct things in one ~1900-line `tiff.c`:

### (a) Baseline TIFF reader/writer + multi-page volume
- `tiff_write` / `tiff_read`: single-strip *uncompressed* baseline TIFF that
  opens in ImageJ/GIMP/Preview/libtiff. Pixel model row-major chunky, all
  channels same type. Sample types u8/u16/u32, s8/s16/s32, f16/f32.
- Reader handles real-world uncompressed TIFFs: either byte order, single/multi
  strip, SHORT or LONG count/offset tags. Rejects compressed (except LZW in the
  volume path), tiled, planar, BigTIFF, mixed depths.
- **Multi-page volume**: `tiff_read_volume` / `tiff_write_volume` — z-stack of
  equally-sized IFDs into one z-major buffer. Supports Compression=1 (none) and
  **5 (LZW, MSB-first, Predictor=none)** via an in-file `lzw_decode`. This is
  the format the Vesuvius surface-detection 320^3 u8 stacks ship in, and the
  format the official PIL/metric reads back.

### (b) Custom near-lossless 2D codec (NOT standard TIFF)
`tiff_compress` / `tiff_decompress` with `tiff_codec_params {tau, step}`. A
per-plane 64x64 float DCT + a binary range coder (CABAC-style: `renc`/`rdec`,
adaptive bit models, Exp-Golomb, EOB/magnitude contexts, scan order). Guarantees
every reconstructed sample within `tau` of the original. ~Half the file. This is
a self-described private blob, decodable only by `tiff_decompress`. It overlaps
conceptually with matter-compressor's job and is largely independent of the
plain TIFF IO.

### Assessment
The baseline reader/writer + LZW volume path is genuinely needed (it ingests the
real training data) and is small. The bundled DCT/range-coder codec is a large,
specialized chunk that may belong with matter-compressor or be droppable
depending on whether anything calls it (no `src/io` consumer does).

---

## 3. jpg / png — vendored minimal codecs (`third-party/jpg`, `third-party/png`)

- **png** (~21KB `png.c`): baseline non-interlaced PNG read/write with a
  **self-contained DEFLATE/INFLATE** (no zlib): writer does fixed-Huffman
  deflate + adaptive per-row filtering + CRC32; reader does full inflate
  (stored/fixed/dynamic) + filter reversal. 8/16-bit, 1–4 channels, color types
  0/2/4/6. Rejects palette/interlaced. 16-bit exchanged native-endian.
- **jpg** (~25KB `jpg.c`): baseline (sequential DCT, Huffman) JFIF only, own DCT
  + standard Annex-K tables (no libjpeg). Write 4:4:4; read 1/3 components,
  4:2:0/4:2:2/4:4:4, restart intervals. No progressive/arithmetic/12-bit.
- Both are deliberately minimal, mirror the `tiff` style, and have round-trip
  tests. **Neither is currently linked into taberna** — they are vendored but
  unused (only `*_test` standalone builds). They look like staged building
  blocks for a future viewer/export path.

## 4. json — tiny JSON parser (`third-party/json/`)

~150-line `json.c` + `json.h`. Parses to a value tree (`json_value`:
null/bool/num/str/arr/obj), borrowed accessors `json_obj_get` / `json_arr_at`
and typed getters `json_as_num/int/str`. Explicitly NOT RFC8259-complete: no
`\u` escapes, no surrogate pairs. Sole purpose: read provenance blobs from the
`.mca` metadata carveout. Consumed by `src/io/mca.c` (`mca_roi_origin` reads
`roi.origin[z,y,x]`). Compiled directly into `taberna_io` (see CMake).

## 5. vtk — Kitware VTK (GUI only, `third-party/vtk/`)

Large external lib, **not** part of the core C pipeline. Used only by the opt-in
Qt6 viewer (`gui/`, `-DTABERNA_GUI=ON`). From `gui/CMakeLists.txt` the required
VTK 9 modules are: `CommonCore`, `CommonDataModel`, `RenderingCore`,
`RenderingVolume`, `RenderingVolumeOpenGL2`, `RenderingOpenGL2`, `GUISupportQt`,
`InteractionStyle`. Classes actually used (grep of `gui/`): `vtkImageData`,
`vtkGPUVolumeRayCastMapper`, `vtkSmartVolumeMapper`, `vtkVolume`,
`vtkVolumeProperty`, `vtkColorTransferFunction`, `vtkPiecewiseFunction`,
`vtkRenderer`, `vtkGenericOpenGLRenderWindow`, `vtkNew`. I.e. a 2x2 GPU
volume-raycast + MPR slice frontend. `gui/volume_source.cpp` bridges
matter-compressor region/slice reads into `vtkImageData`.

---

## 4 (src/io). How src/io wraps these for volume ingest

`src/io` (`taberna_io` static lib) is the experiment-phase volume ingest layer.
It links `tiff`, zlib, `matter_compressor`, `json` (+ OpenMP, mimalloc).

- **tiff_vol.c/.h**: thin wrapper over the vendored `tiff_read_volume` /
  `tiff_write_volume`. `tiff_load_u8` loads a single-channel u8 multi-page TIFF
  into taberna's z-major/x-fastest `u8*` (`v[(z*ny+y)*nx+x]`), transferring
  ownership of `tiff_volume.data` (no copy); rejects multi-channel/non-u8.
  `tiff_save_u8` writes back. Also contains a **constructor that fixes OpenMP
  team size** to the online core count (containers often report affinity=1),
  since every tool links `taberna_io`.
- **nrrd.c/.h**: self-contained minimal NRRD (ASCII header + binary blob).
  Reads 3D volumes, encodings `raw` and `gzip` (via **zlib** `inflateInit2(15+32)`
  auto-detect), types uint8/uint16/float, little-endian only. `nrrd_read_f32`
  loads + casts to f32 in taberna layout. Writers emit `raw` NRRD. Used for the
  friction-free Vesuvius feasibility data (instance-labels-harmonized cubes).
  Axes stored fastest-first (x,y,z) matching taberna layout.
- **mca.c/.h**: wraps matter-compressor's **read** API to pull u8 regions out of
  `.mca` archives (real full-scroll data). `mca_dims` mmaps the file and reads
  geometry; `mca_open`/`mca_close`/`mca_read` give a persistent handle (one mmap
  + node-table parse, many region reads); `mca_load_region` is the per-call
  convenience. `mca_read` fills a malloc'd z-major u8 buffer via
  `mc_archive_read_region`. `mca_metadata`/`mca_roi_origin` borrow the archive's
  JSON provenance and parse `roi.origin` via the `json` lib. `mca_find_region`
  uses `mc_archive_sample_boxes` to locate a material-rich box.

So ingest has three on-disk/remote sources funneled into the same
z-major/x-fastest u8 (or f32) layout: **TIFF stacks** (surface-detection data),
**NRRD** (feasibility cubes), and **.mca** (full scroll, which itself fetches
zarr from S3 via libs3 inside matter-compressor).

## 5. mimalloc tuning (`src/io/mimalloc_tune.c`)

Tiny. When built with `TABERNA_MIMALLOC=ON`, mimalloc is *linked* (not
LD_PRELOAD'd) into every tool through `taberna_io`/`taberna_malloc`, overriding
malloc/free automatically. A `__attribute__((constructor))` runs before `main`
and sets `mi_option_purge_delay = 50` (ms). Rationale documented inline: default
(~10ms) holds reserved segments and inflates RSS between pipeline phases;
purge_delay=0 returns memory eagerly but costs a syscall per free (~50% slower);
50ms is the sweet spot (bounded RSS, no per-free churn). Pure tuning, no API.

---

## 6. C++26 rewrite: keep / replace / rewrite (minimal-deps lens)

| Lib | Recommendation | Rationale |
|-----|----------------|-----------|
| **libs3** | **KEEP / port** (keep libcurl as the one network dep) | Load-bearing and genuinely good; the batch/coalesce/async + SigV4-via-curl design has no cheap std equivalent and is exactly tuned to chunked zarr/mca streaming. The std alternative (aws-sdk-cpp) is enormous — the opposite of minimal. Rewrite the C API as RAII C++26 over libcurl; keep behavior. libcurl is the single justified network dependency. |
| **tiff (baseline + LZW volume)** | **KEEP / rewrite thin** | Needed to ingest real training stacks; small. No good header-only C++ TIFF that's this minimal; libtiff is heavier and adds a dep. Re-implement the ~uncompressed+LZW reader/writer in C++26. |
| **tiff custom DCT/range codec** | **MOVE or DROP** | Large and overlaps matter-compressor. Keep only if a caller needs it (none in `src/io` today); otherwise fold into the compressor module. |
| **nrrd** | **KEEP (trivial), zlib via std/3rd-party** | Format is a 200-line ASCII-header+blob reader; not worth a dependency. Only external need is gzip inflate — keep zlib (or its successor) as the decompression dep. Easy C++26 rewrite. |
| **json** | **REPLACE with one small header-only lib OR keep** | It's ~150 lines but intentionally incomplete (no `\u`). For a clean rewrite, a single header-only JSON lib (e.g. a minimal one) or a hand-written parser is fine; only need is reading `.mca` provenance. Don't pull a heavy JSON framework. Lowest-risk: rewrite the tiny parser in C++26 and keep zero JSON deps. |
| **jpg / png** | **DROP unless the viewer needs them** | Currently unused by taberna (vendored, only self-tests). If an export/screenshot path is added later, reconsider; PNG's self-contained inflate already overlaps the zlib used by nrrd. Don't carry unused codecs into the rewrite. |
| **VTK** | **KEEP, GUI-only, isolated** | Only the opt-in viewer needs it; it's not a core-pipeline dep. Keep it firewalled behind the GUI target with the same ~8 modules. A from-scratch GPU volume raycaster is out of scope; VTK is the pragmatic choice but must stay out of the core. |
| **mimalloc + tuning** | **KEEP (optional)** | Allocator choice is orthogonal to C++26; keep mimalloc as an opt-in link with the same purge_delay=50 tuning (a one-liner in a static init / `std::call_once`). |
| **OpenMP** | KEEP or move to std | The team-size constructor is a workaround; C++26 `std::execution`/`<thread>` could replace it, but OpenMP pragmas are deeply embedded in the compute libs — out of scope for the IO layer. |

**Bottom line for minimal deps**: the irreducible external set is **libcurl**
(network) + **zlib** (gzip) + **VTK/Qt** (GUI only) + optional **mimalloc**.
Everything else (S3 logic, TIFF, NRRD, JSON, the wrappers) is first-party and
small enough to port directly to C++26 rather than swap for a heavier standard
library. The biggest judgment call is the tiff custom codec (move into the
compressor) and jpg/png (drop until proven needed).
