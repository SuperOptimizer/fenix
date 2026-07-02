# Review — unit "io" (src/io/: s3.hpp, zarr.hpp, nrrd.hpp, jpeg.hpp, slice.hpp, surface.hpp, scan_meta.hpp, io.hpp CLI glue)

Overall assessment: the module is in decent shape for its declared maturity — the S3 retry/backoff design, the export-scroll prefetch pipeline, and the absent-vs-failed typing on the *remote* path are all sound and the concurrency in `export_scroll` (queue + CV + cancel) is correct. The severe problems cluster in exactly the places the project's own invariants call out: the **local** fetch path silently converts open failures into air, a **short/truncated chunk blob silently becomes air**, and the thread-local libcurl handle design leaks a CURL easy handle per ephemeral `parallel_for_io` thread — unbounded over a multi-hour scroll export. Below that there is a tier of untrusted-input robustness gaps (zarr dtype/endianness/fill_value, NRRD sizes, fxsurf dims overflow, `std::stoll` throwing under `-fno-exceptions`) that are real fuzz-surface bugs given the first-party-decoder mandate.

## [high/resource-safety] Thread-local CURL easy handle leaks once per ephemeral fetch thread — unbounded over a scroll export

**Verdict:** CONFIRMED — s3.hpp:121-144 stores a raw thread_local CURL* with no RAII; curl_easy_cleanup appears nowhere in src/ (grep-verified). parallel.hpp:192-196 shows parallel_for_io spawns and joins fresh std::threads on every call, and zarr.hpp:165 routes every read_zarr_region remote fetch through it (fetch_threads=24), so each spawned worker leaks one easy handle at thread exit. io.hpp:277-314 confirms export-scroll's 8 producers call read_zarr_region once per region across a tens-of-thousands-of-regions worklist in a multi-hour unattended run, making the leak unbounded and unaccounted by the RAM budget. The 'lives for the thread's lifetime' comment assumes long-lived threads and is not a documented intentional leak (unlike SharePool at s3.hpp:91, which is explicitly annotated as such).

**Fix notes:** Fix is correct. Two magnitude/implementation notes: (1) parallel.hpp:183 clamps the team to the work span, and a default export region is ~8 chunks (io.hpp:259), so the leak is ~7 handles per region, not ~23 — still unbounded. Leaked state is the easy-handle struct + buffers, not sockets/TLS (those live in the shared pool), so the multi-GB estimate is plausible but high-end. (2) In the RAII holder, keep the existing one-time-setopt lambda as the initializer, and there is no destruction-order hazard with shared_pool() since the share is intentionally never cleaned up; curl_easy_cleanup on a share-attached handle safely detaches it.

**Location:** src/io/s3.hpp:121-144 (`detail::thread_handle`), interacting with src/core/parallel.hpp:176-199 (`parallel_for_io`) and src/io/zarr.hpp:165

**Evidence:**
```cpp
inline CURL* thread_handle() {
    thread_local CURL* h = [] {
        CURL* c = curl_easy_init();
        ...
        return c;
    }();
    return h;
}
```
`h` is a raw pointer with no destructor — `curl_easy_cleanup` is never called. The comment says "Lives for the thread's lifetime", which was designed for long-lived threads. But `read_zarr_region` fetches via `parallel_for_io`, which **spawns fresh `std::thread`s on every call and joins them** (parallel.hpp:194-198). Each new thread creates its own easy handle on first `http_get` and leaks it at thread exit.

**Failure scenario:** `fenix export-scroll` on a remote zarr: 8 producer threads each call `read_zarr_region` per region, each spawning ~24 fetch threads (`fetch_threads = 24`). Every region read therefore leaks ~23 easy handles (each tens of KB of libcurl state). A full-scroll export is tens of thousands of regions → hundreds of thousands of leaked handles → multi-GB RSS growth over an unattended multi-hour export, on top of the byte-budgeted region buffers. Memory the RAM-budget accounting knows nothing about.

**Suggested fix:** wrap the thread-local in a small RAII holder (`struct H { CURL* c; ~H(){ if(c) curl_easy_cleanup(c);} }; thread_local H h{...};`) so the handle is destroyed at thread exit. Connection/TLS reuse across short-lived threads is already provided by the SharePool (`CURL_LOCK_DATA_CONNECT`), so cleanup does not sacrifice the warm-cache goal. Alternatively give `parallel_for_io` a persistent pool, but the RAII fix is the minimal correct change.

**Outcome:** fixed — `thread_handle()` in src/io/s3.hpp now wraps the CURL easy handle in a `ThreadHandle` RAII struct (move-disabled) whose destructor calls `curl_easy_cleanup`; the thread_local is now `thread_local ThreadHandle t; return t.h;` instead of a bare `thread_local CURL*`. All setopt calls unchanged (still set once in the constructor, no per-call reset). Verified via full build + `test_zarr`/`test_trace_stream` (exercises `parallel_for_io` fan-out through `read_zarr_region`).

## [high/correctness] Local `fetch_object` conflates ANY open failure with "absent" — transient IO error silently becomes air

**Verdict:** CONFIRMED — At src/io/zarr.hpp:30-31: `std::ifstream f(...); if (!f) return std::nullopt;` conflates every open failure (EMFILE/EACCES/EIO/ENOMEM) with ENOENT, and the slurp at line 32 is unchecked so a partial read yields a short buffer. Callers do not compensate: read_zarr_region (zarr.hpp:178-180) treats nullopt OR a short blob (`blob->size() >= ccount*esz` check) as fill-value air; copy_zarr_region_local (zarr.hpp:286) silently omits the chunk; segment/trace_stream.hpp:25-29 consumes the result. The local path runs with unbounded parallelism (fetch_threads=0 for local, zarr.hpp:150-159) and is exercised by ingest-zarr/export-scroll via io.hpp:86,101,112,180,286. This violates the explicitly documented invariant in src/io/CLAUDE.md ("Absent ≠ fetch-failed ... other → retry then hard-fail — never silent air") and root CLAUDE.md §2.4; the function's own doc comment (zarr.hpp:24-26) promises the distinction the local branch fails to implement. Not a documented stub — the module Status lists zarr.hpp as implemented and "validated".

**Fix notes:** Proposed fix is correct in direction. Refinements: (1) use ::open(O_RDONLY)/fstat + read loop (or std::fopen with errno) since -fno-exceptions rules out iostream exception state; treat only ENOENT (and arguably ENOTDIR mid-path) as absent; (2) per the io invariant, transient errno (EMFILE/EAGAIN/EINTR/EIO) should get retry+backoff before hard-failing, matching the s3.hpp remote path; (3) verify completeness against fstat st_size rather than an expected byte count — fetch_object is generic (also fetches .zarray, whose size is unknown a priori); the caller-side ccount*esz check at zarr.hpp:179 can then become a hard error for present-but-short chunks instead of silent air; (4) adjacent same-class bug worth fixing in the same pass: copy_zarr_region_local (zarr.hpp:296-297) never checks the ofstream after write, so a full-disk/EIO write also silently succeeds.

**Location:** src/io/zarr.hpp:30-31

**Evidence:**
```cpp
std::ifstream f(root + "/" + sub, std::ios::binary);
if (!f) return std::optional<std::vector<u8>>(std::nullopt);  // missing chunk = air
```
`ifstream` open failure does not distinguish ENOENT from EMFILE / EACCES / EIO / ENOMEM. The remote path carefully separates 404 from hard failure (s3.hpp), but the local path — used for every local zarr read, including the ingest and export paths — treats *every* failure as a missing chunk.

**Failure scenario:** during a local-zarr `export-scroll`/`ingest-zarr`, fd exhaustion (plausible: 8 producers × up to `cpu_budget()` fetch threads each holding an open file, plus the leaked CURL handles from the finding above) or a flaky disk returns EMFILE/EIO on open. The chunk is silently written into the `.fxvol` as fill-value air. This is the exact violation the root CLAUDE.md flags as non-negotiable: "A transient fetch error must never silently become air."

**Suggested fix:** open with a mechanism that exposes errno (e.g. `::open`/`std::fopen` and check `errno == ENOENT`), returning `std::nullopt` only for ENOENT and `err(Errc::fetch_failed, ...)` for everything else. Also check the *read* succeeded (`f.good()` / byte count) after the istreambuf slurp — a mid-read EIO currently yields a short buffer (see next finding).

**Outcome:** fixed — `fetch_object` in src/io/zarr.hpp now opens via `std::fopen` (errno-authoritative under -fno-exceptions), classifies ENOENT/ENOTDIR as absent (nullopt) and every other errno as `err(Errc::fetch_failed, ...)`; the read loop checks `ferror()` after each `fread` so a mid-read EIO also hard-fails instead of returning a silent short buffer. Regression tests added (see test_zarr entry below): `zarr_unreadable_chunk_is_hard_error_not_fill` (EACCES via chmod 000) and `zarr_genuinely_missing_chunk_is_still_legal_fill` (control case — ENOENT still legally fills). Verified: full build + ctest (66/70 pass, 4 pre-existing unrelated failures).

## [high/correctness] Short/truncated chunk blob is silently treated as absent → fill-value air

**Verdict:** CONFIRMED — At src/io/zarr.hpp:179: `present = blob && blob->size() >= ccount*esz` demotes a fetched-but-short chunk to absent, and lines 190-195 fill it with fill_value with no error or log — the function returns success. The scenario is reachable: fetch_object (zarr.hpp:30-33) slurps a local file via istreambuf_iterator and returns success with however many bytes exist, and copy_zarr_region_local writes chunks with a plain ofstream (zarr.hpp:290-296), NOT atomic write-temp-rename, so a crash mid-copy leaves exactly the truncated chunk file described; every later read silently returns air for it. This directly violates the io module CLAUDE.md invariant ("Absent ≠ fetch-failed … never silent air") and root CLAUDE.md §2.4, so it is not intentional (the line-115 comment sanctions fill only for OMITTED chunks). Only the 'hides blosc misread as raw' sub-claim is off: read_zarray rejects non-null compressors (zarr.hpp:105-107), so compressed chunks never reach this path — that does not refute the core defect.

**Fix notes:** The three-way check is right; use the existing Errc::decode_error (core/error.hpp) or Errc::io_error, and set the failed flag via the same CAS pattern as the fetch-failure path (lines 173-177) so fail_msg has a single writer. Change the comparison to strict equality (blob->size() == ccount*esz); note that when blosc support lands the size check must apply to the DECOMPRESSED payload, not the raw blob. Include chunk id (c.cz/c.cy/c.cx with m.sep) plus got-vs-expected byte counts in the message. Two related hardening items worth fixing in the same pass: (1) root cause — copy_zarr_region_local should write chunks via write-temp-rename (an ofstream failure after open is also currently unchecked: `of.write(...)` at zarr.hpp:296 never tests the stream), per the io CLAUDE.md atomicity invariant; (2) the .zarray blob path (read_zarray) is unaffected but fetch_object's local branch cannot distinguish ENOENT from EACCES/EIO on open — both return 'absent'; a stat/errno check would keep the absent-vs-failed distinction honest for local roots too.

**Location:** src/io/zarr.hpp:179-180

**Evidence:**
```cpp
const bool present = blob && blob->size() >= static_cast<usize>(ccount) * esz;
const u8* data = present ? blob->data() : nullptr;
```
A chunk that *was fetched* but whose byte count is smaller than `chunks.count() * dtype_size` — a truncated local file (crash mid-write, see the `copy_zarr_region_local` finding), a mid-read EIO short slurp from the ifstream path, or an unexpected server body — is silently demoted to "not present" and the whole chunk reads as fill value. No error, no log.

**Failure scenario:** `copy_zarr_region_local` crashed mid-chunk-write on a previous run (its writes are not atomic, next finding), leaving a 100 KB file where 262144 bytes are expected. Every subsequent `read_zarr_region` over that slab returns air for that chunk; the corruption propagates into `.fxvol` archives and downstream stages with zero indication. Also silently hides a wrong-size chunk that would indicate a compressed (blosc) source being misread as raw.

**Suggested fix:** distinguish three cases: `blob == nullopt` → fill (correct); `blob->size() == expected` → decode; anything else → `Errc::decode_error` hard fail ("chunk z.y.x: got N bytes, expected M"). Oversized blobs should also fail — with raw-only support, a size mismatch almost certainly means a compressed chunk.

**Outcome:** fixed — `read_zarr_region` in src/io/zarr.hpp now does the three-way check exactly as suggested: `nullopt` → fill stays; `size == expected` (strict equality, catches oversized too) → decode; anything else → hard error via the existing `failed`/`fail_msg` CAS (message includes chunk id, got/expected byte counts), surfaced as `Errc::fetch_failed` by the existing wrapper at the end of the function. Regression test added: `zarr_truncated_chunk_is_hard_error_not_fill`. Verified via test_zarr (7/7 pass) + full ctest.

## [medium/resource-safety] `copy_zarr_region_local` chunk writes are unchecked and non-atomic

**Verdict:** unverified (medium/low)

**Location:** src/io/zarr.hpp:290-297

**Evidence:**
```cpp
std::ofstream of(dpath, std::ios::binary | std::ios::trunc);
if (!of) { ... }
of.write(reinterpret_cast<const char*>((*got)->data()), static_cast<std::streamsize>((*got)->size()));
copied.fetch_add(1, std::memory_order_relaxed);
```
The result of `of.write` (and the implicit flush at destruction) is never checked, and the file is written in place rather than via the module's own write-temp-rename invariant (io/CLAUDE.md: "Atomic write-temp-rename").

**Failure scenario:** ENOSPC or an IO error mid-write leaves a truncated chunk file, the function still reports success (`copied` incremented, no `failed` set). Combined with the short-blob-→-air finding above, the truncated chunk later reads back as silent air. A crash mid-copy leaves the same time bomb.

**Suggested fix:** write to `dpath + ".tmp"`, check `of.write` and `of.close()`/`of.good()`, then `std::filesystem::rename` — same pattern already used correctly in `write_fxsurf`. Set `failed`/`fail_msg` on any write error.

**Outcome:** fixed — `copy_zarr_region_local` in src/io/zarr.hpp now writes to a per-task temp path (`dpath + ".tmp" + i`, unique per parallel_for task index so concurrent tasks never collide), checks `of.write`/`of.close()` and removes the temp file + sets `failed` on any write error, then `std::filesystem::rename`s into place on success (checking `rename`'s error_code too). Matches the `write_fxsurf` atomicity pattern per the io/CLAUDE.md invariant. Regression test added: `zarr_copy_local_propagates_truncated_source_chunk_error` (verifies a truncated source chunk cannot silently propagate as valid data through the copy path). Verified via test_zarr + full ctest.

## [medium/correctness] Zarr dtype validation missing: 8-byte and big-endian dtypes silently decode as garbage

**Verdict:** unverified (medium/low)

**Location:** src/io/zarr.hpp:69-89 (`dtype_size`, `cast_dtype`), src/io/zarr.hpp:92-111 (`read_zarray` never validates dtype)

**Evidence:**
```cpp
inline usize dtype_size(const std::string& dt) {
    if (dt.size() < 3) return 1;
    return static_cast<usize>(dt[2] - '0');  // |u1 ->1, <u2->2, <f4->4
}
...
    return static_cast<T>(*p);   // cast_dtype fallthrough
```
`read_zarray` accepts any dtype string. For `<i8`/`<u8`(int64)/`<f8`, `dtype_size` returns 8 but `cast_dtype` has no 8-byte branch and falls through to `static_cast<T>(*p)` — reading only the low byte of each element. Big-endian dtypes (`>u2`, `>f4`) are decoded as little-endian with no check. A non-digit third char (e.g. dtype `"|b1"`... fine, but `"<c8"`) can produce a bogus element size used in chunk-size arithmetic.

**Failure scenario:** pointing `ingest-zarr` at a `<f8` or `>u2` OME-Zarr (both occur in the wild) produces a volume full of wrong values — no error, no warning — which is then encoded into `.fxvol` and consumed by every downstream stage. Silent data corruption on a supported-looking input.

**Suggested fix:** validate in `read_zarray`: byte-order char must be `|` or `<`, kind∈{u,i,f}, size∈{1,2,4}; otherwise `err(Errc::unsupported, "zarr dtype " + dt)`. This is one `if` and closes the whole class.

## [medium/correctness] `fill_value` is never parsed from `.zarray` — nonzero-fill zarrs read missing chunks as 0

**Verdict:** unverified (medium/low)

**Location:** src/io/zarr.hpp:92-111 (`read_zarray`), src/io/zarr.hpp:40 (`ZarrMeta::fill` defaults to 0 and is never assigned)

**Evidence:**
```cpp
struct ZarrMeta { ... f32 fill = 0.0f; ... };
inline Expected<ZarrMeta> read_zarray(const std::string& root) {
    ...
    m.dtype = detail::json_string(js, "dtype");
    // fill_value: never read
```
`read_zarr_region` then uses `const T fillv = static_cast<T>(m.fill);` — always 0.

**Failure scenario:** an OME-Zarr with `"fill_value": 255` (or any nonzero fill; some prediction/mask exports use nonzero fills) has every omitted chunk read back as 0 instead of the declared fill. The struct field exists, so callers reasonably believe fill is honored. Wrong voxel values with no error.

**Suggested fix:** parse `fill_value` with a `json_number`-style extractor (one already exists in scan_meta.hpp — hoist it) and assign `m.fill`; if `fill_value` is non-numeric (`"NaN"`, string), reject as unsupported rather than defaulting to 0.

## [medium/bug] slice/video CLI: `std::stoll`/`std::stof`/`std::stoi` throw — process aborts under -fno-exceptions; slice index unbounded → OOB read

**Verdict:** unverified (medium/low)

**Location:** src/io/slice.hpp:137, 141-145, 158, 184-185 (and `video_cmd` equivalents)

**Evidence:**
```cpp
const s64 idx = std::stoll(std::string(args[2]));
...
o.vmin = std::stof(opt_get(args, "min", "0"));
```
The project builds with `-fno-exceptions`; libc++ (built with exceptions) will throw `std::invalid_argument` from `stoll`/`stof` into a no-exceptions TU → `std::terminate`/abort. Every other CLI entry point uses the `std::from_chars` helpers. Additionally `idx` is never range-checked: `build_slice` calls `slice_at(raw, a, idx, r, c)` → `v(idx, r, c)` with no bounds check.

**Failure scenario:** `fenix slice vol.fxvol z abc out.jpg` → immediate abort instead of a usage `Error`. `fenix slice vol.fxvol z 999999 out.jpg` on a 512³ volume → out-of-bounds reads across the whole slice (heap over-read; garbage image at best, crash at worst).

**Suggested fix:** replace all `sto*` calls with the existing `parse_i`/`from_chars` helpers (which are already the module convention), and validate `0 <= idx < slice_geom(...).frames` before building the slice, returning `Errc::invalid_argument` otherwise.

**Outcome:** fixed — added `template<class T> Expected<T> parse_num(std::string_view)` in src/io/slice.hpp (from_chars-based, requires the WHOLE token consumed so "12abc" is rejected rather than silently truncated — io.hpp's existing `from_chars` call sites don't check this, so this is stricter than the module's current convention). Replaced every `std::stoll`/`std::stof`/`std::stoi` in `slice_cmd` and `video_cmd` (index, min/max/alpha/thresh/quality/fps/step) with `parse_num`, returning `Errc::invalid_argument` instead of aborting. Added the range check `0 <= idx < slice_geom(...).frames` in `slice_cmd`, returning a clean error on OOB instead of an unchecked read. Manually verified end-to-end with a real `fenix` binary: OOB index, non-numeric index, non-numeric option values, and trailing-garbage numbers all now return clean errors (exit 1, no abort) instead of crashing or reading OOB.

## [medium/bug] NRRD header sizes unvalidated — negative or overflowing `sizes` drive allocation/read with a bogus count

**Verdict:** unverified (medium/low)

**Location:** src/io/nrrd.hpp:50, 70-71 (and `stream_nrrd_f32` at 104)

**Evidence:**
```cpp
h.nx = sizes[0]; h.ny = sizes[1]; h.nz = sizes[2];   // no >0 / product check
...
Volume<f32> vol(Extent3{nz, ny, nx});
const s64 n = nz * ny * nx;
```
NRRD is an untrusted external format (the module's declared fuzz surface). Negative sizes pass straight through; `nz*ny*nx` can overflow s64 for large crafted values (signed overflow = UB), and a merely huge-but-valid product turns into `std::vector<T> buf(static_cast<usize>(n))` → under `-fno-exceptions`, `bad_alloc` cannot propagate → abort.

**Failure scenario:** a fuzzer (or a corrupted file) with `sizes: -1 4 4` or `sizes: 3000000000 3000000000 3000000000` → UB/abort instead of `Expected` error. `read_nrrd_u8` and `nrrd_max` share the same hole via `stream_nrrd_f32`.

**Suggested fix:** in `parse_nrrd_header`, require each size `> 0` and check the product against a sane ceiling (e.g. `nx <= 1<<18` per axis per the project's own volume bound, and product * type_bytes bounded against the file size or a hard cap), returning `Errc::decode_error` otherwise.

## [low/correctness] `.fxsurf` reader: `nu * nv` can overflow s64 before the bounds check

**Verdict:** unverified (medium/low)

**Location:** src/io/surface.hpp:65

**Evidence:**
```cpp
if (nu < 0 || nv < 0 || nu * nv > (s64{1} << 30)) return err(Errc::decode_error, "bad fxsurf dims");
```
`nu` and `nv` are raw s64s read from the file. `nu * nv` with e.g. `nu = nv = 0x2000000000` overflows (UB) and can wrap to a small/negative value that passes the check, after which `Surface s(nu, nv)` attempts an enormous allocation → abort under `-fno-exceptions`.

**Failure scenario:** fuzzed/corrupted `.fxsurf` with two large dims → signed-overflow UB and/or allocation abort instead of a clean decode error.

**Suggested fix:** bound each dimension individually before multiplying: `if (nu < 0 || nv < 0 || nu > (1<<20) || nv > (1<<20) || nu*nv > (1<<30)) ...` (or compare via `nu > limit / std::max<s64>(nv,1)`).

## [low/concurrency] `fdct8x8` lazily initializes a function-local static matrix with a non-atomic flag — data race if two threads ever encode JPEGs

**Verdict:** unverified (medium/low)

**Location:** src/io/jpeg.hpp:122-132

**Evidence:**
```cpp
static double M[8][8];
static bool init = false;
if (!init) {
    for (...) M[u][x] = ...;
    init = true;
}
```
Unlike a magic-static (`static const auto M = compute();`), this hand-rolled pattern has no synchronization: two threads can both see `init == false` and race on `M`, or one can see `init == true` while `M`'s stores are not yet visible → torn coefficients → corrupt JPEG output, plus a formal data race (UB, TSan CI job will flag it).

**Failure scenario:** today's callers (`slice_cmd`, `trace_surface` render) are single-threaded, so this is latent — but the first parallel slice/preview export (an obvious future step for the `video` path) silently produces corrupted frames on first use.

**Suggested fix:** make it a magic static: `static const std::array<std::array<double,8>,8> M = []{ ... }();` (thread-safe init is guaranteed), or `constexpr`-compute it (cos isn't constexpr-friendly pre-C++26 libc++ — the lambda-init magic static is the simple fix).

## [low/bug] `video_cmd`: unchecked `fwrite` to the ffmpeg pipe; early ffmpeg death → SIGPIPE kills the process; out-path interpolated unescaped into the shell command

**Verdict:** unverified (medium/low)

**Location:** src/io/slice.hpp:210-233

**Evidence:**
```cpp
std::snprintf(cmd, sizeof cmd, "ffmpeg ... \"%s\"", ..., out.c_str());
std::FILE* pipe = ::popen(cmd, "w");
...
std::fwrite(im.px.data(), 1, im.px.size(), pipe);   // unchecked
```
If ffmpeg exits early (bad codec on this box, disk full), the next `fwrite` raises SIGPIPE — default action terminates the whole process, so the `pclose` status handling and the `Errc::io_error` path are never reached. `fwrite`'s return is also never checked. The output path is quoted but not escaped (a path containing `"` breaks the command); `snprintf` truncation for long paths is unchecked.

**Failure scenario:** `fenix video vol z out.mp4 enc=nvenc` on a box where NVENC probes OK but fails at real resolution → ffmpeg exits → fenix dies from SIGPIPE with no error message mid-export.

**Suggested fix:** ignore SIGPIPE around the pipe writes (`signal(SIGPIPE, SIG_IGN)` once, or `write` with MSG_NOSIGNAL semantics via a manual pipe+fork), check `fwrite` return and bail with an Error, and validate/escape `out` (reject paths containing `"`), checking the `snprintf` return against `sizeof cmd`.

**Outcome:** partially fixed — `video_cmd` in src/io/slice.hpp now scopes `signal(SIGPIPE, SIG_IGN)` around the write loop (restoring the previous handler after `pclose`) and checks `fwrite`'s return against the expected byte count, breaking out and returning `Errc::io_error` on a short write instead of continuing silently or dying to SIGPIPE. Did NOT implement the path-escaping/snprintf-truncation hardening (out-of-scope low-severity hardening not in my assigned finding list; flagging for a follow-up pass if desired).
