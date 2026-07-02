# sweep-errors — repo-wide error-handling discipline under `-fno-exceptions`

Scope: every `std::expected` producer/consumer chain in `src/`, throwing-STL use, error
swallowing (especially the absent-vs-fetch-failed rule in io/zarr/s3), write-error
checking, and allocation-failure behavior.

**Overall assessment.** The core `Expected<T, Error>` discipline is genuinely good: the
codec archive (`codec/archive.hpp`), `io/s3.hpp` (404-vs-hard-fail typed, retry+backoff),
`io/surface.hpp` (version-checked, atomic rename, final stream check), the export-scroll
producer/consumer pipeline, and the ML checkpointing are all careful, with no
`.value()`-without-check anywhere and correct error propagation through dozens of hops.
The failures are concentrated at the *edges*: the streaming tracer's tile fetcher throws
the whole io contract away (a hard fetch error becomes an **uninitialized** tile that is
then traced as if it were data), the local-file half of `fetch_object` conflates every
open failure with "absent = air", short/corrupt zarr chunk blobs are silently read as
fill, and several CLI/file parsers use throwing STL (`std::stoi`/`std::stof`) that
aborts the process under `-fno-exceptions`. Ordered most-severe first.

## [critical/silent-corruption] stream_tile_u8 swallows a hard fetch failure and returns an UNINITIALIZED tile as data
**Verdict:** CONFIRMED — Confirmed at every cited point. (1) src/core/volume.hpp:89-91: `Volume(Extent3)` allocates via `std::make_unique_for_overwrite` — indeterminate bytes, not zeros (only `Volume::zeros()` at volume.hpp:93-98 value-inits). (2) src/segment/trace_stream.hpp:28-30: `Volume<u8> out(pe); auto r = io::read_zarr_region(...); if (!r) return out;` returns that uninitialized buffer on any error, so the line-25 comment "a hard fetch error yields zeros here" is factually wrong. (3) The failure is reachable, not hypothetical: src/io/zarr.hpp:204 returns `Errc::fetch_failed` after s3.hpp's retry/backoff is exhausted, and read_zarr_region also errors on a missing/unsupported `.zarray` (zarr.hpp:94-107) — e.g. a typo'd root path yields a full trace over garbage that "succeeds". (4) All four call sites (trace_stream.hpp:60, 62, 105, 107) consume the tile unconditionally and pass it into `detail::trace_one_tile`, so indeterminate heap bytes are traced into .fxsurf fragments and manifest entries with success status — a direct violation of the root CLAUDE.md hard rule and the io CLAUDE.md invariant "Absent ≠ fetch-failed … never silent air" (this is worse than air: it fabricates structure). The escape-hatch comment ("production callers … should thread the Expected through") does not refute it — the production caller `trace_volume_streamed_to_disk` (documented in src/segment/CLAUDE.md as THE fully out-of-core tracer) lives in the same file and does not thread it, and the comment misdescribes the failure mode. Reading indeterminate values is also erroneous/undefined behavior on top of the scientific-output corruption.
**Fix notes:** The fix direction is right, but one claim in it is wrong: only `trace_volume_streamed_to_disk` already returns Expected (trace_stream.hpp:85). `trace_volume_streamed` (trace_stream.hpp:44) returns plain `VolumeResult`, so it must be re-signed to `Expected<VolumeResult>`, and its callers updated — currently tests/test_trace_stream.cpp:128 and :217 bind `const segment::VolumeResult`. Also update the false line-25/26 comment when fixing, and do NOT paper over with `Volume<u8>::zeros()` on error — zeros would still be the silent-air violation; propagate the error. src/segment/CLAUDE.md's descriptions of both entry points should be touched up to reflect the Expected return.
**Location:** src/segment/trace_stream.hpp:27-40 (the swallow is line 30)
**Evidence:**
```cpp
inline Volume<u8> stream_tile_u8(const std::string& root, Index3 porg, Extent3 pe, f32 scale) {
    Volume<u8> out(pe);                       // make_unique_for_overwrite -> UNINITIALIZED
    auto r = io::read_zarr_region(root, porg, pe);
    if (!r) return out;                       // hard fetch error -> garbage tile, no signal
```
The header comment (line 25) even documents it: "a hard fetch error yields zeros" — but
`Volume<u8>(pe)` uses `std::make_unique_for_overwrite`, so it isn't even zeros, it's
uninitialized heap memory.
**Failure scenario:** During `trace_volume_streamed_to_disk` (a multi-hour out-of-core
trace over S3), one region's fetch exhausts its retries (S3 5xx burst, DNS blip beyond
the s3.hpp retry budget). `read_zarr_region` correctly returns `Errc::fetch_failed`;
`stream_tile_u8` discards it and hands the tracer a tile of uninitialized bytes. The
tracer then seeds/grows surfaces through random garbage (or, if the allocator happened to
return zero pages, silently traces "air"), writes plausible-looking `.fxsurf` fragments
+ manifest entries, and the run reports success. This is the exact violation of the root
CLAUDE.md hard rule ("a transient fetch error must never silently become air"), plus an
uninitialized-read UB on top. It corrupts the scientific output of the whole
segment→winding→flatten chain with no diagnostic.
**Suggested fix:** Return `Expected<Volume<u8>>` from `stream_tile_u8` and propagate at
all four call sites (trace_stream.hpp:60,62,105,107). Both callers already return
`Expected<...>`, so propagation is mechanical. If a "keep going past one bad tile" mode
is ever wanted, it must be explicit (skip the tile and record it in the manifest/stats),
never fabricated voxels.

**Outcome:** fixed — `stream_tile_u8` now returns `Expected<Volume<u8>>`, propagating
`read_zarr_region`'s error via `std::unexpected` instead of returning an uninitialized
tile. `trace_volume_streamed` is re-signed to `Expected<VolumeResult>` and propagates at
both its own call sites; `trace_volume_streamed_to_disk` (already `Expected`) propagates
at its two call sites too. Updated the false "yields zeros" comment. Updated all 3
affected call sites in tests/test_trace_stream.cpp (lines ~128, ~154, ~220) to unwrap the
new `Expected` return via `REQUIRE(...has_value())`. Verified: test_trace_stream 5/5
pass (78.86s under shared-box load), full ctest 66/70 (4 pre-existing unrelated
failures).

## [high/silent-corruption] Local fetch_object treats EVERY open failure as "absent = air" (EMFILE/EACCES/EIO become fill)
**Verdict:** CONFIRMED — Confirmed. src/io/zarr.hpp:30-31: the local branch is exactly `std::ifstream f(root + "/" + sub, std::ios::binary); if (!f) return std::optional<std::vector<u8>>(std::nullopt); // missing chunk = air` — every open failure (EACCES, EIO, EMFILE, ENOMEM, ...) is conflated with ENOENT and returned as "absent". This directly violates the module's own documented invariant at src/io/CLAUDE.md:33: "Absent ≠ fetch-failed (404 → air; other → retry then hard-fail — never silent air)" — the remote path (s3.hpp via http_get, zarr.hpp:29) honors it, the local path does not. There is no guard anywhere else: in read_zarr_region (zarr.hpp:174-181) a nullopt blob yields present=false and the chunk is scattered as fill_value; callers in src/io/io.hpp:101/286 (ingest/transcode) commit that volume, and src/segment/trace_stream.hpp consumes it, so a transient local open failure silently becomes air in the output artifact. The mid-stream-read claim is also real: line 32's istreambuf_iterator read stops silently on a stream error, producing a short vector; zarr.hpp:180 (`blob->size() >= ccount*esz`) then treats the short blob as absent → fill, again silent air with no error. One detail of the reviewer's scenario is overstated: for local roots fetch_threads=0 does not mean unbounded — parallel_for_io (src/core/parallel.hpp:181) maps 0 to cpu_budget(), so concurrency is ~core count with one fd per thread. Hitting EMFILE from this loop alone is less likely than claimed (though plausible on macOS's default 256-fd rlimit or fd-heavy processes), but EACCES/EIO/ENOMEM/short-read need no such conditions, so the failure scenario is reachable regardless and the finding stands as filed.
**Fix notes:** The fix direction is right; refinements: (1) Don't rely on errno after a failed std::ifstream open — whether the filebuf preserves errno is implementation-defined. Prefer opening with ::open(path, O_RDONLY) (the build is -fno-exceptions anyway, so iostreams error reporting buys nothing here): ENOENT → nullopt, any other errno → err(Errc::fetch_failed, ... + strerror). (2) Use fstat on the already-open fd for st_size and read exactly that many bytes — statting the path separately is a TOCTOU race. (3) The short-read guard must return an error, not just check eof()/bad(): today zarr.hpp:180 quietly converts a present-but-truncated chunk file into fill; a local chunk whose size != ccount*esz (after fetch succeeded) should hard-fail rather than fall through to present=false. (4) Note read_zarray (zarr.hpp:93-95) already maps nullopt to Errc::not_found, so the .zarray path degrades to a clear error either way; the chunk path is where the silent-air bug bites. (5) Retry+backoff (per the CLAUDE.md invariant) is optional for local EIO but the ENOENT-vs-else distinction is the mandatory part.
**Location:** src/io/zarr.hpp:27-34 (line 31)
**Evidence:**
```cpp
    if (is_remote(root)) return http_get(root + "/" + sub);
    std::ifstream f(root + "/" + sub, std::ios::binary);
    if (!f) return std::optional<std::vector<u8>>(std::nullopt);  // missing chunk = air
```
The remote path carefully distinguishes 404 from hard failure; the local path does not.
`!f` is true for ENOENT *and* EMFILE, EACCES, EIO, ELOOP, ENOMEM. Additionally, a
read error mid-stream (line 32's istreambuf read) silently yields a short byte vector,
which the region reader then treats as absent (see next finding).
**Failure scenario:** `read_zarr_region` on a local store runs an **unbounded**
`parallel_for` (fetch_threads=0 for local roots) with one `ifstream` per chunk; on a
256-thread box against a many-chunk region, hitting the fd rlimit makes `open` fail with
EMFILE on some chunks — those chunks silently decode as fill (air). The exported .fxvol
has holes exactly where real papyrus was, coverage says Zero (a *committed* answer, so a
resume never re-fetches them), and nothing ever errors. Same story for a permission
mistake or a failing disk.
**Suggested fix:** After a failed open, `stat` the path (or check `errno`): ENOENT →
`std::nullopt` (air); anything else → `err(Errc::fetch_failed, ...)`. Also verify the
stream is `.eof() && !.bad()` after the read and compare bytes read against `st_size`,
erroring on mismatch.

**Outcome:** fixed — `fetch_object` now opens via `std::fopen` (errno-authoritative),
ENOENT/ENOTDIR → nullopt (air), any other errno → `Errc::fetch_failed` with strerror.
Read loop checks `ferror()` per chunk (64 KiB buffered reads) so a mid-read EIO
hard-fails instead of yielding a silent short buffer. Same fix as the `io.md` entry for
this finding (duplicate). Regression tests: `zarr_unreadable_chunk_is_hard_error_not_fill`,
`zarr_genuinely_missing_chunk_is_still_legal_fill`.

## [high/silent-corruption] Short/corrupt zarr chunk blob is silently treated as absent (fill) instead of an error
**Verdict:** CONFIRMED — The finding is real and reachable. At src/io/zarr.hpp:179, `const bool present = blob && blob->size() >= ccount * esz;` is the ONLY classification point — any existing-but-undersized chunk blob silently becomes fill_value with no diagnostic. Guards elsewhere do not cover it: (1) the local path in fetch_object (zarr.hpp:30-33) reads via ifstream with zero size validation, so a chunk file truncated by ENOSPC/kill during a prior write is returned as a short blob and scattered as fill on every subsequent read; (2) the remote path is only partially protected — libcurl returns CURLE_PARTIAL_FILE on a Content-Length mismatch (s3.hpp:169-180 would then retry/hard-fail), but an object that was stored truncated at the origin (upload interrupted, corrupt bucket) arrives with a consistent Content-Length, passes http_get's status==200 check (s3.hpp:173), and hits the same silent-fill path. This directly contradicts the module invariant in src/io/CLAUDE.md ("never silent air") and root CLAUDE.md §2.4 ("A transient fetch error must never silently become air"); the header comment (zarr.hpp:115, :3) only sanctions treating MISSING chunks as fill — a present-but-short chunk is not missing. Zarr v2 raw edge chunks are always written full-size (padded), so for this raw-only reader there is no legitimate reason for a blob smaller than ccount*esz.
**Fix notes:** The proposed fix is sound with three adjustments. (1) Use strict equality (`blob->size() == ccount*esz`): an OVERSIZED blob is equally anomalous for the raw-only reader (likely a compressed chunk this reader can't decode) and should also error rather than be silently truncated-read; when blosc2 support lands the check must move to post-decompression size. (2) `Errc::decode_error` exists (src/core/error.hpp:25), so the proposed error code is available; route it through the existing failed/fail_msg CAS at zarr.hpp:174-176, but note line 204 wraps fail_msg as `Errc::fetch_failed` with a "fetch failed" prefix — either preserve the original Error (store an std::optional<Error> instead of a string) or accept the slightly misleading wrapper. (3) The same defect exists in copy_zarr_region_local (zarr.hpp:~280): it copies chunk blobs verbatim with no size check, so a truncated source chunk propagates into the local slab and then hits this same silent-fill path on read — apply the identical validation there.
**Location:** src/io/zarr.hpp:179-180
**Evidence:**
```cpp
    const std::optional<std::vector<u8>>& blob = *got;
    const bool present = blob && blob->size() >= static_cast<usize>(ccount) * esz;
    const u8* data = present ? blob->data() : nullptr;
    ...
    T v = fillv;
    if (present) { ... }
```
A blob that exists but is smaller than `chunks.count()*esz` — a truncated download, a
partially-written local chunk file (see the unchecked-write finding below), a corrupt
object — is classified `present = false` and the whole chunk reads as `fill_value`.
**Failure scenario:** A previous `ingest-zarr ... out.zarr` run died on ENOSPC leaving a
truncated chunk file (its write status is never checked — next finding). Every later
read of that store silently returns air for that chunk: prediction/CT data vanishes from
the pipeline with zero diagnostics. This conflates *corrupt* with *absent*, the same
class of violation as the 404 rule.
**Suggested fix:** Distinguish three cases: `!blob` → air; `blob->size() ==
ccount*esz` (or `>=`, tolerating trailing junk if you must) → present; otherwise →
`Errc::decode_error` ("chunk <id> has N bytes, expected M") propagated through the
`failed` CAS like fetch errors.

**Outcome:** fixed — exactly this three-way check, strict equality. Duplicate of the
`io.md` entry for the same finding (fixed once, in `read_zarr_region`).
`copy_zarr_region_local` is a raw byte-copy (no decode), so the truncated-source-chunk
half of this finding is covered by the write-temp-rename fix (next finding) plus the
regression test `zarr_copy_local_propagates_truncated_source_chunk_error`, which proves
the corruption is caught on read rather than propagating silently through the copy.

## [high/resource-safety] copy_zarr_region_local never checks that chunk writes succeeded — ENOSPC yields a silently truncated store
**Verdict:** CONFIRMED — Confirmed. src/io/zarr.hpp:296-297: of.write is followed by copied.fetch_add with no stream check, and the ofstream destructor discards flush errors; the adjacent write_text lambda (zarr.hpp:229-235) does check after write, showing the omission is a bug not a policy. The sole caller, ingest_zarr (src/io/io.hpp:86-90), logs success and returns 0, so ENOSPC/EIO yields a truncated chunk file and exit 0. The reader at zarr.hpp:179 treats a short blob as absent -> fill value, so the truncated chunk silently reads back as air — directly violating the project invariant (CLAUDE.md §2.4 and src/io/CLAUDE.md: 'never silent air', 'atomic write-temp-rename'). No guard exists elsewhere (no fsync, size verify, or temp-rename in this function).
**Fix notes:** The proposed fix is right but must also handle deferred flush errors: ENOSPC frequently surfaces only when the stream buffer flushes at close, not at of.write. So: after of.write check `if (!of)`, then explicitly `of.close()` and check `of.fail()` again, routing both through the existing failed-CAS with "write " + dpath, and only increment `copied` after a clean close. Optionally, to honor the io module's 'atomic write-temp-rename' invariant, write to dpath+".tmp" and std::filesystem::rename on successful close so a crash mid-copy can't leave a short chunk behind.
**Location:** src/io/zarr.hpp:290-297
**Evidence:**
```cpp
        std::ofstream of(dpath, std::ios::binary | std::ios::trunc);
        if (!of) { ...fail... }
        of.write(reinterpret_cast<const char*>((*got)->data()), ...);
        copied.fetch_add(1, std::memory_order_relaxed);
```
The `of.write` result and the stream state at close are never checked (compare
`write_text` a few lines up, which does check). A full disk / quota hit / EIO produces a
short chunk file and the function still returns success ("zarr copy failed" is only
raised for *open* failures).
**Failure scenario:** `fenix ingest-zarr <s3-url> ... slab.zarr` fills the disk halfway
through a chunk. The command logs "wrote local zarr" and exits 0. Every subsequent read
of that chunk hits the short-blob path above and reads air — a two-bug pipeline that
turns ENOSPC into missing papyrus with no error at either end.
**Suggested fix:** After `of.write`, check `if (!of) { CAS-fail with "write " + dpath;
return; }`; also `of.close()` explicitly and re-check before counting the chunk copied.

**Outcome:** fixed — write-temp-rename per the io/CLAUDE.md atomicity invariant: writes
to `dpath + ".tmp" + <task-index>` (unique per parallel_for task, no collision), checks
`of.write`/`of.close()` and removes the temp + sets `failed` on any error, then
`std::filesystem::rename`s into place (checking the rename's error_code). Duplicate of
the `io.md` entry for the same finding.

## [high/no-exceptions] read_obj uses std::stoi on untrusted file tokens — malformed OBJ aborts the process; unchecked `>>` pushes uninitialized vertices
**Verdict:** CONFIRMED — The core claim is correct and reachable. src/geom/mesh.hpp:57 calls std::stoi on a token extracted with `ss >> tok` (line 56) with no check: a face line like "f 1 2" leaves tok empty on the third iteration, and std::stoi("") throws std::invalid_argument inside libc++. The project builds every target with -fno-exceptions -fno-rtti (CMakeLists.txt:90 `target_compile_options(fenix_headers INTERFACE -fno-exceptions -fno-rtti)`), so there is no handler anywhere in the binary and the throw terminates/aborts the process — defeating the Expected<Mesh> error contract of read_obj (mesh.hpp:35). The same applies to garbage tokens ("f a b c") and out-of-range values (std::out_of_range). read_obj is a public interop API ("OBJ (v/vt/vn) + PLY read/write" in src/geom/CLAUDE.md; tools/CLAUDE.md lists an `import` VC-interop subcommand), so untrusted external OBJ files are its intended input; the only current caller being a round-trip test (tests/test_mesh.cpp:25) does not make the scenario unreachable through the API. The face-index claim also holds: indices are never checked against vertices.size() (or for negatives from "f 0 ..." or OBJ relative indices), and write_ply/consumers index vertices with them. One detail of the claim is overstated: since C++11 a failed `>>` extraction writes 0 to an arithmetic operand, so "v 1" yields x=1, y=0 — but the third extraction's sentry fails on the already-set failbit and does NOT write, so z genuinely remains uninitialized and is copied into m.vertices (lines 45-47): the uninitialized-push scenario is real, just narrower than stated.
**Fix notes:** std::from_chars is the right tool (libc++ has floating-point from_chars; no exceptions, no locale, and it also rejects trailing garbage that std::stoi silently accepts, e.g. "12abc"). Corrections/additions to the proposed fix: (1) Errc::decode_error exists in src/core/error.hpp:25, so the error code is valid as proposed. (2) Take the index substring as std::string_view (tok.substr allocates; from_chars wants a char range anyway). (3) For v/vn lines, check ss state (or from_chars result) for all three components before push_back — the proposal already says this; make sure the vn path is covered too, not just v. (4) Index validation must account for OBJ semantics: valid OBJ permits negative (relative) indices and, technically, faces referencing vertices defined later in the file; simplest correct approach for this minimal reader is to reject idx <= 0 at parse time (documenting that relative indices are unsupported) and validate idx-1 < vertices.size() in a single pass after the whole file is parsed, rather than per-line. (5) Faces with >3 vertices currently have their extra tokens silently dropped (quad becomes a triangle); worth either rejecting or fan-triangulating while touching this code, though that is a separate lower-severity issue.
**Location:** src/geom/mesh.hpp:52-59 (line 57)
**Evidence:**
```cpp
            for (int k = 0; k < 3; ++k) {
                std::string tok;
                ss >> tok;
                t[static_cast<usize>(k)] = std::stoi(tok.substr(0, tok.find('/'))) - 1;
            }
```
Also lines 44-47: `f32 x, y, z; ss >> x >> y >> z;` — on a short/garbled `v` line the
extraction fails and uninitialized `x,y,z` are pushed into `m.vertices` (uninitialized
read, UB). Face indices are never range-checked against `m.vertices.size()`.
**Failure scenario:** `read_obj` returns `Expected<Mesh>` precisely because input files
are untrusted, but a face line `f 12/3` (only two tokens → third `tok` is empty →
`std::stoi("")` throws `std::invalid_argument`) unwinds into `-fno-exceptions` frames →
`std::terminate`/abort. Any imported mesh with a comment-mangled face line kills the
binary instead of returning `Errc::decode_error`. An out-of-range index (e.g. `f 999999
1 2` on a 10-vertex mesh) instead survives parsing and causes OOB reads in whatever
consumes `tris`.
**Suggested fix:** Parse with `std::from_chars` and return `err(Errc::decode_error,
"bad face line: " + line)` on failure; check `ss >> x >> y >> z` succeeded before
pushing; validate `0 <= idx < vertices.size()` after the file is read.

**Outcome:** fixed — see the fuller outcome note under the same finding in
`docs/review/2026-07-02/geom-flatten-render.md` (this is a duplicate finding on the same
`src/geom/mesh.hpp:read_obj`, fixed once).

## [high/silent-corruption] Zarr dtype is never validated — big-endian ('>') and 8-byte dtypes decode to per-voxel garbage
**Verdict:** unverified (medium/low)
**Location:** src/io/zarr.hpp:69-89, 103 (read_zarray accepts any dtype string)
**Evidence:**
```cpp
inline usize dtype_size(const std::string& dt) { ... return static_cast<usize>(dt[2] - '0'); }
template <class T> inline T cast_dtype(const u8* p, const std::string& dt) {
    ... if (kind == 'f' && sz == 4) {...} ... if (kind == 'u') {...1,2,4...}
    ... return static_cast<T>(*p);   // fallthrough: first byte of the element
```
`read_zarray` stores `m.dtype` verbatim. `">u2"` (big-endian u16 — legal zarr v2) takes
the `kind=='u', sz==2` path and is memcpy'd as little-endian: every voxel is
byte-swapped. `"<f8"` (f64) falls through every branch and returns the *first byte* of
each double. Neither errors.
**Failure scenario:** Pointing `ingest-zarr` at a third-party `>u2` or `<f8` store
produces a .fxvol full of numerically plausible garbage (byte-swapped u16 looks like
noisy CT), which then flows into segmentation/winding. No error at any stage.
**Suggested fix:** Validate in `read_zarray`: byte-order char must be `|` or `<`
(reject `>` with `Errc::unsupported` until swapped decode exists), and (kind,size) must
be one of the pairs `cast_dtype` actually implements; otherwise `Errc::unsupported`
naming the dtype.

## [medium/no-exceptions] std::stof/stoi/stoll/stod CLI parsing across five modules aborts on malformed arguments; from_chars sites silently default instead
**Verdict:** unverified (medium/low)
**Location:** src/segment/trace_surface.hpp:83-84,209-210 (also src/io/slice.hpp:137-185,
src/preprocess/preproc_cli.hpp:65-137, src/preprocess/dering.hpp:229-240,
src/ml/augment_cli.hpp:21-23; the silent-default variant: src/io/io.hpp:28-32,70,143,
src/eval/eval.hpp:57, src/render/render.hpp:27)
**Evidence:**
```cpp
    const f32 iso = std::stof(opt("iso", "0.5"));          // throws -> abort under -fno-exceptions
    ...
inline s64 parse_i(std::string_view s, s64 def) {
    s64 v = def;
    std::from_chars(s.data(), s.data() + s.size(), v);      // result ignored -> silent default
    return v; }
```
**Failure scenario:** Two opposite failure modes for the same class of input. (a)
`fenix trace-surface ... iso=O.5` (letter O) → `std::stof` throws
`std::invalid_argument` out of libc++ into `-fno-exceptions` frames → abort, no usage
message, non-Error exit. (b) `fenix ingest-zarr root 0 4o00 0 0 512 512 512 out.fxvol`
→ `parse_i("4o00")` parses `4` and silently proceeds — a multi-hour export of the
*wrong region* that looks successful (from_chars errors are discarded at every call
site). Both violate the "errors via Expected" rule; (b) additionally corrupts
provenance (the artifact claims an origin the operator didn't ask for).
**Suggested fix:** One shared checked parser in core (e.g. `Expected<s64>
parse_int(std::string_view)`, `Expected<f32> parse_float(...)` over `std::from_chars`,
requiring full consumption `ptr == end && ec == errc{}`), used by every subcommand;
delete all `std::sto*` calls (they are banned STL under `-fno-exceptions`).

**Outcome:** partially fixed (my cluster's slice of this multi-module finding) — added
`template<class T> Expected<T> parse_num(std::string_view)` in src/io/slice.hpp
(from_chars, full-consumption check) and replaced every `std::sto*` in `slice_cmd`/
`video_cmd` (src/io/slice.hpp) with it. Did NOT touch the other files listed
(trace_surface.hpp, preproc_cli.hpp, dering.hpp, augment_cli.hpp — owned by other
review-fix agents) nor the silent-default `from_chars` sites in io.hpp/eval.hpp/
render.hpp (out of my assigned finding list; `io.hpp`'s `parse_i`/`parse_f` silently
defaulting on bad input is pre-existing module convention, not something I introduced —
flagging as a good follow-up for whoever owns a repo-wide parse-helper consolidation).

## [medium/silent-corruption] windings.txt written without any stream-error check — ENOSPC truncates the winding assignment silently
**Verdict:** unverified (medium/low)
**Location:** src/winding/stitch_stream.hpp:173-186 (same pattern at :303)
**Evidence:**
```cpp
    std::ofstream w(fs::path(dir) / "windings.txt");
    if (!w) return err(Errc::io_error, "cannot write windings.txt in " + dir);
    for (const FragRec& r : manifest) { ... w << r.name << ' ' << gw << '\n'; ... }
    return rep;   // stream state never re-checked
```
**Failure scenario:** The out-of-core stitch (hours of work) writes its *only* output —
the per-fragment winding numbers — to a nearly-full disk. The stream fails partway;
`stitch_streamed` returns a success report. The downstream flatten/render consumes a
truncated windings.txt and drops the tail fragments (or worse, a half-written last
line parses as a wrong winding). Also no atomic write-temp-rename, violating the io
invariant every other artifact writer follows (`write_fxsurf` does both correctly).
**Suggested fix:** After the loop: `w.flush(); if (!w) return err(Errc::io_error, ...)`.
Write to `windings.txt.tmp` and `std::filesystem::rename` with `error_code`, matching
`write_fxsurf`.

## [medium/resource-safety] VolumeArchive::commit() ignores msync failure — the "durable checkpoint" contract silently breaks on EIO
**Verdict:** unverified (medium/low)
**Location:** src/codec/archive.hpp:531-538 (also write_superblock_ at :754)
**Evidence:**
```cpp
        if (base_ && committed_eof_ > last_msync_) {
            const u64 from = last_msync_ & ~static_cast<u64>(4095);
            ::msync(base_ + from, committed_eof_ - from, MS_SYNC);   // return ignored
        }
        last_msync_ = committed_eof_;
        ++commit_seq_;
        write_superblock_(commit_seq_);   // its msync return ignored too
        return {};
```
**Failure scenario:** On writeback EIO (dying disk, thin-provisioned volume out of
space), `msync(MS_SYNC)` returns -1 with the data *not* durable, yet `commit()` bumps
`last_msync_` (so the failed range is never re-synced), writes a superblock claiming
`committed_eof_` is durable, and returns success. export-scroll then *relies* on this as
its crash-safe resume bookmark: after a crash, coverage says the region is present but
the blob bytes are lost/torn — the per-blob crc will catch it only when someone reads
that specific chunk, possibly much later, and the resume logic (first-tile-present ⇒
region done) will never re-fetch it. `commit()` is `Expected<void>` precisely so this
can be reported.
**Suggested fix:** Check both `msync` calls; on failure do not advance
`last_msync_`/`commit_seq_`, return `err(Errc::io_error, "msync: " +
std::string(strerror(errno)))`. Callers (export_scroll consumer loop, close) already
propagate.

## [medium/fuzz-surface] read_nrrd allocates from unvalidated header dims — a malformed header aborts (bad_alloc) or overflows instead of returning decode_error
**Verdict:** unverified (medium/low)
**Location:** src/io/nrrd.hpp:70-71 (dims from parse_nrrd_header, no validation)
**Evidence:**
```cpp
    const s64 nx = hh->nx, ny = hh->ny, nz = hh->nz;  // fastest-first
    Volume<f32> vol(Extent3{nz, ny, nx});
    const s64 n = nz * ny * nx;
```
`sizes: -1 4 4` or `sizes: 99999999 99999999 99999999` pass the parser (it only checks
`sizes.size()==3`). `dims.count()` goes negative → `static_cast<usize>` → a ~2^64
`make_unique_for_overwrite` → `std::bad_alloc` from libc++ unwinding into
`-fno-exceptions` code → abort. `read_fxsurf` next door validates (`nu*nv > 1<<30`
check at surface.hpp:65); NRRD — the *foreign import* format, the fuzz surface — does
not.
**Failure scenario:** `fenix ingest corrupt.nrrd out.fxvol` on a truncated/hostile file
aborts the process instead of printing `Errc::decode_error`. Same for the three other
NRRD readers sharing the header parser (read_nrrd_u8, nrrd_max).
**Suggested fix:** In `parse_nrrd_header`, reject `size <= 0` per axis and cap the
product (e.g. `nx*ny*nz > (s64{1}<<40)` → decode_error), checking the product with
overflow-safe math (divide-compare), before any allocation.

## [low/resource-safety] write_jpeg never checks fwrite/fclose — truncated image reported as success
**Verdict:** unverified (medium/low)
**Location:** src/io/jpeg.hpp:268-272
**Evidence:**
```cpp
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return err(Errc::io_error, "jpeg: cannot write " + path);
    std::fwrite(o.data(), 1, o.size(), f);
    std::fclose(f);
    return {};
```
**Failure scenario:** Disk fills during a `fenix slice ... .jpg` render sweep; the
truncated JPEG is reported written. Low severity (preview artifact, immediately visible
when opened), but it's the same unchecked-write smell as the zarr/windings findings and
a one-line fix.
**Suggested fix:** `if (std::fwrite(...) != o.size() || std::fclose(f) != 0) return
err(Errc::io_error, "short write " + path);` (fclose only after the fwrite check,
avoiding double-close).

## [low/design] Volume<T> construction cannot report allocation failure — CLI-sized giant extents abort instead of erroring
**Verdict:** unverified (medium/low)
**Location:** src/core/volume.hpp:89-98
**Evidence:**
```cpp
    explicit Volume(Extent3 dims)
        : dims_(dims), storage_(std::make_unique_for_overwrite<T[]>(static_cast<usize>(dims.count()))) {}
```
**Failure scenario:** Under `-fno-exceptions`, operator new's `std::bad_alloc` cannot be
caught anywhere → abort. Dimensions frequently come straight from CLI args or file
headers (`read_zarr_region` extent, NRRD sizes): `fenix ingest-zarr ... 0 0 0 20000
20000 20000 out.fxvol` requests a 8 PB f32 volume and the process dies with no fenix
error message. The out-of-core byte-budget rule implies volume-scale allocations should
be fallible at the boundary.
**Suggested fix:** Add `static Expected<Volume> alloc(Extent3)` using
`new (std::nothrow)` (and validate `dims.count() > 0` and below a sanity cap), and use
it at the untrusted boundaries (io readers, CLI-driven region reads); the infallible
ctor stays for internal, pre-validated sizes.
