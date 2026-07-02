# apps-tools-tests — apps/driver.cpp, tools/, tests/ (harness + test quality)

Overall assessment: the driver is appropriately tiny and the split/unity include discipline is
correct (SPLIT links module TUs as objects into the executable, so static registrars are not
dropped). The test suite is far better than a stub — test_fxvol in particular is exemplary
(corruption, recovery, COW isolation, cache-budget tests). The serious problems are at the
edges: the CLI subcommand layer parses numbers with throwing `std::sto*` under
`-fno-exceptions` (abort on any malformed flag or OBJ file), the project's #1 invariant
(fetch-failed must never become air) has zero test coverage while the zarr reader *currently
violates it* for truncated chunks, the `fuzz` preset cannot produce a fuzzer (no harnesses
exist and the flag wiring breaks every `main()`-bearing target), and the recipe runner rejects
ordinary multi-line TOML and silently mis-splits comma-containing stage args.

## [high/bug] CLI/stage arg parsing and OBJ import use throwing std::sto* under -fno-exceptions — malformed input aborts the process

**Verdict:** CONFIRMED — CMakeLists.txt:90 applies -fno-exceptions -fno-rtti to all TUs in the default (non-ML, CI-canonical) build, and src/io/slice.hpp:137 calls std::stoll(std::string(args[2])) on a raw CLI token with no validation; slice.hpp:141-145,158,178-185, segment/trace_surface.hpp:83-86,209-210, preprocess/dering.hpp:229-240, preprocess/preproc_cli.hpp:65-137, ml/augment_cli.hpp:21-23, ml/inference.cpp, and geom/mesh.hpp:57 all do the same. Under -fno-exceptions the libc++ sto* throw path terminates the process; in FENIX_ML builds exceptions are on (CMakeLists.txt:87-91) but nothing catches, so uncaught std::invalid_argument -> std::terminate — the crash is reachable in every configuration. It is not intentional: the project error model is std::expected (root CLAUDE.md §2.3), slice_cmd itself returns typed errors for every other bad argument (slice.hpp:131,134,153), and the codebase already uses non-throwing std::from_chars in core/config.hpp:115, io/io.hpp:30,42,70,143,385, render/render.hpp:27, eval/eval.hpp:57. read_obj (geom/mesh.hpp:35-63) returns Expected<Mesh> yet aborts on any malformed or short face line ('f a/b/c', or stoi("") when <3 indices) — a crash on foreign-file input via the documented OBJ import path. None of these are documented stubs.

**Fix notes:** The proposed Expected<T> parse_num<T>(std::string_view) over std::from_chars is the right fix and matches existing in-repo idiom. Corrections/additions: (1) the helper must require ec == std::errc{} AND ptr == s.end() so trailing garbage like "12abc" is rejected, unlike the existing io.hpp from_chars sites which silently ignore errors and leave the default value — consider migrating those to the helper too; (2) floating-point from_chars needs a recent libc++ (FP overloads landed ~LLVM 20) — fine given the always-latest-Clang invariant, but worth a comment; (3) in read_obj also handle the short-face-line case (fewer than 3 tokens) and the >> extraction failures for v/vn floats (currently silently push zeros) while touching that function; (4) note the FENIX_ML nuance in the finding: with ML on, exceptions are globally enabled, so the failure mode is uncaught-exception terminate rather than -fno-exceptions abort — same crash, fix applies identically; (5) the clang-tidy/grep CI ban on std::sto* is sensible and cheap.

**Location:** src/io/slice.hpp:137 (plus slice.hpp:141-145,158,184; src/segment/trace_surface.hpp:83-86,210; src/preprocess/dering.hpp:229-239; src/preprocess/preproc_cli.hpp; src/ml/augment_cli.hpp; src/geom/mesh.hpp:57 — 49 call sites total)

**Evidence:**
```cpp
const s64 idx = std::stoll(std::string(args[2]));          // slice.hpp:137
o.vmin = std::stof(opt_get(args, "min", "0"));             // slice.hpp:141
t[static_cast<usize>(k)] = std::stoi(tok.substr(0, tok.find('/'))) - 1;  // mesh.hpp:57, read_obj
```
The whole program is compiled with `-fno-exceptions -fno-rtti` (CMakeLists.txt:90) and the
error model is `std::expected<T, fenix::Error>`. `std::stoll/stof/stoi` throw
`std::invalid_argument`/`std::out_of_range` from inside libc++; under `-fno-exceptions` user
frames the unwind terminates/aborts.

**Failure scenario:** `fenix slice vol.fxvol z abc out.jpg` → `std::stoll("abc")` throws →
process aborts with no fenix error message. Worse, `read_obj` (geom/mesh.hpp:57) runs
`std::stoi` on face tokens of an *imported foreign file*: any OBJ with a malformed face line
(`f a/b/c ...`, or an empty token) kills the process instead of returning
`Expected<Mesh>` — an untrusted-file-input crash, exactly the fuzz surface class.

**Suggested fix:** add a `core` helper `Expected<T> parse_num<T>(std::string_view)` built on
`std::from_chars` (already used in config.hpp) and replace all 49 `std::sto*` call sites;
ban `std::sto[a-z]+` via clang-tidy or a grep CI check.

## [high/correctness] The "transient fetch error must NEVER become air" invariant has zero test coverage — and the zarr reader currently violates it for truncated chunks

**Verdict:** CONFIRMED — Confirmed, not refuted. (1) The code path exists exactly as claimed: src/io/zarr.hpp:178 does `const bool present = blob && blob->size() >= ccount*esz;` and the loop below writes `fillv` for every voxel when `!present` — a chunk that EXISTS but is short is silently converted to fill/air. (2) The scenario is reachable: for local roots, fetch_object (zarr.hpp:27-34) opens the file with ifstream and returns whatever bytes are there — a truncated file (interrupted copy by external tooling; fenix's own writer in copy_zarr_region_local at zarr.hpp:290-296 writes with plain ofstream, NOT write-temp-rename, so even a killed fenix copy leaves short chunk files) returns a short blob with no error. For remote roots, curl's CURLE_PARTIAL_FILE catches mid-transfer truncation, but a short-at-rest object returns HTTP 200 with a short body (s3.hpp:173) and flows into the same fill path. (3) It violates the module's own documented contract, not just the root invariant: src/io/CLAUDE.md:33 says "Absent ≠ fetch-failed (404 → air; other → retry then hard-fail — never silent air)", and the adjacent code comment at zarr.hpp:173 ("hard fetch failure — record, do NOT treat as air") shows the intent that the short-blob branch escapes. (4) Test-gap claim verified: tests/test_zarr.cpp has exactly three tests (present-chunk read, subregion, blosc rejection) and no test asserts truncated-chunk → error; grep across tests/ finds zero references to s3.hpp, http_get, or fetch_failed. Not a documented stub — io/CLAUDE.md's TODO list (line 66) covers blosc/v3/SigV4, not this.

**Fix notes:** Proposed fix is correct and minimal: in read_zarr_region, only `!*got` (nullopt = 404/missing file) is air; a present blob with size < ccount*esz should set `failed` with Errc::fetch_failed (message should include the chunk id and got-vs-expected sizes). Two additions: (a) copy_zarr_region_local (zarr.hpp:~280-296) should get the same size check before copying — today it propagates a short source chunk verbatim, planting the corruption in the local slab — and its ofstream write should be write-temp-rename per the root CLAUDE.md atomicity rule, which is itself a producer of exactly these truncated files. (b) The remote short-read test is hard to do hermetically without an HTTP stub; the local truncated-file test is trivial (truncate the chunk written by make_zarr to <64 bytes) and covers the shared code path, so prioritize that; a 404-vs-error fetch test needs either a tiny loopback server or can be deferred with the gap noted in tests/CLAUDE.md. Also use `!=` rather than `<` if you want to reject oversized chunks too (raw zarr chunks are exactly ccount*esz; oversize also indicates corruption), though `<` is the correctness-critical half.

**Location:** tests/test_zarr.cpp:33 (coverage gap); live defect at src/io/zarr.hpp:180

**Evidence:** test_zarr.cpp covers exactly three cases: present chunk, *missing* chunk → fill,
and blosc rejection. There is no test that a present-but-unreadable or short chunk produces
`Errc::fetch_failed` instead of fill. Meanwhile the implementation does:
```cpp
const bool present = blob && blob->size() >= static_cast<usize>(ccount) * esz;
...
T v = fillv;
if (present) { ... }
```
A chunk file/object that EXISTS but is truncated (partial download, interrupted copy, disk
error mid-file) yields `present == false` → the whole chunk is silently written as
`fill_value` — silent air, the exact matter-compressor failure mode the conventions call fatal.
`io/s3.hpp`'s 404-vs-failure distinction is also completely untested (no test includes s3.hpp).

**Failure scenario:** during a TB-scale ingest one S3 GET returns a short body (connection cut
after headers, or a partially-synced local mirror file); that 64³ region of the scroll becomes
zeros in the output archive with no error, poisoning everything downstream. No test in the
suite would catch either the current behavior or a future regression.

**Suggested fix:** in zarr.hpp treat `blob && blob->size() < expected` as a hard
`Errc::fetch_failed` (only `!blob` / 404 is air). Add tests: (a) truncate a present chunk file
and assert `read_zarr_region` errors; (b) a local-file s3/fetch test asserting 404→absent vs
short-read/permission-denied→fetch_failed.

## [high/design] The fuzz preset is unbuildable and zero fuzz harnesses exist for the implemented untrusted-input parsers

**Verdict:** CONFIRMED — CMakePresets.json:67-69 sets FENIX_SANITIZE="address,fuzzer"; CMakeLists.txt:157-161 applies -fsanitize=address,fuzzer as INTERFACE compile AND link options on fenix_headers, which is linked PRIVATE by the fenix binary (CMakeLists.txt:~199) and by all ~69 test binaries (CMakeLists.txt:229-255). Each of those defines its own main(); -fsanitize=fuzzer at link pulls libFuzzer's main from libclang_rt.fuzzer.a, so the build fails at link with duplicate main (the reason -fsanitize=fuzzer-no-link exists). grep for LLVMFuzzerTestOneInput hits zero source files — only tests/CLAUDE.md:12 (which mandates harnesses 'for every parser/codec/container path') and review docs. The untrusted-byte parsers are real and shipped (src/codec/rans.hpp, src/codec/lossless.hpp, src/codec/entropy.hpp; fxvol/zarr/NRRD/OBJ in src/io and src/geom) and no test feeds them corrupt input. Not an exonerating stub: tests/CLAUDE.md affirmatively documents fuzzing as the property-oracle home, and a preset that cannot build is a defect regardless of stub status. Additionally worse than claimed: "fuzz" exists only as a configure preset — no build preset — so root CLAUDE.md's documented `cmake --build --preset fuzz` fails even before the link error, and the fuzz CI workflow ci.yml:71 defers to does not exist (.github/workflows contains only ci.yml).

**Fix notes:** Proposed fix is correct in shape; refinements: (1) put plain -fsanitize=fuzzer (compile+link) only on the fuzz_*.cpp targets — since fenix is header-only, per-target flags instrument all included library code in that TU, so no interface-wide fuzzer-no-link is strictly needed; if the fuzz preset should still ASan-instrument tests, keep FENIX_SANITIZE=address and add fuzzer per-target. (2) Also add the missing "fuzz" entry to buildPresets (and testPresets exclusion) — today `cmake --build --preset fuzz` fails at preset lookup, independent of the link error. (3) tests/CLAUDE.md specifies AFL++ (afl-clang-fast, persistent mode) as the runner; LLVMFuzzerTestOneInput harnesses are compatible with both AFL++ and libFuzzer, so write the harnesses runner-agnostic and gate the -fsanitize=fuzzer link behind the preset. (4) Highest-value first harnesses: rans_decode/lossless_decode (pure byte-blob entry, reachable from any .fxvol) and fxvol open+read superblock/radix; then .zarray JSON, NRRD header, OBJ/PLY. (5) Update the stale Status section of tests/CLAUDE.md and add the deferred fuzz CI workflow or remove the ci.yml:71 claim.

**Location:** CMakePresets.json:67; tests/ (no fuzz_*.cpp anywhere)

**Evidence:** `"name": "fuzz"` sets `FENIX_SANITIZE: "address,fuzzer"`, which CMakeLists.txt
applies as `-fsanitize=address,fuzzer` INTERFACE-wide on `fenix_headers` — i.e. onto the
`fenix` binary and all ~69 test binaries, every one of which defines its own `main()`.
`-fsanitize=fuzzer` links libFuzzer's `main`, so every target either fails to link (duplicate
main) or is nonsense; and no `LLVMFuzzerTestOneInput` harness exists in the repo (the only
mention is prose in tests/CLAUDE.md). tests/CLAUDE.md states "Property/invariant oracles live
in the fuzzers" — so the declared home of property testing does not exist, while real
untrusted-byte parsers are already implemented and shipped: fxvol superblock/radix
(archive.hpp), zarr `.zarray` JSON, NRRD header, OBJ/PLY (mesh.hpp), `rans_decode`,
`lossless_decode`, scan-meta JSON. `rans_decode`/`lossless_decode` are never fed corrupt bytes
by any test despite being reachable from any on-disk .fxvol blob.

**Failure scenario:** `cmake --preset fuzz && cmake --build --preset fuzz` (documented in root
CLAUDE.md §4) fails at link; meanwhile a crafted .fxvol/OBJ/.zarray exercising an unchecked
decode path (e.g. rANS symbol table with counts summing past the renorm bound) ships
unfuzzed.

**Suggested fix:** add `tests/fuzz_*.cpp` harnesses (fxvol open+read, rans_decode,
lossless_decode, read_obj, .zarray parse, NRRD header) built as separate targets with
`-fsanitize=fuzzer` only on those, `fuzzer-no-link` (or nothing) on the rest; make the preset
build only fuzz targets.

## [medium/bug] Recipe runner: multi-line TOML arrays are rejected and comma-containing stage args are silently split apart

**Verdict:** unverified (medium/low)

**Location:** apps/driver.cpp:103-117; src/core/config.hpp:27-41,76-88

**Evidence:** `Config::parse` is strictly line-based (`for (std::string line; std::getline...)`
requiring `=` on every non-section line), and `get_array` does
`std::getline(ss, item, ',')` with no quote awareness. Driver `run` builds stage args from
`cfg->get_array(sname + ".args")`.

**Failure scenario 1:** a conventionally formatted recipe
```toml
stages = [
  "segment",
  "wind",
]
```
fails: line `stages = [` stores value `"["`, then `"segment",` has no `=` →
`decode_error: bad config line` — every multi-line recipe is unusable.
**Failure scenario 2 (silent):** `wind.args = ["--dims=64,64,64", "--out=o"]` → comma split
yields `--dims=64`, `64`, `64`, `--out=o` — the stage runs with wrong parameters (or aborts
via finding 1) instead of erroring. Also `get_int` (config.hpp:64-67) parses via f64 then
casts: `threads = 3.9` silently becomes 3; s64 values >2^53 corrupt.

**Suggested fix:** make the parser continue a value across lines until brackets balance
(or accumulate the whole file and tokenize); make `get_array` quote-aware; parse ints with
`from_chars<s64>` directly and reject non-integral input.

## [medium/hygiene] Driver dispatch, `info`, and the recipe runner have zero test coverage

**Verdict:** unverified (medium/low)

**Location:** apps/driver.cpp:37-142; tests/test_core.cpp:15

**Evidence:** the only registry-adjacent test is `stage_registry_roundtrip` (register/find).
Nothing exercises: verbosity-flag stripping, `help`/unknown-command exit codes, `fenix info`
coverage counting, or the `run` path (recipe load → stage lookup → per-stage args → error
propagation). The recipe runner is plain functions calling `Config` + registry — trivially
testable in-process without spawning the binary.

**Failure scenario:** finding 4 (multi-line recipes rejected, args mis-split) is exactly the
class of bug this gap already let through; any regression in `run`'s arg plumbing or exit
codes ships green.

**Suggested fix:** add `tests/test_driver.cpp` that registers a probe stage, writes a temp
recipe (incl. one multi-line array + one quoted comma arg), and asserts the stage received the
exact args and that unknown stage / failing stage return the documented codes (refactor the
`run` body into a testable `fenix::run_recipe(path, Context&)` helper if needed).

## [low/design] Test harness: a binary with zero registered cases passes, and the failure counter is not thread-safe

**Verdict:** unverified (medium/low)

**Location:** src/core/test.hpp:32-43,27-30

**Evidence:** `run_all()` over an empty `cases()` prints `0/0 cases passed` and returns 0.
`failures()` is a plain `static int` incremented by `CHECK`/`REQUIRE`.

**Failure scenario:** a PCH/refactor/macro regression that stops `TEST` registration (or a
test file whose TESTs get `#ifdef`'d out) turns that binary permanently, silently green in
CTest. Separately, any future test that calls `CHECK` inside `parallel_for` (test_log already
runs `parallel_for` in tests, just without CHECKs in the body) races the counter and can
undercount failures.

**Suggested fix:** `if (cases().empty()) { println("no tests registered"); return 1; }`; make
`failures()` a `std::atomic<int>`.

## [low/hygiene] Tautological assertion in test_substrate — seed sensitivity is not actually tested

**Verdict:** unverified (medium/low)

**Location:** tests/test_substrate.cpp:16

**Evidence:**
```cpp
CHECK(a.next_u32() != c.next_u32() || true);  // different seed: almost surely differs
```
`X || true` is always true; this CHECK cannot fail. A PCG implementation that ignored the seed
entirely would pass this file.

**Suggested fix:** compare a few draws: `bool differ = false; for (int i=0;i<8;++i) differ |=
(a.next_u32() != c.next_u32()); CHECK(differ);`

## [low/performance] `fenix info` walks every chunk serially through coverage() — minutes on a full-scroll archive

**Verdict:** unverified (medium/low)

**Location:** apps/driver.cpp:78-85

**Evidence:** triple loop over the chunk extent calling `a->coverage({cz,cy,cx})` (a per-chunk
radix-tree walk) on one thread.

**Failure scenario:** PHerc Paris 3 scale (70k×40k×40k → ~1094×625×625 ≈ 4.3×10⁸ chunks): the
"inspect an artifact" command performs hundreds of millions of page-table walks serially —
minutes of wall time for a metadata query.

**Suggested fix:** have `VolumeArchive` expose aggregate coverage counts computed by a single
radix-tree traversal (counting leaf sentinels in bulk), or at minimum parallelize the loop.

## [low/hygiene] `--threads` is documented (root CLAUDE.md §2.4) but parsed nowhere; Context.threads is dead

**Verdict:** unverified (medium/low)

**Location:** apps/driver.cpp:60; src/core/core.hpp:60

**Evidence:** root CLAUDE.md: "thread count via `--threads`/Context". No code parses
`--threads` (grep: zero hits); `Context.threads` stays 0; only `winding/diffeo_fit.hpp:211`
reads a threads field, from its own cfg. The only working control is the `FENIX_THREADS` env
var inside `cpu_budget()`.

**Failure scenario:** `fenix wind --threads 8 ...` — the flag is either rejected by a stage's
arg loop or silently ignored; the documented CLI contract does not exist.

**Suggested fix:** strip `--threads N` in driver.cpp alongside the verbosity flags, set
`ctx.threads` (and thread it into `init_thread_limits`/`cpu_budget`), or delete the claim from
CLAUDE.md.

## [low/hygiene] tools/proto is the documented "tool sprawl" smell: 8 unregistered one-off Python scripts

**Verdict:** unverified (medium/low)

**Location:** tools/proto/ (render_collage.py, surf_metrics.py, render_fenix.py, render_volume.py, render_fenix_big.py, render_big_sheet.py, slim_flatten.py, render_slim.py)

**Evidence:** tools/CLAUDE.md Gotchas: "resist accumulating experimental one-off binaries …
New experiments = a subcommand or a test, not a new binary." tools/proto holds eight ad-hoc
render/flatten prototype scripts, none mentioned in any CLAUDE.md, with no marker of which are
current vs abandoned — the exact taberna current-vs-abandoned confusion the doc warns about.

**Suggested fix:** either promote what's load-bearing into `fenix` subcommands (slice/video
already exist) and delete the rest, or add a tools/proto/CLAUDE.md declaring each script's
status and expiry.
