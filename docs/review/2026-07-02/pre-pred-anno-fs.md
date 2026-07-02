# Review — unit `pre-pred-anno-fs` (src/preprocess/, src/predictions/, src/annotate/, src/fs/)

Overall assessment: the numerical kernels (dering, wiener, guided, MUSICA, FFT, Otsu, phase-corr) are
algorithmically correct against their cited methods — axis mappings in the separable passes and FFT are
right, the dering vote/median logic matches the fysics design, and the f32→u8 round-clamp on output is
correct at both ends of the range. The real problems are at the seams: every CLI stage in this unit parses
numbers with throwing `std::sto*` in a `-fno-exceptions` binary (abort on a typo'd arg), the fxfs read path
does per-byte cache-locked sampling while its own comments claim a block-caching optimization that does not
exist, fxfs silently quantizes non-u8 archives, and `Umbilicus::center` is a linear scan sitting inside
per-voxel loops in `winding` (accidental O(N·Z)). `annotate` and `predictions` are honest stubs; the small
implemented parts have NaN/UB edges noted below.

## [high/bug] CLI numeric parsing uses throwing std::stoi/stof/stod under -fno-exceptions — malformed args abort the process

**Verdict:** CONFIRMED — Confirmed. CMakeLists.txt:90 applies -fno-exceptions -fno-rtti to fenix_headers INTERFACE (all targets). preproc_cli.hpp:65-66,83-84,107,116,134-137 and dering.hpp:229-240 parse user-supplied option values (cli::opt / opt returns the raw substring after '=') with std::stof/stoi/stoll/stod. These are non-inline libc++ exports, so they compile under -fno-exceptions but throw std::invalid_argument/std::out_of_range at runtime; with zero catch frames in the -fno-exceptions binary the throw reaches std::terminate → abort. E.g. `fenix dering in.fxvol out.fxvol slab=abc` or `fenix deconv in out sigma=` aborts the process. Not a stub: src/preprocess/CLAUDE.md Status says these stages are "Implemented + wired to CLI", and the root CLAUDE.md §2.3 error model mandates Expected<T, fenix::Error>. The in-tree correct idiom exists at core/config.hpp:113-118 and core/parallel.hpp:50 (std::from_chars).
**Fix notes:** Proposed fix is sound. Corrections/additions: (1) make cli::parse<T> generic over ints and floats via std::from_chars (floating-point from_chars is available in current libc++, which the always-latest-Clang toolchain guarantees; older libc++ lacked it); (2) also require full-string consumption (ptr == end) so "1.5x" is rejected, matching neither stof's lenient prefix parse nor silently ignoring trailing junk; (3) include the option key and offending value in the Error message for usability; (4) the helper belongs where both preproc_cli.hpp and dering.hpp can use it — either cli:: in preproc_cli.hpp with dering.hpp including it, or better in core/ next to the existing from_chars users, since other CLI stages likely want it too (worth a quick grep for std::sto* across src/ to sweep any other stages in the same PR); (5) run_deconv/run_denoise/run_aircut/run_musica/dering's run must propagate the parse failure as std::unexpected before invoking the kernel.

**Location:** src/preprocess/preproc_cli.hpp:65-66,83-84,107,116,134-137; src/preprocess/dering.hpp:229-240
**Evidence:**
```cpp
const f32 sigma = std::stof(cli::opt(args, "sigma", "1.0"));   // preproc_cli.hpp:65
p.slab_z = std::stoi(opt("slab", "512"));                       // dering.hpp:231
```
CMakeLists.txt:90 applies `-fno-exceptions -fno-rtti` to all fenix targets. `std::stoi/stof/stod` throw
`std::invalid_argument`/`std::out_of_range` from inside libc++ on non-numeric or out-of-range input; the
exception unwinds into frames compiled `-fno-exceptions` → `std::terminate` → abort.
**Failure scenario:** `fenix dering in.fxvol out.fxvol slab=abc` (or `sigma=` with an empty value, or a
value overflowing the type) aborts the whole pipeline instead of returning
`Expected<..., fenix::Error>{invalid_argument}` — a direct violation of the §2.3 error-model invariant.
**Suggested fix:** add a `cli::parse<T>(std::string_view) -> Expected<T>` helper on `std::from_chars`
(core already does exactly this in `core/config.hpp:115` and `core/parallel.hpp:50`) and use it for every
option in `run_deconv`/`run_denoise`/`run_aircut`/`run_musica`/`run_dering`. (The same pattern exists in
io/slice, geom/mesh, segment/trace_surface, ml/ — out of this unit's scope but worth a sweep.)

**Outcome:** fixed — added `cli::parse<T>(key, tok) -> Expected<T>` to `src/preprocess/preproc_cli.hpp`
(from_chars, full-token-consumption required) and used it for every option in `run_deconv`
(also now validates `sigma > 0`/`reg > 0`, fixing the separate low-severity Wiener NaN finding in
this same file), `run_denoise`, `run_aircut`, `run_musica`. `dering.hpp` gets its own equivalent
`detail::parse_opt<T>` (kept local — it's a leaf header, not including preproc_cli.hpp) used via a
`FENIX_DERING_PARSE` macro for all 12 `DeringParams` fields. `geom/mesh.hpp`'s `read_obj` throwing-
`stoi` is fixed under Finding 4 above (same unit's other listed callers — io/slice, segment/
trace_surface, ml/ — are out of this cluster's assigned files and were left for their owning
agents). Verified: `fenix dering slab=abc`-style malformed args now return
`Expected` errors instead of aborting (checked by direct `from_chars`-failure-path inspection; no
dedicated CLI-abort test was added since these are argv-parsing wrappers around already-tested
kernels, not new logic).

## [high/performance] fxfs fx_read samples per BYTE through the locked block cache; the "current block" caching its own comment claims does not exist

**Verdict:** CONFIRMED — Confirmed by direct reading. fx_read's loop (src/fs/fxfs.cpp:102-113) calls g->ar.sample_f32(0,z,y,x) once per output byte with no state carried across iterations — no BlockCache::Ref, key, or block-coord variable exists anywhere in the function, so the comment at fxfs.cpp:96-100 ("The 'current block' is cached across the loop so a contiguous x-run (16 voxels) costs one block16 call") describes code that was never written; src/fs/CLAUDE.md:33-34 repeats the same false claim as present-tense fact (its own TODO at line 37/48 lists only the *different* "block-aligned gather" as future). The per-byte cost is exactly as claimed: sample_f32 (src/codec/archive.hpp:450-459) calls block16 every time; block16 (archive.hpp:401-404) pays the lazy-init branch, block_key_ hash, and BlockCache::get, which takes a per-shard std::mutex lock_guard, a hash-map find, and a shared_ptr copy (atomic refcount) per call (src/codec/block_cache.hpp:33-44). Plus the div/mods at fxfs.cpp:104-106 and archive.hpp:451,453, and the f32 widen → lround → clamp at archive.hpp:455 / fxfs.cpp:111-112, which is a pure identity round-trip for u8-native data (the cache stores native u8 per archive.hpp:396-398). So a 128 KiB FUSE read really is 131,072 lock acquisitions touching only ~35 distinct 16³ blocks, and concurrent readers contend on the 16 shard mutexes. Nothing guards or mitigates this: main() (fxfs.cpp:135) only sizes the cache; the kernel page cache helps re-reads, not first-pass whole-volume streaming, which is the module's stated purpose (fs/CLAUDE.md:4-7, "mmap/numpy.memmap" consumers). Not refutable as a documented stub — the docs claim the optimization exists.
**Fix notes:** The proposed fix is sound with four refinements. (1) Gate the memcpy fast path on the archive's source dtype being u8: the BlockCache stores f32-in-bytes for non-u8 archives (archive.hpp:399-400, 410), so raw byte copy is only valid for u8 — VolumeArchive currently keeps src_dtype_ private and fxfs hardcodes u8 (fxfs.cpp:136), so either expose a dtype()/is_u8_native() accessor or keep the sample_f32 fallback for non-u8. (2) Holding one Ref and copying x-runs still leaves a key-hash+lock every 16 bytes; for large reads the better shape is the block-major pattern already proven in gather_box (archive.hpp:462+): enumerate the covering 16³ blocks of [offset, end), fetch each exactly once, and scatter its rows into buf — that's one lock per 4 KiB block instead of per 16-byte run. (3) The copyable run is min(16 - x%16, X - x, bytes remaining) — a C-order linear run crosses the y-row (and volume-width) boundary, not just the block boundary, and X need not be a multiple of 16; the row-clamp must be explicit. (4) Fix BOTH stale docs: the comment block at fxfs.cpp:96-100 and the false present-tense sentence in src/fs/CLAUDE.md:33-34 (per root CLAUDE.md §5.2.7, keep the dir CLAUDE.md current). The u8 fast path also correctly drops the lround/clamp identity round-trip, preserving the -EIO-on-error invariant as long as block16 failures still return -EIO.

**Location:** src/fs/fxfs.cpp:96-113
**Evidence:**
```cpp
// ... The "current block" is cached across the loop so a contiguous x-run (16 voxels) costs one block16 call ...
for (u64 i = 0; i < n; ++i) {
    const u64 lin = static_cast<u64>(offset) + i;
    const s64 x = static_cast<s64>(lin % static_cast<u64>(X));
    ...
    auto v = g->ar.sample_f32(0, z, y, x);
    if (!v) return -EIO;
    const s32 q = static_cast<s32>(std::lround(*v));
```
There is no current-block variable. Each byte pays: `block16()` (lazy-init branch, key hash, sharded cache
lock, `shared_ptr` copy in `BlockCache::get`), two u64 div/mods, an f32 widen, `lround`, and a re-clamp of
data that is already u8. A 128 KiB FUSE read = 131,072 lock acquisitions for data that lives in at most
~35 cached blocks. The whole point of this fs (CLAUDE.md: mmap/whole-volume reads for external tools) makes
the read path the hot path; this is roughly two orders of magnitude off, and under multithreaded FUSE the
cache-shard locks become the contention point.
**Failure scenario:** `cp mnt/volume.raw /dev/shm/` or a numpy.memmap full-volume scan crawls at
lock-acquisition speed instead of memcpy speed; concurrent readers serialize on the cache shards.
**Suggested fix:** hold the `BlockCache::Ref` for the current 16³ block across the loop; for u8-native
archives `memcpy` the contiguous x-run within the block (up to 16 bytes at once, no f32 round-trip) and
call `block16` only when crossing a block boundary. Fix or delete the stale comment either way.

**Outcome:** fixed (with box-verification caveat) — `fx_read` in `src/fs/fxfs.cpp` now walks the
request in block-local x-runs: for each run's first voxel it calls `block16()` once and `memcpy`s
the contiguous bytes covered by that call (clamped by the block boundary, the row/y boundary since
X need not be a multiple of 16, and the remaining request size) for u8-native archives — the raw
byte copy per fix_notes refinement (1), gated on `src_dtype()==DType::u8`; non-u8 archives keep the
old per-voxel `sample_f32`+`lround` fallback (fxfs currently hardcodes u8 mounting anyway). Per
fix_notes refinement (2) this is the row-clamped block-major shape, not merely "16-byte runs
ignoring the row boundary". Did NOT implement refinement (2)'s further suggestion of reusing
`gather_box`'s multi-block enumeration (that's a bigger restructure; the per-run block16 call
already collapses ~131k calls/128KiB read to ~1 per 16 bytes, the stated two-orders-of-magnitude
fix) or fix the stale doc comments beyond rewriting them to describe the real fast path (fix_notes
(4)). **NOTE (per the task instructions): FENIX_FS=OFF on this Mac (no libfuse3), so this could not
be compiled via the normal CMake target.** Verified two ways instead: (a) `clang++ -fsyntax-only`
against the real `codec::VolumeArchive`/`BlockCache` headers on an extracted copy of the exact
read-loop logic (type-checks clean, no diagnostics) — confirms `src_dtype()`, `block16(lod,
ChunkCoord)`, and the `BlockCache::Ref`/`Block` API are used correctly; (b) careful manual review of
the block/row/request clamping arithmetic. This still needs a real build+mount smoke test on a Linux
box with libfuse3 (`FENIX_FS=ON`) before merge — flagging for box verification as instructed.

## [medium/correctness] fxfs silently serves round-clamped u8 for non-u8 archives (and meta.toml lies about it)

**Verdict:** unverified (medium/low)

**Location:** src/fs/fxfs.cpp:111-112,136,141
**Evidence:**
```cpp
st.esz = 1;   // u8 archives (dtype-agnostic store; these scrolls are u8)
...
"dims_zyx = [...]\ndtype = \"u8\"\n..."
...
const s32 q = static_cast<s32>(std::lround(*v));
buf[i] = static_cast<char>(static_cast<u8>(q < 0 ? 0 : q > 255 ? 255 : q));
```
The archive is dtype-tagged (`VolumeArchive::src_dtype()`, archive.hpp:259) and the file header comment
claims "the element type is carried in via --dtype (default u8)" — but no `--dtype` parsing exists (argv[2:]
goes straight to fuse). Mounting an f32 archive (e.g. a prediction field — predictions ARE stored as lossy
.fxvol per predictions/CLAUDE.md) produces a `volume.raw` of the wrong size whose bytes are the f32 values
round-clamped to 0..255, and a `meta.toml` asserting `dtype = "u8"`. That is silent data corruption of
exactly the absent-vs-wrong kind §2.4 forbids; a `[0,1]` probability field reads back as all 0s/1s.
**Failure scenario:** `fenix-fxfs sheet_prob_f32.fxvol mnt && python -c "np.memmap('mnt/volume.raw',...)"`
→ garbage field, no error anywhere.
**Suggested fix:** in `main`, check `ar->src_dtype()`; refuse non-u8 with a loud error until `--dtype`
lands (or implement it: esz from dtype, raw memcpy of native bytes, honest meta.toml). Also fix the
top-of-file comment claiming the option exists.

## [medium/performance] Umbilicus::center is a linear scan invoked per voxel via polar() — accidental O(N·Z) in the winding hot path

**Verdict:** unverified (medium/low)

**Location:** src/annotate/umbilicus.hpp:24-26 (callers: src/winding/winding_field.hpp:31, src/winding/spiral_fit.hpp:68)
**Evidence:**
```cpp
usize i = 1;
while (i < z.size() && z[i] < zq) ++i;
```
`estimate()` emits one control point per z-slice, so `z.size() == Z`. `winding_init` calls
`umb.polar(z,y,x,...)` for every voxel, and `polar` calls `center` which scans from the front every time.
For a 2048³ volume that is up to 2048 scan iterations × 8.6e9 voxels ≈ 1.7e13 wasted compares; at scroll
scale (Z up to 70k) it is catastrophic. The struct documents `sorted ascending by z` — the invariant for a
binary search is already there.
**Failure scenario:** `winding_init` on any volume with a per-slice umbilicus runs minutes-to-hours in
`center()` instead of seconds in the actual field math.
**Suggested fix:** `std::upper_bound` on `z` in `center()` (O(log n)); additionally hoist `center(z)` out of
the per-(y,x) loops in `winding_init`/`spiral_fit_from_field` (it is constant per slice).

## [medium/performance] dering pass 1 traverses the volume plane-strided (z innermost) and both passes are single-threaded

**Verdict:** unverified (medium/low)

**Location:** src/preprocess/dering.hpp:84-104 (pass 1), 177-192 (pass 2)
**Evidence:**
```cpp
for (s64 y = 0; y < d.y; y += ss) {
    for (s64 x = 0; x < d.x; x += ss) {
        ...
        for (s64 z = 0; z < d.z; ++z) {
            const f32 v = static_cast<f32>(vol(z, y, x));
```
With ZYX x-fastest storage, the inner z loop strides by Y·X·sizeof(T) per step (a full plane — 1 MiB for
512² f32), so essentially every accumulate is a cache+TLB miss across the entire volume. Every sibling
kernel in this module (musica, guided, aircut) is z-major and uses `parallel_for_z`; dering's two
full-volume passes are the only serial, cache-hostile ones and dominate `fenix dering` wall time. The
comment ("slab locality") argues the order avoids recomputing the (sector,radius) map per z — but that map
is only Y·X entries.
**Failure scenario:** `fenix dering` on a 1024³ f32 volume spends ~all its time in pass-1 memory stalls,
single-core, while the machine idles.
**Suggested fix:** precompute a per-(y,x) `(sector, rbin)` map once (Y·X × 8 bytes), traverse z-major, and
parallelize over z with per-slab (or per-thread, reduced) accumulators; pass 2 is embarrassingly parallel —
wrap in `parallel_for_z`.

## [low/correctness] Wiener deconv with reg<=0 produces 0/0 NaN that fast-math then writes out as silent garbage

**Verdict:** unverified (medium/low)

**Location:** src/preprocess/deconv.hpp:76; src/preprocess/preproc_cli.hpp:66
**Evidence:**
```cpp
const f32 inv = h / (h * h + reg);  // Wiener
```
`run_deconv` accepts `reg` from the CLI with no validation. For high frequencies
`h = exp(-2π²σ²f²)` underflows to 0.0f, so `reg=0` yields `0/0 = NaN`; the inverse FFT smears NaN across
the whole volume; `cli::write`'s `std::clamp(NaN,...) + 0.5f → static_cast<u8>` is UB on NaN (and
`isnan` guards would be unreliable under `-ffast-math` anyway). Result: a fully corrupt `.fxvol` written
with exit code 0.
**Failure scenario:** `fenix deconv in.fxvol out.fxvol reg=0` → garbage archive, success log line.
**Suggested fix:** validate `reg > 0` (and `sigma > 0`) in `run_deconv`, returning
`Errc::invalid_argument`.

## [low/performance] guided box_mean is O(N·r) per axis, not the O(N) the He-Sun-Tang filter promises — and serial

**Verdict:** unverified (medium/low)

**Location:** src/preprocess/guided.hpp:34-38
**Evidence:**
```cpp
for (s64 t = 0; t < n; ++t) {
    f32 s = 0;
    for (s64 k = -r; k <= r; ++k) s += line[static_cast<usize>(reflect(t + k, n))];
```
The header comment sells this as "the fysics-recommended O(N) edge-preserving denoiser", and
preprocess/CLAUDE.md names the guided box filter the hottest kernel — but the window is re-summed per
output sample (O(N·(2r+1)) per axis, ×6 box_mean calls per guided_filter), and nothing is parallelized.
Fine at the default r=2; noise-matched larger radii scale linearly, and the whole-scroll streaming wrapper
(TODO) will inherit it.
**Failure scenario:** `fenix denoise ... r=8` runs ~5× slower than r=2 for identical output quality math.
**Suggested fix:** running-sum sliding window (add leading, subtract trailing — true O(N)) and
`parallel_for` over the (o1,o2) line grid.

## [low/correctness] predictions::normalize sorts external fields containing NaN (UB) and reads flat()[0] of a possibly-empty field

**Verdict:** unverified (medium/low)

**Location:** src/predictions/field.hpp:26-33
**Evidence:**
```cpp
std::vector<f32> v(field.flat().begin(), field.flat().end());
std::ranges::sort(v);
...
lo = hi = field.flat()[0];
```
Prediction fields come from external ML tooling and can legitimately contain NaN. `sort` with NaN violates
strict weak ordering — genuine UB (libc++ can scramble or crash) — and this module can't lean on
`std::isnan` filtering because fenix compiles `-ffast-math`. MinMax also dereferences element 0 with no
empty check.
**Failure scenario:** normalize(Percentile) on an externally-produced field with NaN padding → scrambled
percentiles or a crash; either way the "conditioned" data term fed to the fit is wrong.
**Suggested fix:** filter non-finite values by bit pattern (exponent bits all-set on the u32 image —
fast-math-safe) while copying into `v` / while scanning min/max; return the input (or an Error) for empty
fields. `std::nth_element` twice instead of a full sort while at it.

## [low/bug] Umbilicus::center divides by zero on duplicate z control points

**Verdict:** unverified (medium/low)

**Location:** src/annotate/umbilicus.hpp:26
**Evidence:**
```cpp
const f32 t = (zq - z[i - 1]) / (z[i] - z[i - 1]);
```
`estimate()` cannot produce duplicates, but the struct is the documented target of TOML load (module
CLAUDE.md) and is a plain public SoA anyone can fill. Two control points at the same z make the divisor 0
→ inf/NaN center → NaN radius/theta from `polar()` poisoning the entire cylindrical frame — the
"highest-leverage annotation" failing silently, with no fast-math-reliable NaN check downstream.
**Failure scenario:** a hand-edited umbilicus TOML with a repeated z row → the whole winding field is NaN.
**Suggested fix:** skip zero-length segments in `center` (return the endpoint), and `FENIX_ASSERT`
strictly-increasing z at load/construction time.
