# sweep-oom-ooc — memory-scale & out-of-core discipline (all of src/)

Scope: cross-cutting sweep for full-volume/full-surface materialization, s64-vs-s32 index math,
byte-budget/backpressure, and mmap-vs-read, judged against the 70k×40k×40k envelope.

**Overall assessment.** The flagship out-of-core paths are genuinely disciplined: `export-scroll`
(bounded producer queue, region-at-a-time, resumable), `trace_volume_streamed[_to_disk]`
(tile+halo, fragments to disk), `stitch_streamed[_3d]` (slab/tile-local fields), the mmap'd
Morton radix page table, and the byte-budgeted SIEVE `BlockCache`. The s64 index audit is
essentially clean — `Extent3`/strides are s64 everywhere, and the few `int`s are grid/patch-local
(G ≤ 1400) or thread counts; I found no three-int-multiply-before-widen bug. The problems are at
the edges: (1) the OOC tracer's tile fetch **silently converts a hard fetch failure into zeros**
— a direct violation of the project's own "absent ≠ fetch-failed" hard rule; (2) the `.fxvol`
mmap reservation caps any archive at 1 TiB, below a plausible whole-scroll compressed size;
(3) a family of CLI stage entry points (`eval`, `render`, `transcode`, preprocess, `trace-surface`,
slice/augment) decode entire archives to **dense f32** (`read_volume()`), quadrupling RAM on
u8-native data and making those stages secretly in-core; and (4) the eval metric chain stacks
~20 B/voxel of transients (dual f32 EDTs + an 8 B/voxel union-find parent) on top of the f32
pair, so `fenix eval` OOMs on volumes that inference already handles. Findings below, most
severe first.

## [critical/correctness] OOC tracer tile fetch silently turns a hard fetch failure into air
**Verdict:** CONFIRMED — src/segment/trace_stream.hpp:30 discards the Expected from io::read_zarr_region, which returns Errc::fetch_failed for hard fetch failures distinctly from absent chunks (src/io/zarr.hpp:173-177, 204 — 'hard fetch failure — record, do NOT treat as air'). The early-returned tile is UNINITIALIZED (Volume<u8> out(pe) uses make_unique_for_overwrite, src/core/volume.hpp:91); the in-file comment claiming 'a hard fetch error yields zeros here' is wrong — the zero-init lives inside read_zarr_region's own discarded output (zarr.hpp:123). Both callers (trace_stream.hpp:60/62 and :105/107) are the production OOC tracers per src/segment/CLAUDE.md ('the FULLY out-of-core tracer', tested), not stubs, and neither threads the error. Direct violation of root CLAUDE.md §2.4 ('a transient fetch error must never silently become air') and the io CLAUDE.md invariant; the S3-blip → silent-missing-wraps → exit 0 scenario is reachable, plus the uninit read is UB.
**Fix notes:** Proposed fix direction is right (return Expected<Volume<u8>> and propagate) with corrections: (1) No Volume<u8>::zeros needed in stream_tile_u8 — read_zarr_region already zero-inits its output and writes fill_value for legitimately absent chunks; on success stream_tile_u8 overwrites every voxel, and on failure it now just propagates, so the uninit path disappears entirely. (2) trace_volume_streamed currently returns plain VolumeResult — its signature must become Expected<VolumeResult>, updating callers in tests/test_trace_stream.cpp (:128, :217) and the direct stream_tile_u8 use at :152. trace_volume_streamed_to_disk already returns Expected so propagation is trivial there. (3) The extra tile-level retry/backoff loop is optional hardening: s3.hpp's fetch_object already does exponential-backoff retry + a low-speed stall watchdog per src/io/CLAUDE.md, so the minimal correct fix is propagation; if added, mirror export_scroll's capped producer retry for consistency. (4) Also fix the false comment at trace_stream.hpp:25-26 ('yields zeros').
**Location:** src/segment/trace_stream.hpp:30 (`stream_tile_u8`)
**Evidence:**
```cpp
inline Volume<u8> stream_tile_u8(const std::string& root, Index3 porg, Extent3 pe, f32 scale) {
    Volume<u8> out(pe);
    auto r = io::read_zarr_region(root, porg, pe);
    if (!r) return out;   // hard fetch error -> all-zero tile, no error propagated
```
The comment admits it: "a hard fetch error yields zeros here — production callers that must
honour 'absent != fetch-failed' should thread the Expected through." But the production callers
ARE this file: `trace_volume_streamed` (line 60) and `trace_volume_streamed_to_disk` (line 105)
call it unconditionally and never see the failure. `io::read_zarr_region` correctly distinguishes
404-absent (fill) from `Errc::fetch_failed` (hard error with retries in `s3.hpp`), and this
wrapper throws that distinction away. Note also `Volume<u8> out(pe)` uses
`make_unique_for_overwrite` — the early-return tile is **uninitialized memory**, not even zeros.
**Failure scenario:** a multi-hour whole-scroll streamed trace hits one sustained S3 blip
(DNS flap, 5xx storm outlasting the s3.hpp retry budget) on one tile → that tile reads as
garbage/air → the tracer seeds nothing there → wraps silently missing from the fragment set and
the manifest, with exit code 0. This is exactly the matter-compressor lesson the conventions doc
calls fatal ("a transient blip poisoning a TB output").
**Suggested fix:** return `Expected<Volume<u8>>` (and zero-init via `Volume<u8>::zeros` for the
legitimate all-absent case), propagate through `trace_volume_streamed[_to_disk]` (both already
have `Expected`-shaped or trivially-changeable signatures), and add a retry/backoff loop like
`export_scroll`'s producer.

## [high/design] .fxvol mmap reservation hard-caps any archive at 1 TiB — below whole-scroll size
**Verdict:** CONFIRMED — Verified in code: src/codec/archive.hpp:86 fixes kFxReserve at 1 TiB, map_() (line 644) maps exactly that, and ensure_() (line 652) hard-errors "archive exceeds reservation" on any growth past it; create() (line 102) performs no dims-based check. The scenario is reachable via a shipped tool built for exactly this: export-scroll (src/io/io.hpp:125-215) creates the archive at the FULL zarr shape and streams a whole level; root CLAUDE.md targets 2^18/axis volumes (PHerc Paris 3 ~1.12e14 voxels u8), and measured codec ratios (~6-13x at q2-q8, codec CLAUDE.md) plus the LOD pyramid (~+14%) and documented COW orphan bloat (io.hpp:151-153, up to ~45% of committed bytes at commit=1) put the whole-scroll archive plausibly past 1 TiB. Resume genuinely cannot help since the file itself cannot grow. The "Phase-1 limitation" comment does not refute: docs/design/fxvol-v4-layout.md §9 shows all remaining phases DONE or dropped, and none lifts the reservation — the limitation has no scheduled fix, and there is no create-time rejection, so a multi-day export fails mid-run.
**Fix notes:** Direction is right; corrections: (1) mremap-style growth is NOT portable — the codebase explicitly supports macOS (archive.hpp:653 __APPLE__ branch) and macOS has no mremap; also base_-relative pointers held across a remap would need care under concurrent appends. Prefer the dims-derived fixed reservation. (2) Sizing "chunk-grid worst case" naively (raw bytes + headroom) can exceed user VA for the max-supported 2^18/axis dims (2^54 B raw > 128 TiB x86-64 47-bit VA); clamp the reservation to a platform VA budget and only then refuse at create() if the worst case still exceeds it — with the refusal message telling the user to split or use a coarser level. (3) open() must also size correctly: dims live in the superblock, so map a small fixed window (one superblock page) first, read dims, then remap at the computed reservation. (4) Keep the sanitizer small-reserve path (line 84) but apply the same create-time check so ASan tests fail fast, not mid-write. (5) Worst-case estimate must include the LOD pyramid (~x8/7), page-table nodes, and COW orphan headroom (bounded by commit batching, but include margin), and use an incompressible-data bound (compressed tile can slightly exceed raw), not an assumed ratio.
**Location:** src/codec/archive.hpp:86 (`kFxReserve`), error at archive.hpp:652
**Evidence:**
```cpp
inline constexpr u64 kFxReserve = 1ull << 40;  // 1 TiB (caps a single archive; Phase-1 limitation)
...
if (want > detail::kFxReserve) return err(Errc::io_error, "archive exceeds reservation");
```
PHerc Paris 3 is ~1.12e14 voxels u8. At quality q2 (~10× ratio measured) the masked scroll is
several TiB compressed; even q8 (~45× on the smooth crop; dense regions compress worse) plus the
LOD pyramid and COW page-table orphans can plausibly cross 1 TiB. `export-scroll` — the tool
whose whole point is whole-scroll archives — will then hard-fail mid-export after days of
transfer, and no resume can help because the file itself cannot grow.
**Failure scenario:** `fenix export-scroll s3://... 0 scroll.fxvol q=2` on a full scroll runs for
days, hits the reservation at ~1 TiB, and every subsequent run fails at the same point.
**Suggested fix:** size the reservation from `dims` at create/open (chunk count × worst-case blob
bound, or simply a much larger constant — VA space is free with `MAP_NORESERVE`), or implement
re-`mmap` growth (offsets are file-relative, only `base_` consumers need re-derivation). At
minimum, compute the worst-case size up front and refuse at *create* time, not mid-export.

## [high/performance] `fenix eval` is in-core at ~30+ bytes/voxel: dense-f32 loads + dual EDT + union-find parent
**Verdict:** CONFIRMED — The scenario is real and unguarded. src/eval/eval.hpp:29-36 `load_f32` calls `a->read_volume()`, which is `read_volume_as<f32>` (src/codec/archive.hpp:331) — a dense f32 decode of the whole archive, 4x inflation for u8-native scroll data. In `score_pair` (eval.hpp:70-92) both f32 volumes `pv`/`gv` stay alive for the entire scoring (r.dims reads pv->dims() at line 90 after all metrics run), the u8 masks add 2 B/voxel, and `official_score` then runs `nsd` which allocates two more full-volume u8 surface masks plus TWO full-volume f32 `edt_squared` buffers simultaneously (src/eval/nsd.hpp:38-41) — peak during NSD alone on a 2048^3 pair is ~68.7 GB (f32 pair) + 17.2 GB (masks) + 17.2 GB (surfaces) + 68.7 GB (EDTs) ≈ 172 GB. The VOI/topo CC passes each allocate std::vector<s64> parent(n) = 68.7 GB plus an s32 label volume (src/geom/connected_components.hpp:16-30). No dimension guard or windowing exists anywhere in eval/. This also collides with the root CLAUDE.md hard rule "Out-of-core is a hard rule ... Every stage is block + halo + stitch" (§2.4), and eval/CLAUDE.md's own performance note "Windowed/cropped evaluation for large volumes" is unimplemented. The best refutation candidate — docs/design/ml-accel-and-distillation.md §6 (commit ab03fb8) — documents only a SPEED limitation ("single-threaded → slow on 1024^3 ... use 256^3/512^3 crops") and prescribes crop-based eval as the intended Phase-1 workflow, but it never documents or guards the memory blowup, and eval/CLAUDE.md lists these paths under "Implemented + tested", not stubbed. A user pointing `fenix eval` at a large pred/gt pair hits the OOM with no warning.
**Severity adjusted to:** medium
**Fix notes:** Downgrading to medium: docs/design/ml-accel-and-distillation.md §6 explicitly establishes small-crop (256^3/512^3) evaluation as the intended Phase-1 workflow ("the right unit anyway — you score the held-out TEST crops"), so the OOM path is off the documented happy path, and eval is an offline measurement tool, not the pipeline. Corrections to the proposed fix: (1) `read_volume_as<u8>` exists (archive.hpp:298) but u8-native loading only helps .fxvol; the .nrrd path (io::read_nrrd) and 0..1 softmax probability maps are genuinely f32, so the robust fix is streaming per-chunk threshold-on-decode (decode chunk -> compare -> write u8 mask), which kills the dense f32 for both dtypes — note this requires two passes or a running max for the auto 0.5*peak threshold (peak() currently needs the full volume). (2) "Free each intermediate" needs `r.dims = pv->dims()` captured BEFORE releasing pv/gv (eval.hpp:90 reads it last), and nsd() should free each surface mask right after its EDT is built (halves NSD peak). (3) NSD tiling needs a ceil(tau) z-halo (tau default 2, cheap) since EDT distances cross tile boundaries; the VOI contingency table is only tileable AFTER a global CC labeling, so the CC block+boundary-merge pass is the prerequisite, and connected_components' parent array could independently drop from s64 to s32-per-block with the merge pass. (4) Cheapest immediate mitigation consistent with the design doc: a hard-fail/warning in score_pair when dims.count() exceeds a byte-budget estimate, pointing users at crop-based eval, until windowing lands. Note also the design doc's instruction NOT to prematurely rework shared geom/topo primitives for eval — the CC merge-pass change should be justified on its own or scoped to eval.
**Location:** src/eval/eval.hpp:29 (`load_f32`), src/eval/nsd.hpp:40, src/geom/connected_components.hpp:30
**Evidence:** `load_f32` does `a->read_volume()` (dense f32 of the whole archive — 4×
inflation on u8 data), for **both** pred and gt; `official_score` then runs `nsd` (two full-volume
u8 surface masks + **two full-volume f32 `edt_squared` buffers**), `voi_union` (two 26-conn
`connected_components`, each allocating `std::vector<s64> parent(n)` = **8 B/voxel** + a full-n
s32 `root_label` + s32 labels), and `topo_score` (three more CC passes + a bg copy).
**Failure scenario:** eval of a 2048³ pair: 2×34 GB f32 volumes + 17 GB masks + 68 GB EDTs +
69 GB CC parent ⇒ far over a 128 GB box — the *scoring* step OOMs on a volume the ML inference
path handles in ~34 GB. The eval CLAUDE.md promises "windowed/cropped evaluation for large
volumes" but no windowing exists in `eval.hpp`.
**Suggested fix:** load as native u8 (`read_volume_as<u8>` + threshold during decode, never a
dense f32), free each intermediate before the next metric, and/or implement the promised
z-window/tile evaluation (NSD and the VOI contingency table are both tileable; CC needs a
block+boundary-merge pass, which `connected_components` already half-implements via its slab
phases).

## [high/design] Eight CLI stage entry points decode the whole archive to dense f32 (`read_volume()`), 4× RAM on u8-native data
**Verdict:** CONFIRMED — codec/archive.hpp:331 — `read_volume(s64 lod=0)` is exactly `read_volume_as<f32>(lod)`, a dense full-LOD0 f32 decode. All eight cited call sites verified to call it on the whole archive: src/io/io.hpp:389 (transcode), src/render/render.hpp:35 (plus render.hpp:44 `winding_init` returning a second full-dims Volume<f32> per winding/winding_field.hpp:22, plus a third dense pass in unroll — triple dense f32 confirmed), src/preprocess/preproc_cli.hpp:26, src/eval/eval.hpp:33, src/io/slice.hpp:99, src/segment/trace_surface.hpp:31, src/preprocess/dering.hpp:221, src/ml/augment_cli.hpp:27. No caller checks `src_dtype()` (accessor exists, archive.hpp:259) and no size/RAM guard exists on any path, so `fenix transcode <2048³-u8>.fxvol out.fxvol 8` reachably allocates the 34 GB f32 buffer. The scenario is not just reachable — the codebase itself documents it as wrong: archive.hpp:292-296 ('a u8 archive → Volume<u8>, 8 GiB for a 2048³, NOT 34 GiB f32'), the v5 format comment at archive.hpp:41 ('source-dtype tag (native-dtype read path; no f32 widen)'), and preproc_cli.hpp's own write() comment ('never widen to f32'). The v5 native-dtype machinery was built precisely to avoid this and these callers bypass it. Partial mitigations that do NOT refute: render/ and preprocess/ are documented STUBs, and several tools legitimately need f32 compute or f32-native [0,1] inputs (transcode's scale255 path) — but needing f32 math per-tile/slab does not justify a whole-volume f32-resident load, and eval/segment/slice/ml/io-transcode are not stubs. The finding stands as written.
**Fix notes:** Fix direction is correct; corrections/additions: (1) The dtype-aware branch must keep the f32 path when src_dtype()==f32 — transcode's scale255 flow assumes a [0,1] f32 field, so branch on src_dtype(), don't unconditionally read u8. (2) Additional related defect in transcode (io.hpp:405): the non-scale255 path writes via the f32 convenience overload `write_volume(vol->view())`, which retags the OUTPUT archive src_dtype_ as f32 (archive.hpp:272,290) even when the input was u8 — losing the native-u8 decoded-tile cache path (archive.hpp:407) downstream. The fix should read u8 AND write via write_volume<u8> to preserve the tag. (3) 'convert per-tile where f32 math is needed' is nontrivial: most downstream kernels (unroll, marching cubes, dering, deconv, eval::threshold) take VolumeView<const f32>; the cheap interim win is hold dense u8 + widen per-slab (dering already slabs via slab_z, eval::threshold can template on T trivially), with true chunk-streaming via read_chunk_as as the longer-term fix, as proposed. (4) For stages whose algorithm genuinely needs the whole f32 field resident at once, dtype-aware load alone doesn't cap peak RAM — only the out-of-core rewrite does; the fix note's two-tier framing (dtype-aware now, streaming later) is the right shape.
**Location:** src/io/io.hpp:389 (`transcode`), plus src/render/render.hpp:35,
src/preprocess/preproc_cli.hpp:26, src/eval/eval.hpp:33, src/io/slice.hpp:99,
src/segment/trace_surface.hpp:31, src/preprocess/dering.hpp:221, src/ml/augment_cli.hpp:27
**Evidence:** `auto vol = a->read_volume(0); // decode LOD0 dense` — `read_volume()` defaults to
`read_volume_as<f32>`. Every scroll archive is u8 (`src_dtype` exists precisely so reads stay
native, per the v5 format and the project's "no f32 widening of u8" rule), yet these stages widen
the entire volume to f32. `fenix render` is the worst: dense f32 volume + a second full-volume
f32 `winding_init` field + the unroll scatter = 3 dense f32 buffers.
**Failure scenario:** `fenix transcode big.fxvol out.fxvol 8 scale255` on a 2048³ u8 archive
allocates 34 GB f32 + another 8.6 GB u8 copy, where 8.6 GB u8 (+ per-tile f32) suffices; a 4096³
archive (275 GB f32) is impossible on any current box though it is only 69 GB u8. None of these
stages can ever run at scroll scale, and they OOM ~4× earlier than necessary today.
**Suggested fix:** make the archive-load helpers dtype-aware: `read_volume_as<u8>` when
`src_dtype()==u8` (all of transcode/slice/augment/render can work from u8 directly; eval
thresholds can be applied on u8; preprocess kernels can convert per-tile). Longer term these
stages should stream chunk-wise (the archive already supports `read_chunk_as`/`gather_box_f32`).

## [medium/performance] `write_level_` buffers the ENTIRE level's compressed payloads in RAM before the serial commit
**Verdict:** unverified (medium/low)
**Location:** src/codec/archive.hpp:598 (`std::vector<Enc> enc(ntiles)`)
**Evidence:**
```cpp
struct Enc { std::vector<u8> payload; bool zero; };
std::vector<Enc> enc(static_cast<usize>(ntiles));
parallel_for(0, ntiles, [&](s64 i) { ... enc[i].payload = encode_tile_dct<T>(...); });
for (s64 i = 0; i < ntiles; ++i) { ... commit_encoded_(...); }
```
Phase 1 encodes every 64³ tile of the level into its own heap buffer and keeps ALL of them until
phase 2 drains them serially. Peak RAM = the source volume + the whole compressed level (+ heap
fragmentation from millions of small vectors). For `write_volume` on a large in-RAM volume this
adds the full compressed size (GBs) on top of the already-resident source, and it scales linearly
with volume — an avoidable non-out-of-core buffer inside the container itself.
**Failure scenario:** `fenix ingest` of an 8 GiB u8 2048³ at low q adds ~1–4 GiB of retained
payload buffers at peak, right when the box is already holding the dense source; larger ingests
scale this linearly until OOM.
**Suggested fix:** pipeline it — bounded ring of encoded payloads (like `export_scroll`'s
producer/consumer), or commit in fixed-size batches of tiles (encode N in parallel, commit N,
free, repeat).

## [medium/performance] Rotation/ensemble TTA holds up to four full-volume f32 buffers concurrently
**Verdict:** unverified (medium/low)
**Location:** src/ml/infer.hpp:552 (`predict_surface_rots`), also 483–511, 610–655
**Evidence:** `predict_surface_rots` keeps `acc` (f32 volume) + `wsum` (f32 vector, `d.count()`)
+ each member's `prob` + `valid` (both full f32 volumes) alive at once = 16 B/voxel, and each
inner `predict_surface` allocates its own dense `prob` accumulator on top. `predict_surface_tta`
and `predict_surface_scales` stack the same way (acc + member prob + inner accumulator, plus
`resample_f32`'s extra contiguous copy). The dense accumulator itself is a documented TODO
("out-of-core accumulators for whole-scroll inference"), but the ensemble wrappers *multiply* it
3–4× beyond that acknowledged baseline.
**Failure scenario:** rot-TTA on a 2048³ crop: 4×34 GB ≈ 137 GB CPU RAM — OOM on the 128 GB GPU
box for a volume plain inference handles in ~34–68 GB.
**Suggested fix:** accumulate in place per member (add into `acc` slab-by-slab as each member's
tiles finish rather than materializing member `prob` first), reuse one scratch volume across
members, and store `wsum` as u8/u16 validity counts (it only ever holds small sums of weights).

## [medium/performance] export-scroll air-skip fallback can dense-load a near-full-res pyramid level as the occupancy map
**Verdict:** unverified (medium/low)
**Location:** src/io/io.hpp:177 (`for (s64 k = 5; k >= 1; --k)`)
**Evidence:**
```cpp
for (s64 k = 5; k >= 1; --k) {
    auto m = read_zarray(root + "/" + std::to_string(level_int + k));
    if (!m) continue;
    auto o = read_zarr_region<u8>(root + ..., {0,0,0}, m->shape);  // ENTIRE level, dense
```
The loop tries level+5 (÷32, fine) down to level+1 (÷2). If the source pyramid is shallow (only
one level above the export level — common for cropped uploads or prediction zarrs), it fetches
and holds the ENTIRE half-resolution level dense: ~⅛ of the full data. For a whole scroll that is
a multi-TB download+allocation before the export even starts.
**Failure scenario:** `export-scroll` against a two-level pyramid of a large scroll stalls
downloading TBs for the "cheap" occupancy prefilter, then OOMs allocating it.
**Suggested fix:** bound the occupancy map by bytes, not by level index: skip any candidate level
whose `shape.count()` exceeds a budget (e.g. 4 GiB), falling back to "process everything".

## [medium/design] `connected_components` allocates 12 B/voxel of auxiliary dense arrays; geom's block+halo claim is unimplemented for CC/EDT/morphology
**Verdict:** unverified (medium/low)
**Location:** src/geom/connected_components.hpp:30 (`parent`), :132 (`root_label`)
**Evidence:** `std::vector<s64> parent(n)` (8 B/voxel) plus `new s32[n]` `root_label` (4 B/voxel,
written only at root slots — almost entirely wasted) plus the s32 labels output. geom/CLAUDE.md:
"workhorses on huge volumes — parallel + cache-friendly + out-of-core-aware (operate on blocks +
halo)", but CC, `edt_squared` (f32 whole-volume in+out) and `morphology` (two full copies) are all
whole-volume. CC is the single largest transient in the eval chain (see the eval finding) and in
`topo::betti_numbers` (called 3× per `official_score`).
**Failure scenario:** any CC on a 2048³ mask needs 69 GB for `parent` alone; `betti_numbers`
repeats it three times sequentially.
**Suggested fix:** cheap wins first: `u32` parent indices for volumes < 4.3e9 voxels (halves it),
replace `root_label` with a hash map or reuse `parent`'s storage for the compact ids, and free
`parent` before allocating the labels output where possible. Real fix per the module's own doc:
block-local CC + boundary union pass (the slab machinery is already there — generalize slabs to
blocks and drop the global parent array).

## [low/performance] `finalize()` holds an O(all-real-chunks) pending vector and walks the dense chunk grid; `fxinfo` repeats the dense walk
**Verdict:** unverified (medium/low)
**Location:** src/codec/archive.hpp:353 (`pending`), :360–366 (dense enumeration), src/io/io.hpp:448–455
**Evidence:** pass 1 loops `cz×cy×cx` over the FULL chunk grid per LOD (`slot_read_` each) and
accumulates `Pending{slot,off,len}` (24 B) for every REAL chunk plus a per-LOD `items` vector.
At Paris-3 dims that is ~4.3e8 grid probes and up to ~10 GB of `pending` for a dense archive —
inside the tool specifically meant for the biggest (SEALED whole-scroll) archives.
**Failure scenario:** finalizing a whole-scroll archive on a modest box adds ~10 GB of index
bookkeeping and minutes of dense grid probing (also in `fxinfo`, which triple-loops the grid per
LOD).
**Suggested fix:** stream pass 2 per Morton batch (allocate+copy blobs in fixed-size batches so
`pending` stays bounded), and/or walk the radix tree (only populated leaves) instead of the dense
chunk grid for both finalize and fxinfo coverage counts.

## [low/performance] `build_eulerian_winding_field`: serial full-grid re-gauge reduction inside every GS iteration; per-cell KdTree rasterization
**Verdict:** unverified (medium/low)
**Location:** src/winding/patch_field.hpp:235 (serial sum), :194–204 (b rasterization)
**Evidence:** after each parallel red-black sweep, the mean is computed by a *serial* triple loop
over the whole coarse grid (`for z/y/x sum += W(z,y,x)`), i.e. O(iters × cells) single-threaded —
with `iters=80` this can rival the parallel sweeps' total cost on large grids. The b-field
rasterization does one `tree.nearest` per coarse cell over the full domain even when `band>0`
would mask most of them out (mask is applied to the solve, not the rasterization). Note the
full-domain grid itself (W + bz/by/bx + mask ≈ 17 B/coarse-voxel) is inherently in-core — fine
for the in-RAM path since `stitch_stream.hpp` is the OOC answer, but the serial reduction hurts
exactly the largest in-RAM fields.
**Suggested fix:** fold the sum into the color sweeps (per-thread partials), or re-gauge every
K iterations; skip `tree.nearest` for cells outside the band mask.

## [low/hygiene] `predictions::normalize` Percentile copies the whole field and fully sorts it
**Verdict:** unverified (medium/low)
**Location:** src/predictions/field.hpp:26
**Evidence:** `std::vector<f32> v(field.flat().begin(), field.flat().end()); std::ranges::sort(v);`
— a full-volume f32 copy (+4 B/voxel) and a serial O(n log n) sort to extract two percentiles;
the output volume is also allocated before the copy, so peak is 3× the field. `ml::detail::
pct_bounds` already shows the cheap pattern (histogram percentiles, O(n), no copy).
**Suggested fix:** histogram-based percentiles (or `std::nth_element` twice on a subsample); the
module is a STUB but this is the function every prediction ingest will call.

---

### s64 / index-math audit (clean)
Checked all index arithmetic against the 2¹⁸/axis envelope: `Extent3::count()`, `VolumeView`
strides/offsets, zarr chunk math, archive Morton keys (21-bit part1by2 covers the 2¹²-per-axis
chunk grid), OccMap bin keys (bins ≤ 2²⁰/axis, products < 2⁶³), `stitch_streamed_3d` edge keys,
and VOI label packing — all s64/u64 with no int-multiply-before-widen. The `int`s that exist
(grid G ≤ 1400, patch counts, thread counts, LOD indices) are bounded by construction. No finding.

### Byte-budget / backpressure (mostly honored)
`BlockCache` is byte-budgeted+sharded (SIEVE) and the default 256 MiB is applied lazily;
`export_scroll` bounds its prefetch queue; `read_zarr_region` bounds remote fan-out. The
bypasses are the dense-decode entry points (findings above), not the substrate.
