# Code review — unit `segment` (src/segment/: detectors, CT-valley oracle, tracer/grower, patch graph, MWS, CLI stages)

Overall the module is in good health: the grower (`grow.hpp`) is careful about NaN traps (`normalized()` guards zero, degenerate frames fall back explicitly), the red-black ARAP parallelization is genuinely race-free, the OccMap key packing fits s64 at 2^18-per-axis, and the tiled/streamed tracers share one tile body so they cannot drift. The serious problems are concentrated at two places: the out-of-core streaming path violates the project's own "fetch error must never become air" hard rule (with a comment admitting it), and the CT-valley injectivity guards ignore the documented `ct_ds` coarse-CT coordinate mapping that the data term itself honors — combining two individually-recommended knobs silently disables the wrap-crossing guard. The rest is a mix of halo off-by-a-couple at tile seams, `-fno-exceptions` CLI foot-guns, s32 linear-index truncation, and two performance landmines (serial Hessian pass, O(N²) finite-difference tracer).

## [high/silent-data-corruption] Streamed tracer turns a zarr fetch failure into an all-zero (air) tile

**Verdict:** CONFIRMED — src/segment/trace_stream.hpp:28-30 zero-initializes Volume<u8> out(pe) and does `if (!r) return out;`, converting a hard read_zarr_region failure into an all-air tile. Both callers — trace_volume_streamed (line 60/62) and trace_volume_streamed_to_disk (line 105/107) — consume it unconditionally, so a tile whose fetch hard-fails is silently traced as empty and its wraps are dropped. read_zarr_region (src/io/zarr.hpp:117+) already distinguishes missing-chunk=fill from hard failure (failed flag → Expected error, after the s3 layer's retry/backoff), and it supports remote s3://http roots (fetch-thread pool, zarr.hpp:152-159), so the transient-network/corrupt-chunk scenario is reachable. This directly violates root CLAUDE.md §2.4 ("a transient fetch error must never silently become air") and src/io/CLAUDE.md invariants ("Absent ≠ fetch-failed ... never silent air"). The in-code comment (lines 25-26) admits the gap and defers to "production callers", but the production callers in the same file do not thread the Expected — an acknowledged violation is not an intentional invariant or a documented stub; segment/CLAUDE.md presents these functions as THE out-of-core tracer path. Only-tests-call-it-today does not refute reachability.

**Fix notes:** Fix direction is correct with one correction: retry+backoff does NOT need adding to read_zarr_region — it already exists in the s3 fetch layer (src/io/s3.hpp: exponential-backoff retry + stall watchdog, per io/CLAUDE.md); the only change needed is to stop swallowing the error. Concretely: stream_tile_u8 → Expected<Volume<u8>> (return std::unexpected(r.error())); trace_volume_streamed → Expected<VolumeResult>; trace_volume_streamed_to_disk propagates through its existing Expected<StreamToDiskStats>. Update the now-false comment at trace_stream.hpp:24-26 and segment/CLAUDE.md, and adjust callers in tests/test_trace_stream.cpp (lines ~128, 152, 217, 224, 260, 298).

**Location:** src/segment/trace_stream.hpp:27-38 (used by `trace_volume_streamed` :60-62 and `trace_volume_streamed_to_disk` :105-107)

**Evidence:**
```cpp
// Missing chunks read as the zarr fill (air) inside read_zarr_region; a hard fetch error yields zeros
// here — production callers that must honour "absent != fetch-failed" should thread the Expected through.
inline Volume<u8> stream_tile_u8(const std::string& root, Index3 porg, Extent3 pe, f32 scale) {
    Volume<u8> out(pe);
    auto r = io::read_zarr_region(root, porg, pe);
    if (!r) return out;
```
**Failure scenario:** A transient S3/HTTP error (or corrupt chunk) while streaming a multi-TB scroll makes `read_zarr_region` return an error; `stream_tile_u8` swallows it and returns a zero-filled block. `trace_one_tile` then sees no prediction and no CT in that tile, seeds nothing, and the tile's wraps are simply missing from the output — no error, no log. Root CLAUDE.md §2.4 makes this a hard rule: "A transient fetch error must never silently become air." Note `trace_volume_streamed_to_disk` already returns `Expected<...>` — the plumbing exists but the error is dropped one level down. The self-aware comment does not make this acceptable in the function two production entry points actually call.

**Suggested fix:** Make `stream_tile_u8` return `Expected<Volume<u8>>` and propagate in both streamed tracers (`trace_volume_streamed` should also become `Expected<VolumeResult>`); retry+backoff belongs in `read_zarr_region` per the io contract, but a hard failure must abort the trace.

## [high/correctness] CT-valley guards sample the CT view with prediction-space coords, ignoring `ct_ds`

**Verdict:** CONFIRMED — The code matches the claim exactly. DataField::value() (src/segment/grow.hpp:141-149) maps prediction-space coords into the coarse CT grid via cp=(p+0.5)/ct_ds-0.5 when ct_ds!=1, but both crosses_valley call sites bypass that mapping: the ARAP data-snap guard passes raw vertex coords P,q (grow.hpp:378: `crosses_valley(fld.ct, P, q, p.ct_barrier, 0.5f)`) and the growth barrier passes raw S.at(uu,vv)/place (grow.hpp:960). crosses_valley (src/segment/ct_valley.hpp:66-86) does trilinear sampling of the view it is handed with the coords as-is — it has no ct_ds parameter. So with a coarse CT (ct_ds=2), every barrier probe reads the CT at 2x the intended position, clamping at the border for the upper 7/8 of the volume, silently neutering or misdirecting the guard. Reachability: no in-repo caller currently combines ct_ds>1 with ct_barrier>0 (grep: test_grow.cpp:178 sets ct_ds=2 with ct_barrier=0; test_trace_fit.cpp:62 sets ct_barrier=0.12 with ct_ds=1; GrowParams is not wired to any CLI tool yet), so it is latent today — but both knobs are public GrowParams members and the module CLAUDE.md explicitly recommends them as the paired production configuration (the coarse CT term "The grower samples it via DataField::ct_ds / GrowParams::ct_ds" plus "the tracer's GrowParams::ct_barrier ... lifts paris4 to ~0..13"). The documented intended config triggers the bug with no warning; that is a reachable failure, not an unreachable/guarded one, so the finding stands.

**Severity adjusted to:** medium

**Fix notes:** The ct_coord helper is the right fix: add DataField::ct_coord(Vec3f) implementing the existing (p+0.5)/ct_ds-0.5 mapping (refactor value() to use it) and map BOTH endpoints at both call sites (grow.hpp:378 and :960). Two corrections to the proposal: (1) sample_step does NOT need scaling by 1/ct_ds once the coords are mapped — the segment length L shrinks by ct_ds automatically, so 0.5 is then 0.5 coarse voxels, which matches the coarse grid's actual resolution (scaling it as proposed would oversample a grid that has no sub-voxel detail); scaling sample_step is only needed in the alternative where coords stay full-res, which is not the proposed helper route. (2) Note that ct_sheetness_coarse mean-pools + smooths, so valley prominence on the coarse grid is shallower than full-res — ct_barrier~0.12 was tuned at ct_ds=1 and will likely need re-tuning (lowering) for ds=2; worth a comment or a test combining the knobs (the finding correctly notes no test exercises the pair). The interim FENIX_ASSERT(ct_barrier<=0 || ct_ds==1) is a reasonable stopgap but per project convention it compiles out in release; a runtime expected-error or the real fix is preferable. Downgraded to medium only because no current in-repo caller or CLI reaches the combination; the defect itself is real.

**Location:** src/segment/grow.hpp:378 (ARAP data-snap guard) and :960 (growth barrier)

**Evidence:**
```cpp
const bool cross = p.ct_barrier > 0.0f && fld.has_ct() && segment::crosses_valley(fld.ct, P, q, p.ct_barrier, 0.5f);
...
if (V(uu, vv)) jumped = segment::crosses_valley(fld.ct, S.at(uu, vv), place, p.ct_barrier, 0.5f);
```
`DataField::value()` (grow.hpp:141-149) carefully maps prediction coords into the coarse CT grid when `ct_ds > 1` (`cp = (p+0.5)/ct_ds - 0.5`), but both `crosses_valley` call sites pass full-resolution `P`/`place` straight into `fld.ct`.

**Failure scenario:** The module's own CLAUDE.md recommends both knobs: the coarse CT term (`ct_sheetness_coarse` ds=2 + `GrowParams::ct_ds=2`, used in test_grow.cpp:174-180) and `ct_barrier≈0.12` ("the touch-proof winding fix", used in test_trace_fit.cpp:62). Set them together on a 512³ pred / 256³ CT: every valley probe with a coordinate above 256 clamps at the CT border (flat profile → no saddle ever detected → the barrier is silently OFF for 7/8 of the volume), and everywhere else the profile is read at 2× the intended physical position. The wrap-fusing over-merge this guard exists to prevent (paris4: two whole wraps collapsing) comes back with no warning. Today's tests only ever use one knob at a time, which is why this is latent.

**Suggested fix:** Add a `DataField::ct_coord(Vec3f)` helper (the existing `ct_ds` mapping) and route both `crosses_valley` call sites — and any future `count_air_valleys` use of `fld.ct` — through it; also scale `sample_step` by `1/ct_ds`. Alternatively `FENIX_ASSERT(p.ct_barrier <= 0 || p.ct_ds == 1)` until fixed.

## [medium/correctness] `structure_tensor_sheetness` halo can undershoot the true stencil radius → reflect-boundary contamination at interior tile seams

**Verdict:** unverified (medium/low)

**Location:** src/segment/structure_tensor.hpp:40-41

**Evidence:**
```cpp
const f32 sig_sum = p.sigma_grad + p.sigma_tensor + std::max(0.0f, unsharp_sigma);
const s64 halo = std::max<s64>(4, static_cast<s64>(std::ceil(3.0f * sig_sum)) + 1);
```
The actual dependency radius of an interior output voxel is `ceil(3σ_unsharp) + ceil(3σ_grad) + 1 (gradient_at central difference) + ceil(3σ_tensor)` — `gaussian_blur` (core/filter.hpp:18) uses `r = ceil(3σ)` per pass. `ceil(3(a+b+c)) + 1` can be up to 2 less than `ceil(3a)+ceil(3b)+ceil(3c)+1`.

**Failure scenario:** σ_grad=1.4, σ_tensor=2.4, unsharp_sigma=1.4 (all plausible config values; sigmas are "config-only, tune per dataset"): required halo = 5+5+1+8 = 19, computed halo = ceil(15.6)+1 = 17. Because `gaussian_blur` uses REFLECT boundary handling at the padded-tile edge, the two-voxel shortfall doesn't merely truncate the Gaussian tail — it mirrors halo data into the stencil, so interior voxels within ~2 voxels of every 256³ tile seam get biased tensor components → visible sheetness seams that the whole-volume version doesn't have (breaking the stated "interior results match up to tail truncation" contract). With the current defaults (integral sigmas) it happens to be exact, which is why tests pass.

**Suggested fix:** `halo = max(4, ceil(3*unsharp_sigma) + ceil(3*sigma_grad) + 1 + ceil(3*sigma_tensor))`, i.e. sum the per-stage radii instead of taking the radius of the summed sigmas.

## [medium/crash] `trace-surface`/`render-sheet` CLI stages: `std::stof/stoi/stoll` throw under `-fno-exceptions`; unchecked `sscanf` reads uninitialized floats

**Verdict:** unverified (medium/low)

**Location:** src/segment/trace_surface.hpp:84-86, 209-210 (stof/stoi/stoll), :216 (sscanf)

**Evidence:**
```cpp
const f32 iso = std::stof(opt("iso", "0.5"));
const int step = std::max(1, std::stoi(opt("step", "1")));
...
f32 z, y, x; std::sscanf(ss.c_str(), "%f,%f,%f", &z, &y, &x); seed = {z, y, x};
```
**Failure scenario:** `fenix trace-surface pred.nrrd out.ply iso=0.5x` — `std::stof` (defined in the libc++ dylib, so it compiles fine under `-fno-exceptions`) throws `std::invalid_argument`; with no handler possible in this binary the process aborts via `std::terminate` instead of returning `Expected`. Root CLAUDE.md: any throw path is a bug. Separately, `seed=100,200` (missing x) leaves `x` uninitialized and `sscanf`'s return value is ignored → the walk starts from an indeterminate coordinate (UB, reads of an uninitialized f32).

**Suggested fix:** Use the first-party arg parsing (`std::from_chars` wrapped in `Expected`, as the conventions require) for all numeric options, and check `sscanf(...) == 3` (or parse seed with from_chars too), returning `err(Errc::invalid_argument, ...)` on failure.

## [medium/bug] `ct_sheetness_coarse` reads out of bounds when a dimension is smaller than `ds`; `compute_normal_field` produces a zero-extent volume

**Verdict:** unverified (medium/low)

**Location:** src/segment/structure_tensor.hpp:124,138; src/segment/grow.hpp:90

**Evidence:**
```cpp
const Extent3 dd{std::max<s64>(1, d.z / ds), ...};   // dd.z clamped UP to 1
...
acc += static_cast<f32>(cv0(z * ds + dz, y * ds + dy, x * ds + dx));  // dz in [0,ds)
```
and in `compute_normal_field`: `const Extent3 dd{d.z / ds, d.y / ds, d.x / ds};` with no clamp at all.

**Failure scenario:** `ct_sheetness_coarse` on a thin slab (e.g. d.z=1, ds=2 — a boundary crop or a 2D-ish test volume): `dd.z` is clamped to 1, so the pooling loop reads `cv0(1, y, x)` — one full plane past the end of the allocation (heap over-read; garbage densities in the air gate). `compute_normal_field` with `d.z < ds` instead builds a `Volume<f32>` with extent 0 and feeds it to `structure_tensor`/`NormalField::at`, whose trilinear border path clamps to `d-1 = -1` (out-of-bounds). The tilers guard `pe >= 16`, but `ct_sheetness_coarse`/`compute_normal_field` are public API called directly (test_grow, tools).

**Suggested fix:** Clamp the pooling window to the source extent (`for dz < min(ds, d.z - z*ds)` with the divisor adjusted), or reject inputs with any dim < ds via `Expected`/`FENIX_ASSERT`; give `compute_normal_field` the same `max(1,...)` + guard.

## [medium/correctness] `build_signed_affinity` stores voxel linear indices as s32 — overflows above 2³¹ voxels

**Verdict:** unverified (medium/low)

**Location:** src/segment/affinity.hpp:25-26, 41-44

**Evidence:**
```cpp
std::vector<s32> node_of;   // voxel linear index -> node id, or -1 (background)
std::vector<s32> voxel_of;  // node id -> voxel linear index
...
if (g.node_of[i] >= 0) g.voxel_of[static_cast<usize>(g.node_of[i])] = static_cast<s32>(i);
```
**Failure scenario:** Any foreground volume larger than 2³¹ voxels (≈1291³ — small next to the 2¹⁸-per-axis design point and even next to a single 2048³ working block): `static_cast<s32>(i)` wraps negative/garbage, so `voxel_of` maps nodes back to the wrong (or negative) voxels, silently corrupting every downstream label write from `mws_partition`. Node ids in `SignedEdge` are `u32` with the same ceiling. This directly violates the conventions' "s64 required for linear indices" rule.

**Suggested fix:** Make `voxel_of` (and node ids if per-wrap instance segmentation is ever run above ~4G foreground voxels) `s64`; at minimum `FENIX_ASSERT(d.count() <= std::numeric_limits<s32>::max())` at entry so it fails loudly, and note the supervoxel (SNIC) coarsening as the intended path at scale.

## [medium/performance] `hessian_sheet`: the per-voxel eigen pass is fully serial, and the pass materializes 4 whole-volume side buffers per scale

**Verdict:** unverified (medium/low)

**Location:** src/segment/hessian.hpp:37-68

**Evidence:**
```cpp
std::vector<f32> ra(n, 0), ss(n, 0);
std::vector<u8> bright(n, 0);
std::vector<Vec3f> nrm(n);
f32 max_s = tol::eps;
for (s64 z = 1; z < d.z - 1; ++z)        // <-- plain serial triple loop
    for (s64 y = 1; y < d.y - 1; ++y)
        for (s64 x = 1; x < d.x - 1; ++x) {
            ... sym_eig3<f32>(...); std::sort(ord...);
```
**Failure scenario:** A `sym_eig3` (50-sweep Jacobi worst case) plus a `std::sort` per voxel on ONE core: at 512³ this is minutes per sigma while every sibling detector uses `parallel_for_z`; the module CLAUDE explicitly says "per-voxel, embarrassingly parallel (OpenMP)". Memory: per sigma the pass holds work(4B) + ra(4) + ss(4) + bright(1) + nrm(12) + output sheet(4) + normal(12) ≈ 41 B/voxel → ~44 GB at 1024³, the exact "whole-volume multi-buffer pass = OOM bug at scale" the module CLAUDE calls out (structure tensor got its tiled variant; Hessian did not).

**Suggested fix:** Parallelize with `parallel_for_z` (the only cross-voxel coupling is `max_s` — reduce per-slice maxima). Fold the `Ra/S/bright/nrm` buffers away by doing a two-pass max_s (or a per-tile max) and computing plateness in the same loop; longer term give it the `structure_tensor_sheetness`-style tiled form.

## [medium/performance] `trace_patch` finite-difference gradient is O(N²) per iteration — full loss re-evaluated for each of 3N parameters

**Verdict:** unverified (medium/low)

**Location:** src/segment/tracer.hpp:89-99

**Evidence:**
```cpp
for (usize i = 0; i < params.size(); ++i) {
    params[i] = o + h;
    const f32 lp = loss(params);   // loss() is a full O(N) sweep over the grid
    params[i] = o - h;
    const f32 lm = loss(params);
```
**Failure scenario:** The loss is a sum of strictly local terms (data + 4-neighbour Laplacian + 2 edge terms), yet each of the 3N central differences re-evaluates all N cells: cost per AdamW step is 6N·O(N) = O(N²), ×200 iterations. A modest 100×100 patch (N=10⁴): ~6·10⁸ full-cell evaluations ≈ 10¹² trilinear samples — hours for one patch, versus milliseconds with the analytic (or locally-scoped FD) gradient. Also `h = 1e-3f` against coordinates of magnitude ~10³ voxels loses the perturbation entirely in f32 (`1024 + 0.001` rounds to 1024 under fast-math f32), so on real volumes the gradient of the data term is exactly 0 and the fit silently does nothing.

**Suggested fix:** Differentiate analytically (data term via `sample_trilinear_grad`, Laplacian/edge terms are closed-form) or at minimum restrict the FD loss re-evaluation to the ≤5 cells whose terms touch parameter i and use a relative step `h = max(1e-3, |o|*1e-4)`.

## [low/correctness] `make_patch` winding-angle stats break across the atan2 ±π branch cut

**Verdict:** unverified (medium/low)

**Location:** src/segment/patch_graph.hpp:137, 144-146

**Evidence:**
```cpp
const f32 turn = std::atan2(p.y - c.y, p.x - c.x) / two_pi;   // in [-0.5, 0.5]
...
ang_sum += static_cast<f64>(turn);
amin = std::min(amin, turn); amax = std::max(amax, turn);
```
**Failure scenario:** A patch whose angular footprint straddles the −y axis (turn jumps −0.5→+0.5) gets `ang_lo≈−0.5`, `ang_hi≈+0.5` (a full-turn extent for a 10° patch) and `ang_mean≈0` — pointing at the opposite side of the scroll. These fields are documented as "thaumato's f_init" for the winding fit; any consumer seeding from `ang_mean` mis-initializes exactly the wraps that cross the cut (a fixed azimuth, so a consistent stripe of the scroll).

**Suggested fix:** Accumulate the mean via `atan2(Σsin, Σcos)` and compute the extent by unwrapping turns relative to the first cell's angle (or store cos/sin sums and derive lo/hi from the unwrapped values).

## [low/bug] `binkey` negative bin coordinates alias across axis fields of the packed key near the volume origin

**Verdict:** unverified (medium/low)

**Location:** src/segment/grow.hpp:854-857 (same scheme at :1067, :1097; surface_quality :779)

**Evidence:**
```cpp
auto binkey = [&](Vec3f c, int oz, int oy, int ox) {
    return (static_cast<s64>(c.z * ibin) + oz) * BS * BS + (static_cast<s64>(c.y * ibin) + oy) * BS +
           (static_cast<s64>(c.x * ibin) + ox);
};
```
**Failure scenario:** `fold_conflict` probes the 26-neighbourhood with offsets −1. For a point in bin x=0 (coordinate < bin_size, reachable since the margin `mgn` can be smaller than `bin_size` claims near clamped tile edges — and y/z bin 0 likewise), `x_bin−1 = −1` borrows into the y field: the probed key equals `(z, y−1, x_max)` — a different, far-away bin. Effect: occasional spurious fold-conflicts (rejected growth) or missed conflicts along the low-coordinate faces of every tile. Rare and self-limited, but it makes the injectivity guard behave differently at tile origin faces than elsewhere.

**Suggested fix:** Bias bins by a constant (+1 or +BS/2) before packing, or early-continue when any offset bin coordinate is negative.

## [low/performance] `mws_partition` mutex lists are never deduplicated and `blocked()` re-finds every stale entry — quadratic on volume-scale signed graphs

**Verdict:** unverified (medium/low)

**Location:** src/segment/partition.hpp:34-59

**Evidence:**
```cpp
auto blocked = [&](s32 ra, s32 rb) {
    for (s32 v : mx[static_cast<usize>(ra)]) if (find(v) == rb) return true;
    ...
mb.insert(mb.end(), ms.begin(), ms.end());   // concat, no dedup, no path-compressed rewrite
```
**Failure scenario:** On a voxel-level RAG (the intended input: every foreground voxel a node, a repulsive edge per across-normal contact) the roots of touching wraps accumulate mutex lists proportional to their contact area; each subsequent attractive edge between big clusters scans both full lists (with a `find` per element). Wrap-scale contact areas of 10⁵–10⁶ entries make the edge loop effectively O(E·M) — hours where the reference Mutex Watershed is near linear-log.

**Suggested fix:** Keep mutexes as a hash set keyed on the canonicalized root pair (rehome/dedup entries on merge, small-into-big), the standard MWS implementation trick; or defer to the planned supervoxel coarsening and assert a node-count ceiling meanwhile.
