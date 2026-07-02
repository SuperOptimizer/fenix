# Review: eval-topo-postproc — src/eval/, src/topo/, src/postproc/

Overall the unit is in good health. The recently-multithreaded metric reductions (NSD
counts in `nsd.hpp`, the VOI contingency tables in `score.hpp::voi_union`) are race-free:
per-chunk locals with a serial merge, integer-exact, correct chunk boundaries (`n*c/nchunks`
stays within s64 even at 70k×40k×40k). Degenerate inputs are handled deliberately and
correctly (both-empty ⇒ 1.0, empty-pred ⇒ NSD 0 because `geom::edt_squared` uses a finite
`edt_big = 1e18f` sentinel instead of Inf — fast-math-safe); VOI log(0) cannot occur since
only observed joint cells (c ≥ 1) are iterated and marginals ≥ joint. The Betti pipeline
(closed-cube χ, 26-conn b0, 6-conn-bg cavities, b1 by Euler-Poincaré) uses a consistent
(26,6) duality and is correct. The one real correctness bug is that **both VOI
implementations swap the split and merge components** — invisible to `total()` and to the
existing tests, but directly contradicting the module's own "VOI_merge ≫ split" invariant.
The rest is hygiene/perf: a silently-ignored `--baseline` CI gate, a duplicated
`majority_filter` with divergent border semantics, and O(n·r³) ball morphology.

## [high/correctness] VOI split and merge components are swapped (both `voi` and `voi_union`)

**Verdict:** CONFIRMED — Confirmed by direct math and code inspection. src/eval/metrics.hpp:72 adds pjoint*log(pg/pjoint) to r.merge, but Σ p(s,g)log(p(g)/p(s,g)) = H(seg,gt) − H(gt) = H(seg|gt), which per the struct's own definition (metrics.hpp:43, split = H(seg|gt)) is the SPLIT term; line 73 symmetrically puts H(gt|seg) (=merge) into r.split. The inline comments on both lines are wrong. Concrete trace: seg=single label vs gt=many labels gives ps=1, pg=pjoint ⇒ r.merge=0, r.split=H(gt)>0 — a pure cross-wrap merge reported as split, exactly inverted. The identical swapped pattern exists in voi_union (src/eval/score.hpp). Not intentional: eval/CLAUDE.md mandates asymmetric emphasis 'VOI_merge≫split', so field orientation is a module contract. Not guarded: tests/test_eval_geom.cpp:55-63 checks only total() (symmetric, unaffected), despite its comment 'merge term up'; official_score() also uses only total(). The failure is reachable by any consumer reading Voi.split/Voi.merge.

**Severity adjusted to:** medium

**Fix notes:** Proposed swap is correct for both metrics.hpp voi() and score.hpp voi_union(). Additionally: (1) fix the wrong inline comments on those lines in both files (pg term = H(seg|gt), ps term = H(gt|seg)); (2) add directional assertions to the existing test's merge case (v_merge.merge > 0.1 && v_merge.split < 1e-9, plus the transposed call asserting the reverse) and a mirrored directional test for voi_union. Severity note: currently latent (all callers use total(), which is unaffected), so medium rather than high — but it must be fixed before the documented VOI_merge≫split weighting is implemented.

**Location:** src/eval/metrics.hpp:72-73 and src/eval/score.hpp:78-79

**Evidence:**
```cpp
// metrics.hpp
r.merge += pjoint * std::log(pg / pjoint);   // H(gt|seg)   <-- actually H(seg|gt)
r.split += pjoint * std::log(ps / pjoint);   // H(seg|gt)   <-- actually H(gt|seg)
// score.hpp voi_union — identical expressions/assignments
r.merge += pjoint * std::log(pg / pjoint);  // H(gt|pred)
r.split += pjoint * std::log(ps / pjoint);  // H(pred|gt)
```
Derivation: Σ p(s,g)·log(p(g)/p(s,g)) = H(seg,gt) − H(gt) = **H(seg|gt)** — the
over-segmentation (split) term (it conditions on gt). The code adds it to `merge`.
Symmetrically, the `ps` expression is H(gt|seg) = merge, assigned to `split`.

**Failure scenario:** seg = a single label covering all foreground, gt = many labels — a
pure cross-wrap **merge**, the exact catastrophe eval/CLAUDE.md calls out. The code
reports `split = H(gt) > 0, merge = 0` (inverted). Any consumer implementing the module's
documented asymmetric weighting ("VOI_merge ≫ split") — e.g. a weighted score
`w_m·merge + w_s·split`, a merge-triggered alarm, or postproc's `connect_fragments`
being evaluated on the merge axis — punishes splits and waves catastrophic merges
through. `total()` and hence `official_score` are unaffected, which is why
`test_eval_geom.cpp` (checks `total()` only) passes.

**Suggested fix:** swap the two `+=` targets in both functions (merge takes the `ps`
term, split the `pg` term) and add a directional unit test: all-merged-seg vs blocked-gt
must give `merge > 0, split ≈ 0`, and the transpose the reverse.

## [medium/hygiene] `eval-set --baseline` is advertised as the CI regression gate but silently ignored

**Verdict:** unverified (medium/low)

**Location:** src/eval/eval.hpp:172 (option loop; usage strings at 155-159)

**Evidence:**
```cpp
// usage: fenix eval-set <manifest.toml> <split> [...] [--baseline B.json]
...
else if (a == "--json") json = true;
// --baseline reserved (regression gate); left as a TODO hook so the CLI is stable.
```
The header comment sells `--baseline` as "flags regressions ... the CI quality gate", the
usage/error string lists it, but the parse loop drops it (and its argument) on the floor
and the command exits 0.

**Failure scenario:** CI runs `fenix eval-set manifest.toml test --baseline base.json`
expecting a nonzero exit on an `official` drop. A model regression lands; eval-set prints
scores, performs no comparison, returns 0; the "overfitting firewall" green-lights the
regression indefinitely. This is exactly the "stub that silently returns success instead
of erroring" class.

**Suggested fix:** until implemented, make `--baseline` a hard error
(`err(Errc::unimplemented, "--baseline not implemented yet")`) and remove it from the
usage string; unknown options should also error rather than be ignored.

## [medium/design] Duplicate `majority_filter` with divergent border semantics (postproc re-rolls geom)

**Verdict:** unverified (medium/low)

**Location:** src/postproc/morph.hpp:20 vs src/geom/morphology.hpp:14

**Evidence:** `geom::majority_filter` counts neighbours with `av.at_clamped(...)`
(clamp-to-edge: border voxels see replicated copies of themselves), while
`postproc::majority_filter` bounds-checks and treats out-of-bounds as background
("out-of-bounds neighbours = background", morph.hpp header). Same name, same defaults
(`iters=1, thresh=14`), different results on every border voxel.

**Failure scenario:** a border fg voxel with a moderately full neighbourhood: clamped
counting inflates its count (edge replication) and keeps it; bg-OOB counting erodes it.
A recipe smoothing a mask via `geom::` and an eval/postproc path using `postproc::`
silently disagree on every face of every block — and in a block+halo out-of-core pass,
per-block borders make the divergence interior. This directly violates geom/CLAUDE.md
("do NOT let modules re-roll" primitives) and postproc/CLAUDE.md ("Morphology (via
geom)"), and is the exact taberna duplicate-primitive smell core exists to kill.

**Suggested fix:** keep ONE implementation in `geom` with an explicit border-policy
parameter (background-OOB is the right default for masks — clamped edge replication
biases the curvature flow at volume faces); make `postproc` call it; delete the copy.

## [medium/performance] `ball_dilate/erode/close/open` are O(n·r³) brute force; geom's EDT gives O(n)

**Verdict:** unverified (medium/low)

**Location:** src/postproc/morph.hpp:100-128 (`detail::ball_morph`)

**Evidence:**
```cpp
for (s64 dz = -r; dz <= r && !hit; ++dz)
  for (s64 dy = -r; dy <= r && !hit; ++dy)
    for (s64 dx = -r; dx <= r; ++dx) { if (dz*dz+dy*dy+dx*dx > r2) continue; ... }
```
Per output voxel a full (2r+1)³ scan; `ball_close`/`ball_open` do it twice.

**Failure scenario:** `ball_close(mask, 5)` on a modest 1024³ eval crop ≈ 2 × 10⁹ voxels
× ~700 in-ball probes ≈ 1.4×10¹² memory-indirect probes — hours where the exact same
result is two EDT passes: dilation = `edt_squared(mask) <= r²`, erosion =
`edt_squared(~mask) > r²` (with the module's OOB-is-background convention), each O(n)
via the Felzenszwalb-Huttenlocher EDT already in `geom/edt.hpp`. The early-out `!hit`
only helps deep inside/outside; the whole shell band (where these masks live — thin
sheets are nearly all shell) pays close to full cost.

**Suggested fix:** implement `ball_dilate/erode` as EDT thresholds on top of
`geom::edt_squared`; keep the brute-force path only for tiny r if benchmarks favour it.
`connect_fragments` deserves the same treatment eventually (per-label EDT or a two-pass
nearest-label EDT) but is at least gated to gap voxels.

## [low/performance] Serial full-volume loops in metric hot paths (dice/iou/voi, jacobian_fold_fraction, threshold/peak)

**Verdict:** unverified (medium/low)

**Location:** src/eval/metrics.hpp:17,31,53; src/eval/deformation.hpp:20-39; src/eval/eval.hpp:43,51

**Evidence:** `dice`, `iou`, and `voi` iterate `for (s64 i = 0; i < n; ++i)` on one
thread; `jacobian_fold_fraction` is a serial triple loop over the entire displacement
field (three f32 volumes); `detail::threshold`/`peak` are serial. Meanwhile the sibling
metrics (`nsd`, `voi_union`) were parallelized with the per-chunk-local pattern.

**Failure scenario:** `fenix eval` on a real crop spends its non-EDT time single-core:
`score_pair` calls the parallel `official_score` and then serial `dice` + `iou` over the
same n voxels; `jacobian_fold_fraction` — the GT-free fold guard meant to run on
whole-fit displacement fields — is the largest single-threaded loop in the module.
No wrong results, just multi-minute evals that should take seconds on a 27-CPU budget.

**Suggested fix:** reuse the `nsd.hpp` chunked-Counts reduction for dice/iou/threshold/
peak/fold-fraction (integer counters, so the merge stays exact); `voi` can adopt
`voi_union`'s per-chunk map merge.

## [low/hygiene] `jacobian_fold_fraction` never checks that dz/dy/dx dims match

**Verdict:** unverified (medium/low)

**Location:** src/eval/deformation.hpp:14-17

**Evidence:**
```cpp
inline f64 jacobian_fold_fraction(VolumeView<const f32> dz, VolumeView<const f32> dy,
                                  VolumeView<const f32> dx, f32 det_floor = ...) {
    const Extent3 d = dz.dims();
    if (d.z < 3 || d.y < 3 || d.x < 3) return 0.0;
```
Only `dz.dims()` is consulted; `dy`/`dx` are then indexed with dz's extents through the
unchecked `operator()`.

**Failure scenario:** caller passes component fields from two different LODs or a ZYX/XYZ
transposed loader (the #1 predecessor foot-gun): if `dy` is smaller, this reads out of
bounds (heap over-read, garbage det values — under fast-math no NaN backstop); if merely
transposed, it silently computes a wrong fold fraction, defeating the one GT-free
invertibility guard. Cheap to prevent.

**Suggested fix:** `FENIX_ASSERT(dy.dims() == d && dx.dims() == d);` (and ideally return
an `Expected`/0-with-error-log in release given the function is a safety metric).

## [low/performance] `score_pair` holds both full f32 volumes alive through the entire scoring pass

**Verdict:** unverified (medium/low)

**Location:** src/eval/eval.hpp:70-92 (`detail::score_pair`), 29-36 (`load_f32`)

**Evidence:** `pv`/`gv` (f32, 4 B/voxel each) stay in scope while `official_score` runs,
which itself materializes: 2 u8 masks, 2 u8 surface masks, 2 f32 EDTs (8 B/voxel), two
CC passes each allocating an s64 `parent` array (8 B/voxel) plus s32 labels, and
`betti_numbers` twice more. Peak footprint ≈ 30-40 B/voxel with 8 B/voxel of it (the f32
sources) needed only to produce the masks and `dims()`.

**Failure scenario:** evaluating a 2048³ pair (a normal validation chunk) peaks well over
300 GB; dropping the f32 volumes right after `threshold` (they contribute 64 GB there)
is free headroom. Not a correctness bug — eval/CLAUDE.md already lists windowed
evaluation as TODO — but the trivial early-release is worth taking now.

**Suggested fix:** compute `dims`/thresholds, build `pm`/`gm`, then `pv.reset()/gv.reset()`
(or scope the loads) before calling `official_score`; longer-term, add the planned
windowed/cropped evaluation.
