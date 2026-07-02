# Review — unit "winding" (src/winding/: diffeo fit, SVF flow, spiral model, patch/Eulerian fields, OOC stitch)

Overall assessment: the module is in good shape for its stage of life. The RK4 discrete adjoint in
`flow.hpp` is mathematically correct (I re-derived every stage coefficient: kb4 = a·dt/6, kb3 = a·dt/3 +
sb4·dt, kb2 = a·dt/3 + sb3·dt/2, kb1 = a·dt/6 + sb2·dt/2, all match the unrolled forward), the
analytic winding backward in `diffeo_fit.hpp` matches the forward chain term-by-term (atan2 partials,
affine-transpose adjoint, Mi outer-product accumulation), the red-black Gauss–Seidel solvers are
race-free under `parallel_for_z`, and the OOC stitch alignment algebra (sign/offset via median-variance,
first-owner-wins) is consistent. The real problems are: an unguarded fixed-size trajectory buffer in the
flow adjoint (stack smash for `flow_steps > 32`), a genuine angle-wrapping defect where the fit losses
compare the branch-cut-discontinuous model winding against cut-free integer wrap labels (latent until
the P2 bridge fits real data — but `fit_bridge.hpp` already ships), an O(hole²) fill loop that
effectively hangs on holes near `max_hole`, an O(#slices) umbilicus lookup inside the fit's innermost
loop, and a chirality sign disagreement between the two documented "views of the same scalar".

## [high/bug] `flow_point_backward` overflows its fixed `traj[33]` stack buffer for `flow_steps > 32`

**Verdict:** CONFIRMED — Confirmed by reading the code. src/winding/flow.hpp:82 declares `Vec3f traj[33];` with the steps<=32 contract only in a comment, and the forward loop at flow.hpp:91 writes `traj[i+1]` for i in [0,steps) — steps>32 is an out-of-bounds stack write (UB, -fno-exceptions, no FENIX_ASSERT, no clamp). The value is unvalidated end-to-end: DiffeoFitConfig::flow_steps (diffeo_fit.hpp:44, bare int, default 8) is copied to SpiralModel at diffeo_fit.hpp:157 and drives flow_point_backward at diffeo_fit.hpp:92; SpiralModel::flow_steps (spiral_model.hpp:22) is likewise unchecked, and a repo-wide grep shows no bounds check anywhere. fit_spiral_diffeo is public API of a header-only library, so the scenario is reachable by any caller; winding/CLAUDE.md confirms .fxmodel persistence (P6) will feed this from disk. Not a documented stub — flow.hpp/diffeo_fit.hpp are the "implemented + validated" fit core. Minor partial refutation: the steps=0 sub-claim is benign — dt=sign/0 is computed but both loops skip and the Inf is never used; the function just returns its input.

**Severity adjusted to:** medium

**Fix notes:** Direction is right, two corrections: (1) FENIX_ASSERT is compiled out in -Ofast release per root CLAUDE.md §2.3, so an assert alone leaves release builds unprotected — also introduce `inline constexpr int kMaxFlowSteps = 32;`, size the buffer `traj[kMaxFlowSteps + 1]`, and clamp or reject in fit_spiral_diffeo before copying cfg.flow_steps into the model; the future .fxmodel read boundary should reject via std::expected<T, fenix::Error> (project error convention), not assert. (2) Drop the steps=0/Inf part of the fix rationale: with steps=0 both loops skip and dt is never used, so it is dead-value, not a propagating Inf (still fine to require steps>=1 in the assert). Severity downgraded to medium because today no CLI/recipe/deserializer feeds flow_steps externally — all in-repo call sites hardcode 6-10 — so it is latent until P6 persistence lands, at which point it becomes high.

**Location:** src/winding/flow.hpp:82 (used via src/winding/diffeo_fit.hpp:92, `DiffeoFitConfig::flow_steps` / `SpiralModel::flow_steps`)

**Evidence:**
```cpp
inline Vec3f flow_point_backward(const FlowField& f, Vec3f q0, int steps, f32 sign, Vec3f a_out, FlowGradAccum& ga) {
    const f32 dt = sign / static_cast<f32>(steps);
    Vec3f traj[33];  // cache the forward trajectory (steps <= 32)
    ...
    for (int i = 0; i < steps; ++i) { ... traj[i + 1] = p; }
```
The `steps <= 32` contract exists only in a comment. `DiffeoFitConfig::flow_steps` and
`SpiralModel::flow_steps` are plain unvalidated `int`s (default 8), and `flow_steps` is slated to be
persisted in `.fxmodel` (P6), i.e. it will eventually arrive from disk.

**Failure scenario:** a caller (or a future `.fxmodel` reader) sets `flow_steps = 64` for a finer
integration; `fit_spiral_diffeo` → `winding_backward` → `flow_point_backward` writes `traj[33..64]`
past the array — stack corruption / UB inside the multithreaded backward, with `-fno-exceptions` and no
assert to catch it. `steps = 0` additionally gives `dt = sign/0` (Inf, which fast-math then mangles).

**Suggested fix:** `FENIX_ASSERT(steps >= 1 && steps <= 32);` at the top of both `flow_point` and
`flow_point_backward` (or clamp in `DiffeoFitConfig`/model setter), and reject out-of-range
`flow_steps` at any future deserialization boundary. Alternatively size the buffer from `steps` via a
small stack arena / `std::array` + cap enforced by the config type.

## [high/correctness] Fit losses compare the branch-cut-discontinuous model winding to cut-free integer wrap labels — irreducible ±1 seam along θ = π

**Verdict:** CONFIRMED — Confirmed. spiral_model.hpp:55-57: winding_at = r_ideal/dr − atan2(qy,qx)/2π + offset is, along one physical sheet, a per-turn staircase jumping +1 exactly at the canonical-frame cut θ=±π; winding_offset (diffeo_fit.hpp P[7]) is a single constant gauge and cannot absorb a spatial step. The labels the fit consumes are not cut-aligned: patch_field.hpp:257 assign_windings_from_field rounds a per-cluster mean of the cut-free Eulerian θ (and the discrete segment::assign_windings path is likewise one integer per cluster), and fit_bridge.hpp:43-46 pins every subsampled cell of a patch to that one integer plus one CoWindingGroup per patch. Any wrap spanning a full turn necessarily crosses θ=±π, giving an irreducible ±1 residual on one side of the −x half-plane in both the target term and the group-variance term; the gradient seed (diffeo_fit.hpp, seed = coeff*(winding_at(p) − ref)) backprops that phantom error into dr/affine/flow. No mitigation (wrapped residual, guard band, cut split) exists in diffeo_fit.hpp/fit_bridge.hpp, and src/winding/CLAUDE.md documents no such known limitation. The masking claim is verified verbatim: tests/test_diffeo_fit.cpp:112-134 generates targets with r = dr*(w + θ/2π) (staircase boundary at the atan2 cut by construction) and samples θ only in [−π+0.25, π−0.25] (held-out ±0.6) — never touching the cut — while the real-data path (tests/test_trace_fit.cpp on paris4, roadmap P2) feeds arbitrary-boundary cluster integers into this exact loss.

**Fix notes:** The proposed directions are sound but need corrections: (1) The cut is MODEL-FRAME-DEPENDENT — θ is computed in canonical space after umbilicus shift, flow, and affine inverse, all of which change during optimization, so a one-shot split/guard-band computed in patches_to_constraints drifts as the fit rotates/translates the frame; the split or guard must be evaluated under the current model inside the loss (or refreshed periodically), not statically at bridge time. (2) Option (a) (continuous per-cell Eulerian targets, (θ_cell − mn)/dth) covers only the field-assignment path; the discrete CT-valley assign_windings path (used by test_trace_fit) has no continuous θ and still needs (b)-style handling. Continuous targets also inherit field wobble — weight them softer than snapped integers. (3) CoWindingGroups need the same treatment (split at the cut or a cut-invariant variance), which the fix mentions only for targets. (4) Do NOT substitute a wrap-tolerant residual e ← e − round(e): it would erase genuine one-wrap assignment errors that the target term exists to correct. A clean per-point alternative: inside the loss, unwrap θ consistently across each patch (choose the atan2 branch θ±2π that keeps the patch angularly contiguous under the current model) before forming W, which makes both loss terms cut-invariant without moving constraint construction into the fit.

**Location:** src/winding/spiral_model.hpp:55 (`theta = std::atan2(q.y, q.x)`), consumed by src/winding/diffeo_fit.hpp:102/248 and src/winding/fit_bridge.hpp:43

**Evidence:**
```cpp
const f32 theta = std::atan2(q.y, q.x);
const f32 shifted_radius = r_ideal - dr_per_winding * theta / two_pi;
return shifted_radius / dr_per_winding + winding_offset;
```
`winding_at` is discontinuous by exactly 1 across the canonical-frame cut θ = ±π (the −x half-plane):
along one continuous physical sheet, `W` jumps by +1 where the sheet crosses the cut. The wrap labels
the fit consumes are NOT cut-aligned: `assign_windings_from_field` rounds a per-patch/per-cluster mean
of the continuous (cut-free) Eulerian θ, and the CT-valley discrete assign has its own arbitrary gauge.
`patches_to_constraints` then pins every subsampled cell of a patch to one integer
(`out.targets.push_back({p.pos[i], static_cast<f32>(p.wrap)})`) and puts the whole patch in one
`CoWindingGroup`.

**Failure scenario:** on real data (the P2 milestone: paris4), essentially every wrap is a full turn and
crosses the model's cut once. For each such patch, the winding-target loss carries an irreducible
squared error of ~1 for all cells on one side of the cut, and the co-winding variance term penalizes a
genuine model-side jump it can never remove. The gradient (`seed = coeff*(winding_at(p) − ref)`)
pushes `dr`, the affine, and the flow lattice to "compress" a phantom one-winding gap concentrated
along the −x half-plane — a systematic seam warped into the recovered deformation and a biased
`dr_per_winding`. The existing synthetic test masks this because its targets are generated from the
model's own `winding_at` (hence cut-aligned by construction).

**Suggested fix:** make the residual cut-consistent. Options, most principled first: (a) emit
CONTINUOUS per-cell targets from the same field that produced the assignment (the Eulerian θ is
cut-free), instead of one rounded integer per patch; (b) in `patches_to_constraints`, split each patch's
targets/groups at the model cut (cells with θ within a guard band of ±π go to a sub-group with a ±1
target correction determined from θ's sign); (c) at minimum, exclude cells within an angular guard band
of the cut from both loss kinds and split `CoWindingGroup`s at the cut. Document the invariant that
wrap labels must be gauge-aligned with the model's atan2 branch.

## [medium/performance] `fill_surface_from_field` Laplacian warm-start is O(hole²) — up to ~10¹⁰ ops on a `max_hole`-sized hole

**Verdict:** unverified (medium/low)

**Location:** src/winding/patch_field.hpp:389

**Evidence:**
```cpp
// Laplacian warm-start across the hole from the valid banks.
for (int it = 0; it < static_cast<int>(comp.size()) + fp.smooth_iters; ++it)
    for (s64 ci : comp) { ... }
```
The pass count is the hole AREA (`comp.size()`, admitted up to `fp.max_hole = 200000`), not the hole
diameter, so the loop is O(|comp|²): 200k² ≈ 4·10¹⁰ neighbour-gather iterations, single-threaded,
inside the per-round cosegment refine (`cosegment_refine` calls this per sheet per round).

**Failure scenario:** one large-but-admitted dropout (e.g. a 400×500-cell river on a real paris4 wrap)
makes `cosegment_refine` appear to hang for hours; several such holes make Stage D unusable at scale.

**Suggested fix:** iterate O(diameter): run passes until max positional delta < eps with a cap of
`nu + nv` (or seed by BFS distance from the valid banks — one pass in BFS order gives a good
warm-start — then `fp.smooth_iters` smoothing passes). Also consider `parallel_for` over `comp` with
Jacobi (double-buffer) instead of in-place Gauss–Seidel.

## [medium/performance] `Umbilicus::center` is a linear scan — O(#slices) inside the fit's innermost loop

**Verdict:** unverified (medium/low)

**Location:** src/annotate/umbilicus.hpp:24-26, hot via src/winding/spiral_model.hpp:34 / src/winding/diffeo_fit.hpp:62

**Evidence:**
```cpp
usize i = 1;
while (i < z.size() && z[i] < zq) ++i;
```
`Umbilicus::estimate` emits one control point PER z-slice (`u.z.push_back(z)` for every z), so on a
real scroll the polyline has 40k–70k knots. `winding_at`/`winding_backward` each call `center(p.z)`
once, and they run per constraint per AdamW iteration (Stage 1: 500 iters × potentially millions of
bridged cells) — the fit becomes O(iters × constraints × slices), and the annotate CLAUDE.md's "tiny
data; trivial" assumption is false in this loop.

**Failure scenario:** fitting paris4 with an auto-estimated umbilicus (70k knots) and 10⁶ constraints:
each gradient step does ~7·10¹⁰ float compares in `center` alone — the "out-of-core streamed fit"
becomes umbilicus-lookup-bound by orders of magnitude.

**Suggested fix:** `std::lower_bound` on the sorted `z` (the struct already documents "sorted ascending
by z"); or resample the umbilicus to a uniform-z table and index directly. One-line fix, belongs in
`annotate/umbilicus.hpp` but is load-bearing for winding.

## [medium/correctness] `winding_init` chirality disagrees with `SpiralModel::winding_at` and the documented shifted_radius convention

**Verdict:** unverified (medium/low)

**Location:** src/winding/winding_field.hpp:32 vs src/winding/spiral_model.hpp:56

**Evidence:**
```cpp
// winding_field.hpp
wv(z, y, x) = r / p.pitch + th.value / two_pi;
// spiral_model.hpp
const f32 shifted_radius = r_ideal - dr_per_winding * theta / two_pi;
```
Root CLAUDE.md and winding/CLAUDE.md define the unifying scalar as `shifted_radius = ‖yx‖ − θ/2π·dr`.
`winding_init` uses `+θ/2π` — the opposite chirality. The module doctrine is explicitly "keep
winding-field and diffeomorphic-fit as two views of IT, not two pipelines", and `render.hpp` unrolls
with `winding_init` while the fit/model uses the minus convention.

**Failure scenario:** (1) used as the documented warm-start `seed` for `build_eulerian_winding_field`
on a scroll wound in the model's convention, the seed's angular ramp has the wrong sign — the GS solve
must first undo it, i.e. the warm start actively fights convergence. (2) A `.fxmodel`-based render and
a `winding_init`-based render of the same scroll disagree on which neighbouring wrap is "next"
(mirror-ordered wraps), a silent cross-view inconsistency exactly of the kind the CLAUDE warns about.

**Suggested fix:** flip `winding_init` to `r / pitch − θ/2π` (matching the documented convention and
`winding_at`), and re-baseline the tests that consume it (test_winding/test_unroll/test_pipeline use it
only against monotonicity/self-consistency, so the sign flip should be behaviour-preserving there).

## [medium/correctness] `winding_backward` silently assumes an identity gap-expander while the forward applies `gap.inverse`

**Verdict:** unverified (medium/low)

**Location:** src/winding/diffeo_fit.hpp:71-77

**Evidence:**
```cpp
// gap held fixed (identity for the first slice) => r_ideal = r, a_r = a_ri.
const f64 r_ideal = static_cast<f64>(m.gap.inverse(static_cast<f32>(r)));
...
const f64 a_r = gW * (1.0 / dr);
```
The forward (`winding_at` and this function's own `r_ideal`) routes through `gap.inverse(r)`, but the
adjoint treats d(r_ideal)/dr as 1. For a non-empty `gap.logits`, the true factor is
`1/exp(logits[k])` per segment, so `a_qy/a_qx` (and everything downstream: affine grads, flow scatter,
`g_ty/g_tx`) are scaled wrong per point, non-uniformly across radius.

**Failure scenario:** roadmap P3 (or any caller warm-starting from a previous fit whose model carries
gap logits) passes a `SpiralModel` with `gap.logits.size() > 0` into `fit_spiral_diffeo`. Nothing
errors; the fit runs with gradients that are wrong by up to the per-winding scale factor, converging to
a subtly wrong deformation — the classic "silently wrong data" failure the FD gradient gate can't catch
because the gate's model has an empty gap.

**Suggested fix:** `FENIX_ASSERT(m.gap.logits.empty())` (and a documented precondition) until P3
implements the chain term `a_r *= d(gap.inverse)/dr = 1/exp(logits[k])` plus the `g_dr`/logit adjoints.

## [medium/resource-safety] `stitch_streamed`/`stitch_streamed_3d` never verify the `windings.txt` write succeeded; missing fragments are silently assigned the gauge-min winding

**Verdict:** unverified (medium/low)

**Location:** src/winding/stitch_stream.hpp:173-185 and 303-315

**Evidence:**
```cpp
std::ofstream w(fs::path(dir) / "windings.txt");
if (!w) return err(Errc::io_error, "cannot write windings.txt in " + dir);
...
w << r.name << ' ' << gw << '\n';
...
return rep;   // stream state never checked after writing
```
Only the OPEN is checked. Under `-fno-exceptions`, an ofstream reports ENOSPC/quota/IO failure purely
via stream state, which is dropped here. Separately, any manifest fragment absent from `global` is
written as `gmin - gmin = 0` (line 179 / 309) — a fabricated winding, not an error.

**Failure scenario:** the OOC stitch of a full scroll (weeks of upstream tracing) hits ENOSPC halfway
through streaming `windings.txt`; the function returns a healthy `StitchStreamReport` and downstream
consumes a truncated winding table — silent corruption of the exact artifact this stage exists to
produce. Likewise a corrupt/truncated `manifest.txt` line is skipped by `read_manifest`'s
partial-parse (line 63), and the fragment quietly lands at winding 0 — absent-vs-failed conflation, the
project's cardinal sin.

**Suggested fix:** after the write loop: `w.flush(); if (!w) return err(Errc::io_error, ...)`; write to
a temp file + rename per the project's atomic-write convention. Treat a fragment with no winding as an
error (or at least count + report it in `StitchStreamReport`) instead of defaulting to 0, and make
`read_manifest` return `Expected` distinguishing "no file" from "unparseable line".

## [medium/hygiene] AdamW keeps optimizer state in f32, violating the stated f64-optimizer-state policy

**Verdict:** unverified (medium/low)

**Location:** src/core/optimize.hpp:47 (consumed by src/winding/diffeo_fit.hpp:196 and fit.hpp:76)

**Evidence:**
```cpp
std::vector<f32> m_, v_;
```
Root CLAUDE.md §2.3: "f64 only for accumulation-sensitive spots (optimizer state, large reductions)";
winding/CLAUDE.md: "f32 compute, f64 for optimizer accumulation". The fit dutifully accumulates
per-constraint gradients in f64 (`FitGrad`), then feeds an optimizer whose first/second moments are
f32. `v_` accumulates `g*g` with `beta2 = 0.999` — a long-tailed f32 EMA of squared f32 grads, exactly
the accumulation-sensitive state the policy calls out; small-gradient flow-lattice cells lose their
second-moment signal to f32 rounding over the 500-iteration Stage 1 (and more at P5 scale).

**Failure scenario:** at real scale (10⁶ constraints, small per-cell flow grads ~1e-4), `v_` updates of
`0.001·g²` ≈ 1e-11 underflow relative to the running f32 moment, freezing the effective per-parameter
LR at a stale value — slow/uneven convergence that looks like a modelling problem.

**Suggested fix:** make `m_`, `v_` (and the bias-correction `pow` chain) f64 in `AdamW` (params stay
f32); this is the module's own documented contract.

## [low/correctness] `fit_loss` ignores `r_min` while the gradient path enforces it — reported loss and optimized objective differ

**Verdict:** unverified (medium/low)

**Location:** src/winding/diffeo_fit.hpp:117 (`(void)r_min;`) vs :70 (`if (r < r_min) return;`)

**Evidence:** `winding_backward` drops any constraint with canonical radius `< r_min` (default 2), but
`fit_loss` sums all of them, and the co-winding `means` (diffeo_fit.hpp:233) include near-umbilicus
points whose gradients are then discarded — the variance term's gradient no longer matches its value.

**Failure scenario:** constraints near the umbilicus (θ ill-conditioned, W noisy) inflate
`initial_loss`/`final_loss` and shift group means; the fit "converges" but the reported final loss
plateaus at a floor the optimizer cannot see, confusing loss-based schedule decisions (the annealed
spring / coarse→fine schedule is an open ADR keyed on loss).

**Suggested fix:** apply the same `r < r_min` skip (and group-mean exclusion) in `fit_loss` and in the
`means` computation, so value and gradient describe the same objective.

## [low/bug] `fill_surface_from_field` does grid index math in `int` — overflows for surfaces beyond ~2³¹ cells

**Verdict:** unverified (medium/low)

**Location:** src/winding/patch_field.hpp:345-363

**Evidence:**
```cpp
const int G = static_cast<int>(S.nu), H = static_cast<int>(S.nv);
...
const usize id = static_cast<usize>(v * G + u);   // int multiply, then widen
```
`Surface::nu/nv` are s64 (core keeps them 64-bit for a reason: per-wrap surfaces of a full scroll reach
nu ≈ 2πr ≈ 250k, nv ≈ 70k). `v * G + u` is evaluated in `int`; for `G·H > 2³¹` (or even `v·G > 2³¹`)
this is signed overflow — UB — before the cast widens it. Same pattern in the flood fill,
component walk, warm-start, and edge checks throughout the function, and in
`field_consistency_snap`'s caller path.

**Failure scenario:** P6 generates full-scroll per-wrap `.fxsurf` surfaces; cosegment/fill on one of
them computes garbage indices (or traps under UBSan), corrupting `S.coord`/`S.valid` — precisely the
s32-on-2¹⁸-volumes trap the root conventions call out.

**Suggested fix:** keep `G/H` as s64 and do all id math in s64 (the surrounding code already uses
`std::vector<s64>` for the ids — only the arithmetic is narrow).

## [low/hygiene] The module umbrella `winding.hpp` omits half the module — diffeo_fit, patch_field, cosegment, stitch_stream, fit_bridge never reach the unity TU

**Verdict:** unverified (medium/low)

**Location:** src/winding/winding.hpp:6-12

**Evidence:** the umbrella includes only `fit/flow/relax/transforms/spiral_fit/spiral_model/
winding_field`. `diffeo_fit.hpp`, `patch_field.hpp`, `cosegment.hpp`, `stitch_stream.hpp`, and
`fit_bridge.hpp` are included by no non-test header in `src/` (verified by grep), so the single-TU
driver build and the split-build unit (`winding.cpp` → `winding.hpp`) never compile them.

**Failure scenario:** CI's `-Weverything -Werror`, clang-tidy, and include-cleaner gates run on code
the driver TU actually sees; the module's capstone (the diffeo fit) and the OOC stitch are only
compiled via test binaries, so a warning/tidy/IWYU regression in them ships past the driver-build gates.
The `winding::run` stage also cannot call its own fit without new includes.

**Suggested fix:** add the five missing includes to `winding/winding.hpp` (header-only convention says
the umbrella pulls in the whole module).
