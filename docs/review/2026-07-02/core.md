# Code review — unit "core" (src/core/, all headers)

Overall assessment: src/core/ is in good shape for a substrate — the Volume/VolumeView
index math is genuinely 64-bit throughout, ZYX ordering is consistent (I verified the
gaussian_blur axis permutations, sampling stencils, and Surface layout against the
convention), the stage registry uses function-local statics (safe static-init order,
idempotent re-registration for the split build), the thread pools have no
shutdown/lost-wakeup hazards (parallel_for_io is join-before-return with a simple
relaxed fetch_add cursor, which is correct), and the cgroup budget parsing handles the
"max" and v1 cases. The real problems found are: one out-of-bounds heap access in the
shared Gaussian blur for small-dimension lines, a cross product that silently computes
the reverse-order (left-handed) cross in the declared right-handed frame, a
convergence threshold in the one-and-only eigensolver that never fires in f32 (so the
per-voxel hot path always runs 50 full Jacobi sweeps), and several allocation/UB edges
that conflict with the -fno-exceptions error model.

## [high/bug] gaussian_blur reads out of bounds when the kernel radius reaches the line length (single-application reflect)

**Verdict:** CONFIRMED — Confirmed by direct inspection. src/core/filter.hpp:29 reflects only once, so it maps only [-(n-1), 2n-2] into range; the len<ksz branch at filter.hpp:67-73 generates sample indices t+j in [-r, len-1+r], and when r > len-1 the reflected index is still out of range (reflect(-r,len)=r>=len; reflect(len-1+r,len)=len-1-r<0), reading past both ends of the heap `line` buffer allocated at filter.hpp:42. Reachable: gaussian_blur has no min-dimension guard, and structure_tensor_sheetness (src/segment/structure_tensor.hpp:85) calls it with sigma_tensor=2.0 (r=6) on padded tiles whose dims equal the volume dims for thin volumes (e.g. a 4-slice z-crop gives pd.z=4 < 6); preprocess/deconv.hpp:27 and segment/hessian.hpp:32 likewise pass arbitrary sigma on arbitrary volumes. No guard exists anywhere upstream. The len>=ksz paths are indeed safe (single reflection suffices when len >= 2r+1).

**Fix notes:** The iterated-reflect fix is correct and preserves kernel normalization; the n==1 early-out is genuinely required (with n=1, i=1 maps 1 -> -1 -> 1 and loops forever). The alternative fix "clamp the effective kernel radius to len-1" is subtly wrong as stated: truncating the tap range while keeping the full normalized kernel drops weight mass, biasing output low on small lines — it would need the truncated weights renormalized per line. Recommend the while-loop reflection (or equivalently a periodic-fold closed form) as the single fix.

**Location:** src/core/filter.hpp:29 (reflect), used at :62 and :70 (the `len < ksz` branch)

**Evidence:**
```cpp
auto reflect = [](s64 i, s64 n) { return i < 0 ? -i : (i >= n ? 2 * (n - 1) - i : i); };
...
} else {                                   // len < ksz
    for (s64 t = 0; t < len; ++t) {
        f32 acc = 0;
        for (s64 j = -r; j <= r; ++j) acc += kp[j + r] * lp[reflect(t + j, len)];
```

**Failure scenario:** `reflect` applies the reflection exactly once, so it only maps
indices in `[-(n-1), 2n-2]` back into range. In the `len < ksz` branch the index runs
over `t + j ∈ [-r, len-1+r]`; whenever `r > len - 1` (i.e. an axis dimension
`≤ ceil(3σ)`), `reflect(-r, len)` returns `r ≥ len` and `reflect(len-1+r, len)`
returns `len-1-r < 0` — both index off the ends of the heap `line` buffer. Concrete:
`gaussian_blur(v, 2.0f)` (r = 6) on any volume with a dimension ≤ 6 — e.g. a thin
z-crop, a remainder tile, or `preprocess/deconv.hpp:27` / `segment/structure_tensor.hpp:85`
on a slab — reads `lp[-3]`/`lp[9]` on a 4-element vector: garbage filter output in
release, ASan crash in CI. The `len >= ksz` paths are safe (there `r ≤ (len-1)/2`);
only the small-line branch is broken.

**Suggested fix:** loop the reflection until in-range (`while (i < 0 || i >= n) { i = i < 0 ? -i : 2*(n-1)-i; }`,
with an `n == 1` early-out), or clamp the effective radius to `len - 1` in the small-line branch.

**Outcome:** fixed — `reflect()` in `src/core/filter.hpp` now iterates until in-range (per fix_notes,
not the truncate-and-renormalize alternative) with an `n<=1` early-out returning 0. Added
`tests/test_filter.cpp` (5 cases) covering dims of 1/2/4 at sigma=2.0, mass-preservation on a
small-line non-constant field, and a normal-size sanity check. Verified with ASan against the
pre-fix `reflect` that the bug is a real heap-buffer-overflow at the old `filter.hpp:70`, and that
the new test both catches it (against the buggy code) and passes (against the fix).

## [medium/correctness] cross(a, b) computes b×a — the frame is declared right-handed but the cross is left-handed

**Verdict:** unverified (medium/low)

**Location:** src/core/vec.hpp:36-39

**Evidence:**
```cpp
template <class T>
constexpr Vec3<T> cross(Vec3<T> a, Vec3<T> b) {
    // ZYX cross product (right-handed).
    return {a.y * b.x - a.x * b.y, a.x * b.z - a.z * b.x, a.z * b.y - a.y * b.x};
}
```

**Failure scenario:** with `a = ẑ = {1,0,0}` and `b = ŷ = {0,1,0}` this returns
`{0,0,1} = x̂`, i.e. `z × y = +x`. In the right-handed frame that docs/conventions.md
line 11 mandates, `ẑ × ŷ = −x̂`. Component-wise the correct storage-order formula is
`r.z = a.x*b.y − a.y*b.x` etc.; the implementation is its exact negation, so
`cross(a,b) ≡ b × a`. Every caller today (`segment/grow.hpp:666,739`,
`render/surface_render.hpp:33`, `eval/mesh_quality.hpp:132`, ...) is internally
consistent so nothing visibly breaks yet — but conventions.md pins physical meaning to
the normal's sign ("recto = +", "normals point toward the umbilicus"), and the first
person who implements a formula from a paper (or exports MC33/OBJ normals to an
external viewer, or hand-derives a winding sign) using the mathematical right-hand
rule gets a silent global orientation flip. This is precisely the predecessor
normal-orientation bug class this header exists to kill.

**Suggested fix:** negate all three components so `cross` matches the right-hand rule
in the declared frame (`{a.x*b.y - a.y*b.x, a.z*b.x - a.x*b.z, a.y*b.z - a.z*b.y}`),
audit the ~13 call sites for sign-sensitivity (most are `normalized(cross(...))` used
consistently and just flip together), and add a TEST asserting `cross(ẑ, ŷ) == −x̂`.

## [medium/performance] sym_eig3 convergence test is absolute 1e-20 — never fires in f32, so every per-voxel eigensolve runs all 50 sweeps

**Verdict:** unverified (medium/low)

**Location:** src/core/eig.hpp:30,33

**Evidence:**
```cpp
for (int sweep = 0; sweep < 50; ++sweep) {
    T off = std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]);
    if (off < T(1e-20)) break;
    ...
        if (std::abs(a[p][q]) < T(1e-20)) continue;
```

**Failure scenario:** structure-tensor entries on papyrus voxels have magnitudes
~1e2–1e8; after ~3 sweeps the off-diagonals bottom out at roundoff, ≈ eps·‖A‖ ≈
1e-4…1e0 for f32 — never below the absolute 1e-20 threshold, and the per-rotation skip
never triggers either. So `sym_eig3<f32>` (called per voxel in
`segment/structure_tensor.hpp:99,224` and `segment/hessian.hpp:53` — the whole-scroll
hot path) always executes 50 full sweeps (150 rotations, 300 sqrts) where ~3 sweeps
suffice: roughly 10–15× wasted work in the eigensolve. (Air voxels with exact-zero
tensors do break early, so benchmarks on sparse test data understate the cost.)

**Suggested fix:** make the threshold relative to the matrix scale, e.g.
`const T scale = |azz|+|ayy|+|axx|+...;` and break on
`off <= scale * (sizeof(T)==4 ? 1e-6 : 1e-13)` (same for the per-element skip), or cap
sweeps at ~8 — Jacobi on 3×3 converges quadratically.

## [medium/resource-safety] Volume/Arena allocate via throwing operator new under -fno-exceptions — OOM terminates instead of returning Expected; Volume dims unvalidated

**Verdict:** unverified (medium/low)

**Location:** src/core/volume.hpp:89-91 (also :96, :103), src/core/arena.hpp:19

**Evidence:**
```cpp
explicit Volume(Extent3 dims)
    : dims_(dims),
      storage_(std::make_unique_for_overwrite<T[]>(static_cast<usize>(dims.count()))) {}
...
Arena(usize capacity_bytes)
    : cap_(capacity_bytes),
      base_(static_cast<u8*>(::operator new(capacity_bytes, std::align_val_t(64)))) {}
```

**Failure scenario:** the project error model is `Expected<T, Error>` with
`-fno-exceptions`; §2.4 mandates "byte-budgeted RAM with backpressure". `Volume<T>` is
exactly where multi-GB allocations happen (a 2048³ f32 block is 32 GiB), and on
allocation failure libc++'s `operator new` throws `bad_alloc`, which cannot unwind
through -fno-exceptions frames → `std::terminate`, aborting a multi-hour out-of-core
run with no error, no retry, no eviction. Additionally `Volume(Extent3{-1, 10, 10})`
computes `count() = -100`, and `static_cast<usize>` turns it into ~1.8e19 → guaranteed
alloc failure → same abort path, with no FENIX_ASSERT on non-negative dims.

**Suggested fix:** allocate with `new (std::nothrow)` /
`::operator new(n, align, std::nothrow)` and either (a) add a
`static Expected<Volume> create(Extent3)` factory for fallible large allocations and
make the raw ctor FENIX_ASSERT dims ≥ 0, or (b) at minimum fail with a logged fatal
message naming the size. Same treatment for Arena.

## [low/bug] Arena::alloc_n size-overflow and alloc offset-overflow allow a too-small block to be handed out

**Verdict:** unverified (medium/low)

**Location:** src/core/arena.hpp:28-29, :38

**Evidence:**
```cpp
usize p = (offset_ + (align - 1)) & ~(align - 1);
if (p + bytes > cap_) return nullptr;
...
return static_cast<T*>(alloc(n * sizeof(T), ...));
```

**Failure scenario:** `alloc_n<f32>(n)` with attacker/corruption-derived `n` (e.g. a
count read from an untrusted container) computes `n * sizeof(T)` in usize — for
`n = 2^62+1`, this wraps to 4 bytes, `alloc` succeeds, and the caller writes `n`
elements past the arena → heap corruption instead of the nullptr the API promises.
`p + bytes` can likewise wrap when `bytes` is near SIZE_MAX, defeating the capacity
check. `align` is also assumed to be a power of two with no assert.

**Suggested fix:** in `alloc_n`, reject `n > SIZE_MAX / sizeof(T)`; in `alloc`, check
`bytes > cap_ - p` (subtraction form, no overflow) after checking `p <= cap_`, and
`FENIX_ASSERT(std::has_single_bit(align))`.

## [low/bug] Pcg32::bounded(0) is modulo-by-zero UB

**Verdict:** unverified (medium/low)

**Location:** src/core/rng.hpp:39

**Evidence:**
```cpp
constexpr u32 bounded(u32 bound) {
    u32 threshold = (~bound + 1u) % bound;
```

**Failure scenario:** `bound == 0` → integer division by zero (UB; SIGFPE on x86). Any
caller computing a bound from data (e.g. `r.bounded(candidates.size())` on an empty
candidate set — the natural pattern for the material-rich box draws this class exists
for) crashes or worse. Current call sites pass constants (`ml/augment.hpp:262`), but
the API is a landmine.

**Suggested fix:** `FENIX_ASSERT(bound > 0)` plus a defined release behavior
(`if (bound == 0) return 0;`).

## [low/correctness] Config::get_int round-trips through f64 — out-of-range cast is UB and large s64 values silently lose precision

**Verdict:** unverified (medium/low)

**Location:** src/core/config.hpp:64-67

**Evidence:**
```cpp
[[nodiscard]] std::optional<s64> get_int(const std::string& key) const {
    auto v = get_float(key);
    return v ? std::optional<s64>(static_cast<s64>(*v)) : std::nullopt;
}
```

**Failure scenario:** config/recipe files are fuzz surface. `key = 1e300` parses as a
valid f64, and `static_cast<s64>` of a value outside `[INT64_MIN, INT64_MAX]` is UB
(UBSan trap; unspecified garbage in release) instead of a decode error. Separately, an
integer key above 2^53 (e.g. a byte budget or a 64-bit id) is silently rounded, and a
fractional value (`threads = 3.9`) silently truncates to 3 rather than erroring.

**Suggested fix:** parse integers directly with `std::from_chars<s64>` on the raw
token; return nullopt (or better, distinguish "absent" from "malformed") when the
token isn't an exact integer in range.

## [low/concurrency] Context::cancelled is a plain bool — cross-thread cancellation is a data race by construction

**Verdict:** unverified (medium/low)

**Location:** src/core/core.hpp:61

**Evidence:**
```cpp
struct Context {
    int threads = 0;          // 0 => hardware_concurrency
    bool cancelled = false;
```

**Failure scenario:** cancellation is inherently written by one thread (signal
handler/driver/GUI) and polled by workers inside `parallel_for` bodies. With a plain
`bool` that is a data race (UB), and the compiler may legally hoist the load out of a
polling loop so cancellation is never observed. The Context is stub-marked, but the
field's type is the API every stage will code against — it should be born correct.

**Suggested fix:** `std::atomic<bool> cancelled{false};` (relaxed loads/stores
suffice); note Context then needs to be passed by reference, which it already is.

## [low/hygiene] Test-harness failure counter is not atomic — CHECK inside a parallel_for body can lose increments and produce a false PASS

**Verdict:** unverified (medium/low)

**Location:** src/core/test.hpp:27-30, :60

**Evidence:**
```cpp
inline int& failures() {
    static int n = 0;
    return n;
}
...
::fenix::test::failures()++;
```

**Failure scenario:** the natural way to test parallel kernels is
`parallel_for(0, n, [&](s64 i){ CHECK(out[i] == expect[i]); })`. Two racing failing
CHECKs can lose an increment (and it's UB besides); worse, `run_all` decides PASS/FAIL
by `failures() == before`, so lost increments can turn a failing parallel test into a
reported PASS — the worst possible harness failure mode. tests/ already mixes
parallel_for with CHECK-adjacent code (tests/test_log.cpp:59).

**Suggested fix:** make the counter `std::atomic<int>` (add a `note_failure()` helper
so the macros don't need to know).

## [low/design] FENIX_ASSERT is defined in core.hpp after all leaf headers are included, so Volume/Arena/sampling cannot use it — the "only voxel accessor" has no debug bounds checking

**Verdict:** unverified (medium/low)

**Location:** src/core/core.hpp:40-51 (definition); src/core/volume.hpp:52-53, :73-75 (would-be users)

**Evidence:**
```cpp
// volume.hpp:52 — "Unchecked access (hot path). Debug builds still catch OOB via FENIX_ASSERT callers."
constexpr T& operator()(s64 z, s64 y, s64 x) const { return data_[offset(z, y, x)]; }
```
while `FENIX_ASSERT` is only defined in core.hpp *after* `#include "core/volume.hpp"`.

**Failure scenario:** volume.hpp's own comment delegates OOB catching to
"FENIX_ASSERT callers", but the macro is textually unavailable to every substrate
header (volume, arena, sampling, surface — none can assert), and in practice most
callers don't assert either. Debug/ASan builds therefore only catch volume OOB when
the bad offset happens to leave the whole allocation — an off-by-one that stays inside
the buffer (the classic halo bug) is silent even in debug. `VolumeView::crop` likewise
accepts an origin/extent that exceeds `dims()` without any debug check.

**Suggested fix:** move FENIX_ASSERT into core/error.hpp (or its own tiny
core/assert.hpp) so leaf headers can use it; add debug-only bounds asserts in
`operator()`, `at_clamped` (empty-view case is UB via `std::clamp(hi < lo)` too), and
`crop`.
