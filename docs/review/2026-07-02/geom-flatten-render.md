# Review — unit `geom-flatten-render` (src/geom/, src/flatten/, src/render/)

Overall assessment: the implemented code is small, readable, and mostly follows the
project conventions (ZYX, s64 indices, `Expected`, explicit connectivity). The
connected-components slab-parallel union-find and the KD-tree are correct as written.
However there are three genuinely serious defects: the marching-tetrahedra quad case
emits a topologically wrong (bowtie) triangulation on every 2-2 sign split — directly
contradicting the header's "watertight by construction" claim; `render_surface`
ignores the validity mask when forming tangents, silently sampling garbage normals at
every valid-region border (the exact VC-sentinel foot-gun `docs/conventions.md` calls
out); and `read_obj` can abort the whole process on malformed input via `std::stoi`
under `-fno-exceptions`. Several medium findings concern the "EDT is exact" invariant
(f32 breaks it past 4096 voxels), the render stage widening a u8 archive to a whole
f32 volume, and the SLIM/ARAP mislabel with no fold-over guard despite the flatten
CLAUDE.md declaring the Jacobian guard mandatory. `geom/CLAUDE.md` also badly
overstates what exists (maxflow, Dijkstra3D, skeletonize, signed EDT, MC33, PLY read
are all absent).

## [high/bug] Marching-tetrahedra quad case triangulates across the wrong diagonal — holes + double-cover in every 2-2 sign split

**Verdict:** CONFIRMED — The finding is correct and reachable. At /Users/forrest/fenix/src/geom/marching.hpp:43-53 crossings are collected in fixed tedges order {01,02,03,12,13,23}; for every 2-in/2-out split the face-adjacency cycle of the four crossings is cross[0]-cross[1]-cross[3]-cross[2] (verified analytically for all three edge-pair configurations: e.g. inside={0,1} gives cross=[p02,p03,p12,p13], and p03/p12 share no tet face). Lines 66-67 emit (c0,c1,c2),(c0,c2,c3), whose shared edge c0-c2 is a quad EDGE not a diagonal, producing a bowtie. Numerical verification (random field values, all six inside-pairs): 24-38% of the true quad area is uncovered (hole) and 12-26% double-covered, per tet. Nothing guards this: the caller src/segment/trace_surface.hpp:94 feeds real fields directly to marching_tetrahedra, the module is documented as implemented in full (src/geom/CLAUDE.md 'Implemented in full (not stubbed)'), the header itself claims 'watertight by construction' (marching.hpp:3), and the existing test tests/test_marching.cpp only checks triangle count and vertex-radius deviation — bowtie vertices still lie on tet edges near the sphere, so the test passes despite the bug. Any smooth isosurface hits 2-2 splits constantly, so cracks and folded strips are pervasive.

**Fix notes:** The proposed fix — emit(cross[0],cross[1],cross[3]); emit(cross[0],cross[3],cross[2]) — is correct: it fans across the true diagonal c0-c3 of the cycle c0,c1,c3,c2 and works for all three 2-2 configurations since the collection order always yields that cycle. The closedness golden test (every welded edge shared by exactly 2 triangles on a sphere SDF) is the right regression guard; per tests/CLAUDE.md, weld with a tolerance, not bit-exact positions. One addition: neither the current code nor the fix orients triangles consistently with the field sign (the 3-crossing case at line 64 has the same issue) — outward-normal consistency is a separate, lower-severity concern worth a follow-up if flatten/eval consume orientation; watertightness (edge-shared-by-2) is achieved by the fix regardless.

**Location:** src/geom/marching.hpp:65-68 (with the edge order at :43-44)

**Evidence:**
```cpp
static constexpr std::array<std::array<int, 2>, 6> tedges = {
    {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
...
} else if (nc == 4) {
    emit(cross[0], cross[1], cross[2]);
    emit(cross[0], cross[2], cross[3]);
}
```

**Failure scenario:** For any tet with a 2-in/2-out sign split (e.g. vertices {0,1}
below iso, {2,3} above), the crossings land on edges 02,03,12,13 and are collected in
`tedges` enumeration order: `cross = [p02, p03, p12, p13]`. The true quad cycle
(crossings adjacent iff they share a tet face) is `p02–p03–p13–p12`, i.e. cyclic order
is `c0,c1,c3,c2` — the code's fan `(c0,c1,c2),(c0,c2,c3)` triangulates across BOTH
diagonals. Verified numerically on a planar quad: point (0.9,0.5) inside the quad is
covered by neither triangle (hole), point (0.1,0.5) is covered by both (overlap). This
holds for all three 2-2 configurations because `tedges` enumeration always lists the
quad as `c0,c1,c2,c3` with `c2/c3` swapped relative to the cycle. Every extracted
isosurface has cracks and folded double-surface strips wherever a tet is split 2-2 —
i.e. across the bulk of any smooth surface — despite the header claiming the mesh is
"watertight by construction". Downstream: eval surface metrics and flatten input are
silently corrupted.

**Suggested fix:** `emit(cross[0], cross[1], cross[3]); emit(cross[0], cross[3], cross[2]);`
(fan in the true cyclic order), and add a golden/fuzz test asserting the extracted
mesh of a sphere SDF is closed (every edge shared by exactly 2 triangles after vertex
welding).

**Outcome:** fixed — `src/geom/marching.hpp`'s `nc==4` branch now emits `(c0,c1,c3)`+`(c0,c3,c2)`
(the true diagonal) per the fix_notes. Added `marching_tetrahedra_sphere_is_watertight` to
`tests/test_marching.cpp`: welds coincident vertices (tolerance-quantized) and asserts every
UNDIRECTED edge is shared by exactly 2 triangles on a fully-interior sphere SDF (R+1 < volume
half-extent, so no boundary tets are cut). Skips a small number (<1%) of pre-existing,
diagonal-fix-independent zero-area triangles (crossings landing exactly on a shared tet vertex —
present identically before and after the fix, confirmed by direct comparison) rather than treating
them as a failure. Per-triangle winding-consistency (a separate, lower-severity concern the
verifier flagged as a follow-up — neither pre- nor post-fix code orients by field sign) is
explicitly NOT asserted. Verified: the test fails against the pre-fix diagonal (many undirected
edges shared by !=2 triangles) and passes against the fix.

## [high/correctness] render_surface computes tangents from invalid neighbours' zero coords — garbage normals at every valid-region border

**Verdict:** CONFIRMED — src/render/surface_render.hpp:31-32 computes tangents from neighbour coords with no is_valid check (the line-30 comment claims skipping but none exists). src/core/surface.hpp:27 value-initializes coord to {0,0,0} and only set() (lines 39-42) writes coords+validity, so invalid cells sit at the world origin — no NaN sentinel by project rule (core CLAUDE.md: "validity masks not NaN"), so the zero flows silently into the tangent. The producer src/flatten/extract_wrap.hpp:17 explicitly leaves invalid cells wherever a ray exits bounds (line 37) or finds no winding crossing, so valid/invalid adjacency at wrap ends is guaranteed in real use; at such pixels the tangent is c−origin, the normal is normalized garbage, and all 2N+1 layers sample wrong voxels while the pixel stays marked valid. The existing test (tests/test_surface_render.cpp:42) only checks the offset-0 layer, where s=c regardless of the normal, so it cannot detect this. render/CLAUDE.md's "STUB" status does not shield it: render_surface is implemented, tested, and is the module's live layer-stack path. Also confirmed: no per-pixel validity is returned, and both invalid pixels and degenerate-normal skips (line 35) leave silent 0.0f in the stack — the magic-sentinel leak docs/conventions.md forbids.

**Fix notes:** Fix direction is correct; three corrections: (1) when falling back to one-sided differences, also propagate line 35's degenerate-normal `continue` into the returned mask — any non-rendered pixel must be marked invalid, not left as silent zeros; (2) do not unconditionally wrap u: Surface has no closed-in-u flag, and blind periodic wrap is wrong for partial patches — add a closed_u flag to Surface (extract_winding_surface sets it; scale_u=2π/nu spans a full turn) or take periodicity as a LayerParams/render argument; (3) return the per-pixel validity with the stack (e.g. a {nv,nu} u8 mask in a small RenderResult struct) rather than telling callers to reuse surf.valid, since render invalidates pixels the surface considers valid.

**Location:** src/render/surface_render.hpp:31-33

**Evidence:**
```cpp
// Tangents from neighbours (skip across invalid cells where possible).
const Vec3f tu = surf.at(clampu(u + 1), v) - surf.at(clampu(u - 1), v);
const Vec3f tv = surf.at(u, clampv(v + 1)) - surf.at(u, clampv(v - 1));
```

**Failure scenario:** Despite the comment, validity is never checked. `Surface`
default-initializes `coord` to `Vec3f{0,0,0}` for invalid cells (core/surface.hpp),
so at any valid cell whose (u±1,v) or (u,v±1) neighbour is invalid, the tangent is
computed against the world origin — e.g. `tu ≈ -at(u-1,v)`, a vector of magnitude
tens of thousands pointing at (0,0,0). The resulting normal is normalized garbage, and
all 2N+1 layers at that pixel sample the wrong voxels. Since `extract_winding_surface`
produces invalid cells exactly where the wrap ends/tears, every border of the valid
region — where the recoverable text often is — gets a silently corrupted layer stack
(no error, plausible-looking values). This is precisely the magic-sentinel leak
`docs/conventions.md` §Validity forbids.

**Suggested fix:** Consult `surf.is_valid` for each neighbour; fall back to a
one-sided difference using the valid side (skip the pixel if neither side is valid),
and wrap u (periodic) rather than clamp for closed wraps. Also emit/return the
per-pixel validity mask alongside the stack (the render CLAUDE.md contract says
coords+normals+validity).

**Outcome:** fixed (partial per fix_notes scope) — `render_surface` now returns a `RenderResult{stack,
mask}`: tangents use one-sided differences that only consult valid neighbours (never an invalid
cell's zero-initialized coord), a pixel is marked invalid in the returned `{nv,nu}` mask if neither
tangent has a usable side or the normal is degenerate (propagating the old silent `continue`), and
callers must consume the mask rather than reusing `surf.is_valid`. Did NOT implement the closed_u
periodic-wrap flag (fix_notes item 2 — needs a new `Surface::closed_u` field, which is a
cross-cutting core/surface.hpp change outside this cluster's scope); u still clamps at the
boundary, which is conservative (may mark a valid periodic seam invalid, never fabricates data).
Updated `tests/test_surface_render.cpp`'s existing case for the new return type and added
`render_surface_border_no_garbage_normal`: an isolated 5-cell valid strip forces the one-sided path
at u=4, and asserts sampled values stay near the sheet's own coordinate (not a huge garbage jump
toward the origin) wherever the mask is set.

## [high/resource-safety] read_obj aborts the process on malformed files (std::stoi under -fno-exceptions) and reads uninitialized floats; face indices unvalidated

**Verdict:** CONFIRMED — Confirmed at src/geom/mesh.hpp:57: std::stoi on an unsanitized token under -fno-exceptions (CMakeLists.txt:90) — any non-numeric/empty/overflowing face token ('f a b c', 'f 1 2', 'f 99999999999') terminates the process instead of returning Expected, violating the project's expected/no-except invariant (root CLAUDE.md §2.3). mesh.hpp:45-47: for 'v 1 2' the third operator>> sentry fails (eofbit) so z is never written and an indeterminate f32 is pushed. mesh.hpp:52-59: indices are never validated against vertices.size() and negative OBJ relative indices are mangled by the unconditional '- 1'; write_ply (mesh.hpp:87) and tri consumers then index OOB. Reachable: read_obj is the documented public OBJ interop entry point (src/geom/CLAUDE.md lists OBJ/PLY read/write for mesh interop; geom is 'implemented in full', not a stub), so untrusted external files hit this path directly. No guard exists in the function or any caller (only current caller is tests/test_mesh.cpp:25).

**Fix notes:** Proposed fix is correct. Additions: (a) also check stream state after the 'v'/'vn' float extractions (or parse floats with from_chars too) and return err(Errc::decode_error) on short lines — the fix as written only covers face indices; (b) minor claim correction: the uninitialized-z push is C++26 'erroneous behavior' (defined fill value) rather than classic UB, and mid-line num_get failures store 0 — severity-neutral, still silent data corruption; (c) OBJ negative indices are legal (relative to current vertex count) — either resolve them as idx = vertex_count + i or reject explicitly, but don't just subtract 1; (d) validate index bounds against the FINAL vertex count only if vertices are guaranteed to precede faces — safest is a post-parse pass over all tris, since OBJ permits forward references; (e) optionally reject/triangulate faces with >3 vertices instead of silently dropping extra tokens.

**Location:** src/geom/mesh.hpp:44-58 (stoi at :57)

**Evidence:**
```cpp
f32 x, y, z;
ss >> x >> y >> z;                     // extraction failure -> x/y/z uninitialized
m.vertices.push_back(Vec3f{z, y, x});
...
std::string tok;
ss >> tok;                              // may fail -> tok empty
t[static_cast<usize>(k)] = std::stoi(tok.substr(0, tok.find('/'))) - 1;
```

**Failure scenario:** OBJ is an untrusted interop input (fuzz surface). (1) A line
`f a b c` or `f 1 2` makes `std::stoi` see a non-numeric/empty token; stoi reports
errors by throwing `std::invalid_argument`/`std::out_of_range`, which under
`-fno-exceptions` libc++ turns into an immediate `abort()` — the whole pipeline dies
instead of returning `Expected` (`f 99999999999` aborts too via out_of_range). (2) A
line `v 1 2` leaves `z` uninitialized and pushes an indeterminate f32 (UB,
uninitialized read). (3) Parsed indices are never validated against
`m.vertices.size()`; a hostile/corrupt file with `f 1 2 1000000` (or OBJ's legal
negative relative indices, which this parser mangles) yields out-of-bounds `tris`
that any consumer (slim triangle loop, PLY writer round-trip, normals) dereferences
OOB.

**Suggested fix:** Parse with `std::from_chars` checking `ec`, return
`err(Errc::decode_error, ...)` on any short/garbled line, and after the parse loop
validate `0 <= t[k] < vertex_count` (handle or reject negative indices explicitly).

**Outcome:** fixed — `read_obj` in `src/geom/mesh.hpp` rewritten: `v`/`vn` lines check the stream
state after each float extraction (`from_chars`-based `parse_f32_tok`) and error on short lines
(covers fix_notes item (a), not just faces); face indices parse via `from_chars` (`decode_error` on
bad/empty/overflowing tokens, never `std::stoi`); negative OBJ relative indices resolve as
`vertex_count + i` rather than being blindly `-1`'d (item c); index bounds are validated per-face
against the vertex count seen so far (this reader requires vertices to precede faces — forward refs
rejected, per item d's "simplest correct approach"); polygon faces with >3 vertices are
fan-triangulated instead of silently dropping vertices (the separate low-severity finding below).
Added 7 new cases to `tests/test_mesh.cpp`: short face line, non-numeric index, index overflow,
out-of-range index, short vertex line (all must return an Expected error, verified they don't
abort), negative relative index resolution, and quad fan-triangulation.

## [medium/correctness] EDT in f32 is not exact beyond ~4096-voxel distances — violates the module's "EDT is exact" invariant

**Verdict:** unverified (medium/low)

**Location:** src/geom/edt.hpp:19-44 (f32 throughout; `edt_big` at :15)

**Evidence:**
```cpp
inline void edt1d(const f32* f, s64 n, f32* d, s64* v, f32* z) {
    ...
    s = ((f[q] + static_cast<f32>(q) * static_cast<f32>(q)) - (f[v[k]] + vk * vk)) /
        (2.0f * static_cast<f32>(q) - 2.0f * vk);
```

**Failure scenario:** geom/CLAUDE.md states "EDT is exact", and volumes go to 2^18 per
axis. Squared distances are integers, exactly representable in f32 only up to 2^24 —
i.e. distances up to 4096 voxels. Beyond that, `f[q] + q*q` rounds (at q=2^18, q² ≈
6.9e10 has ulp 4096), parabola intersections `s` are computed with error, and the
lower envelope can pick the wrong parabola, so `d[q]` is off by up to thousands (and
the propagated y/z passes compound it). Any distance field on a scroll-scale block
with sparse seeds (sheet-repair gap bridging, skeletonization radii, touching-sheet
separation on a full slice) is silently inexact — while eval/NSD code relying on the
documented exactness treats it as ground truth. Additionally `f[q]+q²` when
`f[q]=edt_big` (1e18, ulp ≈ 1.4e11) absorbs q² entirely, which only works by accident
of the envelope ordering.

**Suggested fix:** Run `edt1d` in f64 (or integer arithmetic with s64 squared
distances, the FH-standard formulation) and store the result volume as f32 only at the
end if desired — this is exactly the "f64 for accumulation-sensitive spots"
conventions carve-out. Add a brute-force-vs-EDT property test with seeds >4096 voxels
apart.

## [medium/performance] render stage widens a u8 archive to a whole in-core f32 volume and writes the f32 output — the exact u8→f32 blow-up the archive v5 path was built to avoid

**Verdict:** unverified (medium/low)

**Location:** src/render/render.hpp:35 (and the f32 output write at :50-52)

**Evidence:**
```cpp
auto vol = a->read_volume();   // == read_volume_as<f32>(0) regardless of src_dtype
...
auto out = codec::VolumeArchive::create(outp, img.dims(), codec::DctParams{});
if (auto w = out->write_volume(img.view()); !w) ...  // img is Volume<f32>
```

**Failure scenario:** `VolumeArchive::read_volume()` is the f32 convenience overload;
codec/archive.hpp v5 exists precisely so a u8 scroll archive decodes natively
(`read_volume_as<u8>`, "no f32 widen" per its own comments and the project's standing
rule). `fenix render` on a 2048³ u8 crop allocates 34 GiB instead of 8 GiB — OOM or
swap-death on a typical box — and the whole-volume mean/unroll pass then runs on the
widened copy. The unrolled texture (mean of u8-range CT) is then persisted as an f32
.fxvol, 4× the ink-input storage for data with u8 dynamic range.

**Suggested fix:** Dispatch on `a->src_dtype()`: `read_volume_as<u8>` for u8 archives
(threshold/unroll templated on T or taking `VolumeView<const u8>`), and quantize the
unrolled mean image to u8 (round, clamp) before `write_volume(VolumeView<const u8>)`.

## [medium/correctness] slim.hpp claims SLIM / local injectivity but implements plain ARAP with no fold-over guard — the flatten CLAUDE.md calls the Jacobian guard mandatory

**Verdict:** unverified (medium/low)

**Location:** src/flatten/slim.hpp:151-152 (local step); header claim at :1-4

**Evidence:**
```cpp
const f64 ang = std::atan2(j10 - j01, j00 + j11);
const f64 cs = std::cos(ang), sn = std::sin(ang);
```
and the energy at :127 uses `E1 / (det * det + 1e-12)` — sign-blind.

**Failure scenario:** flatten/CLAUDE.md invariants: "Guard injectivity / no fold-over
(Jacobian non-inversion) … the Jacobian guard is mandatory." The local/global loop
here is classic ARAP: the closest-rotation fit plus an unconstrained Laplacian solve
can and does produce inverted triangles (det J < 0) on high-distortion wraps
(near-tears, strongly curved regions of a spiral wrap). Nothing detects or penalizes
inversion — no line search with a flip barrier (SLIM's defining feature), no det-J
floor check against the named `tol` fold constant — and the reported
`energy_final` uses `det²`, so a mirrored patch reports a *good* (low) energy. A
folded UV chart flows straight into rendering, where two sheet regions overwrite each
other in the texture with no error.

**Suggested fix:** Minimum: after the loop, assert/report `det J > 0` per triangle
(count of flipped tris in FlatMesh) and fail/flag if any are inverted. Proper: add the
SLIM reweighting or a flip-preventing line search on the symmetric-Dirichlet energy.
Rename the file/API to arap_* until it actually is SLIM.

## [medium/correctness] unroll accumulates mean CT in f32 — count saturates at 2^24 and sums drift; also a fully serial whole-volume triple loop

**Verdict:** unverified (medium/low)

**Location:** src/render/unroll.hpp:34,37-45

**Evidence:**
```cpp
std::vector<f32> count(static_cast<usize>(height * width), 0.0f);
...
for (s64 z = 0; z < d.z; ++z)   // serial over the entire volume
    ...
        iv(0, z, col) += ct(z, y, x);
        count[static_cast<usize>(z * width + col)] += 1.0f;
```

**Failure scenario:** Per-(z,col) bins collect ny·nx/width voxels; on a 40k×40k slice
with a few hundred columns that is 10⁶–10⁷ contributions. `count += 1.0f` silently
stops incrementing at 2²⁴ (16.7M), and the f32 running sum loses low bits long before
that (ulp of a 10⁸-magnitude sum is 8), giving a systematic darkening/bias of the mean
that varies per bin — silent corruption of the "straighten the spiral" output.
Conventions explicitly require f64 for large reductions. Separately, the scatter loop
and the min/max scan are single-threaded over the whole volume in a stage whose rows
(z) are perfectly independent — this is the stage's entire cost.

**Suggested fix:** Accumulate sum and count in f64 (or s64 count), divide once at the
end; parallelize with `parallel_for_z` (each z owns disjoint `iv(0,z,*)`/`count[z*width..]`
rows, so no races).

## [medium/bug] marching_tetrahedra vertex indices truncate to s32 with no overflow check

**Verdict:** unverified (medium/low)

**Location:** src/geom/marching.hpp:57

**Evidence:**
```cpp
const s32 base = static_cast<s32>(mesh.vertices.size());
```

**Failure scenario:** The extractor emits 3 fresh vertices per triangle (soup, no
sharing). A field of just ~1300³ cells with a dense surface (or any block a few
thousand voxels on a side of a wavy multi-sheet scroll mask) exceeds 2³¹ vertices;
`base` wraps negative and `tris` silently indexes garbage/negative — downstream OOB in
write_obj (indices printed wrong) or any consumer loop. No FENIX_ASSERT guards it.

**Suggested fix:** `FENIX_ASSERT(mesh.vertices.size() < ...max)` at minimum, or make
`Mesh::tris` s64 like `slim.hpp`'s `FlatMesh::tri` already is (the two mesh types
disagree on index width today).

## [medium/hygiene] geom/CLAUDE.md claims the toolkit is "Implemented in full (not stubbed)" — most of it does not exist

**Verdict:** unverified (medium/low)

**Location:** src/geom/CLAUDE.md:5-6 (and the Public API list)

**Evidence:** CLAUDE.md lists signed EDT, morphology open/close/hole-fill, "Marching
cubes — MC33", mesh cleanup (SAT self-intersection, largest-CC), OBJ **and PLY read**/
write, Dijkstra3D, maxflow (BK + Dinic), and TEASAR/Lee skeletonization as implemented
"in full". The directory contains none of: dijkstra3d, maxflow, skeletonize, signed
EDT, mesh cleanup, PLY *read*; and the isosurface is marching *tetrahedra*, not MC33.

**Failure scenario:** The working agreement makes per-dir CLAUDE.md the API source of
truth; an agent (or reviewer) planning segment/postproc work against this contract
will assume `geom::maxflow`/`dijkstra3d` exist and design around them, or "fix" call
sites to use APIs that were never written. This review's own assignment was misled by
it.

**Suggested fix:** Rewrite the Status & Public API sections to reflect reality
(implemented: unsigned squared EDT, CC, majority/dilate/erode, marching tetrahedra,
Mesh + OBJ write/read + PLY write, KD-tree; everything else TODO), per §5.2.7 of the
root CLAUDE.md.

## [low/hygiene] render CLI usage, stage description, and includes contradict what the stage does; arg parse errors silently ignored

**Verdict:** unverified (medium/low)

**Location:** src/render/render.hpp:22, 25-31, 61 (includes at :9-10)

**Evidence:**
```cpp
log(LogLevel::error, "usage: fenix render <in.fxvol> <out.nrrd> [pitch=8] [samp=4]");
...
std::from_chars(s.data(), s.data() + s.size(), v);   // ec discarded
...
FENIX_REGISTER_STAGE(render, "unroll a .fxvol volume to a flattened NRRD image", ...)
```

**Failure scenario:** The stage writes a `.fxvol` and the code comments "We never
write NRRD", yet the usage string, registered description, and `io/nrrd.hpp` include
all say NRRD — a user following `fenix render in.fxvol out.nrrd` gets a .fxvol-format
file named `.nrrd` that NRRD tools reject. The usage text `[pitch=8]` also invites
typing literally `pitch=8`, which `from_chars` fails to parse; the error code is
discarded, so the user's pitch silently reverts to the default with no warning.
`render/surface_render.hpp` is included but unused by the stage.

**Suggested fix:** Fix usage/description to `<out.fxvol>`, drop the stale includes,
and log a warning (or hard error) when `from_chars` returns a non-empty `ec`/doesn't
consume the whole token.

## [low/correctness] read_obj silently keeps only the first 3 indices of polygon faces

**Verdict:** unverified (medium/low)

**Location:** src/geom/mesh.hpp:52-59

**Evidence:**
```cpp
} else if (tag == "f") {
    std::array<s32, 3> t{};
    for (int k = 0; k < 3; ++k) { ... }
    m.tris.push_back(t);
}
```

**Failure scenario:** Quad-face OBJs are ubiquitous in interop (VC exports quads).
`f a b c d` parses a,b,c and silently discards d — half the surface's triangles vanish
with no error, and the mesh looks "valid" (a comb of disconnected triangles). Silent
data loss on the exact interop path this reader exists for.

**Suggested fix:** Read all indices on the line and fan-triangulate (v0,vi,vi+1), or
return an error for >3 vertices until polygons are supported.

**Outcome:** fixed — folded into the read_obj rewrite above: all face-line indices are now read
into a vector and fan-triangulated `(v0,vi,vi+1)`, covered by the new
`read_obj_fan_triangulates_quad_faces` test.
