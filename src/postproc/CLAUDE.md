# postproc — CLAUDE.md

## Purpose
Binary surface-mask cleanup — the Kaggle-winning post-processing toolkit. See
`docs/research/research-core.md` (postproc), `surface-detection-kaggle` notes in
`research-build.md`.

## Public API & key types
- `cleanup.hpp` (`fenix::postproc`):
  - `Volume<u8> remove_small_components(VolumeView<const u8> mask, s64 min_size, geom::Conn conn = TwentySix)`
    — dust removal via `geom::connected_components`; drops fg CCs below `min_size` voxels.
  - `Volume<u8> fill_holes(VolumeView<const u8> mask)` — cavity fill: 6-conn CC on the
    background, flood-marks any bg component touching the volume border, fills the rest.
- `morph.hpp` (`fenix::postproc`) — taberna `morph.c` lineage, all double-buffered/alias-safe,
  out-of-bounds neighbours treated as background:
  - `Volume<u8> majority_filter(VolumeView<const u8> mask, int iters = 1, int thresh = 14)`
    — iterated 27-neighbour (incl. self) vote; `thresh=14` = strict majority = discrete
    mean-curvature flow, the single biggest LB score gain. Iterate 6–10× (~+0.01 LB).
  - `Volume<u8> plug_pinholes(VolumeView<const u8> mask)` — bg voxel with all 6 face
    neighbours fg → fg.
  - `Volume<u8> connect_fragments(VolumeView<const u8> mask, int r, geom::Conn conn = TwentySix)`
    — bg voxel that sees ≥2 DISTINCT fg CCs (via `geom::connected_components`) within
    Euclidean radius `r` → fg; bridges wrap-fragment gaps without blanket-dilation's
    SurfaceDice cost.
  - `Volume<u8> ball_dilate/ball_erode/ball_close/ball_open(VolumeView<const u8> mask, int r)`
    — Euclidean-ball structuring element morphology (`detail::ball_morph`).
- `postproc.hpp` — umbrella header, includes `cleanup.hpp` + `morph.hpp`, and the
  self-registered `postproc` CLI stage (`FENIX_REGISTER_STAGE`) — **stage entry point
  `run()` is currently a stub** (`stage_unimplemented`); the primitives above are usable
  as a library today but not yet wired into a driver-invokable pipeline stage.
- **Not yet written** (design intent only, no code): normal-aware `inplane_close`; sheet
  repair (per-CC height-map rebuild over a PCA tangent plane, Laplace-inpaint gaps →
  watertight disk); topo surgery (PH-guided tunnel fill — window screen via `topo` Betti
  → localize H1 loops via `cubical_features` on the inverted window → precise dilate-fill,
  vs. blunt global dilation that kills SurfaceDice).

## Inputs / outputs & formats
In: binary/label surface masks (`.fxvol`, or any `VolumeView<const u8>` in-core). Out:
cleaned masks (`Volume<u8>` / `.fxvol`). All current functions operate purely in-core on
`VolumeView`/`Volume<u8>` — no `.fxvol` container IO lives in this module yet (that's
`postproc`'s stage-wiring TODO, via `io`).

## Dependencies
Intra: `core` (`Volume`/`VolumeView`, `parallel_for_z`, `Expected`), `geom`
(`connected_components`, `Conn`). `topo` (Betti/PH) will be a dependency once topo surgery
is written but is not `#include`d yet. Third-party: none.

## Invariants & numerics
Double-buffered alias-safe morphology (`majority_filter` swaps two full buffers per
iteration; all others allocate a fresh output volume — none mutate `mask` in place).
Connectivity per `conventions.md` (fg default 26-conn, bg dual 6-conn in `fill_holes`).
Mask values treated as `{0, nonzero}`→`{0,1}`; outputs always canonicalized to exactly
`u8{0}`/`u8{1}`. Repairs must not tank SurfaceDice (precise > blunt) — this is why
`connect_fragments` requires two *distinct* CC labels in radius `r` rather than dilating
everything.

## Performance notes
All morphology ops are OpenMP-parallel per Z-slice (`parallel_for_z`), single dense pass
per iteration/call — no acceleration structure (ball ops are brute-force over the `r`-ball
per voxel, `O(voxels · r³)`). `connect_fragments`/ball ops recompute distance checks
per-voxel rather than precomputing an EDT; fine at current CC sizes, revisit if `r` grows.
Everything is in-core today; out-of-core windowing (needed for topo surgery, and for
running morphology over 70k³-class volumes) is unimplemented.

## Gotchas / pitfalls
fenix's thesis is to prevent wrap-merges at segmentation (signed affinity + the
diffeomorphic fit) rather than repair post-hoc — postproc is for mask-output/eval paths
and the Kaggle surface-detection task, not the primary unrolling correctness.

## Status & TODO
**Implemented** (ported from taberna `morph.c`/lineage, see commit `03e9a35`):
`cleanup.hpp` (`remove_small_components`, `fill_holes`) + `morph.hpp` (`majority_filter`,
`plug_pinholes`, `connect_fragments`, `ball_{dilate,erode,close,open}`). These are
callable library functions; no dedicated unit tests found in this module yet (verify
against `tests/` before relying on "tested" status).

**Stub**: `postproc.hpp`'s registered CLI stage (`run()`) — returns `stage_unimplemented`.
No `.fxvol` in/out wiring exists yet.

**Not started**: `inplane_close` (normal-aware in-plane close), `sheet_repair`
(PCA-tangent-plane height-map rebuild + Laplace inpaint — HARD, taberna's version failed
on curved/porous ridges, see Gotchas), `topo_surgery` (PH-localized tunnel fill, needs
`topo::cubical_features`, not yet `#include`d here).

Next steps: wire `run()` to read/write `.fxvol` and chain the existing morphology
primitives into a configurable pipeline (recipe-driven op order + params); then tackle
sheet_repair/topo_surgery. No open ADRs specific to this module yet — PCA-height-map
limits and windowed-surgery params are open design questions, not yet written up.
