# Multiscale multi-instance surface predictions — the plan

Directive (forrest, 2026-07-05): surface models at scales 1/2/4/8/16/32/64/128× that
predict not just "surface vs not" but **which surface** — detection + instance
segmentation in one head, at every scale.

## 1. Label encoding: mod-k wrap coloring

Absolute wrap indices (0..~75) are not locally learnable (a patch can't know its global
winding). What is learnable: **relative wrap identity** — adjacent wraps must get
different labels. Encoding:

- classes: `0 = trusted background`, `1..k = wrap color (1 + wrap mod k)`, `255 = ignore`
  (unlabeled), exactly extending today's {bg, sheet, ignore} tri-state.
- `k = 4` default: a wrap's color differs from its ±1 and ±2 neighbours (collision only
  at ±k); k is a config knob, revisit after the first confusion matrices.
- The wrap index source is THE FITTED SPIRAL MODEL: per-mesh-cell absolute wrap via the
  transported residual (corpus_bridge unwrap: `res = winding_at + round(ref − winding_at)`
  BFS over the uv grid, gauge = per-component base). Model corpus_v7/v8 (holdout 2.06
  windings) is accurate enough for mod-k coloring: an error must exceed ±0.5 winding
  LOCALLY to miscolor, and holdout error is dominated by smooth regional bias, which
  cancels in the local unwrap.

## 2. Pipeline changes (in order)

**P0 — LOD2 ring anomaly (BLOCKER).** The 4× smoke ring shows sheet 0.554 / bg nan
(native: 2–4% / present). Diagnose via feed_reader.py's real API (not `Ring`) before any
ladder training. Suspects: label-band offsets vs the 4× mesh-cell density; the trusted-bg
shell placement at scaled thickness; or logging semantics. Verdict gates everything.

**P1 — `wrap-label` stage** (winding module; consumes model + meshes):
`fenix wrap-label model=<fxmodel> surf=<fxsurf>... [k=4] [inplace|out-dir]` —
per cell: unwrap turn over the uv grid (corpus_bridge::detail machinery), absolute wrap =
round(base + turn) via the model gauge, store `1 + wrap mod k` in a new fxsurf u8
channel (`wrapcolor`; version-bump fxsurf or reuse conf slot with a flag — DECIDE: new
channel, version 4, tri-state preserved). One-time batch over the corpus per scroll.

**P2 — rasterizer** (ml/rasterize.hpp): when the mesh has a wrapcolor channel, the sheet
band rasterizes that color instead of constant 1; trusted-bg and ignore unchanged; trust
grids still gate per-tile. Feeder passes it through untouched (labels are already u8).

**P3 — trainer** (tools/train/train.py): `--classes k+1` (softmax CE + per-class dice);
class-balanced weights (colors are ~equal by construction, bg dominates as today); eval
adds a **contact-separation metric**: on val patches where two colors touch, fraction of
contact voxels with correct color on both sides — THE metric this whole idea exists for.
Band-limited eval per scale (GT coords × msc) as in the multiscale plan.

**P4 — the ladder** (multiscale-surface.md table): train rungs with instance labels from
the start — coarse-first (16×, 32×: minutes each, several wraps per patch = strongest
ordering signal), then 8×/4×/2×, then 1× (longest). Weights: `surface_w4_l<k>.fxweights`.
64×/128× = density rungs, binary labels still (wraps unresolvable) — the ladder is
instance-colored where wraps resolve (1×–16×), density above.

**P5 — consumers**:
- patch graph / cosegment: Δwrap read from channel differences at contacts (replaces or
  corroborates CT-valley counting).
- tracer gates: the winding gate gets a LEARNED prior (reject frontier cells whose
  predicted color mismatches the patch's color track).
- the spiral fit: per-color dense terms at coarse scales — the P5 dense-data roadmap item
  lands with instance structure already in it.
- predictions module: `.fxvol` u8 channel = argmax color; prob channels optional.

## 3. Eval + firewall

Same discipline at every rung: 5 held-out val meshes, wrap-labeled by the same model,
never fed to training. Metrics: per-class surface dice, contact-separation, and binary
collapse (any-color vs bg) to compare against today's binary models — the ladder must
first MATCH binary detection before instance gains count.

## 4. Risks / open questions

- Model gauge errors near the umbilicus (few wraps, high curvature) may miscolor inner
  wraps → mask cells with |residual − round(residual)| > 0.35 to ignore (confidence gate).
- k=4 collisions at ±4 wraps are invisible to the loss — acceptable (contacts are ±1).
- fxsurf version bump ripples through io/surface consumers — keep channel optional.
- The 0.554 anomaly may be a rasterize-band bug that ALSO affects native rings subtly —
  diagnose before concluding it's LOD-specific.

## 5. Execution order

1. P0 diagnosis (pod, feed_reader inspection + one-mesh raster stats)
2. P1 wrap-label stage + unit test (synthetic spiral: colors alternate correctly)
3. P2 raster + P3 trainer changes; smoke on native 1× ring (small)
4. P4 coarse rungs 16×/32× first; report contact-separation vs binary baseline
5. P5 consumers as results justify
