# Multiscale surface predictions — the full pyramid (scales 1..128)

Directive (forrest, 2026-07-04): surface predictions at ALL of 1/2/4/8/16/32/64/128×
(2.4 µm → 307 µm). One ResEncUNet per scale, same arch, per-scale weights; the feeder's
`canon=`/`msc=` tokens (2026-07-04) make every rung a one-line recipe on the SAME corpus.

## The ladder

| scale | pitch (µm) | source | canon= | um= | msc= | thickness | semantic |
|---|---|---|---|---|---|---|---|
| 1 | 2.4 | zarr/0 | 2.4 | 2.4 | 1 | 6 | individual sheets (today's model) |
| 2 | 4.8 | zarr/1 | 4.8 | 4.8 | 0.5 | 3 | individual sheets |
| 4 | 9.6 | zarr/2 | 9.6 | 9.6 | 0.25 | 2 | individual sheets |
| 8 | 19.2 | zarr/3 | 19.2 | 19.2 | 0.125 | 1–2 | sheets, ~12 vox/wrap |
| 16 | 38.4 | zarr/4 | 38.4 | 38.4 | 1/16 | 1 | sheets marginal (~6 vox/wrap) |
| 32 | 76.8 | zarr/5* | 76.8 | 76.8 | 1/32 | 1 | sheet DENSITY (~3 vox/wrap) |
| 64 | 153.6 | derived* | 153.6 | 153.6 | 1/64 | 1 | density + orientation context |
| 128 | 307.2 | derived* | 307.2 | 307.2 | 1/128 | 1 | whole scroll ≈ 590×255×255 |

\* check upstream pyramid depth; missing levels are mean-downsampled from the deepest
available (the view-scroll LOD builder already produces `<prefix>_l<k>.fxvol`).

## Notes
- Wrap pitch ≈ 100 native voxels ⇒ scales ≥32 cannot separate wraps: labels rasterize to
  near-continuous bands. That is a FEATURE for the fit's dense coarse term (P5): coarse
  models answer "where is papyrus and how dense", fine models answer "which sheet".
- Eval per scale: band-limited eval against the 5 val meshes with GT coords × msc;
  thresholds/thickness scale with the rung. Same holdout firewall at every scale.
- Consumers: flow-pyramid fit data terms (coarsest model first — matches the lattice
  pyramid), global seed finding (16–32×), streamed-tracer coarse pass (4–8×), view-scroll
  overlay at every LOD.
- Registry: `surface_l<k>.fxweights` + TOML entries; predict-surface is already
  resolution-agnostic (patch 128 at every scale).

## Status
- feeder canon=/msc= shipped (commit 6ab95ae); LOD2 (4×) smoke ring running on the pod.
- Next: smoke verdict → full ring + training per rung, coarse rungs first (cheapest,
  biggest architectural unlock); derived-level caches for 64/128×.
