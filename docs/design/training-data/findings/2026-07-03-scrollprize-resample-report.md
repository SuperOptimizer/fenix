# Upstream report: misregistered "-on-<volume>" mesh resamples (PHerc Paris 4)

**Status:** draft for upstream (ScrollPrize) — evidence collected 2026-07-03/04 on the
`-on-20260411134726-2.4um` resample family. Companion to the root-cause finding
[2026-07-03-mesh-volume-misalignment.md](2026-07-03-mesh-volume-misalignment.md).

## Summary

A large fraction of the public PHerc Paris 4 segment meshes, when resampled onto the
2.4 µm volume `20260411134726`, do **not** lie on the papyrus sheet they annotate.
Training a surface model on the raw set stalls at chance; filtering by the QC below
restores normal learning. We ask upstream to re-check the resample transforms for the
listed segments (and to publish the per-segment transform provenance so consumers can
check it mechanically).

## Method

`fenix surf-qc` (src/ml/surf_qc.hpp): sample K=60 valid uv cells per mesh; at each,
compare mean CT (3³ neighborhood) at the surface point against the mean at ±12 voxels
along the local surface normal:  `delta = ct@surface − mean(ct@±12)`.

A correctly registered mesh sits on bright papyrus with darker inter-wrap gaps alongside,
so delta is decisively positive. Well-registered references measure **+8 to +20**:

- PHerc0139 on 2.399 µm: 34/37 meshes pass, typical delta +6.5 … +20.4
- PHerc1667: 16/20 pass; PHerc0172 (7.91 µm): 37/52 pass
- PHercParis4 2025/2026-era meshes traced natively on the 2.4 µm volume: mostly +8 … +14

Independent cross-checks that agree with the delta verdicts:
- normal-profile classification (ridge/edge vs air) and its offset coherence
- inter-mesh consistency (`fenix surf-consist`): overlapping traces of the same sheet
- training-time per-mesh loss (mislabeled meshes stay high-CE late in training)

## Results (PHerc Paris 4, in-frame on `20260411134726-2.4um`)

78 in-frame meshes measured. 50 pass the training gate (delta ≥ +1, kept in the
rehearsal corpus at ≥ +1; 25 clear ≥ +5). **53 fail the strict ≥ +5 gate**, of which
many are flat-to-negative (the mesh is as often in a gap as on a sheet — i.e. not a
small offset but a misregistration):

Worst offenders (delta at/below −3):

| segment | delta |
|---|---|
| 20260623161233 | −7.3 |
| 20260701183137 | −7.1 |
| 20260604223808 | −5.9 |
| 20260623144224 | −4.8 |
| 20260603222816 | −4.7 |
| 20260701183142 | −4.1 |
| 20260603185441 | −3.8 |
| 20231221180251 | −3.3 |
| 20260602230115 | −3.2 |

Full list (delta in parens): 20230702185753 (+3.0), 20231007101619 (−0.3),
20231012184424 (+1.3), 20231016151002 (−1.1), 20231022170901 (+1.2),
20231031143852 (+3.6), 20231221180251 (−3.3), 20260602230115 (−3.2),
20260603005223 (+2.8), 20260603185441 (−3.8), 20260603222816 (−4.7),
20260604223808 (−5.9), 20260623141135 (+3.3), 20260623141649 (+0.3),
20260623143441 (−0.1), 20260623144224 (−4.8), 20260623144957 (−1.1),
20260623151729 (−1.4), 20260623152443 (−2.6), 20260623153216 (+1.1),
20260623154617 (+4.0), 20260623155240 (+0.4), 20260623155914 (+1.6),
20260623161233 (−7.3), 20260623161921 (+3.0), 20260623163339 (−2.1),
20260623165742 (+1.0), 20260623170833 (+4.1), 20260623171400 (−2.1),
20260623171929 (+1.5), 20260701183125 (+4.6), 20260701183126 (+2.1),
20260701183127 (−2.2), 20260701183129 (+4.4), 20260701183130 (−2.9),
20260701183131 (+0.8), 20260701183133 (−2.4), 20260701183135 (−0.5),
20260701183136 (−0.8), 20260701183137 (−7.1), 20260701183138 (+4.7),
20260701183139 (−2.8), 20260701183140 (+3.9), 20260701183141 (+4.3),
20260701183142 (−4.1), 20260701183143 (+0.1), 20260701183144 (+4.7),
20260701183145 (+2.7), 20260701183148 (+2.2), 20260701183149 (+3.9),
20260701183150 (+1.9), 20260701183151 (−0.3), plus one INSUFFICIENT (−5.4).

Additionally, the whole `-on-20230205180739-7.91um` and `-on-20260310170716-45.532um`
resample families read 0.0 everywhere at the sampled points (out-of-frame / empty at the
mesh coords) — likely a coordinate-frame rather than a registration problem.

## Hypotheses

1. Some meshes were traced on a different-era volume (7.91 µm, or an earlier 2.4 µm
   reconstruction) and the cross-frame affine used for the resample is wrong or missing
   for them. The 2023-era ids failing on 2.4 µm while 2025/2026 ids mostly pass is
   consistent with this (our era rule: prefer 2025/2026 meshes for Paris 4).
2. Hand-edits: several segments were manually edited after tracing; edits made against a
   different frame would land off-sheet after resampling.
3. A systematic face-vs-midline convention (+~4 voxel normal offset, measured on passing
   meshes) exists on top of the above — small and correctable, unlike (1)/(2).

## Asks

- Re-derive/verify the per-segment cross-frame transforms for the listed ids; publish
  `transform.json` provenance (source frame, matrix, checksum) with each resample.
- Publish per-segment "traced-on" volume ids so consumers can gate mechanically.
- We are happy to share the QC tooling output (per-uv-tile trust maps, offset fields) for
  any segment on request.

## Addendum (2026-07-04): cross-resolution consistency confirms transform breakage

Comparing the SAME segment's delta across resample resolutions (6 segments measured on
both 2.4 µm and 1.129 µm): segment 20231221180251 FAILS on 2.4 µm (−3.3) but passes
strongly on 1.129 µm (+7.2) — the trace is good, the 2.4 µm resample transform is broken.
20231210121321 shows the inverse (+6.5 vs +0.9). This (a) pins the failures on per-resample
transforms rather than on the traces, and (b) gives consumers a recovery path: use the
resample variant that passes QC (`um=` retargeting) instead of dropping the segment.
