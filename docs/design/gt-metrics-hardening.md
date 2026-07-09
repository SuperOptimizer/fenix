# GT metrics hardening plan (2026-07-09)

Addresses the full measurement-layer audit (what we measure / confidence / gaps). Sequenced
by dependency; each item has a gate. Supersedes the P1.3/P1.4/P2.2 sketches in
gt-quality-adversarial-plan.md with concrete designs. Items M8/M9/M11 are staged, not built here.

## M1 — Wilson CIs + adaptive crop escalation  [GATE: borderline segments re-run, grades stable under reseed]
crop_qc: per-segment Wilson 95% CI on the pap/coh Bernoulli aggregates (each crop's pap is a
proportion over k probes; treat crop medians as the estimator, bootstrap the crop set for the
CI — 8 values is too few for asymptotics, so percentile-bootstrap over crops). scorecard:
assign a tier only when the CI half-width around pap_med is small enough that the tier
doesn't flip at CI edges; else "borderline" → crop_qc re-runs that segment with --crops 16
(and 32 max). repair_corpus improved-gate becomes CI-separated (bootstrap the before/after
probe sets).

## M2 — dense-region dispersion substitute  [GATE: iqr coverage >90% of crops]
Where n_offs is low (peak detector starved), gate dispersion on the ALPHA-offset IQR:
surf-qc profile mode also runs detail::air_edge per probe and emits `alpha-IQR`/`n_alpha`.
scorecard disp() accepts either iqr or alpha_iqr.

## M3 — surface the wide-search refusal  [GATE: caution flag in cards]
repair_corpus already runs the wide-search escalation; when the wide pass measures many
peaks (measured/lattice high) but is rejected by the accept-gate, record
`caution: "coherent-far-peaks-refused"` in the scorecard — the only computed hint of
coherent wrong-wrap capture.

## M4 — consist point-to-surface  [staged: fixes the giants blind spot]
Per candidate pair, re-sample the LARGER mesh's uv grid restricted to the pair's bbox
overlap at fixed stride (2-4 cells) instead of the global byte-budget cloud; distances
become point-to-densified-surface. Bounded memory (overlap-local).

## M5 — recto/verso orientation metric  [GATE: flags a deliberately flipped mesh, passes originals]
New: orientation_check.py. Estimate the scroll axis per z (median of all corpus-mesh points
per z-slab = umbilicus proxy). Per mesh: sign of dot(stencil normal, radial-outward) over
sampled cells → orientation consistency % and majority sign. A mesh mixing signs (or
opposite to the scroll convention) gets an `orientation` field in its card; compose() caps
mixed-orientation at C. Fault-injectable: flip the normals (reverse v axis) and check it flags.

## M6 — sheet-thickness / rasterization fitness  [staged]
surf-qc profile emits median peak FWHM (sheet thickness estimate); cards carry it; the
feeder's band radius can then be validated per scroll (band ≈ thickness/2, not a constant).

## M7 — synthetic threshold calibration  [GATE: cuts derived from corruption sweep, not hand-set]
Extend fault_inject: magnitude sweep (offset 1..8 vox, warp 2..8, shift 6..16) on 3 known-good
segments → the measured pap/iqr/coh response curves give the tier cuts that separate
"acceptable" from each corruption family with margin. Replaces the hand-set 92/85/70 with
data-derived cuts + margins (still to be confirmed on real hand-graded anchors later — the
synthetic sweep calibrates corruption-type sensitivity, not scan-texture variation).

## M10 — sheetness sigma tuning  [GATE: surface-vs-random contrast >= 2x on a known crop]
Sweep sigma_grad x sigma_tensor on the demo crop (EDT-verified geometry), pick the pair
maximizing sheetness contrast at surface points vs random material points. If nothing
reaches 2x, sheetness stays shelved (intensity+consist suffice at 2.4um).

## Staged (design ready, blocked on other work)
- M8 end-to-end ablation: small student trained on graded accept-set vs ungraded corpus;
  the only metric that proves the loop pays. Blocked on: corpus grades complete + accept-set
  export + feeder pairs. THE next big milestone after this hardening lands.
- M9 wrap-identity global prior: spiral/winding fit as the per-mesh wrong-wrap oracle
  (SOTA future tier; the winding module is the natural home).
- M11 model axis: label-audit wired into cards, gated by the fault-injection tripwire.
