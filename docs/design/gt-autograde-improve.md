# Automatically grading & improving the GT segment corpus (2026-07-09)

Plan for a closed-loop system that scores every published segment for training-fitness,
repairs the fixable ones, quarantines the unfixable, and re-grades — with a model in the
loop that gets better as the GT does. Built on primitives that already exist in `src/ml/`
(the four QC oracles + surf-repair) and the new GT corpus ([[gt-segment-corpus]]: 190
`.fxsurf` imported, `-on-<scan>` frame convention solved).

## The core idea

GT quality is **multi-signal and repairable**, not a single pass/fail. A segment can be
(a) well-registered, (b) uniformly offset (repairable by a constant shift), (c) warped
(repairable by a per-uv offset field), (d) locally damaged (partially salvageable by
trust-masking bad tiles), or (e) garbage (quarantine). We already have an independent
oracle for each failure mode and a repair operator for (b)/(c). The job is to **chain
them into a graded, self-improving loop** and make the whole thing run over the corpus
unattended.

## The signals we already have (4 orthogonal oracles)

| oracle | signal | needs | output | catches |
|---|---|---|---|---|
| **surf-qc** (delta) | CT brightness at surface vs ±off along normal | CT | per-mesh Δ; `regions=`→per-uv-tile trust grid (P/F/?) | mesh sits on air/gap not papyrus |
| **surf-qc** (profile) | nearest-prominent-peak offset per cell | CT | `offsets=`→u,v,du TSV | systematic vs random offset (repairable vs damage) |
| **surf-consist** | inter-mesh surface-to-surface distance + normal coherence | ≥2 meshes | AGREE/OFFSET/CROSS verdict; per-mesh normal-coherence | misregistration & physically-impossible crossings (no CT needed) |
| **label-audit** | model confidently contradicts label | a prediction `.fxvol` + meshes | per-uv-tile TSV (miss%), worst-first | label errors the model already "knows" are wrong |

Orthogonality is the point: surf-qc uses CT intensity, surf-consist uses geometry (no CT),
label-audit uses the model. A segment that fails all three is almost certainly bad; one that
passes CT but fails consist is a registration issue; disagreements only label-audit sees are
often the subtle warps. **Consensus across independent oracles = the grade's confidence.**

## The repair operators we already have

- **surf-repair** — per-uv **offset-field** snap-to-ridge (or snap-to-air-edge / model-alpha):
  measures the nearest-prominent-peak offset along the normal per coarse-uv point, rejects
  outliers vs local median, smooths, applies clamped. Upgrades a constant `shift=` to a
  smooth warp correction. Fixes failure mode (c).
- **constant shift** — the `shift=`/`msc=` mesh-level correction (a special case of the above)
  for uniform offset (b). Cheap first attempt.
- **trust-masking** — surf-qc `regions=` emits a per-uv-tile trust grid; the rasterizer blanks
  FAIL tiles to ignore. Salvages partially-damaged meshes (d): keep the good tiles, drop the
  bad ones, no geometry change.

## The loop

```
              ┌─────────────────────────────────────────────┐
  corpus ──►  │ GRADE   surf-qc(Δ) + surf-consist + coherence│
              │         → per-segment scorecard (JSON)        │
              └──────────────┬──────────────────────────────┘
                             │ triage by grade
        ┌────────────────────┼─────────────────────┬───────────────┐
     A: pass             B: uniform-offset      C: warped        E: garbage
   (use as-is)          (surf-repair shift)   (surf-repair       (quarantine,
        │                     │                offset-field)      never train)
        │                     └────────┬─────────┘                     │
        │                              ▼                               │
        │                        RE-GRADE the repaired mesh            │
        │                     (did Δ go positive? consist AGREE?)      │
        │                     ┌────────┴────────┐                      │
        │                  improved           still bad ──────────────►│
        │                     │                                        │
        │                     ▼                       D: partial: trust-mask
        └──────────────► ACCEPTED GT SET ◄──────────  bad tiles, keep good
                             │
                             ▼
        (optional) train/fine-tune on ACCEPTED set → label-audit the corpus
                   with the improved model → surfaces subtle errors the
                   intensity/geometry oracles missed → feed back to GRADE
```

Two nested loops:
1. **Repair loop (fast, no training):** grade → repair → re-grade. Pure CT+geometry.
   Converges the corpus to "best achievable with intensity/geometry alone."
2. **Model-in-the-loop (slow, needs a train step):** train on the accepted set → label-audit
   finds residual errors → re-grade. Each turn the model is cleaner so its disagreements are
   more trustworthy (the classic "good model finds bad labels" flywheel). This is where GT and
   model co-improve.

## The scorecard (per-segment, machine-readable JSON)

One record per segment, the atom the loop reads/writes:
```
{ scroll, segment, mesh_url, scan, res, n_valid_cells,
  qc_delta,                 # surf-qc mean Δ (on-sheet brightness)
  qc_trust: {pass, fail, unk},   # per-uv-tile P/F/? counts (regions=)
  offset_field_stat: {median_du, std_du, systematic?},  # profile mode
  consist: [{other, verdict, med_dist}],  # AGREE/OFFSET/CROSS vs neighbors
  normal_coherence,
  audit_miss_pct,           # label-audit (once a model exists)
  grade: A|B|C|D|E,
  repair: {applied, kind, before_delta, after_delta},
  decision: use|repair|trust-mask|quarantine }
```
JSON so the loop is resumable/inspectable and the inspection scrubber/panel can render any
segment's card + overlay on demand (the visual court of appeal for borderline grades).

## Grade thresholds (starting points, tune per scroll)

- **A (use):** qc_delta ≥ +8, ≥90% trust tiles PASS, no CROSS, normal-coherence high.
- **B (shift):** qc_delta ≤ +3 but offset_field systematic (low std_du) → try constant shift,
  re-grade, promote to A if delta jumps.
- **C (warp):** offset_field high-std but structured → surf-repair offset-field, re-grade.
- **D (trust-mask):** mixed trust grid (some tiles PASS, some FAIL) → keep PASS tiles only.
- **E (quarantine):** delta ≤0 AND (CROSS or scrambled normals or repair didn't help). Also
  the tiny-fragment segments (n_valid < threshold) start here pending review.

Thresholds are per-scroll because resolution differs (2.4µm vs 7.91µm change the voxel-space
numbers). Calibrate on a handful of hand-verified segments per scroll (the overlay panel).

## What's already built vs what to build

BUILT: all 4 oracles, surf-repair (3 modes), trust-grid rasterizer integration, the .fxsurf
corpus, import-tifxyz coordscale, the visual inspection tools.

TO BUILD (thin orchestration, mostly Python/shell around existing subcommands):
1. **grade-corpus driver** — run surf-qc(Δ,regions,offsets) + surf-consist over all segments
   (grouped by scroll+scan for consist neighbors + CT reuse), emit the scorecard JSON.
   Parallel, resumable (skip cards already written). Pure orchestration.
2. **triage+repair driver** — read scorecards, apply the B/C/D repair operators, re-grade the
   repaired mesh, update the card. The repair loop.
3. **per-scroll threshold calibration** — 3-5 hand-verified segments/scroll via the overlay
   panel → set the A/B/C/D/E cuts.
4. **label-audit integration** — once a model exists (teacher first), run predict-surface →
   label-audit over the corpus, fold miss% into the scorecard. The model-in-the-loop turn.
5. **accept-set export** — emit the pairs.txt (fxsurf, ct, trust, shift) the feeder consumes,
   from grade==A/B/C + trust masks. Closes to training.

## Open questions / decisions

- **surf-consist needs neighbors** — only works where ≥2 segments overlap. Coverage varies
  per scroll (Scroll-4 dense, others sparse). Where there's no neighbor, grade on CT+model only.
- **Where CROSS means "which one is wrong"** — surf-consist flags the pair; disambiguating the
  culprit needs the CT oracle (whichever has the worse delta) or the model. Encode the tiebreak.
- **Trust-mask granularity** — per-uv-tile (256²) may be coarse; finer costs more QC compute.
- **Do we repair in place or keep provenance** — keep the original .fxsurf + a repaired variant
  + the offset field, so a repair is auditable/reversible (never silently overwrite GT).
- **The tiny fragments** (0.2M cells vs 47M median) — auto-quarantine by size, or are small
  genuine partial sheets still useful? Decide the n_valid floor.

## First concrete step (no training, no scale risk)

Calibrate + validate on ONE scroll (PHercParis4, 70 segments, densest overlap so surf-consist
is informative, and the scan we have): run the grade driver, hand-check 5 gradings against the
overlay panel, tune thresholds, run the repair loop, measure how many segments move A/B/C/D/E
and how many repairs actually improve delta. That proves the loop end-to-end on the best-covered
scroll before running the other four.
```
