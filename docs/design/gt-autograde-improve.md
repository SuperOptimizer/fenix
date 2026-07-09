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

---

## IMPLEMENTATION PATH (2026-07-09, autonomous build order)

The evaluation is **multi-axis**; a segment is training-grade only if it passes all axes. The
crop path ([[crop_qc]] `tools/labelqc/crop_qc.py`) is the WAN-free primitive that unblocks
everything (per-crop local CT boxes, no whole-scroll fetch). Build order, each step verified
before the next:

**Axis inventory (evidence each uses — orthogonal by design):**
| axis | signal | needs CT? | status |
|---|---|---|---|
| alignment (brightness) | surf-qc delta/profile | yes | DONE (crop_qc) |
| alignment (sheetness) | structure_tensor_sheetness — "is there a planar SHEET here" not just bright | yes | lib ready, add to crop_qc |
| intrinsic mesh health | eval/mesh_quality.hpp — holes/folds/self-intersect/tearing/distortion | NO | built as lib, needs CLI |
| inter-mesh consistency | surf-consist AGREE/OFFSET/CROSS + normal-coherence | no | built, wire in |
| model disagreement | label-audit (needs a prediction) | via model | built, flywheel (later) |

**Steps:**

1. **[repair demo] Before/after visual proof.** seg3 from the scan (ridge_med 44% — a real
   warp) → surf-repair → before(red)/after(green) mesh overlay on CT (`rep_overlay.py` ready).
   Confirms repair visibly moves the surface to the ridge. GATE: repair improves a genuinely
   bad segment.

2. **[mesh-qual CLI] Expose eval/mesh_quality.hpp as `fenix mesh-qual <fxsurf> [--json]`.**
   The CT-free intrinsic-health axis — folds/holes/self-intersection/tearing/distortion/
   degeneracy. ~40-line thin wrapper (the library is complete). WAN-free, instant, catches
   corrupt meshes the CT oracles miss. GATE: runs on a corpus .fxsurf, emits sane JSON.

3. **[sheetness in crop_qc] Add structure-tensor sheetness as a 2nd alignment signal.** For
   each crop, compute sheetness on the local CT box, sample it at the surface points → a
   "sits-on-actual-sheet" score alongside brightness delta. GATE: sheetness separates the
   44%-ridge seg from the 76% one more cleanly than brightness alone.

4. **[unified scorecard] One JSON schema per segment across ALL axes** (crop_qc alignment +
   mesh-qual health + surf-consist + later label-audit) with a composite A/B/C/D/E grade that
   requires passing every axis. Update grade_corpus.py/repair_corpus.py to read/write it.

5. **[consist pass] Wire surf-consist per scroll** (all meshes of a scroll at once; finds
   overlaps itself) → fold AGREE/OFFSET/CROSS into the scorecard. CROSS/OFFSET disambiguated
   by the alignment axis (worse-delta mesh is the culprit).

6. **[corpus sweep] Run the full multi-axis grade on PHercParis4 (70 segs)** via crops +
   mesh-qual + consist. Emit the grade distribution + a contact-sheet of the worst N for visual
   audit. GATE: distribution is sane, hand-check 3 extremes against the overlay panel.

7. **[repair loop] Triage+repair the B/C segments, re-grade, measure movement.** How many
   promote A/B/C→A, how many quarantine. Provenance-preserving (original + repaired + offset
   field kept).

8. **[accept-set] Export the pairs.txt (fxsurf, ct, trust, shift) for grade∈{A,B,C}** — the
   feeder-ready training set. Closes loop 1 (no model).

Loop 2 (model-in-the-loop / label-audit flywheel) follows once loop 1 has produced a clean
accept-set and a model is trained on it — deferred but designed (§ above).

Autonomy note: steps 1-8 are all CT-free or crop-local (no whole-scroll WAN), all use
existing verified primitives + thin wrappers. Each has a GATE; on a failed gate, stop and
report rather than proceed.
```
