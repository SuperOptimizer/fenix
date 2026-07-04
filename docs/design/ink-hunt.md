# The ink hunt — automated segment→ink pipeline (PHercParis3, PHerc0332)

Status: 2026-07-04, first full run in flight. Goal (forrest): run the whole ink pipeline
automated with our code on Scroll 2 (PHercParis3) and Scroll 3 (PHerc0332), using the
upstream released models, against the NEW 2025/2026 2.4 µm rescans (never the old 7.91 µm
data the segments were traced on).

## The pipeline (all fenix stages; tools/inkhunt/hunt.sh orchestrates)

1. **Mirror segments** from dl.ash2txt.org (`grab_segments.sh`): obj + mask + meta per
   segment; NO `layers/` (those were rendered from the old scans). 59 Paris3 + 14 0332.
2. **Register old scan → new scan** (one affine per scroll / per source volume):
   - `fenix register-scans` — phase correlation of coarse pyramid levels on a common
     physical pitch → VC-format transform.json. Gets IN-FRAME, not exact. Learned:
     masked-vs-unmasked content makes raw correlation approximate and the mask=1 variant
     unreliable here; zflip=auto arbitration is untrustworthy (confidence degenerate —
     KNOWN ISSUE); force zflip=0 (the scans are consistently mounted so far).
   - `tools/inkhunt/refine_transform.sh` — coordinate descent on the TRUE objective:
     surf-qc delta of a transformed reference segment against the new volume.
     PHercParis3 measured: correlation start −3.7 → refined **+14.9** (native-trace
     quality; well-registered references run +8..+20). Exit 0 iff delta ≥ +5.
3. **Import** (`fenix import-obj`): VC OBJ → .fxsurf uv grid, with the lod0→lod0 affine
   (or a villa transform.json). Segments split across source volumes get their own
   refined transform (PHerc0332: 20231117143551 + 20231027191953).
4. **Frame gate** (`fenix surf-qc`, min_delta=3 per segment): failures are LISTED, never
   silently skipped — a failing segment means its source-volume transform is wrong.
5. **Render** (`fenix render-layers`): 65 layers, **step=1 (native 2.4 µm)** — both
   converted models are canonical-2µm-era. step=3.296 is only for 7.91µm-era models
   (2023 timesformers). Stack z replicate-padded to the DCT block (codec ringing
   otherwise corrupts outer layers — measured).
6. **Ink inference**, both models:
   - `fenix predict-ink` — `ink_3d_dino_guided` (3D UNet; native, bit-validated).
   - `fenix predict-ink2d` — `ink_canonical_2um` (r152 + 3D-FPN "2.5D"; native,
     bit-validated; tiled 256², Hann-blended, 62-layer depth window).
   Weights converted from HF via tools/ml-export/convert_weights.py →
   /workspace/models/{ink,ink2d}.fxweights.
7. **Review** (`fenix project` max/mean projections + `tools/inkhunt/ink_gallery.py`):
   per segment, ink map beside papyrus texture; i/n/u triage; ink-map JPEG size as a
   crude busy-ness sort.

## Volumes
- PHercParis3: new = `20260427095331-2.400um-...-masked.zarr` (open S3); old = volpkg
  `20230210143520` (all 59 segments; standardized zarr `54keV_7.91um_Scroll2A` level 4
  used for registration — frame equality with the mesh volume CONFIRMED by the +14.9).
- PHerc0332: new = `20251211183505-2.399um-...-masked.zarr`; old = standardized
  `53keV_7.91um_Scroll3` — two segment source volumes, refined separately.

## Known issues / next
- register-scans confidence is degenerate (reports ±scale) — debug; until then zflip
  must be forced and refinement is mandatory.
- The third upstream model family (resnet3d-50 `ink_detection_pipeline` /
  PHerc1667-iteration) = parameterize nets/resnet3d depth (task).
- Rotation candidates (scroll remounting) not yet needed — add to register-scans when a
  scroll defeats translation+flip.
- Multi-block eval harness (gate_multi.sh) is the referee pattern for any model claims.
