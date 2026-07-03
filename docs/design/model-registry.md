# Model registry — ScrollPrize/villa model catalog + port status

Status: 2026-07-02 survey of huggingface.co/scrollprize (recent-month models per forrest).
Goal: TRAINING + INFERENCE pipelines for every current model family: surface prediction,
2D ink, 3D ink, fibers, DINO backbone, displacement.

## Port strategy (two tiers)
- **Tier-ts (days):** load the HF checkpoint in Python, `torch.jit` export → `.ts`, run
  through `fenix predict-surface`'s TorchScript path (2461a63) — inference for every model
  without C++ work. Scripts in `tools/ml-export/`.
- **Tier-native (selective):** from-scratch `torch::nn` C++ reimpl in `src/ml/nets/`,
  param-names mirroring the checkpoint, validated bit-identical vs the Python reference —
  the teacher-grade treatment (`resenc_unet.hpp` precedent). For models we run constantly.
- **Training:** the `tools/train/` framework (feed ring + bf16 + KD + QAT + val-ring)
  generalizes per family; each family below lists what the feeder must add.

## Catalog (recent month, newest first)

| model | task / arch | port status | training pipeline needs |
|---|---|---|---|
| `fiber_ink_4class_selfdistill` (07-02) | 3D fibers+ink 4-class UNet-family (256³, self-distilled, p4_4class ddp8) | introspect → likely `resenc_unet` config | GT: fiber direction labels (H/V) + ink — needs a fiber-label rasterize mode |
| `dinovol_v2_ps8_supcon3class_step362500` (07-02) | frozen DINO backbone + 3-class supcon head; ships `avg_fiber_embedding` npz | backbone = `dinovol.hpp` (ps8 ✓); supcon head trivial | SSL/supcon pretraining loop (big; separate phase) |
| `fiber_dinoguided_2class_step010000` (07-02) | 3D fibers 2-class UNet (256³, dino-guided TRAINING signal; inference = plain UNet) | introspect → likely `resenc_unet` config | same as 4-class + dino-embedding guidance loss |
| `fiber_selftrain_teacher_epoch30` (07-02) | frozen fiber teacher UNet | same family | teacher for fiber self-training |
| `copy_displacement_latest` (07-01) | displacement field (copyist alignment) | classify from README/ckpt | TBD |
| `surface_m7_nnunet` (07-01) | surface prediction, nnU-Net 3d_fullres 192³ (fold_0) — likely the NEWEST surface teacher | plans.json → `resenc_unet` config mapping + convert_weights | our existing surface KD pipeline as-is |
| `ink_3d_dino_guided` (06-30) | 3D ink UNet | **native ✓ validated** (`resenc_unet` config) | ink GT rasterize mode |
| `dinovol_v2_ps6_step032350` (06-30) | DINO backbone ps6 variant, **window_global_3d attention (5³ win, 2³ shift, alternating)** | `dinovol.hpp` covers ps8/global; ps6 needs windowed attention added | — |
| `surface_recto_3dunet` (06-30) | surface teacher | **native ✓ validated** (our current KD teacher) | ✓ running |
| `PHerc.1667-iteration-0..5` (06-30) | 2D ink from rendered segments: **resnet3d-50** backbone (62×256² in), HF InkDetectionModel | new config for a resnet3d-50 + decoder (our r152 impl is the same family — parameterize depth) | 2D-ink training: rendered-segment tiles + inklabel PNGs — needs a 2D feed mode |
| `ink_canonical_2um` (06-25) | 2D ink r152 + 3D-FPN decoder | **native ✓ validated** (`resnet3d.hpp`) | 2D-ink training (same feed mode as above) |
| `dinovol_v2_ps8_with_paris4_352500` (06-04) | DINO backbone ps8 (embed 864, depth 24, SwiGLU, mixed RoPE, 4 reg) | `dinovol.hpp` targets exactly this | SSL pretraining loop |

## Priority order (forrest 2026-07-02: surface training first)
1. **surface_m7_nnunet** — candidate replacement/ensemble teacher for the surface KD run
   (introspect plans+ckpt; if ResEnc-compatible → convert_weights; else .ts).
2. **Fiber family** (3 models) — same UNet lineage; unlocks fiber-aware training data.
3. **PHerc.1667 resnet3d-50** — parameterize `resnet3d.hpp` by depth ([3,4,6,3] vs [3,8,36,3]).
4. **dinovol ps6 windowed attention** — extend `dinovol.hpp`; unlocks the newest backbone.
5. **copy_displacement** — classify, then decide.
6. **SSL pretraining loop** (dinovol training) — the largest training-pipeline piece; its own
   design pass.

## Training-pipeline gaps by family
- 3D voxel tasks (surface, 3D ink, fibers): COVERED by train-feed + train.py; fibers need
  a fiber-direction GT mode in the rasterizer (labels from mesh uv axes) + 4-class loss.
- 2D ink (rendered segments): needs a 2D feed mode — tiles from rendered surface volumes
  (62 layers × 256²) + inklabel PNG GT. New, moderate.
- DINO SSL: multi-crop views, EMA teacher, iBOT/DINO losses, LONG runs — its own phase.
