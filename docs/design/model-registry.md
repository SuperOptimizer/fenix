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
| `fiber_ink_4class_selfdistill` (07-02) | 3D fibers+ink 4-class UNet-family (256³, self-distilled, p4_4class ddp8) | **native ✓ smoke-passed** (`model=fibers4`; bit-validation pending) | GT: fiber direction labels (H/V) + ink — needs a fiber-label rasterize mode |
| `dinovol_v2_ps8_supcon3class_step362500` (07-02) | frozen DINO backbone + 3-class supcon head; ships `avg_fiber_embedding` npz | backbone = `dinovol.hpp` (ps8 ✓); supcon head trivial | SSL/supcon pretraining loop (big; separate phase) |
| `fiber_dinoguided_2class_step010000` (07-02) | 3D fibers 2-class UNet (256³, dino-guided TRAINING signal; inference = plain UNet) | **native ✓ smoke-passed** (`model=fibers2`) | same as 4-class + dino-embedding guidance loss |
| `fiber_selftrain_teacher_epoch30` (07-02) | frozen fiber teacher UNet | same family | teacher for fiber self-training |
| `copy_displacement_latest` (07-01) | displacement field (copyist alignment; in=8ch, out=6ch) | encoder loads; decoder transpconv 320→160 differs from standard topology — introspect decoder widths | TBD |
| `surface_m7_nnunet` (07-01) | surface prediction, nnU-Net 3d_fullres 192³ (fold_0) — likely the NEWEST surface teacher | **native ✓ smoke-passed** (`model=m7`: plain naming, 6 stages, no scSE) | our existing surface KD pipeline as-is |
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
5. **ViT surface race (after the CNN KD baseline lands)** — frozen dinovol ps8 backbone +
   light seg decoder (UNETR-style or upsample+linear), trained on the SAME feed ring +
   tri-state GT, judged by the SAME eval-set firewall as the ResEnc student. Head-only
   training is cheap (days). Data is NOT the constraint (forrest, 2026-07-03): 336 meshes
   over 5 volumes = billions of labeled voxels at 2.4um — enough for a from-scratch
   supervised 3D ViT; plus TBs of UNLABELED full-scroll CT for SSL (feeder serves
   mesh-less patches trivially). The constraints are scroll DIVERSITY (~5 volumes —
   firewall must hold out whole scrolls; hurts CNN equally) and COMPUTE (from-scratch =
   weeks on one card; stage for the MI300X/multi-GPU box). Decision tree: head shows
   signal -> full unfreeze or from-scratch supervised ViT on our corpus + fp4 TRT engine;
   head fails on thin-sheet boundaries (ps8 tokens, no fine-scale skips) -> keep the CNN,
   reuse the backbone for guidance losses like ScrollPrize's fiber models.
6. **copy_displacement** — classify, then decide.
7. **SSL pretraining loop** (dinovol training) — the largest training-pipeline piece; its own
   design pass.

### Precision verdict (2026-07-03, MEASURED — tools/ml-export/trt_probe.py, RTX PRO 6000)
- **TRT fp16 engine: ADOPTED for bulk conv-UNet inference — 86ms/patch at 256³ vs 131.7
  eager-best (1.53×)**, flat across batch 1-3. Constraint: TRT tensors cap at 2^31 elements,
  so 256³ engines are STATIC batch ≤3 (dynamic-to-6 profiles find zero conv3d tactics).
- **Quantized conv3d is dead in the whole NVIDIA toolchain**, each lane at a different layer:
  int8 QDQ exports but TRT has no int8 conv3d tactics (build fails); fp8 QDQ can't even
  export (ONNX can't represent FP8-dequant→conv3d); fp4 trace blows the 2GiB protobuf limit.
  torchao int8 conversion = 1.0× (no TRT kernels behind it). Custom CUTLASS fp8/fp4 conv3d
  kernels would be pioneering genuinely unsupported territory — parked unless inference
  cost becomes the binding constraint.
- fp4/fp8 remain GEMM-only: the **dinovol ViT is the fp4 candidate** (modelopt QDQ → TRT
  NVFP4, plausibly 2-3× for backbone inference). Fold into the dinovol port.

## Training-pipeline gaps by family
- 3D voxel tasks (surface, 3D ink, fibers): COVERED by train-feed + train.py; fibers need
  a fiber-direction GT mode in the rasterizer (labels from mesh uv axes) + 4-class loss.
- 2D ink (rendered segments): needs a 2D feed mode — tiles from rendered surface volumes
  (62 layers × 256²) + inklabel PNG GT. New, moderate.
- DINO SSL: multi-crop views, EMA teacher, iBOT/DINO losses, LONG runs — its own phase.
