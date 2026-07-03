# villa training platform — research report (2026-07-03)

Scope: ScrollPrize/villa's MODEL TRAINING code — the pipelines behind the published
surface_recto_3dunet, surface_m7, fiber self-distillation, ink_3d_dino_guided, and
dinovol_v2 models. Explicitly EXCLUDES the neural tracer and lasagna/. Mapped from a
shallow clone by an exploration agent; file/line references verified in that clone.

## Where their training code lives
- `vesuvius/src/vesuvius/models/` — the modern library (BaseTrainer, ~2.6k lines):
  surfaces, fibers teachers, guided ink, dinovol SSL trainers + the published YAML
  configs under `models/configuration/`.
- `scripts/fiber_5class/` — standalone fiber SELF-DISTILLATION trainer (DDP-8).
- `segmentation/models/multi-task-3d-unet/tasks/normals/` — offline mesh→label zarr
  rasterizers (`write_mesh_labels.py`: z-plane triangle intersections, 1-voxel lines,
  one integer class per mesh, NO ignore band and NO background shell at raster time).

## The background/ignore discipline (their answer to our shell bug)
They NEVER trust geometric background:
- Labels carry an explicit `ignore_label` class (e.g. recto/verso uses 3=intersection;
  ink uses 2) — the uncertain near-surface band is a REAL CLASS in the label volume,
  masked out inside every loss (`nnunet_losses.py:159-222` ignore-mask Dice;
  `RobustCrossEntropyLoss` ignore_index).
- BG-patch sampling is FORBIDDEN unless an ignore label exists (config_manager
  raises), and a BG patch is accepted ONLY if it contains ignore-band AND background
  AND zero foreground (`find_valid_patches.py:944-955`) — i.e. background is trusted
  only adjacent to annotated surfaces.
- The fiber self-distillation pipeline adds a **dark-voxel guard**: voxels with RAW
  pre-augmentation intensity < `dark_voxel_thr` (default 90) are forced to background
  (`label_generator.py:447-457`) — and uncertain small instances default to FIBER,
  never background ("uncertain → foreground" bias).
**Validation**: our 2026-07-03 intensity-gated shell fix converged independently on
their dark-guard idea (we gate background-trust by darkness; they gate background-
assignment by darkness). Their `post_dark_mask_fix` checkpoint name = they were bitten
by the same class of bug.

## Cross-frame labels (their answer to our mesh-misalignment finding)
`cross_frame_dataset.py` + `data/affine.py`: labels live in a canonical frame, images
in another; `transform.json` gives the affine. Labels are resampled through the
inverse affine NEAREST at getitem; patch candidates are QC'd by forward-mapping the 8
corners and rejecting degenerate/out-of-bounds label AABBs; the patch cache key
includes a checksum of the affine (re-registration invalidates caches). A comment
records a fixed bug where stride-snapping pushed patches off their FG voxel — same
misalignment class we diagnosed. NOTE: their AABB QC catches degenerate mappings, NOT
a globally-consistent-but-wrong affine → our `surf-qc` (on-sheet brightness) is
complementary and still necessary.

## Losses
- Surfaces ("m7" family): **MedialSurfaceRecall = Dice + SoftSkeletonRecall + CE**
  with skeletons PRECOMPUTED in the dataset — validates our clDice-in-loss direction
  (theirs is skeleton-recall; ours is clDice — same family).
- Ink/fibers teachers: nnUNet Dice+CE with ignore. Deep supervision wrapper.
- Guided ink: frozen dinovol backbone → TokenBook3D prototype matching → scalar guide
  mask fused into the UNet; guide BCE loss vs downsampled foreground with ignore
  handling, weight 0.25.
- Fiber self-distill: CE(label_smoothing 0.1) + softmax Dice(smoothing 0.1, excl bg).

## Augmentation (theirs vs ours)
Theirs: Rot90 (p=0.5, valid isotropic axis pairs) + Mirror; full continuous rotation
COMMENTED OUT; blur/noise/sharpen/contrast/brightness/gamma; SimulateLowResolution;
BlankRectangle (cutout p=0.1). Fiber distill: joint flips+Rot90 with the crucial rule
that a 90° rotation through z SWAPS vertical/horizontal fiber classes.
Ours is a superset geometrically (continuous rotate_z, elastic, scale jitter, SO(3)
knob, ct_degrade, compression) — but we lack **BlankRectangle/cutout** (cheap, worth
adding) and their aniso-aware rotation-axis restriction.

## Optimizer / schedule / scale (published configs)
SGD lr 0.01 (!), wd 3e-5, momentum-nesterov; batch 2 @256³ or 16 @128³; 1000 epochs ×
250 steps; DDP up to 8 GPU; EMA; bf16 autocast (fiber distill: SGD lr 0.005, warmup
1500-2000 → cosine, grad-clip 5.0, EMA student 0.9995 for checkpointing only).
We train AdamW 3e-4 — an SGD arm is a cheap ablation given their consistent choice.

## Self-distillation mechanics (fiber)
Teachers FROZEN (offline distillation, not mean-teacher). Pseudo-labels generated
EVERY STEP on-GPU from the CLEAN crop (inference_mode), THEN jointly augmented with
the image; watershed instancing + PCA orientation for the 4-class fiber labels; ink
teacher optional/fail-soft. No storage of teacher artifacts at all — an alternative
to our precomputed-teacher .fxvol design (theirs pays teacher forward per step ≈ a
second GPU's worth; ours pays storage + the band-sweep).

## Adopt list for fenix (ranked)
1. DONE (converged): intensity gating of background — keep; consider also biasing
   uncertain shell voxels toward ignore rather than background (their asymmetry).
2. Ignore-band as a first-class label + BG-patch rule (only near annotated surfaces):
   fold into rasterize_band_multi semantics + sampler BG draws when we add them.
3. Skeleton-supervision precompute (their SoftSkeletonRecall): our clDice computes
   soft skeletons on GPU per step — fine at our scale; revisit if it grows.
4. Cross-frame affine cache-checksum idea → include mesh/volume identity hashes in
   feeder cache keys (we already reset caches manually; cheap hardening).
5. BlankRectangle (cutout) augmentation op (p≈0.1).
6. SGD-lr-0.01 ablation arm vs AdamW.
7. For Phase C: their per-step on-GPU pseudo-labeling is the storage-free alternative
   — benchmark against precomputed teacher .fxvols before committing the band sweep $.
