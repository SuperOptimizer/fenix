# Whole-scroll surface-prediction fleet — architecture + measurements (2026-07-05)

Target: virtually unroll PHerc0332. First step = whole-scroll surface prediction on the
**2.399µm** volume `s3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr`
(OME-zarr v2, L0 = 33592×15761×15761 u8, 6-level pyramid). Box: 4×RTX 4090 (24 GB), 251 GB RAM,
~108 CPU. Model: `scrollprize/surface_recto_3dunet` → `surface.fxweights` (280M-param 3D ResEnc-UNet).

## Measured
- **Single-GPU predict-surface** (patch=256, overlap=0.5, batch=2, eager fp16), 512³ crop / 27 tiles:
  tta=1 18s · tta=8 79s (~2.9 s/tile) · tta=48 405s (~15 s/tile). VRAM peak ~20.8 GB (batch 2≈3;
  it's genuine 256³ activation memory, not fragmentation — expandable_segments made it WORSE, 23.6 GB).
- **4-GPU scaling: 3.8×** — 4 concurrent 1-process-per-GPU predicts (same 512³ crop) = 108 tiles in
  83s wall vs 79s single-GPU/27 → ~0.77 s/tile aggregate at tta=8. Negligible contention.
- **Occupancy** (from L4 pyramid, ÷16): 2497 real / 8448 bbox tiles = **29.6% non-air**.
- **Whole-scroll ETA**: ~1.18M occupied 256³-stride tiles × 0.77 s ≈ **~10.5 days** at tta=8 on 4 GPUs.

## Decisions
- **TTA=8**, not 48 (~6× faster, most of the octahedral quality per fenix-tta-teacher-ablation).
- **batch=2, default allocator** (no expandable_segments — inflates peak toward the 24 GB ceiling).
- **Drop TensorRT**: TRT 11.1 cannot build fp16 conv3d for this net (BuilderFlag.FP16 removed;
  strongly-typed fp16 fails on missing kernels). The measured 1.5× TRT wins were on older TRT 10.x.
  Eager fp16 4-GPU is the production path. (memory: trt11-no-fp16-conv3d)

## Fleet architecture (two monolithic .fxvol outputs: raw CT + predictions)
- **Stage 1 (raw CT)**: `fenix export-scroll <zarr> 0 rawct.fxvol` — out-of-core, resumable
  (coverage tri-state = bookmark), coarse-pyramid air-skip, prefetch pipeline. One monolithic .fxvol.
- **Stage 2 (predictions)**: predict-surface has NO built-in tiling (loads a whole .fxvol to RAM) —
  the 8.3 TB scroll must be tiled. Per tile: `ingest-zarr <zarr> 0 z y x D H W tile_ct.fxvol` →
  `predict-surface tile_ct.fxvol surface.fxweights tile_pred.fxvol 256 0.5 8 2`. Shard tiles across
  4 GPUs (one process per FENIX_GPU/CUDA_VISIBLE_DEVICES), 1024³ core + 128 halo for seamless
  overlap-blend, resume by skipping tiles whose pred output exists.
- **Monolithic stitch**: a .fxvol archive is SINGLE-WRITER (flock — concurrent appenders corrupt
  coverage), so the 4 GPUs CANNOT co-write one archive. Path: 4 GPUs write per-tile preds in parallel,
  then a SINGLE-WRITER stitch pass merges them into one full-dim predictions .fxvol via
  `VolumeArchive::write_chunk(lod, ChunkCoord, block)` at each tile's chunk offset (I/O-bound, fast).
  Needs a small `fenix stitch`/`predict-scroll` C++ stage (single-writer rule — main instance writes it).

## Validation (in progress 2026-07-05)
2048³ region (8× 1024³ tiles, 2/GPU) near the occupied core (Z0=15488,Y0=6856,X0=6856), fully
occupied (4096/4096 real chunks/tile). Exercises the exact per-tile zarr→ingest→predict flow at scale
before committing to new C++ or the ~10-day run. Script: /root/tilefleet.sh.

## Bottleneck benchmark (4096³ region, L0, tta=8, 4x4090 — 2026-07-05)
Instrumented run (bbox=14336,5632,5632,4096,4096,4096 = 512 regions), /proc+/sys+nvidia-smi sampler:
- **GPU util 93% avg (min 50) — GPU compute is THE bottleneck** (the right answer: expensive resource is busy).
- GPU mem 11.3 GB/card (of 24). CPU 98% (high but not starving GPUs — util stays 93%). RAM 42.7 GB (of 251).
- **Network NOT a bottleneck**: ~0 steady, 644 MB/s peak burst (fast ~5 Gbit uplink; S3 fetches in bursts
  between regions). Disk write ~0 (pre-commit; batches at commit=64). (Contrast: at tta=1 it was
  network/fetch-bound with GPUs idle — the tta multiplier flips the bottleneck to GPU.)
- **KEY INEFFICIENCY — halo waste**: each region gathers core+128-halo and predicts the WHOLE box, but
  writes only the core. At region=512 the gather is 768³ = 125 patches (5³) of which only the 512³ core
  (~27 patches) is kept → **~78% of GPU compute is discarded halo**. Fix = bigger regions:
  region=1024 → 1280³ gather = 729 patches for a 343-patch core = 53% waste (~1.6× faster run);
  region=1536 → ~39% waste. 1280³ gather ≈ 2 GB u8 (fine in 251 GB); patches stay 256³ (VRAM unchanged).
  **Recommend region=1024+ for the production run.** (Halo=patch/2=128 is hardcoded; could also expose it.)
- SSH note: a full-CPU run starves sshd (exit 255 on `ssh <cmd>`); cap with FENIX_THREADS + nice. See
  [[fenix-runpod-gpu-box]].
