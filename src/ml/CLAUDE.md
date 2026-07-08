# ml — CLAUDE.md

## Purpose
In-tree neural inference **and training** via **libtorch (C++)** for surface/sheet
detection, ink detection, and (new) instance (per-wrap) segmentation. Firewalled behind
this module — the rest of fenix consumes its outputs as prediction fields (via
`predictions`). The torch-free half of this module is also the **training data plane**:
a shared-memory ring feeder (`fenix train-feed`) and a family of label-quality QC tools
that stream/rasterize/audit `.fxsurf` corpus meshes against CT, consumed by the Python
training loop in `tools/train/`. See `docs/research/villa-ml.md`, `docs/design/
training-pipeline.md`, `docs/design/model-registry.md`, `docs/design/
multiscale-instance-surface.md`.

## Public API & key types
**Inference** (behind `FENIX_ML`, `ml_api.hpp`/`inference.cpp`):
- `run_predict_surface`, `run_predict_ink`, `run_predict_ink2d`, `run` (the `ml` subcommand
  dispatcher), `predict_surface_window(VolumeView<const u8> ct, weights_path) ->
  Expected<Volume<u8>>` — in-process single-window inference (weights cached by path),
  used by the streamed winding tracer as a per-window ML data term.
- Nets (`nets/`): `resenc_unet.hpp` (nnU-Net ResEnc-UNet + concurrent scSE, used for both
  the surface and DINO-guided ink models, config-driven scSE on/off), `resnet3d.hpp`
  (`ResNet3DInk50` — Bottleneck3D backbone [3,4,6,3] + max-over-depth + 2D decoder),
  `dinovol.hpp` (`DinoVol`/`DinoVolConfig` — from-scratch 3D RoPE/SwiGLU ViT reimpl of
  `dinovol_v2`; reachable today only via the `fenix ml dino-raw` diagnostic hook, not yet
  wired into a production predict stage).
- `tiling.hpp` (torch-free, unit-tested) + `infer.hpp` — sliding-window inference: z-score,
  fp16 forward, softmax/sigmoid, Gaussian-blended overlap, TTA (octahedral-48 group).
- `weights.hpp` — reader for the hand-rolled `.fxweights` flat file (`FXWT` header +
  named/typed/shaped blobs, dtypes f32/f16/bf16/f64/i64/i32/u8/i16/bool); mmap + torch::from_blob.
- `trt_engine.hpp` — `TrtNet` adapter over a built TensorRT `.plan`/`.engine`.

**Training data plane** (torch-free, always built — the bulk of this module by file count):
- `sampler.hpp` — `PatchSampler`/`PatchDraw`: deterministic occupancy-guided draw(seed,i),
  mesh ∝ valid-cell count, cell ~uniform over valid cells + jitter; optional `locality`
  clustering for cache-friendly consecutive draws.
- `surface_index.hpp` — `SurfaceIndex`/`VolumeSurfaceIndex`: two-level R-tree (whole-mesh
  bbox, then 16×16-cell uv-tile bbox) — "which meshes/uv-tiles touch this box" in ~log time.
- `rasterize.hpp` — `rasterize_band_multi`/`rasterize_band`: stamps a sphere per sampled
  surface point into a tri-state (255 sheet / 128 trusted-bg / 0 ignore) label volume,
  union over every mesh touching the patch; instance mode adds colored-sheet classes
  `200+(wrap mod k)` and unverified-instance class `150` (`kLabelSheetUncolored`);
  `label_is_sheet()` covers all sheet variants. Also `TrustGrid`/`read_trust` (per-uv-tile
  QC mask, `fxtrust1` text format) consumed here to blank FAIL tiles to ignore.
- `feed.hpp` — **`fenix train-feed`**: multi-threaded producer into an mmap'd SHM ring
  (`FXRING1` header, fixed-size slots, FREE/WRITING/READY protocol via `std::atomic_ref`)
  that `tools/train/feed_reader.py` consumes zero-copy as numpy. Reads a `pairs.txt`
  (`FeedPair`: fxsurf, ct, optional teacher/trust/wrap sidecars, `um=`/`msc=`/`canon=`/
  `shift=` resampling+correction knobs), groups meshes by CT volume (union-labels every
  mesh sharing a training chunk), gathers CT (native or on-demand zarr-backed
  `io::CachedVolume`), rasterizes GT, applies Otsu-valley-disciplined background
  harvesting/veto (anchored per-patch or volume-level fallback), optional teacher gather,
  augmentation (`aug=0/1/2`), and data-echoing (`echo=`) before publishing to a slot.
  Ships prefetcher threads that walk the deterministic draw sequence ahead of workers.
  Writes a `<ring>.meshes` id map for per-mesh loss telemetry.
- `augment.hpp`/`augment_cli.hpp` — deterministic seedable train-time transforms on
  `Volume<f32>` (+paired label): octahedral (exact 48-sym), rotate_z, elastic, intensity,
  ct_degrade, compression, lowres, an opt-in full-SO(3) knob. `fenix augment` CLI applies
  one op or the full chain to a `.fxvol` for inspection.
- `ingest_band.hpp` — **`fenix band-blocks`** (plan sub-crop grid touching a surface
  band) + **`fenix ingest-band`** (sparse zarr ingest of only near-band 128³ groups) —
  the band-limited data plane for teacher sweeps (~3 orders less volume than whole-bbox).
- `surfaces_cli.hpp` — **`fenix surfaces`**: spatial-index query + optional GT rasterize
  of a box, CLI wrapper over `surface_index`/`rasterize`.

**Label-quality QC (four independent oracles, all torch-free)**:
- `surf_qc.hpp` — **`fenix surf-qc`**: intensity delta-mode (per-mesh PASS/FAIL, or
  `regions=` per-uv-tile trust grid) and profile-mode (nearest-prominent-peak offset
  estimator, ridge/edge/embedded/AIR classification, `offsets=` field dump); also hosts
  `stencil_normal` and `air_edge` (sub-voxel air→material alpha-crossing localization)
  shared by repair/consist/audit.
- `surf_consist.hpp` — **`fenix surf-consist`**: inter-mesh geometric consistency
  (AGREE/OFFSET/CROSS verdicts via point-to-surface distance) + per-mesh normal coherence;
  no CT needed.
- `surf_repair.hpp` — **`fenix surf-repair`**: snap-to-ridge or snap-to-air-edge (`mode=
  ridge|alpha`) per-uv offset-field repair, `side=near|recto|verso` umbilicus-aware face
  disambiguation, `upsample=` grid densification; alpha mode also runs "model-alpha" against
  a sheet-probability `.fxvol` for papyrus-contact repair.
- `label_audit.hpp` — **`fenix label-audit`**: model-vs-mesh disagreement mining
  (per-uv-tile miss%, worst-first TSV) — training-dynamics oracle made queryable.
- Supporting human-in-the-loop tools: `surf_sheet.hpp` (**`fenix surf-sheet`** contact-sheet
  JPEG QC), `stroke_score.hpp` (**`fenix stroke-score`**: blind human strokes vs corpus
  meshes), `qc_chunk.hpp` (**`fenix qc-chunk`**: 3D triage chunk export for the WebGL
  raycaster viewer).

**Model registry** (TOML, `docs/design/model-registry.md`): name, version, TorchScript/
`.fxweights`/`.plan` path, task, I/O tensor specs, patch size, precision.

## Inputs / outputs & formats
In: `.fxvol` volumes (or on-demand zarr-backed `io::CachedVolume`), `.fxsurf` label
meshes + sidecars (`.trust` fxtrust1, `.wrapcolor` fxwcol1), TorchScript/`.fxweights`/
TensorRT `.plan` models, `pairs.txt` corpus manifests. Out: prediction `.fxvol`s
(sheet-prob/normals/distance/ink) → `predictions`; SHM training ring (`FXRING1`) +
`<ring>.meshes`; QC artifacts (trust grids, TSV audits, JPEG contact sheets, `.qcchunk`).

## Dependencies
Intra: `core`, `io`, `codec`, `geom` (R-tree), `preprocess` (Otsu/aircut), `annotate`
(stroke_score). Third-party: **libtorch** (firewalled), **TensorRT** (ADR 0010, firewalled
alongside torch). GPU: multi-GPU-aware; DDP training via `torchrun` (Python side).
WATCH-ITEM: prebuilt libtorch is glibc-based → under musl/Chimera needs a source build.

## Invariants & numerics
Reproducibility-by-default where it matters (fixed seeds, deterministic sampler/feeder —
`draw(seed,i)` and aug seeds are pure functions of index, so runs are resumable and
producer threads partition without coordination). ML math is float/mixed-precision, not
bit-exact. **Training grid is canonically 2.4 µm** (decided 2026-07-02); non-canonical
sources are resampled at feed time (CT/teacher trilinear, GT scaled-then-rasterized —
labels are never interpolated). Augmented CT written to disk stays round-clamped u8, never
f32 (hard project rule). Label semantics: 255 sheet / 128 trusted-background / 0
unlabeled-ignore; instance mode: 150 sheet-unknown-instance / 200+c colored.

## Performance notes
GPU-bound inference; occupancy/mesh-guided sampling avoids decoding empty regions.
Feeder: per-volume decoded-block cache sized above the 256 MiB default (thrashes under
training — measured 81ms→336ms warm-gather degradation), `locality=` clustering turns
cold-feed scatter into fetch-once-use-L-times, `echo=` amortizes gather/raster over K
augmented emissions, prefetcher threads walk the deterministic draw sequence ahead of
workers, `FENIX_ML_ZARR_FETCH_THREADS` capped per-fill to avoid self-congesting S3.
Workers/prefetchers run in `SerialRegion` (nested OpenMP oversubscription measured ~10x
regression otherwise). TensorRT path: 1.53–1.65× over eager fp16 (ADR 0010); static shape
caps batch at patch size. TTA is the single biggest surface-quality lever measured
(corr +0.1–0.17 on 0125).

## Gotchas / pitfalls
- ML is **optional/firewalled** — the classical pipeline must work without it.
- **Never `#include` a torch header** (torch_env/infer/nets/weights/trt_engine) outside
  `inference.cpp` — breaks the build firewall (ADR 0008).
- Both 2D ink nets emit logits at input/4 — must resize probs to tile size before Hann
  blending or tiles paste as unscaled quarter-fills (fixed 2026-07-04).
- Corpus `.fxsurf` meshes are routinely misregistered/warped, not just offset — hence the
  four QC oracles; don't trust a mesh's supervision without running at least surf-qc.
  Bg-shell rasterization needs an Otsu sheet ANCHOR before harvesting background — a
  geometric shell alone can land on a neighboring wrap and collapse training to constant
  output (root-caused 2026-07-03).
- Ring attach must be attach-or-create, never O_TRUNC-on-exists — truncating a live
  consumer's mmap SIGBUSes the training loop.
- Absolute mod-k wrap color is **not locally inferable** from a single training patch (a
  128-vox patch spans ~1 wrap, no anchor for which-of-k) — instance-mode CE must be
  gauge-invariant (min over cyclic shifts), not raw class CE (run-1 finding, 2026-07-05).

## Build firewall (ADR 0008)
**libtorch is parsed in exactly ONE TU: `inference.cpp`.** `ml_api.hpp` is the torch-free
public surface (torch-free stubs when `!FENIX_ML`); `ml.hpp` only registers stages
(torch-free — pulled into `fenix.hpp`) and includes all the always-built torch-free
stages (`augment_cli`, `feed`, `ingest_band`, `label_audit`, `qc_chunk`, `stroke_score`,
`surf_consist`, `surf_qc`, `surf_repair`, `surf_sheet`, `surfaces_cli`); `inference.cpp`
includes `<torch/torch.h>` + nets/weights/infer/trt_engine and defines the entry points.
CMake adds `inference.cpp` to the unity ML target; split build globs it.

## Build / runtime
**Ubuntu/glibc (GPU boxes):** prebuilt CUDA libtorch (`Dockerfile.ml.ubuntu` /
`install-ubuntu.sh --ml`); deps.cmake links plain `.so`s (not `find_package(Torch)` — its
`enable_language(CUDA)` breaks under glibc 2.43 vs CUDA 13 headers). FENIX_ML switches the
toolchain to libstdc++ + exceptions/RTTI (libtorch ABI). **Chimera/musl:** CPU-only source
build (`Dockerfile.ml`, best-effort). VRAM: a 256³ patch needs >8 GB → `patch=128` on 8 GB
cards. TensorRT: `-DFENIX_TRT_LIB=…/libnvinfer.so.11 -DFENIX_TRT_INCLUDE=/opt/trt-include`,
engines built per-box (`tools/ml-export/build_engine.py`), TRT-version/arch-locked.

## Implemented models
- **Surface** (`surface_recto_3dunet`): ResEnc-UNet + scSE, validated bit-identical vs
  PyTorch reference. `fenix predict-surface <in> <weights> <out> [patch] [overlap] [tta]`;
  also reachable per-window via `predict_surface_window` for the streamed tracer.
- **3D ink** (`ink_3d_dino_guided`): same ResEnc-UNet, no scSE, `shared_decoder` + single
  1×1 `task_heads.ink` conv, percentile min-max norm, sigmoid, `ema_model` weights.
  Validated bit-identical. `fenix predict-ink <in> <weights> <out> [patch] [overlap]`.
- **2D ink trio** (all three released ScrollPrize ink models, native): `ink_3d_dino_guided`
  via `predict-ink`; `ink_canonical_2um` (r152 + 3D-FPN + depth-attention, 62-layer window,
  2µm-native) and `resnet50_3um_01122024` (`ResNet3DInk50`, 18-layer window, 256² tiles,
  3µm-native) via `predict-ink2d net=r152|r50`. r50 validated vs HF ref: max|Δ| 4.4e-4,
  corr 0.999998.
- **TensorRT path** (ADR 0010): `predict-surface` accepts `.plan`/`.engine` weights via
  `TrtNet`; engine batch/patch override the CLI's.
- **DinoVol backbone** (`dinovol_v2`): net implemented and exercised via `fenix ml
  dino-raw <weights> <in.raw> <out.raw> <D> <H> <W>`; not yet wired into a production
  predict stage or the training loop.

## Training pipeline & instance segmentation (docs/design/training-pipeline.md,
## multiscale-instance-surface.md — status 2026-07-05)
Binary surface training is mature: `tools/train/train.py` (multi-GPU DDP via `torchrun`,
gauge-invariant-aware), `feed_reader.py` (ring consumer), `inst_train_supervisor.sh` +
`tame_feeders.sh` (restart/health supervision — feeder deaths auto-recover mid-run).
**Instance (per-wrap color) mode is live but young**: `train-feed wrapk=` paints
per-cell wrap colors from `.wrapcolor` sidecars; `train.py --wrapk` trains a (k+1)-class
head. First full instance run (45 meshes, k=8) hit VAL color acc 0.0 despite falling CE —
root-caused to absolute mod-k not being locally inferable; the fix in flight is
gauge-invariant CE (min over cyclic shifts) — see the gotcha above and the multiscale doc
for the P0-P5 rollout plan (rungs: native → 2×/4×/8× → 16-128× density-only).

## TODO
Wire `DinoVol` into a production predict/train path (currently diagnostic-only). Ship
gauge-invariant CE for instance training (run-1 fix, in progress). Out-of-core accumulators
for whole-scroll inference. Quantized conv3d (int8/fp8/fp4): dead in the EXPORT toolchain,
but a hand-written FUSED fp8 implicit-GEMM conv3d wins 2.85–4.13× over cuDNN fp16 on sm120
(2026-07-06 spike, `docs/design/fp8-conv3d-sm120.md` + `tools/ml-export/fp8_conv3d_*.py`);
production path (Fp8Net resident net + calibration + SurfaceDice check) is open work. Open ADRs: model registry schema;
training data schema.
