# ml — CLAUDE.md

## Purpose
In-tree neural inference **and training** via **libtorch (C++)**. Firewalled behind this
module — the rest of fenix consumes its outputs as prediction fields (via `predictions`).
See `docs/research/villa-ml.md` (segmentation/nnU-Net/lasagna), `villa-thaumato.md`
(Mask3D), ink-detection contract.

## Public API & key types
- **Inference** for: surface/sheet + **normals** + distance prediction, **instance
  segmentation** (Mask3D-style), **ink detection** (TimeSformer-style).
- **Training** (in-tree, libtorch): training loop reading patches from `.fxvol` +
  surface/label containers via the **occupancy-guided sampler** (material-rich, rejection-
  free) with deterministic augmentation.
- **Model registry** (TOML): name, version, TorchScript path, task, I/O tensor specs
  (shape/dtype/normalization), patch size, precision, provenance. Numeric formats:
  **f32 + f16 + bf16** (mixed precision).

## Inputs / outputs & formats
In: `.fxvol` volumes + label/surface containers + TorchScript models. Out: prediction
`.fxvol`s (sheet-prob/normals/distance/ink) → `predictions`.

## Dependencies
Intra: `core`, `io`, `codec`, `geom`. Third-party: **libtorch** (the ML dep; firewalled).
WATCH-ITEM: prebuilt libtorch is glibc-based → under musl/Chimera may need source build /
glibc-compat. GPU: multi-GPU-aware (single node).

## Invariants & numerics
Reproducibility-by-default for training where it matters (fixed seeds, stable eval); ML
math is float/mixed-precision (not bit-exact). Keep preprocessing/normalization stable.

## Performance notes
GPU-bound; the occupancy sampler avoids decoding empty regions; out-of-core via the chunk
streamer. Inference batched; the custom uint8 trilinear sampler (with analytic backward)
lives in `core`.

## Gotchas / pitfalls
ML is **optional/firewalled** — the classical pipeline must work without it (predictions
can come from `segment` detectors instead). Don't leak torch types past this module's API.

## Implemented (surface model)
- `torch_env.hpp` — single libtorch include point (device/dtype selection).
- `weights.hpp` — reader for the hand-rolled `.fxweights` flat file + `load_into(module)` by name.
- `nets/resenc_unet.hpp` — **from-scratch `torch::nn` reimplementation** of the nnU-Net
  ResEnc-UNet + concurrent scSE (the `surface_recto_3dunet` arch). Param names mirror the
  checkpoint exactly; **validated bit-identical** against an authoritative PyTorch reference
  (`tools/ml-export/reference.py`, real upstream `dynamic_network_architectures` blocks).
- `tiling.hpp` (torch-free, unit-tested) + `infer.hpp` — sliding-window inference: per-patch
  z-score, fp16 GPU forward, softmax, Gaussian-blended overlap.
- Stages: `fenix predict-surface <in.fxvol|.nrrd> <surface.fxweights> <out> [patch] [overlap]`;
  `fenix ml [info|load-surface|run-raw]` for diagnostics/validation.
- Weight export tooling: `tools/ml-export/` (introspect / convert_weights / reference).

## Build / runtime
**Ubuntu/glibc (this project's GPU boxes):** prebuilt CUDA libtorch (`Dockerfile.ml.ubuntu` /
`install-ubuntu.sh --ml`); deps.cmake links it as plain `.so`s (NOT `find_package(Torch)` — its
`enable_language(CUDA)` is broken by glibc 2.43 vs CUDA 13 headers). FENIX_ML builds switch the
toolchain to libstdc++ + exceptions/RTTI (libtorch ABI). **Chimera/musl:** CPU-only source build
(`Dockerfile.ml`, best-effort). VRAM: a 256³ patch needs >8 GB → use `patch=128` on 8 GB cards.

## Implemented (ink model)
`scrollprize/ink_3d_dino_guided` — same `nets/resenc_unet.hpp` (config-driven): **no scSE**
(plain residual blocks), `shared_decoder` + a single `task_heads.ink` 1×1 conv, percentile-
(0.5/99.5) min-max norm, 1-ch **sigmoid**. `ema_model` weights. **Validated bit-identical** vs
the Python reference. Stage: `fenix predict-ink <in> <ink.fxweights> <out> [patch] [overlap]`.
The DINOv2 guidance was a *training* signal; inference is just the U-Net.

## TODO
The 3D-RoPE DINOv2 backbone (`dinovol_v2`) — same export+reimpl pattern, but a ViT (3D RoPE,
SwiGLU, register tokens), the largest piece. Out-of-core accumulators for whole-scroll
inference; training loop. Open ADRs: model registry schema; training data schema.
