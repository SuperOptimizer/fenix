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

## Status & TODO
STUB. **Image: `Dockerfile.ml`** — libtorch is glibc-based and NOT in Chimera's repos. The
musl-pure path is a CPU-only source build against musl+libc++ via `cmake/scripts/build-libtorch.sh`
(the same script CMake's `auto`/`source` resolver runs; resolves as `fenix::torch`). It is
best-effort — PyTorch assumes glibc in spots — with gcompat-shim / glibc-base fallbacks
documented in `docs/design/docker.md`. ML is opt-in (`-DFENIX_ML=ON`); the core image never
depends on this. Open ADRs: libtorch-on-musl path; model formats; training data schema.
