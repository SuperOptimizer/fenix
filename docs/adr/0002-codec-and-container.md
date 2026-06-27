# ADR 0002 — Codec and Container

**Status:** Accepted (2026-06-27). Supersedes the initial dual-codec (DCT-16³ + wavelet)
idea — **DCT was dropped**.

## Context
Scroll volumes are up to 2¹⁸/axis (PHerc Paris 3 ≈ 70k×40k×40k, ~33% dense), tens of TB,
never resident. We need lossy compression spanning near-lossless → 1000×, progressive over
the network, with rejection-free ML sampling — all out-of-core. Lineages: matter-compressor
(`.mca` container) and c3d (3D CDF 9/7 wavelet). Both MIT; rewrite ground-up.

## Decision
- **One dim-parameterized CDF 9/7 wavelet core** (lifting + bitplane + 8-way interleaved
  rANS), instantiated for **3D-64³** (volumes/prediction fields) and **2D-64²** (images/
  parametric surfaces/texture layers). **Bitplane-progressive in LOD + quality.** Dtypes
  u8/u16/u32/s8/s16/s32/f16/f32 (no f64/u64/s64 dense). **No DCT anywhere.**
- **General lossless codec** (rANS + delta/RLE/bitpacking) for label volumes, masks, exact
  priors. Selected per-archive via a codec-version field.
- **`.fxvol` container:** 64³ chunk = network-IO unit; 2-level page table (2⁶/axis nodes,
  top sparse / 2nd dense / chunk dense) over 2¹² chunks/axis; slot = u64 offset + tri-state
  coverage (NOT_SURE/ZERO/REAL); append-at-EOF + release-store commit + fallocate; self-
  describing chunk headers; adaptive whole/partial fetch.
- **Tolerance-only, non-deterministic** (fast-math/float throughout; correctness = τ/PSNR).
  **SIMD + GPU first-class** (rANS, not serial CABAC). **No-UB-on-any-bytes** hard rule.
- Multi-channel = one scalar archive per channel. Image interop: own PNG/JPEG/TIFF r/w.

## Consequences
+ One codebase for 2D/3D; native LOD + SNR progressiveness; cheap random access + air-skip;
  crash-safe append-while-readable; GPU-friendly.
− No cross-ISA byte reproducibility (accepted). 64³ raises per-subband table overhead vs
  c3d's 256³ → must amortize (merge small-subband tables, drop HF levels at high ratio).
− Near-lossless served by low-ratio wavelet (no separate DCT archival path).
