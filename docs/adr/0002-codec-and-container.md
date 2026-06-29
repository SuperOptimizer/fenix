# ADR 0002 — Codec and Container

**Status:** Accepted (2026-06-27); **amended 2026-06-29** — fenix supports **two selectable
transform codecs** (CDF 9/7 wavelet **and** a separable all-float DCT-16). The codec is still
being settled, so both are kept and compared head-to-head on real CT (PSNR/ratio); this
reverses the earlier DCT-drop. Both share the rANS entropy core, dead-zone quant, the
`.fxvol` container, and the u8..f32 dtype layer; the per-archive codec-version field selects.

## Context
Scroll volumes are up to 2¹⁸/axis (PHerc Paris 3 ≈ 70k×40k×40k, ~33% dense), tens of TB,
never resident. We need lossy compression spanning near-lossless → 1000×, progressive over
the network, with rejection-free ML sampling — all out-of-core. Lineages: matter-compressor
(`.mca` container) and c3d (3D CDF 9/7 wavelet). Both MIT; rewrite ground-up.

## Decision
- **Two lossy transform codecs**, selected per-archive via the codec-version field, sharing
  the rANS entropy core, dead-zone quant + magnitude-category coding, and the u8..f32 dtype
  layer (no f64/u64/s64 dense):
  - **CDF 9/7 wavelet core** (lifting + bitplane + interleaved rANS), dim-parameterized for
    **3D-64³** and **2D-64²**, **bitplane-progressive in LOD + quality**.
  - **Separable all-float DCT-16** (DCT-II, 16³/16² blocks), band-weighted dead-zone quant —
    a rewrite of matter-compressor's `mc_codec` (theirs was an *integer* DCT; ours is float,
    consistent with the tolerance-only/fast-math rule). Strong near-lossless / low-ratio.
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
− Two codecs = more surface to maintain; justified while settling (the head-to-head decides
  the default, or each wins a regime — DCT near-lossless/low-ratio, wavelet high-ratio +
  LOD-progressive). The shared rANS/quant/container/dtype layers keep the extra cost small.
