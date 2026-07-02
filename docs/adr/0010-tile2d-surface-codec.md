# ADR 0010 — 2D DCT-64 tile codec for surfaces (.fxsurf v3)

## Status
Accepted (2026-07-02).

## Context
`.fxsurf` v1 stored coords as raw f32 (12 B/cell). v2 (same day, superseded) used
quantize-1/16 + plane prediction + rANS and reached 9.1× on a real PHercParis4 segment —
but predictive coding of quantized values pays full price for sub-tolerance noise, and
measurement showed v2 was already at the data's entropy at that tolerance.

## Decision
A generic 2D transform codec, `codec/tile2d.hpp`: 64×64 DCT (`codec/dct64.hpp`, orthonormal,
same family as the volume DCT-16) + dead-zone quantization + zigzag/nsig + the shared rANS
lossless substrate. Correctness is tolerance-only and **verified at encode time** per tile
(q-halving up to 4×, then a raw-quantized fallback at τ/4) — never assumed. Three front-ends:
- **scalar** (`encode_field2d`) — texture layers, height/conf fields, validity-aware fill;
- **RGB** (`encode_rgb2d`) — YCoCg decorrelation, per-channel τ/2;
- **coords** (`encode_coords2d`) — the .fxsurf case: per-tile least-squares affine
  (stored quantized 1/16 vox) + tangent-frame projection derived deterministically from the
  quantized affine; 3 correlated channels → 1 height field + 2 near-zero remainders;
  per-channel τ/√3 gives an exact 3D bound (orthonormal frame).

`.fxsurf` v3 uses coords + scalar front-ends; `write_fxsurf` takes `coord_tau`
(default **1/4 voxel** — GT surfaces are only ~±1 voxel accurate, so 1/16 preserves
segmentation noise).

## Measured (real 8512×2484 PHercParis4 segment, 87.7% valid, vs 254 MB tifxyz)
τ=1/16: 9.7× · 1/8: 12.3× · **1/4: 16.1× (default)** · 1/2: 21.6× · 1: 29.5×; mean error at
1/4 is 0.039 vox. Whole 336-mesh training corpus ≈ 1.5 GB tifxyz → ~100 MB.

## Consequences
- This is deliberately a SECOND transform codec (2D) next to the volume DCT-16 tile codec
  (the codec CLAUDE.md guardrail requires this ADR). It reuses the shared substrate
  (rANS/lossless, quant philosophy) and does NOT fork the 3D path.
- Found & fixed en route: `core/vec.hpp cross()` had a broken x-component (`b.x` where
  `b.z` belongs) — every tangent frame/normal/triangle-area in the codebase was silently
  wrong; the tile2d roundtrip test is what caught it.
- v1/v2 files are rejected (project no-compat rule); regenerate by re-importing.
