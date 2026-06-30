# ADR 0005 — Retire the CDF 9/7 wavelet; the DCT-16 tile codec is the sole transform codec

**Status:** Accepted (2026-06-30). **Supersedes the codec choice in [ADR 0002](0002-codec-and-container.md)**
(which had us keep two selectable transform codecs, wavelet + DCT, pending a head-to-head). The
container design from ADR 0002 (`.fxvol`, 64³ chunks, 2-level page table, coverage tri-state,
crash-safe append) is **carried forward unchanged** — only the per-chunk transform codec is settled.

## Context
ADR 0002 (amended 2026-06-29) deliberately kept BOTH a CDF 9/7 wavelet and a separable all-float
DCT-16, to be compared head-to-head on real CT before settling. We ran that comparison and then
optimized both codecs (measured on a 512³ PHerc Paris 4 smooth crop and a 1024³ PHerc0211 dense
scroll-centre region, both 8-bit CT, PSNR-matched throughout):

- DCT structural wins, all quality-neutral: constant-header hoist, DC zigzag-varint, end-of-block
  (frequency-scan), per-block step table, **tile-global rANS tables** (a 64³ tile = 4³ DCT blocks
  share one set of tables — the big one, since per-16³-block tables were ~73% of the payload at high
  q), and a neighbour-magnitude-sum significance context. Cumulative vs the DCT v1: **+63%/+151%/+394%**
  (crop512 q2/q8/q32) and **+32%/+71%/+196%** (dense), with encode ~1.7→4.5 GB/s and decode ~2.4→4.5.
- The wavelet got per-band quant, per-scale context, dead-zone, magnitude-category coding, and a
  vectorized DWT earlier; its per-64³-block tables were already amortized, so it had little headroom
  left from the same tricks (~5-10% at most).

**Result — the tile-DCT now beats the wavelet at iso-quality across the ENTIRE measured range on BOTH
datasets**, e.g. on the dense region: ~40 dB 10.2× vs 6.3× (+62%), ~33 dB 24.2× vs 12.9× (+88%),
~26 dB 88.8× vs 50.6× (+75%) — and it encodes ~2× faster and decodes ~1.5× faster. The wavelet's only
remaining advantage was intrinsic LOD-progressiveness (subbands).

## Decision
- **Remove the CDF 9/7 wavelet transform and the wavelet block codec.** The **DCT-16 tile codec is the
  sole lossy transform codec.** The `.fxvol` archive encodes each 64³ chunk as one DCT tile (bpa=4).
- **LOD is served by an explicit multiscale pyramid**, not wavelet subbands. We had already concluded
  (in the LOD design discussion) that an explicit pyramid is *better* for the tiled / out-of-core
  navigation case: seam-free overviews (the downsample is computed across chunk boundaries before
  re-chunking), independent per-level quality, and trivial random access. So losing the wavelet's free
  subband LOD is not a regression — it's a path we preferred anyway.
- The shared entropy substrate (rANS, dead-zone quant, magnitude-category coding, the sparse-table +
  varint + bit-packer helpers, the u8..f32 dtype layer, the `.fxvol` container) is **kept** — the
  helpers moved from the deleted `block.hpp` into `codec/entropy.hpp`.
- `.fxvol` format version → **2**; readers reject v1 (no migration, per project policy).

## Alternatives considered
- **Keep both codecs (status quo).** Rejected: the DCT now dominates on ratio@quality and speed; a
  second codec is dead weight + a codec-version branch to maintain, with no regime where it wins.
- **Give the wavelet the tile-global treatment first.** Its tables already amortize over 262144
  coeffs, so the expected gain (~5-10%) cannot close the +62-88% gap — not worth deferring the decision.
- **Deprecate but keep the code dormant.** Rejected by the project owner in favour of a clean removal.

## Consequences
- One codec path, simpler container and tests; `BlockParams`→`DctParams` everywhere (archive, io, ml).
- **Open risk — DCT blocking/ringing at high compression.** Mitigated partly already (edge-replicated
  chunk padding killed ~60-unit boundary ringing → ~5). Future work if it bites at aggressive rates:
  decode-side deblock/dering or a lapped/overlap pre/post-filter (8-voxel halo) — see the SOTA
  transform research notes in codec/CLAUDE.md.
- **Validation breadth:** the DCT-wins verdict is from 2 regions (one smooth, one dense). Re-confirm
  opportunistically on more diverse scroll regions; the format-version gate makes a future re-decision
  clean if ever needed.
- Now-affordable follow-ons on the shared tables (still TODO): hybrid-uint small-level tokens,
  context-map clustering, RDOQ (which needs the trustworthy shared-table rate model).
