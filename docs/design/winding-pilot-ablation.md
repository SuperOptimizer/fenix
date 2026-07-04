# The first real-scroll spiral fits — the wind3..7 ablation (PHercParis4, 2026-07-04)

The `winding` stage's first runs on real data: 20 corpus meshes (25 for holdout runs),
stride 8, corpus bridge (per-cell continuous winding from unwrapped mesh turns),
`fit_spiral_diffeo`. Each run isolated one fix; each verdict drove the next commit.

## Runs

| run | axis | objective | capacity | spacing (vox/wrap) | train RMSE (windings) | holdout RMSE |
|-----|------|-----------|----------|--------------------|-----------------------|--------------|
| wind1 (patch bridge) | straight | stepped | global affine + 6×12×12 flow | 4.8 (bogus) | 3.27 (degenerate: all wraps=0) | — |
| wind3 | straight @16346,16346 | stepped (sawtooth) | same | 410.9 (inflated) | 2.98 | — |
| wind4 | **curved** (estimated) | stepped (sawtooth) | same | **102.4 (physical, ~0.25 mm)** | 8.58 | — |
| wind5 | curved | **continuous** | same | 101.3 | 9.02 | **11.16** (385k cells, 5 meshes) |
| wind6 | curved | continuous | **+24 affine bands, 16³ flow** | 101.3 | 6.90 (CORRUPTED — see below) | segfault |
| wind7 | curved | continuous | +24 bands, 16³ flow (FIXED) | 101.3 | 6.91 | **9.63** (0 nonfinite) |
| wind8 | curved | continuous | wind7 + regauge=3 (EM) | 101.3 | 4.92 | **6.65** |

## What each step proved
- **The patch bridge is the wrong model for corpus meshes** (wind1): GP segments wind
  many turns; one-integer-per-patch collapses everything to wrap 0. → corpus bridge
  (unwrap theta/2π per cell; each mesh measures its own pitch via Δr at turn+1).
- **The straight axis was the dominant error** (wind3→4): the estimated umbilicus
  drifts **y 7,878 / x 4,932 voxels** over z 2.7k–74k. A far-off axis inflates radii →
  measured spacing 410.9 (nonsense); the curved axis gives 102 vox ≈ 0.25 mm (physical)
  and spreads the meshes over wraps ~11–71.
- **The stepped readout vs continuous targets left a ±0.5 sawtooth** (wind4→5):
  `winding_at` is the wrap INDEX (branch-cut steps); corpus targets are continuous.
  `winding_cont` (no θ term) removes the floor and the cut-adjacent gradient corruption.
- **holdout ≈ train ⇒ UNDERFIT, not overfit** (wind5): 9.0 train / 11.2 holdout over a
  ~60-wrap span (~15–18% relative). One global affine + one flow z-control-point per
  ~12k slices cannot bend a 72k-z scroll. → P3 capacity: per-z-band residual affines
  (24 bands) + flowz=16.
- **wind6's numbers are invalid**: the post-stage parameter writeback still used the
  pre-bands offset — 6B floats overflowed into flow.vx (ASan-confirmed heap overflow =
  the segfault) AND band params leaked into the lattice (bimodal fit results). The
  companion NaN lesson: a diverged fit puts NaN windings into holdout residuals and
  `nth_element` with NaN comparisons is UB; `std::isfinite` constant-folds away under
  `-ffast-math` — use a magnitude bound (`detail::finite_fm`).

## Measured lever ranking (holdout RMSE, 385k never-seen cells)
wind5 11.16 → wind7 (+bands/flow capacity) 9.63 (−14%) → wind8 (+gauge EM ×3) **6.65**
(−40% total). The EM re-gauging was the single biggest lever: initial per-component bases
were off by up to **3.6 windings** (round-1 max shift), and round 3 still shifted 2.66 —
NOT converged; deeper EM is free improvement.

## Open levers (in expected-value order)
1. **Deeper regauge** (rounds still shifting ~2.7 windings at 3) + interleave earlier.
2. **Gap-expander fitting** — logits still frozen (backward assumes identity);
   per-winding spacing is real (compression varies radially).
3. Constraint balance: dense meshes dominate; per-component weight normalization.
4. More capacity only after 1-2 plateau (capacity was the smaller lever).

## Reproduction
```
fenix umbilicus surf=<25 meshes> out=p4_umb.toml band=2048 stride=8
fenix winding surf=<25 meshes> umb=p4_umb.toml stride=8 bridge=corpus bstride=4 \
              holdout=5 abands=24 flowz=16 flow=16 iters_flow=800 out=corpus.fxmodel
fenix flatten model=corpus.fxmodel out=wrap nu=512 zstep=32
```
