// core/tolerances.hpp — named, documented numeric constants. NO scattered magic literals
// (docs/conventions.md). Tune here, not at call sites. Fast-math is on globally, so these
// are the explicit slack we reason with.
#pragma once

#include "core/types.hpp"

namespace fenix::tol {

// Generic geometric epsilon for "is this ~zero / are these ~equal" in voxel units.
inline constexpr f32 eps = 1e-5f;

// Smaller epsilon for normalized directions / dot products.
inline constexpr f32 dir_eps = 1e-6f;

// Default near-lossless codec max-error (voxel levels) — the τ floor of the quality ladder.
inline constexpr f32 default_tau = 1.0f;

// Iterative-solver convergence: stop when the max per-element update drops below this.
inline constexpr f32 converge_eps = 1e-4f;

// Jacobian determinant floor below which a deformation cell is considered folded/inverted.
inline constexpr f32 fold_det_floor = 1e-3f;

}  // namespace fenix::tol
