// winding/spiral_model.hpp — the unified deformation model: a compositional diffeomorphism
// T: scroll-space <-> straightened spiral-space, built from invertible factors (umbilicus
// shift ∘ SVF flow ∘ per-slice affine ∘ radial gap-expander) + a scalar dr_per_winding.
// shifted_radius/dr gives the continuous winding. This is the .fxmodel core (spiral-v2). All
// factors preserve z and are individually invertible -> the composite is globally fold-free.
// See winding/CLAUDE.md, docs/research/spiral-v2.md.
#pragma once

#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "winding/flow.hpp"
#include "winding/transforms.hpp"

#include <cmath>
#include <numbers>

namespace fenix::winding {

struct SpiralModel {
    annotate::Umbilicus umbilicus;
    FlowField flow;
    int flow_steps = 8;
    bool has_flow = false;
    AffineYX affine;       // per-slice affine (single global here; per-z extension later)
    GapExpander gap;       // radial per-winding scale
    f32 dr_per_winding = 8.0f;
    f32 winding_offset = 0.0f;  // additive winding gauge: the Archimedean readout is ABSOLUTE (r/dr from
                                // the umbilicus) but assigned windings are gauge-relative (0-based per
                                // region); this free scalar reconciles them so dr/affine/flow aren't
                                // hijacked absorbing a large constant. Pure gauge (a constant winding shift).

    // scroll-space voxel -> straightened cartesian (axis-centred, deflowed, de-affined).
    [[nodiscard]] Vec3f to_canonical(Vec3f p) const {
        const Vec3f c = umbilicus.empty() ? Vec3f{p.z, 0, 0} : umbilicus.center(p.z);
        Vec3f q{p.z, p.y - c.y, p.x - c.x};
        if (has_flow) q = flow_point(flow, q, flow_steps, -1.0f);
        return affine.inverse(q);
    }

    // straightened cartesian -> scroll-space voxel (exact inverse of to_canonical).
    [[nodiscard]] Vec3f to_scroll(Vec3f q) const {
        Vec3f p = affine.apply(q);
        if (has_flow) p = flow_point(flow, p, flow_steps, +1.0f);
        const Vec3f c = umbilicus.empty() ? Vec3f{p.z, 0, 0} : umbilicus.center(p.z);
        return {p.z, p.y + c.y, p.x + c.x};
    }

    // Winding number of a scroll-space point: shifted_radius / dr, where
    // shifted_radius = gap_inverse(canonical radius) - dr*theta/2pi (Archimedean). On the
    // spiral this is the integer WRAP INDEX — constant per wrap, stepping at the theta
    // branch cut. Level sets of this are the physical wraps (flatten extracts them).
    [[nodiscard]] f32 winding_at(Vec3f p) const {
        constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
        const Vec3f q = to_canonical(p);
        const f32 r = std::sqrt(q.y * q.y + q.x * q.x);
        const f32 r_ideal = gap.inverse(r);
        const f32 theta = std::atan2(q.y, q.x);
        const f32 shifted_radius = r_ideal - dr_per_winding * theta / two_pi;
        return shifted_radius / dr_per_winding + winding_offset;
    }

    // CONTINUOUS winding coordinate: gap_inverse(r)/dr, no theta term — smooth everywhere
    // (no branch cut). Along a spiral surface this equals the surface's own unwrapped turn
    // + a constant, so it is the readout to fit against CONTINUOUS targets (corpus bridge)
    // and to score held-out meshes; winding_at against such targets leaves an irreducible
    // ±0.5 sawtooth. The Archimedean theta-coupling is carried by the constraints.
    [[nodiscard]] f32 winding_cont(Vec3f p) const {
        const Vec3f q = to_canonical(p);
        const f32 r = std::sqrt(q.y * q.y + q.x * q.x);
        return gap.inverse(r) / dr_per_winding + winding_offset;
    }
};

}  // namespace fenix::winding
