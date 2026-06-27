// winding/flow.hpp — stationary velocity field (SVF) integrated by RK4: the diffeomorphic
// backbone of the unified unrolling method (spiral-v2 IntegratedFlowDiffeomorphism). The
// flow of a smooth ODE is a diffeomorphism; its inverse is integration with negated time,
// guaranteeing a globally invertible, fold-free map. See winding/CLAUDE.md, spiral-v2.md.
#pragma once

#include "core/core.hpp"

namespace fenix::winding {

// Velocity field as three ZYX-aligned component volumes on a (possibly coarse) lattice,
// trilinearly sampled in voxel coordinates.
struct FlowField {
    Volume<f32> vz, vy, vx;

    [[nodiscard]] Vec3f velocity(Vec3f p) const {
        return {sample_trilinear(vz.view(), p), sample_trilinear(vy.view(), p),
                sample_trilinear(vx.view(), p)};
    }
};

// Integrate point `p` through the stationary flow over normalized time [0,1] in `steps`
// RK4 steps. sign=+1 forward (T), sign=-1 inverse (T^-1). Forward then inverse ~= identity.
inline Vec3f flow_point(const FlowField& f, Vec3f p, int steps, f32 sign) {
    const f32 dt = sign / static_cast<f32>(steps);
    for (int i = 0; i < steps; ++i) {
        const Vec3f k1 = f.velocity(p);
        const Vec3f k2 = f.velocity(p + k1 * (dt * 0.5f));
        const Vec3f k3 = f.velocity(p + k2 * (dt * 0.5f));
        const Vec3f k4 = f.velocity(p + k3 * dt);
        p = p + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * (dt / 6.0f);
    }
    return p;
}

}  // namespace fenix::winding
