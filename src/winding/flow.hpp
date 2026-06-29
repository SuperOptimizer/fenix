// winding/flow.hpp — stationary velocity field (SVF) integrated by RK4: the diffeomorphic
// backbone of the unified unrolling method (spiral-v2 IntegratedFlowDiffeomorphism). The
// flow of a smooth ODE is a diffeomorphism; its inverse is integration with negated time,
// guaranteeing a globally invertible, fold-free map. `flow_point_backward` is the discrete
// adjoint of the unrolled RK4 — it scatters the gradient into the (coarse) velocity lattice,
// the core of the differentiable fit (diffeo_fit.hpp). See winding/CLAUDE.md, spiral-v2.md.
#pragma once

#include "core/core.hpp"
#include "core/sampling.hpp"

namespace fenix::winding {

// Velocity field as three ZYX-aligned component volumes on a (possibly COARSE) lattice. The lattice
// covers a domain box: a voxel point p is mapped to lattice coords by to_lat(p) = (p-lat_lo)*lat_scale
// (lat_scale = (L-1)/(hi-lo) per axis), then trilinearly sampled. Velocity VALUES are in voxel units
// (the RK4 integrates in voxel space). Defaults (lo=0, scale=1) make the lattice voxel-resolution,
// reproducing the original behaviour.
struct FlowField {
    Volume<f32> vz, vy, vx;
    Vec3f lat_lo{0, 0, 0};
    Vec3f lat_scale{1, 1, 1};

    [[nodiscard]] Vec3f to_lat(Vec3f p) const {
        return {(p.z - lat_lo.z) * lat_scale.z, (p.y - lat_lo.y) * lat_scale.y, (p.x - lat_lo.x) * lat_scale.x};
    }
    [[nodiscard]] Vec3f velocity(Vec3f p) const {
        const Vec3f l = to_lat(p);
        return {sample_trilinear(vz.view(), l), sample_trilinear(vy.view(), l), sample_trilinear(vx.view(), l)};
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

// Per-thread f64 gradient accumulators into the velocity lattice (one array per component, sized
// vz.dims().count(); indexed by the flat offset from trilinear_stencil — == linear index for a
// contiguous Volume). Reduced across threads by the fit.
struct FlowGradAccum {
    f64* gz = nullptr;
    f64* gy = nullptr;
    f64* gx = nullptr;
};

namespace detail {
// One velocity sample's backward: scatter the velocity-adjoint kbar into the lattice grad (∂v/∂corner
// = trilinear weight) AND return the position-adjoint Jᵀ·kbar (J = ∂v/∂voxel = lattice-grad ⊙ lat_scale).
inline Vec3f flow_sample_backward(const FlowField& f, Vec3f s, Vec3f kbar, FlowGradAccum& ga) {
    const Vec3f l = f.to_lat(s);
    const TrilinearStencil st = trilinear_stencil(f.vz.view(), l);  // same lattice dims for all 3 components
    for (int m = 0; m < 8; ++m) {
        const usize i = static_cast<usize>(st.idx[m]);
        ga.gz[i] += static_cast<f64>(kbar.z * st.w[m]);
        ga.gy[i] += static_cast<f64>(kbar.y * st.w[m]);
        ga.gx[i] += static_cast<f64>(kbar.x * st.w[m]);
    }
    const SampleGrad gz = sample_trilinear_grad(f.vz.view(), l);
    const SampleGrad gy = sample_trilinear_grad(f.vy.view(), l);
    const SampleGrad gx = sample_trilinear_grad(f.vx.view(), l);
    return {(kbar.z * gz.grad.z + kbar.y * gy.grad.z + kbar.x * gx.grad.z) * f.lat_scale.z,
            (kbar.z * gz.grad.y + kbar.y * gy.grad.y + kbar.x * gx.grad.y) * f.lat_scale.y,
            (kbar.z * gz.grad.x + kbar.y * gy.grad.x + kbar.x * gx.grad.x) * f.lat_scale.x};
}
}  // namespace detail

// Reverse-mode (discrete adjoint) of flow_point: given the adjoint `a_out` on the integrated output,
// scatter the gradient into the velocity lattice (`ga`) and return the adjoint on the seed `q0`.
// Exact gradient of the actual unrolled RK4 (discretize-then-optimize); uses the SAME dt as forward.
inline Vec3f flow_point_backward(const FlowField& f, Vec3f q0, int steps, f32 sign, Vec3f a_out, FlowGradAccum& ga) {
    const f32 dt = sign / static_cast<f32>(steps);
    Vec3f traj[33];  // cache the forward trajectory (steps <= 32)
    traj[0] = q0;
    Vec3f p = q0;
    for (int i = 0; i < steps; ++i) {
        const Vec3f k1 = f.velocity(p);
        const Vec3f k2 = f.velocity(p + k1 * (dt * 0.5f));
        const Vec3f k3 = f.velocity(p + k2 * (dt * 0.5f));
        const Vec3f k4 = f.velocity(p + k3 * dt);
        p = p + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * (dt / 6.0f);
        traj[i + 1] = p;
    }
    Vec3f a = a_out;  // adjoint on p_{i+1}, propagated back to p_i each step
    for (int i = steps - 1; i >= 0; --i) {
        const Vec3f pi = traj[i];
        const Vec3f k1 = f.velocity(pi);
        const Vec3f s2 = pi + k1 * (dt * 0.5f);
        const Vec3f k2 = f.velocity(s2);
        const Vec3f s3 = pi + k2 * (dt * 0.5f);
        const Vec3f k3 = f.velocity(s3);
        const Vec3f s4 = pi + k3 * dt;
        // p_{i+1} = pi + (dt/6)(k1+2k2+2k3+k4); process k4..k1 so each stage's s̄ is known when reused.
        Vec3f abar = a;
        const Vec3f kb4 = a * (dt / 6.0f);
        const Vec3f sb4 = detail::flow_sample_backward(f, s4, kb4, ga);
        abar = abar + sb4;
        const Vec3f kb3 = a * (dt / 3.0f) + sb4 * dt;
        const Vec3f sb3 = detail::flow_sample_backward(f, s3, kb3, ga);
        abar = abar + sb3;
        const Vec3f kb2 = a * (dt / 3.0f) + sb3 * (dt * 0.5f);
        const Vec3f sb2 = detail::flow_sample_backward(f, s2, kb2, ga);
        abar = abar + sb2;
        const Vec3f kb1 = a * (dt / 6.0f) + sb2 * (dt * 0.5f);
        const Vec3f sb1 = detail::flow_sample_backward(f, pi, kb1, ga);
        abar = abar + sb1;
        a = abar;
    }
    return a;
}

}  // namespace fenix::winding
