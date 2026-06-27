// segment/tracer.hpp — NLLS surface tracer (VC-style, first-party — no Ceres). Optimizes a
// grid of 3D corner positions to lie on the sheet (high sheetness / surface-prediction)
// while keeping the parameterization regular (smoothness + edge-length terms), via AdamW.
// Produces a Patch (a Surface grid) — the primary constraint for the unified winding fit.
// This is the single-grid fit core; generational region-growing layers on top. See
// segment/CLAUDE.md, docs/research/villa-vc.md.
#pragma once

#include "core/core.hpp"
#include "segment/sheet_field.hpp"

#include <span>
#include <vector>

namespace fenix::segment {

struct TracerParams {
    f32 unit = 1.0f;        // target grid edge length (voxels)
    f32 w_data = 1.0f;      // pull onto the sheet
    f32 w_smooth = 0.5f;    // Laplacian smoothness of the grid
    f32 w_dist = 0.5f;      // edge length ~ unit
    int iters = 200;
    f32 lr = 0.2f;
};

// Trace a nu x nv patch seeded at `seed` with across-sheet `normal`, onto the `sheetness`
// field (values higher on the sheet). Returns the fitted Surface (the Patch).
inline Surface trace_patch(VolumeView<const f32> sheetness, Vec3f seed, Vec3f normal, s64 nu,
                           s64 nv, TracerParams p) {
    // In-plane tangents perpendicular to the normal.
    Vec3f n = normalized(normal);
    Vec3f e = (std::abs(n.x) < 0.9f) ? Vec3f{0, 0, 1} : Vec3f{1, 0, 0};
    Vec3f t1 = normalized(cross(n, e));
    Vec3f t2 = normalized(cross(n, t1));

    const usize N = static_cast<usize>(nu * nv);
    std::vector<f32> params(3 * N);
    auto cidx = [&](s64 u, s64 v) { return static_cast<usize>(v * nu + u); };
    for (s64 v = 0; v < nv; ++v)
        for (s64 u = 0; u < nu; ++u) {
            const f32 du = static_cast<f32>(u) - static_cast<f32>(nu) * 0.5f;
            const f32 dv = static_cast<f32>(v) - static_cast<f32>(nv) * 0.5f;
            const Vec3f c = seed + t1 * (du * p.unit) + t2 * (dv * p.unit);
            const usize i = cidx(u, v) * 3;
            params[i] = c.z;
            params[i + 1] = c.y;
            params[i + 2] = c.x;
        }

    auto getc = [&](std::span<const f32> pr, s64 u, s64 v) {
        const usize i = cidx(u, v) * 3;
        return Vec3f{pr[i], pr[i + 1], pr[i + 2]};
    };

    auto loss = [&](std::span<const f32> pr) {
        f32 data = 0, smooth = 0, dist = 0;
        for (s64 v = 0; v < nv; ++v)
            for (s64 u = 0; u < nu; ++u) {
                const Vec3f c = getc(pr, u, v);
                data += 1.0f - sample_trilinear(sheetness, c);
                // Laplacian smoothness vs 4-neighbour mean.
                Vec3f sum{0, 0, 0};
                int cnt = 0;
                if (u > 0) { sum = sum + getc(pr, u - 1, v); ++cnt; }
                if (u + 1 < nu) { sum = sum + getc(pr, u + 1, v); ++cnt; }
                if (v > 0) { sum = sum + getc(pr, u, v - 1); ++cnt; }
                if (v + 1 < nv) { sum = sum + getc(pr, u, v + 1); ++cnt; }
                if (cnt) {
                    const Vec3f lap = c - sum / static_cast<f32>(cnt);
                    smooth += dot(lap, lap);
                }
                // Edge length terms (to the +u and +v neighbours).
                if (u + 1 < nu) {
                    const f32 len = norm(getc(pr, u + 1, v) - c);
                    dist += (len - p.unit) * (len - p.unit);
                }
                if (v + 1 < nv) {
                    const f32 len = norm(getc(pr, u, v + 1) - c);
                    dist += (len - p.unit) * (len - p.unit);
                }
            }
        return p.w_data * data + p.w_smooth * smooth + p.w_dist * dist;
    };

    // AdamW with central finite-difference gradients (small patches).
    AdamW opt(3 * N, {.lr = p.lr});
    std::vector<f32> g(3 * N);
    const f32 h = 1e-3f;
    for (int it = 0; it < p.iters; ++it) {
        for (usize i = 0; i < params.size(); ++i) {
            const f32 o = params[i];
            params[i] = o + h;
            const f32 lp = loss(params);
            params[i] = o - h;
            const f32 lm = loss(params);
            params[i] = o;
            g[i] = (lp - lm) / (2.0f * h);
        }
        opt.step(params, g);
    }

    Surface surf(nu, nv);
    for (s64 v = 0; v < nv; ++v)
        for (s64 u = 0; u < nu; ++u) surf.set(u, v, getc(params, u, v));
    return surf;
}

}  // namespace fenix::segment
