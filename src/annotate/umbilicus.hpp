// annotate/umbilicus.hpp — the scroll central axis: a polyline of (y,x) per z. Provides
// the cylindrical (radius, theta) frame the whole unrolling is built on, plus a material-
// mask auto-estimate. The highest-leverage annotation (see annotate/CLAUDE.md).
#pragma once

#include "core/core.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace fenix::annotate {

struct Umbilicus {
    std::vector<f32> z, y, x;  // SoA control points, sorted ascending by z

    [[nodiscard]] bool empty() const { return z.empty(); }

    // Linearly-interpolated (y,x) center at slice zq (clamped to the z range).
    [[nodiscard]] Vec3f center(f32 zq) const {
        if (z.empty()) return {zq, 0, 0};
        if (zq <= z.front()) return {zq, y.front(), x.front()};
        if (zq >= z.back()) return {zq, y.back(), x.back()};
        usize i = 1;
        while (i < z.size() && z[i] < zq) ++i;
        const f32 t = (zq - z[i - 1]) / (z[i] - z[i - 1]);
        return {zq, y[i - 1] + t * (y[i] - y[i - 1]), x[i - 1] + t * (x[i] - x[i - 1])};
    }

    // Polar coordinate of voxel (z,y,x) about the axis: radius (voxels) + theta (radians).
    void polar(s64 vz, s64 vy, s64 vx, f32& radius, Radian& theta) const {
        const Vec3f c = center(static_cast<f32>(vz));
        const f32 dy = static_cast<f32>(vy) - c.y;
        const f32 dx = static_cast<f32>(vx) - c.x;
        radius = std::sqrt(dy * dy + dx * dx);
        theta = Radian{std::atan2(dy, dx)};
    }

    // Auto-estimate from a material field: per-z centroid of voxels >= threshold, with a
    // 3-tap moving average over z. Slices with no material carry the previous center.
    static Umbilicus estimate(VolumeView<const f32> material, f32 threshold) {
        const Extent3 d = material.dims();
        Umbilicus u;
        f32 prev_y = static_cast<f32>(d.y) * 0.5f, prev_x = static_cast<f32>(d.x) * 0.5f;
        for (s64 z = 0; z < d.z; ++z) {
            f64 sy = 0, sx = 0;
            s64 cnt = 0;
            for (s64 y = 0; y < d.y; ++y)
                for (s64 x = 0; x < d.x; ++x)
                    if (material(z, y, x) >= threshold) {
                        sy += static_cast<f64>(y);
                        sx += static_cast<f64>(x);
                        ++cnt;
                    }
            if (cnt > 0) {
                prev_y = static_cast<f32>(sy / static_cast<f64>(cnt));
                prev_x = static_cast<f32>(sx / static_cast<f64>(cnt));
            }
            u.z.push_back(static_cast<f32>(z));
            u.y.push_back(prev_y);
            u.x.push_back(prev_x);
        }
        // 3-tap moving average smoothing over z.
        if (u.z.size() >= 3) {
            std::vector<f32> sy = u.y, sx = u.x;
            for (usize i = 1; i + 1 < u.z.size(); ++i) {
                u.y[i] = (sy[i - 1] + sy[i] + sy[i + 1]) / 3.0f;
                u.x[i] = (sx[i - 1] + sx[i] + sx[i + 1]) / 3.0f;
            }
        }
        return u;
    }
};

}  // namespace fenix::annotate
