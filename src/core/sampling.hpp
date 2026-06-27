// core/sampling.hpp — volume sampling kernels. Nearest + trilinear, plus trilinear with
// analytic gradient (the differentiable-fit hot loop). Clamp-to-edge at borders.
// Tricubic/Lanczos land here too; this is the substrate every sampler shares.
#pragma once

#include "core/types.hpp"
#include "core/vec.hpp"
#include "core/volume.hpp"

#include <cmath>

namespace fenix {

template <class T>
f32 sample_nearest(VolumeView<T> v, Vec3f p) {
    return static_cast<f32>(
        v.at_clamped(std::lround(p.z), std::lround(p.y), std::lround(p.x)));
}

// Trilinear sample at fractional (z,y,x), clamped to edge.
template <class T>
f32 sample_trilinear(VolumeView<T> v, Vec3f p) {
    const s64 z0 = static_cast<s64>(std::floor(p.z)), y0 = static_cast<s64>(std::floor(p.y)),
              x0 = static_cast<s64>(std::floor(p.x));
    const f32 fz = p.z - static_cast<f32>(z0), fy = p.y - static_cast<f32>(y0),
              fx = p.x - static_cast<f32>(x0);
    const auto g = [&](s64 dz, s64 dy, s64 dx) {
        return static_cast<f32>(v.at_clamped(z0 + dz, y0 + dy, x0 + dx));
    };
    const f32 c000 = g(0, 0, 0), c001 = g(0, 0, 1), c010 = g(0, 1, 0), c011 = g(0, 1, 1);
    const f32 c100 = g(1, 0, 0), c101 = g(1, 0, 1), c110 = g(1, 1, 0), c111 = g(1, 1, 1);
    const f32 c00 = c000 + (c001 - c000) * fx, c01 = c010 + (c011 - c010) * fx;
    const f32 c10 = c100 + (c101 - c100) * fx, c11 = c110 + (c111 - c110) * fx;
    const f32 c0 = c00 + (c01 - c00) * fy, c1 = c10 + (c11 - c10) * fy;
    return c0 + (c1 - c0) * fz;
}

// Value + analytic gradient d(value)/d(z,y,x) of the trilinear interpolant (for the fit).
struct SampleGrad {
    f32 value;
    Vec3f grad;  // (d/dz, d/dy, d/dx)
};

template <class T>
SampleGrad sample_trilinear_grad(VolumeView<T> v, Vec3f p) {
    const s64 z0 = static_cast<s64>(std::floor(p.z)), y0 = static_cast<s64>(std::floor(p.y)),
              x0 = static_cast<s64>(std::floor(p.x));
    const f32 fz = p.z - static_cast<f32>(z0), fy = p.y - static_cast<f32>(y0),
              fx = p.x - static_cast<f32>(x0);
    const auto g = [&](s64 dz, s64 dy, s64 dx) {
        return static_cast<f32>(v.at_clamped(z0 + dz, y0 + dy, x0 + dx));
    };
    const f32 c000 = g(0, 0, 0), c001 = g(0, 0, 1), c010 = g(0, 1, 0), c011 = g(0, 1, 1);
    const f32 c100 = g(1, 0, 0), c101 = g(1, 0, 1), c110 = g(1, 1, 0), c111 = g(1, 1, 1);
    // x-interpolated faces and their x-derivatives.
    const f32 c00 = c000 + (c001 - c000) * fx, c01 = c010 + (c011 - c010) * fx;
    const f32 c10 = c100 + (c101 - c100) * fx, c11 = c110 + (c111 - c110) * fx;
    const f32 dc00 = c001 - c000, dc01 = c011 - c010, dc10 = c101 - c100, dc11 = c111 - c110;
    const f32 c0 = c00 + (c01 - c00) * fy, c1 = c10 + (c11 - c10) * fy;
    const f32 dc0 = dc00 + (dc01 - dc00) * fy, dc1 = dc10 + (dc11 - dc10) * fy;
    const f32 value = c0 + (c1 - c0) * fz;
    const f32 gx = dc0 + (dc1 - dc0) * fz;          // d/dx
    const f32 gy = (c01 - c00) + ((c11 - c10) - (c01 - c00)) * fz;  // d/dy
    const f32 gz = c1 - c0;                          // d/dz
    return {value, {gz, gy, gx}};
}

}  // namespace fenix
