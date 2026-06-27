// segment/structure_tensor.hpp — structure-tensor sheet detection (the classical front
// end). Produces per-voxel sheetness (scalar) + normal (across-sheet direction) for the
// unrolling fit's data terms. See segment/CLAUDE.md, docs/research/research-core.md.
#pragma once

#include "core/core.hpp"
#include "segment/sheet_field.hpp"

#include <vector>

namespace fenix::segment {

struct StParams {
    f32 sigma_grad = 1.0f;    // pre-smoothing before the gradient (config; no baked default policy)
    f32 sigma_tensor = 2.0f;  // integration scale of the tensor
};

// Compute the structure tensor of `ct` and extract sheetness + normal.
inline SheetField structure_tensor(VolumeView<const f32> ct, StParams p) {
    const Extent3 d = ct.dims();
    const usize n = static_cast<usize>(d.count());

    // Pre-smooth a working copy.
    Volume<f32> work(d);
    for (usize i = 0; i < n; ++i) work.flat()[i] = ct.flat()[i];
    gaussian_blur(work.view(), p.sigma_grad);

    // Tensor components Jzz,Jyy,Jxx,Jzy,Jzx,Jyx from the outer product of the gradient.
    Volume<f32> jzz(d), jyy(d), jxx(d), jzy(d), jzx(d), jyx(d);
    VolumeView<const f32> wv = work.view();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                Vec3f g = gradient_at(wv, z, y, x);
                jzz(z, y, x) = g.z * g.z;
                jyy(z, y, x) = g.y * g.y;
                jxx(z, y, x) = g.x * g.x;
                jzy(z, y, x) = g.z * g.y;
                jzx(z, y, x) = g.z * g.x;
                jyx(z, y, x) = g.y * g.x;
            }
    });

    // Integrate (smooth) each component at the tensor scale.
    for (Volume<f32>* c : {&jzz, &jyy, &jxx, &jzy, &jzx, &jyx}) gaussian_blur(c->view(), p.sigma_tensor);

    SheetField out{Volume<f32>::zeros(d), std::vector<Vec3f>(n)};
    auto svz = jzz.view();
    auto svy = jyy.view();
    auto svx = jxx.view();
    auto vzy = jzy.view();
    auto vzx = jzx.view();
    auto vyx = jyx.view();
    VolumeView<f32> sheet = out.sheetness.view();
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                auto e = sym_eig3<f32>(svz(z, y, x), svy(z, y, x), svx(z, y, x), vzy(z, y, x),
                                       vzx(z, y, x), vyx(z, y, x));
                f32 l0 = e.values[0], l1 = e.values[1];
                sheet(z, y, x) = (l0 - l1) / (l0 + tol::eps);
                out.normal[static_cast<usize>((z * d.y + y) * d.x + x)] = e.vectors[0];
            }
    });
    return out;
}

}  // namespace fenix::segment
