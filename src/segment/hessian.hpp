// segment/hessian.hpp — Hessian-based plate (sheet) detector, Frangi plate variant. Centers
// on the sheet ridge (2nd-derivative peak) better than the gradient structure tensor.
// Multi-scale max over a sigma set. See segment/CLAUDE.md, docs/research/research-core.md.
#pragma once

#include "core/core.hpp"
#include "segment/sheet_field.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace fenix::segment {

struct HessianParams {
    std::vector<f32> sigmas{1.0f, 2.0f};  // scales (config; no baked default policy)
    f32 alpha = 0.5f;                     // plate-vs-line sensitivity
};

// Returns sheetness (Frangi plateness in [0,1]) + across-sheet normal (eigvec of largest|λ|).
inline SheetField hessian_sheet(VolumeView<const f32> ct, HessianParams p) {
    const Extent3 d = ct.dims();
    const usize n = static_cast<usize>(d.count());
    SheetField out{Volume<f32>::zeros(d), std::vector<Vec3f>(n)};
    VolumeView<f32> sheet = out.sheetness.view();

    for (f32 sigma : p.sigmas) {
        Volume<f32> work(d);
        for (usize i = 0; i < n; ++i) work.flat()[i] = ct.flat()[i];
        gaussian_blur(work.view(), sigma);
        VolumeView<const f32> w = work.view();
        const f32 s2 = sigma * sigma;  // gamma=1 scale normalization

        // First pass: eigenvalues (abs-sorted) -> Ra, S, sign, normal; track max S.
        std::vector<f32> ra(n, 0), ss(n, 0);
        std::vector<u8> bright(n, 0);
        std::vector<Vec3f> nrm(n);
        f32 max_s = tol::eps;
        for (s64 z = 1; z < d.z - 1; ++z)
            for (s64 y = 1; y < d.y - 1; ++y)
                for (s64 x = 1; x < d.x - 1; ++x) {
                    const f32 hzz = w(z + 1, y, x) - 2 * w(z, y, x) + w(z - 1, y, x);
                    const f32 hyy = w(z, y + 1, x) - 2 * w(z, y, x) + w(z, y - 1, x);
                    const f32 hxx = w(z, y, x + 1) - 2 * w(z, y, x) + w(z, y, x - 1);
                    const f32 hzy = 0.25f * (w(z + 1, y + 1, x) - w(z + 1, y - 1, x) -
                                             w(z - 1, y + 1, x) + w(z - 1, y - 1, x));
                    const f32 hzx = 0.25f * (w(z + 1, y, x + 1) - w(z + 1, y, x - 1) -
                                             w(z - 1, y, x + 1) + w(z - 1, y, x - 1));
                    const f32 hyx = 0.25f * (w(z, y + 1, x + 1) - w(z, y + 1, x - 1) -
                                             w(z, y - 1, x + 1) + w(z, y - 1, x - 1));
                    auto e = sym_eig3<f32>(hzz * s2, hyy * s2, hxx * s2, hzy * s2, hzx * s2, hyx * s2);
                    // sort eigenpairs by ascending |lambda|.
                    std::array<int, 3> ord{0, 1, 2};
                    std::sort(ord.begin(), ord.end(),
                              [&](int a, int b) { return std::abs(e.values[static_cast<usize>(a)]) <
                                                         std::abs(e.values[static_cast<usize>(b)]); });
                    const f32 l1 = e.values[static_cast<usize>(ord[0])];
                    const f32 l2 = e.values[static_cast<usize>(ord[1])];
                    const f32 l3 = e.values[static_cast<usize>(ord[2])];
                    const usize idx = static_cast<usize>((z * d.y + y) * d.x + x);
                    ra[idx] = std::abs(l2) / (std::abs(l3) + tol::eps);  // plate: small
                    ss[idx] = std::sqrt(l1 * l1 + l2 * l2 + l3 * l3);
                    bright[idx] = l3 < 0.0f ? 1 : 0;  // bright sheet on dark bg
                    nrm[idx] = e.vectors[static_cast<usize>(ord[2])];  // across-sheet = largest|λ|
                    max_s = std::max(max_s, ss[idx]);
                }

        const f32 c = 0.5f * max_s;  // structure half-max (auto per scale)
        const f32 a2 = 2.0f * p.alpha * p.alpha;
        const f32 c2 = 2.0f * c * c;
        for (usize i = 0; i < n; ++i) {
            f32 plate = 0.0f;
            if (bright[i]) {
                plate = std::exp(-(ra[i] * ra[i]) / a2) * (1.0f - std::exp(-(ss[i] * ss[i]) / c2));
            }
            if (plate > sheet.flat()[i]) {  // multi-scale max
                sheet.flat()[i] = plate;
                out.normal[i] = nrm[i];
            }
        }
    }
    return out;
}

}  // namespace fenix::segment
