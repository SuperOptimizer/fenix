// render/unroll.hpp — flatten a volume along its winding field into a 2D image. This is
// the simplest unroller: a forward scatter binning each voxel by (winding -> column,
// z -> row) and accumulating the mean CT value. The "straighten the spiral" output.
// (The full surface ± layer render lives alongside; see render/CLAUDE.md.)
#pragma once

#include "core/core.hpp"

#include <cmath>
#include <vector>

namespace fenix::render {

struct UnrollParams {
    f32 samp = 8.0f;  // output columns per unit winding (pixels per wrap)
};

// Returns a {1, H, W} image: row = z slice, column = winding bin, value = mean CT.
// `winding` and `ct` must share dims. Pixels with no contributing voxel are 0.
inline Volume<f32> unroll(VolumeView<const f32> ct, VolumeView<const f32> winding, UnrollParams p) {
    const Extent3 d = ct.dims();
    FENIX_ASSERT(d == winding.dims());

    f32 wmin = winding.flat().empty() ? 0.0f : winding.flat()[0];
    f32 wmax = wmin;
    for (f32 w : winding.flat()) {
        wmin = std::min(wmin, w);
        wmax = std::max(wmax, w);
    }
    const s64 height = d.z;
    const s64 width = std::max<s64>(1, static_cast<s64>(std::ceil((wmax - wmin) * p.samp)) + 1);

    Volume<f32> img = Volume<f32>::zeros({1, height, width});
    std::vector<f32> count(static_cast<usize>(height * width), 0.0f);
    VolumeView<f32> iv = img.view();

    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const f32 w = winding(z, y, x);
                s64 col = static_cast<s64>(std::lround((w - wmin) * p.samp));
                col = std::clamp<s64>(col, 0, width - 1);
                iv(0, z, col) += ct(z, y, x);
                count[static_cast<usize>(z * width + col)] += 1.0f;
            }
    for (s64 z = 0; z < height; ++z)
        for (s64 col = 0; col < width; ++col) {
            f32 c = count[static_cast<usize>(z * width + col)];
            if (c > 0.0f) iv(0, z, col) /= c;
        }
    return img;
}

}  // namespace fenix::render
