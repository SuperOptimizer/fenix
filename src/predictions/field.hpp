// predictions/field.hpp — ingest + normalize ML- or classically-generated prediction
// fields (sheet-prob, distance, winding-density) into the conventions the unrolling fit
// expects. Either source enters here uniformly; the fit shouldn't care which. See
// predictions/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace fenix::predictions {

enum class Norm { None, MinMax, Percentile };

// Normalize a scalar prediction field into [0,1]. Percentile clips to [p1,p99] first
// (robust to outliers), then linearly maps.
inline Volume<f32> normalize(VolumeView<const f32> field, Norm scheme, f32 plo = 0.01f,
                             f32 phi = 0.99f) {
    const Extent3 d = field.dims();
    Volume<f32> out(d);
    const s64 n = d.count();
    f32 lo, hi;
    if (scheme == Norm::Percentile) {
        std::vector<f32> v(field.flat().begin(), field.flat().end());
        std::ranges::sort(v);
        lo = v[static_cast<usize>(std::clamp<s64>(static_cast<s64>(plo * static_cast<f32>(n)), 0, n - 1))];
        hi = v[static_cast<usize>(std::clamp<s64>(static_cast<s64>(phi * static_cast<f32>(n)), 0, n - 1))];
    } else if (scheme == Norm::MinMax) {
        lo = hi = field.flat()[0];
        for (f32 x : field.flat()) {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
    } else {
        for (s64 i = 0; i < n; ++i) out.flat()[static_cast<usize>(i)] = field.flat()[static_cast<usize>(i)];
        return out;
    }
    const f32 inv = (hi > lo) ? 1.0f / (hi - lo) : 0.0f;
    for (s64 i = 0; i < n; ++i)
        out.flat()[static_cast<usize>(i)] = std::clamp((field.flat()[static_cast<usize>(i)] - lo) * inv, 0.0f, 1.0f);
    return out;
}

// Threshold a (normalized) probability field to a binary surface-prediction mask.
inline Volume<u8> threshold(VolumeView<const f32> field, f32 t) {
    const Extent3 d = field.dims();
    Volume<u8> m = Volume<u8>::zeros(d);
    for (s64 i = 0; i < d.count(); ++i)
        m.flat()[static_cast<usize>(i)] = field.flat()[static_cast<usize>(i)] >= t ? u8{1} : u8{0};
    return m;
}

}  // namespace fenix::predictions
