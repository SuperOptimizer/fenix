// annotate/extract.hpp — carve trustworthy co-winding annotations out of existing traced
// surfaces: keep only high-confidence valid cells, erode away the (least reliable) region
// borders, split into connected components, and emit each big-enough component as a
// CoWindingStroke. The point: an existing patch/trace doesn't have to be good everywhere
// to be useful — its good portions are exact same-winding constraints for the fit.
#pragma once

#include "annotate/annotation.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"

#include <format>
#include <vector>

namespace fenix::annotate {

struct ExtractParams {
    f32 conf_min = 1.0f;    // trust threshold on the surface conf channel (>=1 == on a sheet)
    int erode = 1;          // uv 4-neighbour erosions (grid borders erode too — by design)
    s64 min_cells = 64;     // drop components smaller than this (pre-subsample)
    s64 stride = 4;         // keep every stride-th cell in u and v
    usize max_points = 512; // hard cap per stroke (uniform thinning past it)
};

// Extract co-winding strokes from a traced surface. A surface without a conf channel is
// gated on validity alone.
inline std::vector<CoWindingStroke> extract_trusted(const Surface& s, ExtractParams p = {}) {
    std::vector<CoWindingStroke> out;
    if (s.nu <= 0 || s.nv <= 0) return out;
    const s64 nu = s.nu, nv = s.nv;
    std::vector<u8> mask(static_cast<usize>(nu * nv), 0);
    for (s64 v = 0; v < nv; ++v)
        for (s64 u = 0; u < nu; ++u) {
            const usize i = s.idx(u, v);
            mask[i] = s.valid[i] && (s.conf.empty() || s.conf[i] >= p.conf_min);
        }

    std::vector<u8> next(mask.size());
    for (int e = 0; e < p.erode; ++e) {
        for (s64 v = 0; v < nv; ++v)
            for (s64 u = 0; u < nu; ++u) {
                const usize i = s.idx(u, v);
                next[i] = mask[i] && u > 0 && u + 1 < nu && v > 0 && v + 1 < nv &&
                          mask[s.idx(u - 1, v)] && mask[s.idx(u + 1, v)] &&
                          mask[s.idx(u, v - 1)] && mask[s.idx(u, v + 1)];
            }
        mask.swap(next);
    }

    // 4-connected components over the eroded mask; each survivor becomes one stroke.
    std::vector<s32> comp(mask.size(), -1);
    std::vector<usize> stack;
    s32 ncomp = 0;
    for (usize seed = 0; seed < mask.size(); ++seed) {
        if (!mask[seed] || comp[seed] >= 0) continue;
        const s32 id = ncomp++;
        std::vector<usize> cells;
        stack.assign(1, seed);
        comp[seed] = id;
        while (!stack.empty()) {
            const usize i = stack.back();
            stack.pop_back();
            cells.push_back(i);
            const s64 u = static_cast<s64>(i) % nu, v = static_cast<s64>(i) / nu;
            const s64 nbs[4][2] = {{u - 1, v}, {u + 1, v}, {u, v - 1}, {u, v + 1}};
            for (const auto& nb : nbs) {
                if (nb[0] < 0 || nb[0] >= nu || nb[1] < 0 || nb[1] >= nv) continue;
                const usize j = s.idx(nb[0], nb[1]);
                if (mask[j] && comp[j] < 0) {
                    comp[j] = id;
                    stack.push_back(j);
                }
            }
        }
        if (static_cast<s64>(cells.size()) < p.min_cells) continue;

        CoWindingStroke st;
        st.kind = StrokeKind::patch_extract;
        st.name = std::format("extract-{}", out.size());
        for (usize i : cells) {
            const s64 u = static_cast<s64>(i) % nu, v = static_cast<s64>(i) / nu;
            if (u % p.stride == 0 && v % p.stride == 0) st.points.push_back(s.coord[i]);
        }
        if (st.points.size() > p.max_points) {
            const usize step = (st.points.size() + p.max_points - 1) / p.max_points;
            std::vector<Vec3f> thin;
            for (usize k = 0; k < st.points.size(); k += step) thin.push_back(st.points[k]);
            st.points = std::move(thin);
        }
        if (st.points.size() >= 2) out.push_back(std::move(st));
    }
    return out;
}

}  // namespace fenix::annotate
