// winding/corpus_bridge.hpp — the corpus-mesh → fit bridge. GP/corpus segments are NOT
// single-winding patches: one mesh winds many turns around the scroll. So instead of the
// patch-graph's one-integer-per-patch model (fit_bridge.hpp), each mesh carries its own
// per-cell CONTINUOUS winding: unwrap the angular turn theta/2pi over the uv grid (BFS,
// no branch-cut jumps between neighbouring cells), then place every mesh in one absolute
// Archimedean gauge via base = median(r/spacing - turn). The mesh also MEASURES the wrap
// spacing itself: Delta-r between same-row cells exactly one turn apart. Constraint
// targets are base + turn — dense, multi-wrap, and cross-mesh consistent to first order
// (the diffeo fit's affine/flow/gap factors absorb the rest). See winding/CLAUDE.md.
#pragma once

#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "winding/diffeo_fit.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace fenix::winding {

struct CorpusBridgeParams {
    int stride = 2;        // subsample cells when emitting constraints (on top of any load-time stride)
    f32 spacing = 0.0f;    // wrap spacing (voxels); 0 = auto-estimate from the meshes' own turns
    int min_component = 64;// ignore unwrap islands (disconnected valid regions) smaller than this
};

struct CorpusBridgeOut {
    std::vector<FitConstraint> targets;
    f32 spacing = 0;         // estimated (or passed-through) wrap spacing
    f32 base_lo = 0, base_hi = 0;  // range of per-component base windings (gauge diagnostic)
    s32 components = 0;      // unwrap islands used
};

namespace detail {

// Per-cell unwrapped turn (theta/2pi, continuous across the ±pi cut) for one valid
// component seeded at `seed`, via BFS over 4-neighbour grid adjacency. Returns cell count.
inline s64 unwrap_component(const Surface& s, const annotate::Umbilicus& umb, usize seed,
                            std::vector<f32>& turn, std::vector<u8>& state) {
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    auto theta_of = [&](usize i) {
        const Vec3f p = s.coord[i];
        const Vec3f c = umb.center(p.z);
        return std::atan2(p.y - c.y, p.x - c.x);
    };
    std::vector<usize> queue{seed};
    turn[seed] = theta_of(seed) / two_pi;
    state[seed] = 2;
    s64 n = 0;
    while (!queue.empty()) {
        const usize i = queue.back();
        queue.pop_back();
        ++n;
        const s64 u = static_cast<s64>(i) % s.nu, v = static_cast<s64>(i) / s.nu;
        const s64 nb[4][2] = {{u - 1, v}, {u + 1, v}, {u, v - 1}, {u, v + 1}};
        for (const auto& [uu, vv] : nb) {
            if (uu < 0 || uu >= s.nu || vv < 0 || vv >= s.nv) continue;
            const usize j = s.idx(uu, vv);
            if (state[j] != 1) continue;  // not a fresh valid cell
            // neighbour turn = its own theta, shifted to the branch nearest this cell's turn
            f32 t = theta_of(j) / two_pi;
            t += std::round(turn[i] - t);
            turn[j] = t;
            state[j] = 2;
            queue.push_back(j);
        }
    }
    return n;
}

// Wrap spacing measured by one component: walk each grid row; wherever the row's turn
// advances by >= 1 between two cells, linearly interpolate the radius at turn+1 and take
// Delta-r. Median over all samples.
inline void spacing_samples(const Surface& s, const annotate::Umbilicus& umb, const std::vector<f32>& turn,
                            const std::vector<u8>& state, std::vector<f32>& out) {
    auto radius_of = [&](usize i) {
        const Vec3f p = s.coord[i];
        const Vec3f c = umb.center(p.z);
        return std::sqrt((p.y - c.y) * (p.y - c.y) + (p.x - c.x) * (p.x - c.x));
    };
    for (s64 v = 0; v < s.nv; ++v) {
        for (s64 u0 = 0; u0 < s.nu; ++u0) {
            const usize i0 = s.idx(u0, v);
            if (state[i0] != 2) continue;
            // scan forward for the first cell one full turn later (contiguous valid run)
            for (s64 u1 = u0 + 1; u1 < s.nu; ++u1) {
                const usize ia = s.idx(u1 - 1, v), ib = s.idx(u1, v);
                if (state[ib] != 2) break;
                const f32 ta = turn[ia] - turn[i0], tb = turn[ib] - turn[i0];
                if (std::abs(tb) >= 1.0f) {
                    if (std::abs(ta) < 1.0f && std::abs(tb - ta) > 1e-6f) {
                        const f32 goal = tb > 0 ? 1.0f : -1.0f;
                        const f32 f = (goal - ta) / (tb - ta);
                        const f32 r1 = radius_of(ia) + f * (radius_of(ib) - radius_of(ia));
                        out.push_back(std::abs(r1 - radius_of(i0)));
                    }
                    break;
                }
            }
        }
    }
}

inline f32 median_of(std::vector<f32>& v) {
    if (v.empty()) return 0;
    const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}

}  // namespace detail

inline CorpusBridgeOut corpus_to_constraints(const std::vector<Surface>& sheets,
                                             const annotate::Umbilicus& umb,
                                             CorpusBridgeParams bp = {}) {
    CorpusBridgeOut out;
    struct Comp {
        const Surface* s;
        std::vector<f32> turn;
        std::vector<u8> mask;  // 2 = in this component
        s64 count;
    };
    std::vector<Comp> comps;
    std::vector<f32> sp_samples;
    for (const Surface& s : sheets) {
        std::vector<u8> state(s.valid.begin(), s.valid.end());  // 1 = valid unvisited
        std::vector<f32> turn(state.size(), 0.0f);
        for (usize i = 0; i < state.size(); ++i) {
            if (state[i] != 1) continue;
            std::vector<u8> before = state;
            const s64 n = detail::unwrap_component(s, umb, i, turn, state);
            if (n < bp.min_component) continue;
            Comp c{&s, turn, {}, n};
            c.mask.resize(state.size(), 0);
            for (usize j = 0; j < state.size(); ++j)
                if (state[j] == 2 && before[j] == 1) c.mask[j] = 2;
            if (bp.spacing <= 0) detail::spacing_samples(s, umb, c.turn, c.mask, sp_samples);
            comps.push_back(std::move(c));
        }
    }
    out.components = static_cast<s32>(comps.size());
    out.spacing = bp.spacing > 0 ? bp.spacing : detail::median_of(sp_samples);
    if (out.spacing <= 0 || comps.empty()) return out;

    out.base_lo = 1e30f;
    out.base_hi = -1e30f;
    const int st = std::max(1, bp.stride);
    for (const Comp& c : comps) {
        const Surface& s = *c.s;
        // base winding: median of (r/spacing - turn) over the component — the absolute
        // Archimedean gauge this component lives at (to first order)
        std::vector<f32> bases;
        for (usize i = 0; i < c.mask.size(); ++i) {
            if (c.mask[i] != 2) continue;
            const Vec3f p = s.coord[i];
            const Vec3f ct = umb.center(p.z);
            const f32 r = std::sqrt((p.y - ct.y) * (p.y - ct.y) + (p.x - ct.x) * (p.x - ct.x));
            bases.push_back(r / out.spacing - c.turn[i]);
        }
        const f32 base = detail::median_of(bases);
        out.base_lo = std::min(out.base_lo, base);
        out.base_hi = std::max(out.base_hi, base);
        s64 k = 0;
        for (usize i = 0; i < c.mask.size(); ++i) {
            if (c.mask[i] != 2) continue;
            if ((k++ % st) != 0) continue;
            out.targets.push_back({s.coord[i], base + c.turn[i]});
        }
    }
    return out;
}

}  // namespace fenix::winding
