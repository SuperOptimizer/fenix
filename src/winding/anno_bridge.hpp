// winding/anno_bridge.hpp — the annotation→fit bridge: lower an annotate::AnnotationSet
// into the spiral fit's constraint vocabulary (diffeo_fit.hpp). Strokes with a known
// absolute winding become hard targets; unlabeled strokes become co-winding groups
// (must-links merge components first, and a labeled stroke propagates its winding across
// its component); radial lines become absolute targets when the base winding is known,
// else consecutive-pair relative-winding constraints. Normal hints pass through for the
// dense lasagna term (roadmap P4). Sits beside fit_bridge.hpp (the CT-valley bridge).
#pragma once

#include "annotate/annotation.hpp"
#include "core/core.hpp"
#include "winding/diffeo_fit.hpp"

#include <numeric>
#include <vector>

namespace fenix::winding {

struct AnnoFitInputs {
    std::vector<FitConstraint> targets;
    std::vector<CoWindingGroup> groups;
    std::vector<RelWindingConstraint> rels;
    std::vector<annotate::NormalHint> normals;
};

namespace detail {
struct AnnoDsu {
    std::vector<s32> parent;
    explicit AnnoDsu(usize n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    s32 find(s32 i) {
        return parent[static_cast<usize>(i)] == i ? i : parent[static_cast<usize>(i)] = find(parent[static_cast<usize>(i)]);
    }
    void unite(s32 a, s32 b) { parent[static_cast<usize>(find(a))] = find(b); }
};
}  // namespace detail

inline AnnoFitInputs to_fit_inputs(const annotate::AnnotationSet& a) {
    AnnoFitInputs out;

    // Union strokes across must-links; a component with any labeled stroke is all-absolute.
    detail::AnnoDsu dsu(a.strokes.size());
    for (const annotate::StrokeLink& l : a.links)
        if (!l.cannot) dsu.unite(l.a, l.b);
    std::vector<f32> comp_winding(a.strokes.size(), 0.0f);
    std::vector<u8> comp_labeled(a.strokes.size(), 0);
    for (usize i = 0; i < a.strokes.size(); ++i) {
        if (!a.strokes[i].has_winding) continue;
        const usize r = static_cast<usize>(dsu.find(static_cast<s32>(i)));
        if (!comp_labeled[r]) {
            comp_labeled[r] = 1;
            comp_winding[r] = a.strokes[i].winding;
        } else if (comp_winding[r] != a.strokes[i].winding) {
            FENIX_WARN("winding", "must-linked strokes carry conflicting windings ({} vs {}); keeping the first",
                       static_cast<double>(comp_winding[r]), static_cast<double>(a.strokes[i].winding));
        }
    }
    std::vector<CoWindingGroup> comp_group(a.strokes.size());
    for (usize i = 0; i < a.strokes.size(); ++i) {
        const usize r = static_cast<usize>(dsu.find(static_cast<s32>(i)));
        if (comp_labeled[r]) {
            for (Vec3f p : a.strokes[i].points) out.targets.push_back({p, comp_winding[r]});
        } else {
            auto& g = comp_group[r].points;
            g.insert(g.end(), a.strokes[i].points.begin(), a.strokes[i].points.end());
        }
    }
    for (auto& g : comp_group)
        if (g.points.size() >= 2) out.groups.push_back(std::move(g));

    for (const annotate::RadialLine& r : a.radial_lines) {
        if (r.points.size() != r.offset.size() || r.points.empty()) continue;
        if (r.has_base_winding) {
            for (usize i = 0; i < r.points.size(); ++i)
                out.targets.push_back({r.points[i], r.base_winding + static_cast<f32>(r.offset[i])});
        } else {
            for (usize i = 1; i < r.points.size(); ++i)
                out.rels.push_back({r.points[i - 1], r.points[i],
                                    static_cast<f32>(r.offset[i] - r.offset[i - 1]), r.weight});
        }
    }

    out.normals = a.normals;
    return out;
}

}  // namespace fenix::winding
