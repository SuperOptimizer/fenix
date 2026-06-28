// eval/mesh_quality.hpp — intrinsic, GT-free quality analysis of a traced surface grid (the
// "are our meshes good?" suite). Ports taberna/villa/spiral-v2 metrics adapted to a (u,v) grid:
//   - topology: components + 2D cubical Euler (V-E+F) -> HOLE count (taberna eval/topo.c)
//   - folds: det-J / orientation-flip fraction (taberna unwrap/deform.c jacobian_fold_fraction)
//   - self-intersection: distinct uv-cells sharing a 3D bin (thaumato SAT analog)
//   - distortion / EVENNESS: symmetric-Dirichlet tr(G)+tr(G^-1)-4 of the grid->3D map (spiral-v2)
//   - smoothness: 3-point StraightLoss curvature + normal dihedral (villa-vc StraightLoss)
//   - spacing / TEARING: edge-length median/CV/percentiles, long(tear)/short(collapse) fractions
//   - degeneracy: triangle aspect + min-angle
// All scale-normalized by the median edge so a uniformly-scaled grid scores ideal. See eval/CLAUDE.md.
#pragma once

#include "core/surface.hpp"
#include "core/types.hpp"
#include "core/vec.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace fenix::eval {

struct MeshQuality {
    s64 valid = 0, components = 0, holes = 0;  // holes = components - euler (genus 0)
    f64 euler = 0;
    f64 edge_med = 0, edge_cv = 0, edge_p99_ratio = 0;  // p99/median (tear indicator)
    f64 frac_long = 0, frac_short = 0;                  // edges >1.5x med (tears) / <0.5x med (collapses)
    f64 degen_tri = 0, bad_angle = 0, min_angle_deg = 0;
    f64 fold_detj = 0, self_intersect = 0;              // orientation flips; distant uv overlap
    f64 sdirichlet_mean = 0, sdirichlet_p99 = 0;        // symmetric-Dirichlet (0 = isometric/even)
    f64 curvature_mean = 0, normal_smooth_deg = 0, boundary_frac = 0;
};

namespace detail {
struct UF {
    std::vector<s64> p;
    explicit UF(s64 n) : p(static_cast<usize>(n)) { for (s64 i = 0; i < n; ++i) p[static_cast<usize>(i)] = i; }
    s64 f(s64 x) { while (p[static_cast<usize>(x)] != x) { p[static_cast<usize>(x)] = p[static_cast<usize>(p[static_cast<usize>(x)])]; x = p[static_cast<usize>(x)]; } return x; }
    void u(s64 a, s64 b) { p[static_cast<usize>(f(a))] = f(b); }
};
}  // namespace detail

inline MeshQuality analyze_mesh(const Surface& S) {
    const s64 G = S.nu, H = S.nv;
    MeshQuality q;
    q.valid = S.valid_count();
    if (q.valid < 4) return q;
    auto V = [&](s64 u, s64 v) { return u >= 0 && v >= 0 && u < G && v < H && S.is_valid(u, v); };

    // --- topology: components (4-conn) + 2D cubical Euler V-E+F -> holes ---
    detail::UF uf(G * H);
    s64 nV = 0, nE = 0, nF = 0;
    for (s64 v = 0; v < H; ++v)
        for (s64 u = 0; u < G; ++u) {
            if (!V(u, v)) continue;
            ++nV;
            if (V(u + 1, v)) { ++nE; uf.u(v * G + u, v * G + u + 1); }
            if (V(u, v + 1)) { ++nE; uf.u(v * G + u, (v + 1) * G + u); }
            if (V(u + 1, v) && V(u, v + 1) && V(u + 1, v + 1)) ++nF;
        }
    q.euler = static_cast<f64>(nV - nE + nF);
    {
        std::unordered_map<s64, int> roots;
        for (s64 v = 0; v < H; ++v)
            for (s64 u = 0; u < G; ++u) if (V(u, v)) roots[uf.f(v * G + u)] = 1;
        q.components = static_cast<s64>(roots.size());
    }
    q.holes = q.components - static_cast<s64>(std::llround(q.euler));  // genus-0: each hole drops euler by 1

    // --- edges: spacing / tearing ---
    std::vector<f32> E;
    E.reserve(static_cast<usize>(q.valid * 2));
    for (s64 v = 0; v < H; ++v)
        for (s64 u = 0; u < G; ++u) {
            if (!V(u, v)) continue;
            if (V(u + 1, v)) E.push_back(norm(S.at(u + 1, v) - S.at(u, v)));
            if (V(u, v + 1)) E.push_back(norm(S.at(u, v + 1) - S.at(u, v)));
        }
    f32 med = 1;
    if (!E.empty()) {
        std::vector<f32> t = E;
        std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
        med = std::max(t[t.size() / 2], 1e-6f);
        std::nth_element(t.begin(), t.begin() + (t.size() * 99) / 100, t.end());
        const f32 p99 = t[(t.size() * 99) / 100];
        f64 m = 0; for (f32 e : E) m += e; m /= static_cast<f64>(E.size());
        f64 s2 = 0; for (f32 e : E) s2 += (e - m) * (e - m); s2 /= static_cast<f64>(E.size());
        q.edge_med = med; q.edge_cv = std::sqrt(s2) / (m + 1e-9); q.edge_p99_ratio = p99 / med;
        s64 lo = 0, hi = 0; for (f32 e : E) { if (e > 1.5f * med) ++hi; if (e < 0.5f * med) ++lo; }
        q.frac_long = static_cast<f64>(hi) / static_cast<f64>(E.size());
        q.frac_short = static_cast<f64>(lo) / static_cast<f64>(E.size());
    }

    // --- triangles: degeneracy + min angle ---
    {
        s64 nt = 0, deg = 0, ba = 0; f64 minang = 180;
        auto tri = [&](Vec3f A, Vec3f B, Vec3f C) {
            const f64 ar = 0.5 * norm(cross(B - A, C - A));
            const f64 me = std::max({norm(B - A), norm(C - A), norm(C - B)});
            ++nt;
            if (me > 1e-6 && ar / (me * me) < 0.08) ++deg;
            auto ang = [](Vec3f P, Vec3f Q, Vec3f R) { const Vec3f a = Q - P, b = R - P; const f64 d = norm(a) * norm(b); return d > 1e-9 ? std::acos(std::clamp(static_cast<f64>(dot(a, b)) / d, -1.0, 1.0)) * 180.0 / 3.14159265 : 90.0; };
            const f64 mn = std::min({ang(A, B, C), ang(B, A, C), ang(C, A, B)});
            minang = std::min(minang, mn);
            if (mn < 20.0) ++ba;
        };
        for (s64 v = 0; v + 1 < H; ++v)
            for (s64 u = 0; u + 1 < G; ++u) {
                if (V(u, v) && V(u + 1, v) && V(u, v + 1)) tri(S.at(u, v), S.at(u + 1, v), S.at(u, v + 1));
                if (V(u + 1, v) && V(u + 1, v + 1) && V(u, v + 1)) tri(S.at(u + 1, v), S.at(u + 1, v + 1), S.at(u, v + 1));
            }
        q.degen_tri = nt ? static_cast<f64>(deg) / static_cast<f64>(nt) : 0;
        q.bad_angle = nt ? static_cast<f64>(ba) / static_cast<f64>(nt) : 0;
        q.min_angle_deg = nt ? minang : 0;
    }

    // --- folds (orientation), distortion (symmetric-Dirichlet), curvature, normal smoothness ---
    auto du = [&](s64 u, s64 v) { return (S.at(u + 1, v) - S.at(u - 1, v)) * 0.5f; };
    auto dv = [&](s64 u, s64 v) { return (S.at(u, v + 1) - S.at(u, v - 1)) * 0.5f; };
    auto core = [&](s64 u, s64 v) { return V(u, v) && V(u - 1, v) && V(u + 1, v) && V(u, v - 1) && V(u, v + 1); };
    const f64 m2 = static_cast<f64>(med) * static_cast<f64>(med);
    std::vector<f64> sd;
    std::vector<Vec3f> nrm(static_cast<usize>(G * H));
    std::vector<u8> hasn(static_cast<usize>(G * H), 0);
    f64 curv = 0; s64 ncore = 0;
    for (s64 v = 1; v + 1 < H; ++v)
        for (s64 u = 1; u + 1 < G; ++u) {
            if (!core(u, v)) continue;
            ++ncore;
            const Vec3f a = du(u, v), b = dv(u, v);
            nrm[static_cast<usize>(v * G + u)] = normalized(cross(a, b));
            hasn[static_cast<usize>(v * G + u)] = 1;
            // symmetric-Dirichlet of the (u,v)->3D first fundamental form, scaled by med^2
            const f64 ga = dot(a, a) / m2, gb = dot(a, b) / m2, gc = dot(b, b) / m2;
            const f64 det = ga * gc - gb * gb;
            sd.push_back(det > 1e-6 ? (ga + gc) + (ga + gc) / det - 4.0 : 1e3);
            // 3-point straight/curvature: deviation from neighbour midpoint, both axes (in med units)
            const Vec3f cu = S.at(u, v) - (S.at(u - 1, v) + S.at(u + 1, v)) * 0.5f;
            const Vec3f cv = S.at(u, v) - (S.at(u, v - 1) + S.at(u, v + 1)) * 0.5f;
            curv += (norm(cu) + norm(cv)) / static_cast<f64>(med);
        }
    q.curvature_mean = ncore ? curv / static_cast<f64>(ncore) : 0;
    if (!sd.empty()) {
        f64 m = 0; for (f64 x : sd) m += x; q.sdirichlet_mean = m / static_cast<f64>(sd.size());
        std::nth_element(sd.begin(), sd.begin() + (sd.size() * 99) / 100, sd.end());
        q.sdirichlet_p99 = sd[(sd.size() * 99) / 100];
    }
    // orientation-flip (det-J analog) + normal dihedral
    s64 flips = 0, ftot = 0; f64 dih = 0;
    for (s64 v = 1; v + 1 < H; ++v)
        for (s64 u = 1; u + 1 < G; ++u) {
            if (!hasn[static_cast<usize>(v * G + u)]) continue;
            const Vec3f n = nrm[static_cast<usize>(v * G + u)];
            auto pr = [&](s64 uu, s64 vv) { if (!hasn[static_cast<usize>(vv * G + uu)]) return; ++ftot; const f64 d = std::clamp(static_cast<f64>(dot(n, nrm[static_cast<usize>(vv * G + uu)])), -1.0, 1.0); if (d < 0) ++flips; dih += std::acos(d); };
            pr(u + 1, v); pr(u, v + 1);
        }
    q.fold_detj = ftot ? static_cast<f64>(flips) / static_cast<f64>(ftot) : 0;
    q.normal_smooth_deg = ftot ? dih / static_cast<f64>(ftot) * 180.0 / 3.14159265 : 0;

    // self-intersection: distinct uv-distant cells in the same 3D bin (bin = median edge)
    {
        std::unordered_map<s64, s64> occ;
        occ.reserve(static_cast<usize>(q.valid));
        const f32 ib = 1.0f / std::max(med, 1e-3f);
        const s64 BS = 1 << 20;
        s64 xi = 0, tot = 0;
        for (s64 v = 0; v < H; ++v)
            for (s64 u = 0; u < G; ++u) {
                if (!V(u, v)) continue;
                ++tot;
                const Vec3f c = S.at(u, v);
                const s64 key = static_cast<s64>(c.z * ib) * BS * BS + static_cast<s64>(c.y * ib) * BS + static_cast<s64>(c.x * ib);
                auto it = occ.find(key);
                if (it == occ.end()) { occ[key] = v * G + u; continue; }
                const int ou = static_cast<int>(it->second % G), ov = static_cast<int>(it->second / G);
                if (std::abs(static_cast<int>(u) - ou) + std::abs(static_cast<int>(v) - ov) > 6) ++xi;
            }
        q.self_intersect = tot ? static_cast<f64>(xi) / static_cast<f64>(tot) : 0;
    }
    // boundary fraction
    {
        s64 nb = 0;
        for (s64 v = 0; v < H; ++v)
            for (s64 u = 0; u < G; ++u) {
                if (!V(u, v)) continue;
                if (!(V(u - 1, v) && V(u + 1, v) && V(u, v - 1) && V(u, v + 1))) ++nb;
            }
        q.boundary_frac = q.valid ? static_cast<f64>(nb) / static_cast<f64>(q.valid) : 0;
    }
    return q;
}

}  // namespace fenix::eval
