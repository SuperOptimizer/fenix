// flatten/slim.hpp — global low-distortion parameterization of a traced Surface (ARAP/SLIM family,
// Liu et al. 2008 local/global). Fixes the 3D surface; solves fresh UVs minimizing isometric
// distortion over the whole mesh at once: LOCAL step = closest rotation per triangle, GLOBAL step =
// cotangent-Laplacian solve (first-party CG). Reports symmetric-Dirichlet energy (2 = isometric).
// First-party (no Eigen/libigl). See flatten/CLAUDE.md.
#pragma once

#include "core/surface.hpp"
#include "core/types.hpp"
#include "core/vec.hpp"
#include "flatten/sparse.hpp"

#include <array>
#include <cstdio>
#include <cmath>
#include <vector>

namespace fenix::flatten {

struct FlatMesh {
    std::vector<Vec3f> pos;                 // 3D position per vertex
    std::vector<std::array<f32, 2>> uv;     // optimized 2D parameterization
    std::vector<std::array<s64, 3>> tri;    // triangles (vertex indices)
    f64 energy_init = 0, energy_final = 0;  // area-weighted symmetric-Dirichlet (2 = perfect)
};

namespace detail {
// union-find for the largest connected component
struct DSU {
    std::vector<s64> p;
    explicit DSU(s64 n) : p(static_cast<usize>(n)) { for (s64 i = 0; i < n; ++i) p[static_cast<usize>(i)] = i; }
    s64 find(s64 x) { while (p[static_cast<usize>(x)] != x) { p[static_cast<usize>(x)] = p[static_cast<usize>(p[static_cast<usize>(x)])]; x = p[static_cast<usize>(x)]; } return x; }
    void uni(s64 a, s64 b) { p[static_cast<usize>(find(a))] = find(b); }
};
}  // namespace detail

// Parameterize the valid region of `S` (largest connected component) with `iters` local/global
// sweeps; `cg_iters` caps the inner CG. Initialized from the grid (u,v) layout.
inline FlatMesh slim_parameterize(const Surface& S, int iters = 20, int cg_iters = 600) {
    const s64 G = S.nu;
    // --- vertices + triangles from valid quads ---
    std::vector<s64> vid(static_cast<usize>(G) * static_cast<usize>(S.nv), -1);
    std::vector<Vec3f> pos;
    std::vector<f64> gu, gv;
    auto VID = [&](s64 u, s64 v) -> s64& { return vid[static_cast<usize>(v) * G + u]; };
    for (s64 v = 0; v < S.nv; ++v)
        for (s64 u = 0; u < G; ++u)
            if (S.is_valid(u, v)) { VID(u, v) = static_cast<s64>(pos.size()); pos.push_back(S.at(u, v)); gu.push_back(static_cast<f64>(u)); gv.push_back(static_cast<f64>(v)); }
    // defensive only: drop exactly-degenerate tris so Dm_inv can't divide by zero. Real geometry
    // quality (no slivers) is enforced UPSTREAM in the tracer (segment::surface_quality + cleanup).
    auto good = [&](Vec3f A, Vec3f B, Vec3f Cc) { return 0.5 * norm(cross(B - A, Cc - A)) > 1e-7; };
    std::vector<std::array<s64, 3>> tri;
    for (s64 v = 0; v + 1 < S.nv; ++v)
        for (s64 u = 0; u + 1 < G; ++u) {
            const bool a = S.is_valid(u, v), b = S.is_valid(u + 1, v), c = S.is_valid(u, v + 1), d = S.is_valid(u + 1, v + 1);
            if (a && b && c && good(S.at(u, v), S.at(u + 1, v), S.at(u, v + 1))) tri.push_back({VID(u, v), VID(u + 1, v), VID(u, v + 1)});
            if (b && d && c && good(S.at(u + 1, v), S.at(u + 1, v + 1), S.at(u, v + 1))) tri.push_back({VID(u + 1, v), VID(u + 1, v + 1), VID(u, v + 1)});
        }

    // --- largest connected component ---
    const s64 Nv0 = static_cast<s64>(pos.size());
    detail::DSU dsu(Nv0);
    for (auto& t : tri) { dsu.uni(t[0], t[1]); dsu.uni(t[1], t[2]); }
    std::vector<s64> compcnt(static_cast<usize>(Nv0), 0);
    for (s64 i = 0; i < Nv0; ++i) compcnt[static_cast<usize>(dsu.find(i))]++;
    s64 best = 0;
    for (s64 i = 0; i < Nv0; ++i) if (compcnt[static_cast<usize>(i)] > compcnt[static_cast<usize>(best)]) best = i;
    const s64 root = best;
    std::vector<s64> remap(static_cast<usize>(Nv0), -1);
    FlatMesh M;
    for (s64 i = 0; i < Nv0; ++i)
        if (dsu.find(i) == root) { remap[static_cast<usize>(i)] = static_cast<s64>(M.pos.size()); M.pos.push_back(pos[static_cast<usize>(i)]); }
    std::vector<f64> ggu, ggv;
    for (s64 i = 0; i < Nv0; ++i) if (dsu.find(i) == root) { ggu.push_back(gu[static_cast<usize>(i)]); ggv.push_back(gv[static_cast<usize>(i)]); }
    for (auto& t : tri) if (dsu.find(t[0]) == root) M.tri.push_back({remap[static_cast<usize>(t[0])], remap[static_cast<usize>(t[1])], remap[static_cast<usize>(t[2])]});
    const s64 Nv = static_cast<s64>(M.pos.size());

    // --- per-triangle isometric 2D rest + cotan weights + rest-edge inverse ---
    const s64 Nt = static_cast<s64>(M.tri.size());
    std::vector<std::array<f64, 6>> rest(static_cast<usize>(Nt));   // r0x,r0y,r1x,r1y,r2x,r2y
    std::vector<std::array<f64, 4>> dminv(static_cast<usize>(Nt));  // 2x2 inverse of rest edge matrix
    std::vector<std::array<f64, 3>> wts(static_cast<usize>(Nt));    // w01,w12,w02
    Triplets T;
    auto addedge = [&](s64 i, s64 j, f64 w) { T.add(i, i, w); T.add(j, j, w); T.add(i, j, -w); T.add(j, i, -w); };
    for (s64 t = 0; t < Nt; ++t) {
        const Vec3f P0 = M.pos[static_cast<usize>(M.tri[static_cast<usize>(t)][0])], P1 = M.pos[static_cast<usize>(M.tri[static_cast<usize>(t)][1])], P2 = M.pos[static_cast<usize>(M.tri[static_cast<usize>(t)][2])];
        const f64 e01 = norm(P1 - P0), e02 = norm(P2 - P0), e12 = norm(P2 - P1);
        const f64 x2 = e01 > 1e-9 ? (e01 * e01 + e02 * e02 - e12 * e12) / (2.0 * e01) : 0.0;
        const f64 y2 = std::sqrt(std::max(e02 * e02 - x2 * x2, 1e-12));
        rest[static_cast<usize>(t)] = {0, 0, e01, 0, x2, y2};
        const f64 det = e01 * y2;
        dminv[static_cast<usize>(t)] = {y2 / det, -x2 / det, 0.0, e01 / det};  // inv of [[e01,x2],[0,y2]]
        // cotangent weights (required for correct ARAP parameterization — uniform weights collapse
        // the interior). Clamped >0 for SPD; the upstream tracer guarantees no slivers so they stay
        // bounded. w_ij = cot(angle opposite edge ij) = (adjacent^2 - opposite^2)/(4*area).
        const f64 L01 = e01 * e01, L12 = e12 * e12, L02 = e02 * e02, A4 = 2.0 * det + 1e-12;
        const f64 w01 = std::max((L02 + L12 - L01) / A4, 1e-3);   // opposite v2
        const f64 w12 = std::max((L01 + L02 - L12) / A4, 1e-3);   // opposite v0
        const f64 w02 = std::max((L01 + L12 - L02) / A4, 1e-3);   // opposite v1
        wts[static_cast<usize>(t)] = {w01, w12, w02};
        addedge(M.tri[static_cast<usize>(t)][0], M.tri[static_cast<usize>(t)][1], w01);
        addedge(M.tri[static_cast<usize>(t)][1], M.tri[static_cast<usize>(t)][2], w12);
        addedge(M.tri[static_cast<usize>(t)][0], M.tri[static_cast<usize>(t)][2], w02);
    }
    // Tiny Tikhonov eps*I -> SPD for CG (the b ⊥ constants, so the only nullspace is translation,
    // which a small eps fixes; the resulting constant shift is energy-neutral). A large penalty pin
    // wrecks CG conditioning, and a proximal anchor toward the current UV destabilizes — both avoided.
    const f64 eps = 1e-6;
    for (s64 i = 0; i < Nv; ++i) T.add(i, i, eps);
    const Csr L = Csr::from_triplets(Nv, T);

    // --- init UV = grid layout ---
    std::vector<f64> ux(static_cast<usize>(Nv)), uy(static_cast<usize>(Nv));
    for (s64 i = 0; i < Nv; ++i) { ux[static_cast<usize>(i)] = ggu[static_cast<usize>(i)]; uy[static_cast<usize>(i)] = ggv[static_cast<usize>(i)]; }

    auto energy = [&]() {
        f64 e = 0, area = 0;
        for (s64 t = 0; t < Nt; ++t) {
            const auto& tr = M.tri[static_cast<usize>(t)];
            const f64 d0x = ux[static_cast<usize>(tr[1])] - ux[static_cast<usize>(tr[0])], d1x = ux[static_cast<usize>(tr[2])] - ux[static_cast<usize>(tr[0])];
            const f64 d0y = uy[static_cast<usize>(tr[1])] - uy[static_cast<usize>(tr[0])], d1y = uy[static_cast<usize>(tr[2])] - uy[static_cast<usize>(tr[0])];
            const auto& di = dminv[static_cast<usize>(t)];
            const f64 j00 = d0x * di[0] + d1x * di[2], j01 = d0x * di[1] + d1x * di[3];
            const f64 j10 = d0y * di[0] + d1y * di[2], j11 = d0y * di[1] + d1y * di[3];
            const f64 E1 = j00 * j00 + j01 * j01 + j10 * j10 + j11 * j11, det = j00 * j11 - j01 * j10;
            const f64 a = 0.5 * (rest[static_cast<usize>(t)][2] * rest[static_cast<usize>(t)][5]);
            e += a * 0.5 * (E1 + E1 / (det * det + 1e-12));
            area += a;
        }
        return e / (area + 1e-12);
    };
    M.energy_init = energy();

    // --- local/global iterations ---
    std::vector<f64> bx(static_cast<usize>(Nv)), by(static_cast<usize>(Nv));
    for (int it = 0; it < iters; ++it) {
        std::fill(bx.begin(), bx.end(), 0.0);
        std::fill(by.begin(), by.end(), 0.0);
        for (s64 t = 0; t < Nt; ++t) {
            const auto& tr = M.tri[static_cast<usize>(t)];
            const auto& r = rest[static_cast<usize>(t)];
            const auto& w = wts[static_cast<usize>(t)];
            const s64 i0 = tr[0], i1 = tr[1], i2 = tr[2];
            // closest rotation to the per-triangle Jacobian J = Ds * Dm^-1 (the ARAP local step on the
            // per-triangle isometric energy).
            const auto& di = dminv[static_cast<usize>(t)];
            const f64 d0x = ux[static_cast<usize>(i1)] - ux[static_cast<usize>(i0)], d1x = ux[static_cast<usize>(i2)] - ux[static_cast<usize>(i0)];
            const f64 d0y = uy[static_cast<usize>(i1)] - uy[static_cast<usize>(i0)], d1y = uy[static_cast<usize>(i2)] - uy[static_cast<usize>(i0)];
            const f64 j00 = d0x * di[0] + d1x * di[2], j01 = d0x * di[1] + d1x * di[3];
            const f64 j10 = d0y * di[0] + d1y * di[2], j11 = d0y * di[1] + d1y * di[3];
            const f64 ang = std::atan2(j10 - j01, j00 + j11);
            const f64 cs = std::cos(ang), sn = std::sin(ang);
            auto edge = [&](s64 a, s64 b, f64 wij, f64 drx, f64 dry) {
                const f64 dx = cs * drx - sn * dry, dy = sn * drx + cs * dry;  // R * (rest_a - rest_b)
                bx[static_cast<usize>(a)] += wij * dx; by[static_cast<usize>(a)] += wij * dy;
                bx[static_cast<usize>(b)] -= wij * dx; by[static_cast<usize>(b)] -= wij * dy;
            };
            edge(i0, i1, w[0], r[0] - r[2], r[1] - r[3]);  // edge (0,1)
            edge(i1, i2, w[1], r[2] - r[4], r[3] - r[5]);  // edge (1,2)
            edge(i0, i2, w[2], r[0] - r[4], r[1] - r[5]);  // edge (0,2)
        }
        cg(L, bx, ux, cg_iters, 1e-8);
        cg(L, by, uy, cg_iters, 1e-8);
    }
    M.energy_final = energy();

    M.uv.resize(static_cast<usize>(Nv));
    f64 mnx = 1e300, mny = 1e300;
    for (s64 i = 0; i < Nv; ++i) { mnx = std::min(mnx, ux[static_cast<usize>(i)]); mny = std::min(mny, uy[static_cast<usize>(i)]); }
    for (s64 i = 0; i < Nv; ++i) M.uv[static_cast<usize>(i)] = {static_cast<f32>(ux[static_cast<usize>(i)] - mnx), static_cast<f32>(uy[static_cast<usize>(i)] - mny)};
    return M;
}

}  // namespace fenix::flatten
