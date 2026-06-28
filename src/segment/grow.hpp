// segment/grow.hpp — generational surface region-grower (the VC-style tracer, first-party).
// Grows a (u,v) grid of 3D points outward from a seed across the predicted-surface field:
// each new grid point is extrapolated from its already-placed neighbours and then SNAPPED onto
// the sheet by a short search along the local across-sheet normal (so it locks to the current
// wrap and cannot jump to an adjacent one — the key to not cutting across wraps), with periodic
// Laplacian relaxation + re-snap for regularity. Normals come from the structure tensor of a
// downsampled copy of the field (computed natively here). Produces a Surface (tifxyz/.fxsurf).
#pragma once

#include "core/core.hpp"
#include "core/eig.hpp"
#include "core/sampling.hpp"
#include "core/surface.hpp"
#include "segment/structure_tensor.hpp"

#include <array>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fenix::segment {

// Low-resolution across-sheet normal field (sampled in full-res voxel coordinates).
struct NormalField {
    int ds = 8;
    Volume<f32> nz, ny, nx;
    [[nodiscard]] Vec3f at(Vec3f c) const {
        const Vec3f p{c.z / static_cast<f32>(ds), c.y / static_cast<f32>(ds), c.x / static_cast<f32>(ds)};
        Vec3f n{sample_trilinear(nz.view(), p), sample_trilinear(ny.view(), p), sample_trilinear(nx.view(), p)};
        const f32 l = norm(n);
        return l > 1e-6f ? n / l : Vec3f{1, 0, 0};
    }
};

// Compute the across-sheet normal field natively: mean-downsample `vol` by `ds`, take the
// structure tensor, keep the largest-eigenvalue eigenvector (across-sheet direction).
template <class T>
inline NormalField compute_normal_field(VolumeView<const T> vol, int ds, StParams st = {1.0f, 2.0f}) {
    const Extent3 d = vol.dims();
    const Extent3 dd{d.z / ds, d.y / ds, d.x / ds};
    Volume<f32> small(dd);
    auto sv = small.view();
    const f32 inv = 1.0f / static_cast<f32>(ds * ds * ds);
    parallel_for_z(dd, [&](s64 z) {
        for (s64 y = 0; y < dd.y; ++y)
            for (s64 x = 0; x < dd.x; ++x) {
                f32 acc = 0;
                for (int dz = 0; dz < ds; ++dz)
                    for (int dy = 0; dy < ds; ++dy)
                        for (int dx = 0; dx < ds; ++dx)
                            acc += static_cast<f32>(vol(z * ds + dz, y * ds + dy, x * ds + dx));
                sv(z, y, x) = acc * inv;
            }
    });
    SheetField sf = structure_tensor(small.view(), st);
    NormalField nf{ds, Volume<f32>(dd), Volume<f32>(dd), Volume<f32>(dd)};
    for (s64 i = 0; i < dd.count(); ++i) {
        nf.nz.flat()[i] = sf.normal[static_cast<usize>(i)].z;
        nf.ny.flat()[i] = sf.normal[static_cast<usize>(i)].y;
        nf.nx.flat()[i] = sf.normal[static_cast<usize>(i)].x;
    }
    return nf;
}

struct GrowParams {
    f32 step = 2.0f;          // grid spacing (voxels)
    f32 surf_thresh = 0.15f;  // min field value (in the field's own units) to accept a point
    f32 snap_radius = 4.0f;   // +/- search along the normal; keep < half the inter-wrap spacing
    int max_gen = 4000;       // generation cap
    int grid = 1400;          // (u,v) grid size
    int fold_thresh = 10;     // reject a point landing in a 3D bin owned by a uv-cell >this far away
    f32 bin_size = 2.0f;      // 3D occupancy bin (voxels) for the injectivity guard
    f32 lambda = 3.0f;        // ARAP data weight (pull onto sheet vs. stay isometric)
    int fit_every = 0;        // interleave a light ARAP fit every N generations (0 = grow free first)
    int final_outer = 12;     // outer ARAP iterations in the final global polish
    int final_inner = 25;     // Gauss-Seidel sweeps per outer (more => more global propagation)
};

namespace detail {
// Snap c onto the sheet ridge by maximizing the field along +/- n; parabolic sub-voxel refine.
// Returns {position, field value there, signed offset traveled}.
template <class T>
inline std::tuple<Vec3f, f32, f32> snap_to_sheet(VolumeView<const T> f, Vec3f c, Vec3f n, f32 R) {
    const f32 dt = 0.5f;
    f32 best = -1e30f, bestt = 0;
    for (f32 t = -R; t <= R + 1e-3f; t += dt) {
        const f32 v = sample_trilinear(f, c + n * t);
        if (v > best) { best = v; bestt = t; }
    }
    const f32 vm = sample_trilinear(f, c + n * (bestt - dt));
    const f32 vp = sample_trilinear(f, c + n * (bestt + dt));
    const f32 den = vm - 2.0f * best + vp;
    f32 sub = std::abs(den) > 1e-6f ? 0.5f * (vm - vp) / den : 0.0f;
    sub = std::clamp(sub, -1.0f, 1.0f);
    const f32 tt = bestt + sub * dt;
    const Vec3f pos = c + n * tt;
    return {pos, sample_trilinear(f, pos), tt};
}

// --- small 3x3 helpers for the ARAP local step (component index 0/1/2 == z/y/x) ---
inline f32 cmp(const Vec3f& v, int i) { return i == 0 ? v.z : (i == 1 ? v.y : v.x); }
inline f32 det3(const Mat3f& a) {
    return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) -
           a.m[0][1] * (a.m[1][0] * a.m[2][2] - a.m[1][2] * a.m[2][0]) +
           a.m[0][2] * (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]);
}
// Nearest rotation to C (polar factor) via C * (CᵀC)^(-1/2), with a reflection fix.
inline Mat3f polar_rot(const Mat3f& C) {
    Mat3f CtC{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) CtC.m[i][j] += C.m[k][i] * C.m[k][j];
    const auto e = sym_eig3<f32>(CtC.m[0][0], CtC.m[1][1], CtC.m[2][2], CtC.m[0][1], CtC.m[0][2], CtC.m[1][2]);
    f32 s[3] = {1, 1, 1};
    if (det3(C) < 0) s[2] = -1;  // flip smallest-singular-value direction
    Mat3f is{};
    for (int k = 0; k < 3; ++k) {
        const f32 c = s[k] / std::sqrt(std::max(e.values[k], 1e-8f));
        const Vec3f vk = e.vectors[k];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) is.m[i][j] += c * cmp(vk, i) * cmp(vk, j);
    }
    return C * is;
}

// Local/global ARAP surface fit: treat the (u,v) grid as a flat rest mesh (edge=step) deformed
// onto the sheet. LOCAL step: per-vertex best rotation of the rest 1-ring onto the current
// positions. GLOBAL step: in-place Gauss-Seidel solve placing each vertex to match its rotated
// rest edges to neighbours PLUS a data term pulling it onto the sheet (snap). The global coupling
// distributes distortion over the whole patch — this is what removes the radial fan that purely
// local growth/relaxation produces.
template <class T>
inline void arap_fit(Surface& S, VolumeView<const T> f, const NormalField& nf, const GrowParams& p,
                     int outer, int inner, f32 lambda, bool interior_only = false) {
    const int G = static_cast<int>(S.nu);
    const Extent3 D = f.dims();
    const f32 mgn = p.snap_radius + 2.0f;
    auto inb = [&](Vec3f c) {
        return c.z >= mgn && c.y >= mgn && c.x >= mgn && c.z < static_cast<f32>(D.z) - mgn &&
               c.y < static_cast<f32>(D.y) - mgn && c.x < static_cast<f32>(D.x) - mgn;
    };
    auto rest = [&](int u, int v) { return Vec3f{0.0f, static_cast<f32>(v) * p.step, static_cast<f32>(u) * p.step}; };
    const usize NG = static_cast<usize>(G) * static_cast<usize>(G);
    std::vector<Mat3f> R(NG, Mat3f::identity());
    std::vector<Vec3f> Tg(NG);
    std::vector<u8> hasT(NG, 0);
    const int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};

    for (int o = 0; o < outer; ++o) {
        // data targets: project each vertex onto the sheet
        for (int v = 1; v < G - 1; ++v)
            for (int u = 1; u < G - 1; ++u) {
                if (!S.is_valid(u, v)) continue;
                const Vec3f P = S.at(u, v);
                auto [q, val, tt] = snap_to_sheet<T>(f, P, nf.at(P), p.snap_radius);
                const usize i = S.idx(u, v);
                hasT[i] = (val >= p.surf_thresh * 0.5f && std::abs(tt) < p.snap_radius && inb(q)) ? 1 : 0;
                Tg[i] = hasT[i] ? q : P;
            }
        // local rotations (polar of the rest->current 1-ring covariance)
        for (int v = 1; v < G - 1; ++v)
            for (int u = 1; u < G - 1; ++u) {
                if (!S.is_valid(u, v)) continue;
                Mat3f C{};
                int cnt = 0;
                const Vec3f Pi = S.at(u, v), Ri = rest(u, v);
                for (int k = 0; k < 4; ++k) {
                    const int uu = u + du4[k], vv = v + dv4[k];
                    if (!S.is_valid(uu, vv)) continue;
                    const Vec3f eP = Pi - S.at(uu, vv), eR = Ri - rest(uu, vv);
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b) C.m[a][b] += cmp(eP, a) * cmp(eR, b);
                    ++cnt;
                }
                R[S.idx(u, v)] = cnt >= 2 ? polar_rot(C) : Mat3f::identity();
            }
        // global Gauss-Seidel sweeps
        for (int s = 0; s < inner; ++s)
            for (int v = 1; v < G - 1; ++v)
                for (int u = 1; u < G - 1; ++u) {
                    if (!S.is_valid(u, v)) continue;
                    const usize i = S.idx(u, v);
                    const Vec3f Ri = rest(u, v);
                    Vec3f acc{0, 0, 0};
                    f32 w = 0;
                    int nnb = 0;
                    for (int k = 0; k < 4; ++k) {
                        const int uu = u + du4[k], vv = v + dv4[k];
                        if (!S.is_valid(uu, vv)) continue;
                        ++nnb;
                        const usize j = S.idx(uu, vv);
                        const Vec3f d = Ri - rest(uu, vv);
                        const Vec3f term = (R[i] * d + R[j] * d) * 0.5f;
                        acc = acc + (S.coord[j] + term);
                        w += 1.0f;
                    }
                    if (w == 0) continue;
                    if (interior_only && nnb < 4) continue;  // leave the frontier free to grow
                    if (hasT[i]) { acc = acc + Tg[i] * lambda; w += lambda; }
                    S.coord[i] = acc / w;
                }
    }
}

// Drop points off the sheet or attached by TEAR edges (long edges = the streaks). Tear-aware.
template <class T>
inline void cleanup_outliers(Surface& S, VolumeView<const T> f, const NormalField& nf, const GrowParams& p) {
    const int G = static_cast<int>(S.nu);
    const int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};
    for (int pass = 0; pass < 4; ++pass) {
        std::vector<usize> kill;
        for (int v = 0; v < G; ++v)
            for (int u = 0; u < G; ++u) {
                if (!S.is_valid(u, v)) continue;
                const Vec3f P = S.at(u, v);
                const f32 val = sample_trilinear(f, P);
                int nbr = 0, bad = 0, shortn = 0;
                for (int k = 0; k < 4; ++k) {
                    const int uu = u + du4[k], vv = v + dv4[k];
                    if (uu < 0 || vv < 0 || uu >= G || vv >= G || !S.is_valid(uu, vv)) continue;
                    ++nbr;
                    const f32 d = norm(P - S.at(uu, vv));
                    if (d > 1.8f * p.step) ++bad;       // tear edge
                    if (d < 0.5f * p.step) ++shortn;    // collapsed edge -> sliver source
                }
                if (val < p.surf_thresh * 0.8f || bad >= 2 || shortn >= 1 || nbr <= 1) kill.push_back(S.idx(u, v));
            }
        if (kill.empty()) break;
        for (usize i : kill) S.valid[i] = 0;
    }
}

// Remove sliver triangles (bad aspect, edges OK) by dropping the apex vertex (the one nearest its
// opposite edge). These are mostly ragged-boundary triangles; iterated to convergence. Degenerate
// geometry is killed HERE (in the tracer) so downstream (flatten/codec/render) sees clean meshes.
inline void remove_slivers(Surface& S, const GrowParams& p) {
    const int G = static_cast<int>(S.nu);
    auto distline = [](Vec3f x, Vec3f a, Vec3f b) {
        const Vec3f ab = b - a, ax = x - a;
        const f32 L = norm(ab);
        return L > 1e-6f ? norm(cross(ax, ab)) / L : norm(ax);
    };
    auto aspect = [&](Vec3f A, Vec3f B, Vec3f C) {
        const f32 me = std::max({norm(B - A), norm(C - A), norm(C - B)});
        return me > 1e-6f ? (0.5f * norm(cross(B - A, C - A))) / (me * me) : 0.0f;
    };
    for (int pass = 0; pass < 4; ++pass) {
        std::vector<usize> kill;
        auto consider = [&](int au, int av, int bu, int bv, int cu, int cv) {
            const Vec3f A = S.at(au, av), B = S.at(bu, bv), C = S.at(cu, cv);
            if (aspect(A, B, C) >= 0.08f) return;
            const f32 da = distline(A, B, C), db = distline(B, A, C), dc = distline(C, A, B);
            if (da <= db && da <= dc) kill.push_back(S.idx(au, av));
            else if (db <= dc) kill.push_back(S.idx(bu, bv));
            else kill.push_back(S.idx(cu, cv));
        };
        for (int v = 0; v + 1 < S.nv; ++v)
            for (int u = 0; u + 1 < G; ++u) {
                const bool a = S.is_valid(u, v), b = S.is_valid(u + 1, v), c = S.is_valid(u, v + 1), d = S.is_valid(u + 1, v + 1);
                if (a && b && c) consider(u, v, u + 1, v, u, v + 1);
                if (b && d && c) consider(u + 1, v, u + 1, v + 1, u, v + 1);
            }
        if (kill.empty()) break;
        for (usize i : kill) S.valid[i] = 0;
    }
}

// Fill interior holes: an invalid cell with >=3 valid 4-neighbours is interpolated from them and
// snapped onto the sheet (accepted only if on-sheet and the new edges aren't tears). Iterated.
template <class T>
inline void fill_holes(Surface& S, VolumeView<const T> f, const NormalField& nf, const GrowParams& p) {
    const int G = static_cast<int>(S.nu);
    const int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};
    const Extent3 D = f.dims();
    const f32 mgn = p.snap_radius + 2.0f;
    auto inb = [&](Vec3f c) {
        return c.z >= mgn && c.y >= mgn && c.x >= mgn && c.z < static_cast<f32>(D.z) - mgn &&
               c.y < static_cast<f32>(D.y) - mgn && c.x < static_cast<f32>(D.x) - mgn;
    };
    for (int pass = 0; pass < 6; ++pass) {
        std::vector<std::pair<usize, Vec3f>> add;
        for (int v = 1; v < G - 1; ++v)
            for (int u = 1; u < G - 1; ++u) {
                if (S.is_valid(u, v)) continue;
                Vec3f sum{0, 0, 0};
                int nbr = 0;
                for (int k = 0; k < 4; ++k) { const int uu = u + du4[k], vv = v + dv4[k]; if (S.is_valid(uu, vv)) { sum = sum + S.at(uu, vv); ++nbr; } }
                if (nbr < 3) continue;
                const Vec3f c = sum / static_cast<f32>(nbr);
                if (!inb(c)) continue;
                auto [q, val, tt] = snap_to_sheet<T>(f, c, nf.at(c), p.snap_radius);
                if (val < p.surf_thresh || std::abs(tt) > p.snap_radius || !inb(q)) continue;
                bool tear = false;
                for (int k = 0; k < 4; ++k) { const int uu = u + du4[k], vv = v + dv4[k]; if (S.is_valid(uu, vv) && norm(q - S.at(uu, vv)) > 1.8f * p.step) tear = true; }
                if (!tear) add.emplace_back(S.idx(u, v), q);
            }
        if (add.empty()) break;
        for (auto& [i, q] : add) { S.coord[i] = q; S.valid[i] = 1; }
    }
}
}  // namespace detail

// Geometric quality of a traced Surface — the acceptance gates: consistent spacing, no folds,
// and injectivity (no self-intersection / wrap-crossing). All "want ~0" except coverage (~1).
struct SurfQuality {
    s64 valid = 0;
    f64 spacing_cv = 0, spacing_oor = 0;  // edge-length coeff-of-variation; frac outside [.5,1.5]xmedian
    f64 min_edge = 0, frac_short = 0;      // shortest edge; frac of edges < 0.5xmedian (collapses)
    f64 degen_tri = 0;                     // frac of grid triangles with aspect (area/maxedge^2) < 0.08
    f64 fold_rate = 0;                     // frac of adjacent vertex-normals that flip (local folds)
    f64 overlap = 0, distant_fold = 0;     // frac of points sharing a 3D bin; ... with a uv-distant cell
    f64 coverage = 0;                      // unique bins / points (1 = injective)
};
inline SurfQuality surface_quality(const Surface& S, f32 bin_size, int fold_thresh) {
    const s64 G = S.nu;
    SurfQuality q;
    q.valid = S.valid_count();
    // spacing
    std::vector<f32> E;
    E.reserve(static_cast<usize>(q.valid * 2));
    for (s64 v = 0; v < S.nv; ++v)
        for (s64 u = 0; u < G; ++u) {
            if (!S.is_valid(u, v)) continue;
            if (u + 1 < G && S.is_valid(u + 1, v)) E.push_back(norm(S.at(u + 1, v) - S.at(u, v)));
            if (v + 1 < S.nv && S.is_valid(u, v + 1)) E.push_back(norm(S.at(u, v + 1) - S.at(u, v)));
        }
    f32 med = 0;
    if (!E.empty()) {
        f64 m = 0; for (f32 e : E) m += e; m /= static_cast<f64>(E.size());
        f64 s2 = 0; for (f32 e : E) s2 += (e - m) * (e - m); s2 /= static_cast<f64>(E.size());
        std::vector<f32> tmp = E; std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
        med = tmp[tmp.size() / 2];
        q.spacing_cv = std::sqrt(s2) / (m + 1e-9);
        s64 oor = 0, sh = 0; q.min_edge = 1e30;
        for (f32 e : E) { if (e < 0.5f * med || e > 1.5f * med) ++oor; if (e < 0.5f * med) ++sh; q.min_edge = std::min(q.min_edge, static_cast<f64>(e)); }
        q.spacing_oor = static_cast<f64>(oor) / static_cast<f64>(E.size());
        q.frac_short = static_cast<f64>(sh) / static_cast<f64>(E.size());
    }
    // degenerate-triangle fraction (sliver aspect): the geometry the flattener can't tolerate
    {
        s64 nt = 0, bad = 0;
        auto aspect = [&](Vec3f A, Vec3f B, Vec3f C) {
            const f64 ar = 0.5 * norm(cross(B - A, C - A));
            const f64 me = std::max({norm(B - A), norm(C - A), norm(C - B)});
            return me > 1e-6 ? ar / (me * me) : 0.0;
        };
        for (s64 v = 0; v + 1 < S.nv; ++v)
            for (s64 u = 0; u + 1 < G; ++u) {
                const bool a = S.is_valid(u, v), b = S.is_valid(u + 1, v), c = S.is_valid(u, v + 1), d = S.is_valid(u + 1, v + 1);
                if (a && b && c) { ++nt; if (aspect(S.at(u, v), S.at(u + 1, v), S.at(u, v + 1)) < 0.08) ++bad; }
                if (b && d && c) { ++nt; if (aspect(S.at(u + 1, v), S.at(u + 1, v + 1), S.at(u, v + 1)) < 0.08) ++bad; }
            }
        q.degen_tri = nt ? static_cast<f64>(bad) / static_cast<f64>(nt) : 0;
    }
    // local fold rate via central-difference vertex normals
    auto nrm_at = [&](s64 u, s64 v) -> Vec3f {
        const Vec3f tu = S.at(u + 1, v) - S.at(u - 1, v), tv = S.at(u, v + 1) - S.at(u, v - 1);
        return normalized(cross(tu, tv));
    };
    auto core = [&](s64 u, s64 v) {
        return u > 0 && v > 0 && u + 1 < G && v + 1 < S.nv && S.is_valid(u, v) && S.is_valid(u - 1, v) &&
               S.is_valid(u + 1, v) && S.is_valid(u, v - 1) && S.is_valid(u, v + 1);
    };
    s64 flips = 0, ftot = 0;
    for (s64 v = 1; v + 1 < S.nv; ++v)
        for (s64 u = 1; u + 1 < G; ++u) {
            if (!core(u, v)) continue;
            const Vec3f n = nrm_at(u, v);
            if (core(u + 1, v)) { ++ftot; if (dot(n, nrm_at(u + 1, v)) < 0) ++flips; }
            if (core(u, v + 1)) { ++ftot; if (dot(n, nrm_at(u, v + 1)) < 0) ++flips; }
        }
    q.fold_rate = ftot ? static_cast<f64>(flips) / static_cast<f64>(ftot) : 0;
    // injectivity: 3D bin sharing
    std::unordered_map<s64, s64> occ;
    occ.reserve(static_cast<usize>(q.valid));
    const f32 ib = 1.0f / bin_size;
    const s64 BS = 1 << 20;
    s64 ov = 0, dfold = 0, total = 0;
    for (s64 v = 0; v < S.nv; ++v)
        for (s64 u = 0; u < G; ++u) {
            if (!S.is_valid(u, v)) continue;
            ++total;
            const Vec3f c = S.at(u, v);
            const s64 key = (static_cast<s64>(c.z * ib)) * BS * BS + (static_cast<s64>(c.y * ib)) * BS + static_cast<s64>(c.x * ib);
            auto it = occ.find(key);
            if (it == occ.end()) { occ[key] = static_cast<s64>(v) * G + u; continue; }
            ++ov;
            const int ou = static_cast<int>(it->second % G), ovv = static_cast<int>(it->second / G);
            if (std::abs(static_cast<int>(u) - ou) + std::abs(static_cast<int>(v) - ovv) > fold_thresh) ++dfold;
        }
    if (total) { q.overlap = static_cast<f64>(ov) / static_cast<f64>(total); q.distant_fold = static_cast<f64>(dfold) / static_cast<f64>(total); }
    q.coverage = total ? static_cast<f64>(occ.size()) / static_cast<f64>(total) : 0;
    return q;
}

// Grow a sheet from `seed` (ZYX voxel coords) across the field `f` (high on the sheet).
template <class T>
inline Surface grow_surface(VolumeView<const T> f, const NormalField& nf, Vec3f seed, GrowParams p) {
    const int G = p.grid, C = G / 2;
    const Extent3 D = f.dims();
    const f32 mgn = p.snap_radius + 2.0f;
    auto inb = [&](Vec3f c) {
        return c.z >= mgn && c.y >= mgn && c.x >= mgn && c.z < static_cast<f32>(D.z) - mgn &&
               c.y < static_cast<f32>(D.y) - mgn && c.x < static_cast<f32>(D.x) - mgn;
    };
    Surface S(G, G);
    const usize NG = static_cast<usize>(G) * static_cast<usize>(G);
    std::vector<u8> dead(NG, 0);
    std::vector<Vec3f> Tu(NG), Tv(NG);  // per-vertex tangent frame (parallel-transported)
    auto N = [&](Vec3f c) { return nf.at(c); };
    auto snap = [&](Vec3f c, Vec3f n) { return detail::snap_to_sheet<T>(f, c, n, p.snap_radius); };
    // Re-express frame (tu,tv) in the tangent plane of normal n (Gram-Schmidt) — parallel transport.
    auto transport = [&](Vec3f tu, Vec3f tv, Vec3f n) -> std::pair<Vec3f, Vec3f> {
        Vec3f u = tu - n * dot(tu, n);
        if (norm(u) < 1e-4f) { const Vec3f e = (std::abs(n.x) < 0.9f) ? Vec3f{0, 0, 1} : Vec3f{1, 0, 0}; u = e - n * dot(e, n); }
        u = normalized(u);
        Vec3f v = tv - n * dot(tv, n);
        v = v - u * dot(v, u);
        if (norm(v) < 1e-4f) v = cross(n, u);
        return {u, normalized(v)};
    };
    auto V = [&](int u, int v) { return S.is_valid(u, v); };

    // --- seed patch (3x3), frame parallel-transported from a projected global frame ---
    const Vec3f n0 = N(seed);
    auto [sp, sv0, st0] = snap(seed, n0);
    (void)sv0; (void)st0;
    auto [su, sv] = transport(Vec3f{0, 0, 1}, Vec3f{0, 1, 0}, n0);
    S.set(C, C, sp);
    Tu[S.idx(C, C)] = su; Tv[S.idx(C, C)] = sv;
    for (int dv = -1; dv <= 1; ++dv)
        for (int du = -1; du <= 1; ++du) {
            if (du == 0 && dv == 0) continue;
            const Vec3f c = sp + su * (static_cast<f32>(du) * p.step) + sv * (static_cast<f32>(dv) * p.step);
            if (!inb(c)) continue;
            auto [q, vv, tt] = snap(c, N(c));
            (void)tt;
            if (vv >= p.surf_thresh && inb(q)) {
                S.set(C + du, C + dv, q);
                auto [fu, fv] = transport(su, sv, N(q));
                Tu[S.idx(C + du, C + dv)] = fu; Tv[S.idx(C + du, C + dv)] = fv;
            }
        }

    // 3D occupancy for the injectivity guard: bin -> owning cell index. A new point whose bin
    // (or 26-neighbourhood) is already owned by a uv-DISTANT cell is a self-intersection -> reject.
    std::unordered_map<s64, s64> occ;
    occ.reserve(NG / 4);
    const f32 ibin = 1.0f / p.bin_size;
    const s64 BS = 1 << 20;
    auto binkey = [&](Vec3f c, int oz, int oy, int ox) {
        return (static_cast<s64>(c.z * ibin) + oz) * BS * BS + (static_cast<s64>(c.y * ibin) + oy) * BS +
               (static_cast<s64>(c.x * ibin) + ox);
    };
    auto fold_conflict = [&](Vec3f q, int u, int v) {
        for (int oz = -1; oz <= 1; ++oz)
            for (int oy = -1; oy <= 1; ++oy)
                for (int ox = -1; ox <= 1; ++ox) {
                    auto it = occ.find(binkey(q, oz, oy, ox));
                    if (it == occ.end()) continue;
                    const int ou = static_cast<int>(it->second % G), ov = static_cast<int>(it->second / G);
                    if (std::abs(u - ou) + std::abs(v - ov) > p.fold_thresh) return true;
                }
        return false;
    };
    auto claim = [&](Vec3f q, int u, int v) { occ[binkey(q, 0, 0, 0)] = static_cast<s64>(v) * G + u; };
    for (int dv = -1; dv <= 1; ++dv)
        for (int du = -1; du <= 1; ++du)
            if (V(C + du, C + dv)) claim(S.at(C + du, C + dv), C + du, C + dv);

    const int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};
    for (int gen = 0; gen < p.max_gen; ++gen) {
        std::vector<std::array<int, 2>> todo;
        for (int v = 1; v < G - 1; ++v)
            for (int u = 1; u < G - 1; ++u) {
                if (V(u, v) || dead[static_cast<usize>(v) * G + u]) continue;
                if (V(u - 1, v) || V(u + 1, v) || V(u, v - 1) || V(u, v + 1)) todo.push_back({u, v});
            }
        if (todo.empty()) break;

        int placed = 0;
        for (auto [u, v] : todo) {
            // Predict from each valid neighbour by stepping along ITS transported frame toward
            // (u,v); average. Consistent frames -> no fan; single-neighbour OK -> no stall.
            Vec3f pred{0, 0, 0}, accu{0, 0, 0}, accv{0, 0, 0};
            int np = 0;
            for (int k = 0; k < 4; ++k) {
                const int nu = u + du4[k], nv = v + dv4[k];
                if (!V(nu, nv)) continue;
                const usize ni = S.idx(nu, nv);
                const f32 dU = static_cast<f32>(u - nu), dV = static_cast<f32>(v - nv);
                pred = pred + (S.at(nu, nv) + (Tu[ni] * dU + Tv[ni] * dV) * p.step);
                accu = accu + Tu[ni]; accv = accv + Tv[ni];
                ++np;
            }
            if (np == 0) continue;
            const Vec3f c = pred / static_cast<f32>(np);
            if (!inb(c)) { dead[static_cast<usize>(v) * G + u] = 1; continue; }
            const Vec3f nrm = N(c);
            auto [q, val, tt] = snap(c, nrm);
            if (!inb(q) || val < p.surf_thresh || std::abs(tt) > p.snap_radius - 0.51f) { dead[static_cast<usize>(v) * G + u] = 1; continue; }
            bool ok = true;
            for (int k = 0; k < 4; ++k) {
                const int uu = u + du4[k], vv = v + dv4[k];
                if (V(uu, vv)) { const f32 d = norm(q - S.at(uu, vv)); if (d > 2.5f * p.step || d < 0.5f * p.step) ok = false; }  // no collapse/tear
            }
            if (!ok) { dead[static_cast<usize>(v) * G + u] = 1; continue; }
            if (fold_conflict(q, u, v)) { dead[static_cast<usize>(v) * G + u] = 1; continue; }  // injectivity guard
            S.set(u, v, q);
            claim(q, u, v);
            auto [fu, fv] = transport(normalized(accu), normalized(accv), N(q));
            Tu[S.idx(u, v)] = fu; Tv[S.idx(u, v)] = fv;
            ++placed;
        }

        if (p.fit_every > 0 && (gen % p.fit_every) == 0) detail::arap_fit<T>(S, f, nf, p, 2, 2, p.lambda, /*interior_only=*/true);
        if (placed == 0) break;
    }
    detail::cleanup_outliers<T>(S, f, nf, p);                                       // remove tears/collapses/off-sheet
    detail::remove_slivers(S, p);                                                   // remove bad-aspect slivers
    detail::fill_holes<T>(S, f, nf, p);                                             // fill interior holes
    detail::arap_fit<T>(S, f, nf, p, p.final_outer, p.final_inner, p.lambda, false); // global polish
    detail::cleanup_outliers<T>(S, f, nf, p);                                       // final tear/collapse sweep
    detail::remove_slivers(S, p);                                                   // final sliver sweep
    return S;
}

}  // namespace fenix::segment
