// ml/surf_consist.hpp — `fenix surf-consist`: inter-mesh consistency + physics sanity QC
// (torch-free, no CT needed). Overlapping segments trace the same physical sheets, so where
// two meshes come close their geometry is a label oracle independent of CT intensity:
//   AGREE  median surface-to-surface distance ~0-2 vox  -> both traces confirmed there
//   OFFSET median 2..near vox                           -> at least one is misregistered
//   CROSS  the other mesh appears on BOTH sides at small distance -> physically impossible
//          (sheets never interpenetrate) — one of the traces is damaged
// Plus a per-mesh normal-coherence check (a sheet's normals vary smoothly; scrambled normals
// = degenerate uv grid or corrupt resample).
//
// Distances are point-to-SURFACE: corpus uv grids are coarse (20-vox steps on the VC
// tifxyz lineage), so raw grid-point clouds are 20-80 vox apart and point-to-point distance
// at near=6 is pure sampling noise. A coarse cloud finds the candidate cell; the true
// distance comes from subdividing the bilinear patches around it (~1.5-vox steps), same
// surface model as the rasterizer. Pairs stream (load 2 meshes at a time) — 78 corpus
// meshes would not fit as resident Surfaces.
//   fenix surf-consist <fxsurf...> [k=3000] [near=6] [pts_mb=16]
// One line per mesh (coherence) + one per overlapping pair. Coords must share a voxel space
// (the -on-<volume> corpus resamples are absolute scroll coords, so they do).
#pragma once

#include "core/core.hpp"
#include "core/surface.hpp"
#include "core/vec.hpp"
#include "io/surface.hpp"
#include "ml/surf_qc.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fenix::ml {

namespace detail {

struct MeshCloud {
    std::string path;
    std::optional<Surface> surf;  // kept only when bilinear refinement is needed (pair pass)
    std::vector<Vec3f> pts;       // coarse grid samples (valid cells at stride)
    std::vector<Vec3f> nrm;       // stencil normal per sample (unit; sign is arbitrary)
    std::vector<std::array<s32, 2>> uvs;  // grid coords of each sample (refinement seed)
    f32 spacing = 0;                      // coarse sample spacing (vox)
    f32 lo[3] = {}, hi[3] = {};           // bbox (z,y,x)
    s64 co_n = 0, co_ok = 0;              // normal-coherence tallies
    std::unordered_map<u64, std::vector<u32>> grid;  // cell -> point indices
    f32 cell = 6.0f;

    [[nodiscard]] static u64 key(s64 cz, s64 cy, s64 cx) {
        // 21-bit fields: coords <= 2^18, /cell keeps them far under 2^21; negatives cannot
        // occur (scroll coords are non-negative)
        return (static_cast<u64>(cz) << 42) | (static_cast<u64>(cy) << 21) | static_cast<u64>(cx);
    }
    void build_grid(f32 c) {
        cell = c;
        grid.clear();
        grid.reserve(pts.size() / 2);
        for (u32 i = 0; i < pts.size(); ++i) {
            const Vec3f& p = pts[i];
            grid[key(static_cast<s64>(p.z / cell), static_cast<s64>(p.y / cell), static_cast<s64>(p.x / cell))]
                .push_back(i);
        }
    }
    // nearest coarse sample within `r` of q; returns index or -1
    [[nodiscard]] s64 nearest(Vec3f q, f32 r) const {
        const s64 cz = static_cast<s64>(q.z / cell), cy = static_cast<s64>(q.y / cell),
                  cx = static_cast<s64>(q.x / cell);
        const s64 ring = std::max<s64>(1, static_cast<s64>(std::ceil(r / cell)));
        f32 best = r * r;
        s64 bi = -1;
        for (s64 dz = -ring; dz <= ring; ++dz)
            for (s64 dy = -ring; dy <= ring; ++dy)
                for (s64 dx = -ring; dx <= ring; ++dx) {
                    if (cz + dz < 0 || cy + dy < 0 || cx + dx < 0) continue;
                    const auto it = grid.find(key(cz + dz, cy + dy, cx + dx));
                    if (it == grid.end()) continue;
                    for (u32 i : it->second) {
                        const Vec3f d = pts[i] - q;
                        const f32 dd = d.z * d.z + d.y * d.y + d.x * d.x;
                        if (dd < best) {
                            best = dd;
                            bi = static_cast<s64>(i);
                        }
                    }
                }
        return bi;
    }
    // min distance from q to the bilinear surface within ±2 grid steps of (cu,cv),
    // sampled at <=`fine`-vox spacing (needs `surf`). Also returns the closest point.
    [[nodiscard]] f32 refined_dist(Vec3f q, s32 cu, s32 cv, f32 fine, Vec3f* closest) const {
        const Surface& s = *surf;
        const int su = std::max(1, static_cast<int>(std::ceil(s.scale_u / fine)));
        const int sv = std::max(1, static_cast<int>(std::ceil(s.scale_v / fine)));
        f32 best = 1e30f;
        for (s64 v = cv - 2; v <= cv + 1; ++v)
            for (s64 u = cu - 2; u <= cu + 1; ++u) {
                if (u < 0 || v < 0 || u + 1 >= s.nu || v + 1 >= s.nv) continue;
                if (!s.is_valid(u, v) || !s.is_valid(u + 1, v) || !s.is_valid(u, v + 1) || !s.is_valid(u + 1, v + 1))
                    continue;
                const Vec3f c00 = s.at(u, v), c10 = s.at(u + 1, v), c01 = s.at(u, v + 1), c11 = s.at(u + 1, v + 1);
                for (int j = 0; j <= sv; ++j)
                    for (int i = 0; i <= su; ++i) {
                        const f32 a = static_cast<f32>(i) / static_cast<f32>(su);
                        const f32 b = static_cast<f32>(j) / static_cast<f32>(sv);
                        const Vec3f p =
                            c00 * ((1 - a) * (1 - b)) + c10 * (a * (1 - b)) + c01 * ((1 - a) * b) + c11 * (a * b);
                        const Vec3f d = p - q;
                        const f32 dd = d.z * d.z + d.y * d.y + d.x * d.x;
                        if (dd < best) {
                            best = dd;
                            if (closest) *closest = p;
                        }
                    }
            }
        return std::sqrt(best);
    }
};

// Load a mesh as a coarse cloud (+bbox+coherence); keep_surf keeps the Surface resident for
// bilinear refinement. max_pts bounds the coarse cloud (it only seeds refinement).
inline Expected<MeshCloud> load_cloud(const std::string& path, usize max_pts, bool keep_surf) {
    auto s = io::read_fxsurf(path);
    if (!s) return std::unexpected(s.error());
    MeshCloud mc;
    mc.path = path;
    const f64 cells = static_cast<f64>(s->nu) * static_cast<f64>(s->nv);
    const s64 stride = std::max<s64>(1, static_cast<s64>(std::ceil(std::sqrt(cells / static_cast<f64>(max_pts)))));
    mc.spacing = static_cast<f32>(stride) * std::max(s->scale_u, s->scale_v);
    bool first = true;
    for (s64 v = 0; v < s->nv; v += stride)
        for (s64 u = 0; u < s->nu; u += stride) {
            if (!s->is_valid(u, v)) continue;
            const auto nm = stencil_normal(*s, u, v);
            if (!nm) continue;
            const Vec3f p = s->at(u, v);
            mc.pts.push_back(p);
            mc.nrm.push_back(*nm);
            mc.uvs.push_back({static_cast<s32>(u), static_cast<s32>(v)});
            if (first) {
                mc.lo[0] = mc.hi[0] = p.z;
                mc.lo[1] = mc.hi[1] = p.y;
                mc.lo[2] = mc.hi[2] = p.x;
                first = false;
            } else {
                mc.lo[0] = std::min(mc.lo[0], p.z);
                mc.hi[0] = std::max(mc.hi[0], p.z);
                mc.lo[1] = std::min(mc.lo[1], p.y);
                mc.hi[1] = std::max(mc.hi[1], p.y);
                mc.lo[2] = std::min(mc.lo[2], p.x);
                mc.hi[2] = std::max(mc.hi[2], p.x);
            }
            // coherence vs the next sample along u: |dot| >= 0.7 (~45°) — sign-free, sheets
            // have orientation ambiguity
            if (const auto n2 = stencil_normal(*s, u + stride, v)) {
                ++mc.co_n;
                const f32 d = nm->z * n2->z + nm->y * n2->y + nm->x * n2->x;
                mc.co_ok += std::abs(d) >= 0.7f;
            }
        }
    if (keep_surf) mc.surf = std::move(*s);
    return mc;
}

}  // namespace detail

inline Expected<int> run_surf_consist(std::span<const std::string_view> args, Context&) {
    if (args.empty())
        return err(Errc::invalid_argument, "usage: surf-consist <fxsurf...> [k=3000] [near=6] [pts_mb=16]");
    s64 k = 3000, pts_mb = 16;
    f64 near_r = 6;
    std::vector<std::string> paths;
    for (const auto a : args) {
        auto num = [&](std::string_view key2, auto& v) {
            if (!a.starts_with(key2)) return false;
            const auto t = a.substr(key2.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("k=", k) || num("near=", near_r) || num("pts_mb=", pts_mb)) continue;
        paths.emplace_back(a);
    }
    if (paths.size() < 1) return err(Errc::invalid_argument, "surf-consist: no meshes");
    const usize max_pts = std::max<usize>(4096, static_cast<usize>(pts_mb) * (1u << 20) / (2 * sizeof(Vec3f)));

    // pass 1: bbox + coherence per mesh (loaded one at a time; nothing kept but the summary)
    struct Info {
        std::string path;
        f32 lo[3], hi[3];
        bool ok = false;
    };
    std::vector<Info> infos;
    for (const auto& mp : paths) {
        auto mc = detail::load_cloud(mp, max_pts, /*keep_surf=*/false);
        if (!mc) return std::unexpected(mc.error());
        Info in;
        in.path = mp;
        std::copy(mc->lo, mc->lo + 3, in.lo);
        std::copy(mc->hi, mc->hi + 3, in.hi);
        in.ok = mc->pts.size() >= 16;
        if (!in.ok) {
            std::printf("surf-consist mesh %s  DEGENERATE (%zu usable points)\n", mp.c_str(), mc->pts.size());
        } else {
            const f64 coher = mc->co_n ? 100.0 * static_cast<f64>(mc->co_ok) / static_cast<f64>(mc->co_n) : 0.0;
            std::printf("surf-consist mesh %s  points %zu  spacing %.1f  normal-coherent %.0f%%%s\n",
                        mp.c_str(),
                        mc->pts.size(),
                        static_cast<f64>(mc->spacing),
                        coher,
                        coher < 90.0 ? "  INCOHERENT" : "");
        }
        infos.push_back(in);
    }
    std::fflush(stdout);

    // pass 2: stream bbox-overlapping pairs — A's coarse samples query B's refined surface
    int suspicious = 0;
    for (usize a = 0; a < infos.size(); ++a) {
        if (!infos[a].ok) continue;
        std::vector<usize> partners;
        for (usize b2 = a + 1; b2 < infos.size(); ++b2) {
            if (!infos[b2].ok) continue;
            bool ov = true;
            for (int d = 0; d < 3; ++d)
                ov &= infos[a].lo[d] <= infos[b2].hi[d] + static_cast<f32>(near_r) &&
                      infos[b2].lo[d] <= infos[a].hi[d] + static_cast<f32>(near_r);
            if (ov) partners.push_back(b2);
        }
        if (partners.empty()) continue;
        auto A = detail::load_cloud(infos[a].path, max_pts, /*keep_surf=*/false);
        if (!A) return std::unexpected(A.error());
        for (usize b2 : partners) {
            auto B = detail::load_cloud(infos[b2].path, max_pts, /*keep_surf=*/true);
            if (!B) return std::unexpected(B.error());
            // coarse grid at the sample spacing; a coincident surface's nearest coarse sample
            // is at most ~spacing/sqrt(2) away laterally, so search that far plus near
            B->build_grid(std::max(static_cast<f32>(near_r), B->spacing));
            const f32 R = B->spacing * 0.75f + static_cast<f32>(near_r);
            const usize step2 = std::max<usize>(1, A->pts.size() / static_cast<usize>(k));
            std::vector<f32> dists;
            s64 n_query = 0, n_pos = 0, n_signed = 0;
            for (usize i = 0; i < A->pts.size(); i += step2) {
                ++n_query;
                const s64 j = B->nearest(A->pts[i], R);
                if (j < 0) continue;
                Vec3f cp{};
                const f32 d = B->refined_dist(
                    A->pts[i], B->uvs[static_cast<usize>(j)][0], B->uvs[static_cast<usize>(j)][1], 1.5f, &cp);
                if (d > static_cast<f32>(near_r)) continue;
                dists.push_back(d);
                // side of A the closest B point sits on, along A's normal (only meaningful
                // when the surfaces are locally distinct: skip near-coincident points)
                const Vec3f dv = cp - A->pts[i];
                const f32 sgn = dv.z * A->nrm[i].z + dv.y * A->nrm[i].y + dv.x * A->nrm[i].x;
                if (std::abs(sgn) > 1.0f) {
                    ++n_signed;
                    n_pos += sgn > 0;
                }
            }
            if (dists.size() < 20) continue;  // too little overlap to say anything
            std::sort(dists.begin(), dists.end());
            const f32 med = dists[dists.size() / 2], p90 = dists[dists.size() * 9 / 10];
            const f64 ov_frac = static_cast<f64>(dists.size()) / static_cast<f64>(n_query);
            const f64 coinc = static_cast<f64>(std::count_if(dists.begin(), dists.end(), [](f32 x) {
                                  return x <= 2.0f;
                              })) /
                              static_cast<f64>(dists.size());
            const f64 fpos = n_signed ? static_cast<f64>(n_pos) / static_cast<f64>(n_signed) : 0.0;
            const f64 side_mix = std::min(fpos, 1.0 - fpos);
            // verdicts: duplicates of the same sheet AGREE (~coincident); a stable one-sided
            // offset at 2..near is a misregistration of at least one trace; both sides at
            // meaningful distance = the traces CROSS (impossible for real sheets). Tuned on
            // the Paris4 corpus sweep 2026-07-04: real duplicate traces oscillate ±1-3 vox
            // around each other, so coincidence must VETO the side-mix test (a med-0.8,
            // 94%-coincident pair is an AGREE, not a CROSS — noise flips sides constantly).
            // High coincidence vetoes CROSS: a true crossing shows ~50% coincidence (near the
            // crossing line) with mixed sides; a duplicate trace shows >75% coincidence and its
            // side flips are pure noise. Interleaved traces at med ~3 / coinc ~30% ARE crossings
            // (both labels can't be right there) and stay flagged.
            const char* verdict = "OFFSET";
            if (med <= 2.0f || coinc >= 0.75) verdict = "AGREE";
            if (side_mix > 0.15 && med < 4.0f && coinc < 0.75 && n_signed > 50) verdict = "CROSS";
            if (std::string_view(verdict) != "AGREE") ++suspicious;
            std::printf("surf-consist pair %s <-> %s  overlap %.0f%% (n=%zu)  med %.1f  p90 %.1f  "
                        "coincident %.0f%%  side-mix %.2f  %s\n",
                        infos[a].path.c_str(),
                        infos[b2].path.c_str(),
                        100.0 * ov_frac,
                        dists.size(),
                        static_cast<f64>(med),
                        static_cast<f64>(p90),
                        100.0 * coinc,
                        side_mix,
                        verdict);
            std::fflush(stdout);
        }
    }
    std::printf("surf-consist: %d suspicious pair(s)\n", suspicious);
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_surf_consist = ::fenix::register_stage(
    ::fenix::Stage{"surf-consist",
                   "inter-mesh consistency QC (overlap distance, crossings, normal coherence)",
                   ::fenix::ml::run_surf_consist});
}  // namespace
