// ml/surf_consist.hpp — `fenix surf-consist`: inter-mesh consistency + physics sanity QC
// (torch-free, no CT needed). Overlapping segments trace the same physical sheets, so where
// two meshes come close their geometry is a label oracle independent of CT intensity:
//   AGREE  median surface-to-surface distance ~0-2 vox  -> both traces confirmed there
//   OFFSET median 2..near vox                           -> at least one is misregistered
//   CROSS  the other mesh appears on BOTH sides at small distance -> physically impossible
//          (sheets never interpenetrate) — one of the traces is damaged
// Plus a per-mesh normal-coherence check (a sheet's normals vary smoothly; scrambled normals
// = degenerate uv grid or corrupt resample).
//   fenix surf-consist <fxsurf...> [k=4000] [near=6] [pts_mb=64]
// One line per mesh (coherence) + one per overlapping pair. Coords must share a voxel space
// (the -on-<volume> corpus resamples are absolute scroll coords, so they do).
#pragma once

#include "core/core.hpp"
#include "core/surface.hpp"
#include "core/vec.hpp"
#include "io/surface.hpp"
#include "ml/surf_qc.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fenix::ml {

namespace detail {

struct MeshCloud {
    std::string path;
    std::vector<Vec3f> pts;
    std::vector<Vec3f> nrm;  // stencil normal per point (unit; sign is arbitrary)
    f32 spacing = 0;         // max sample spacing in voxels (distance quantization bound)
    f32 lo[3] = {}, hi[3] = {};  // bbox (z,y,x)
    std::unordered_map<u64, std::vector<u32>> grid;  // cell -> point indices
    f32 cell = 6.0f;

    [[nodiscard]] static u64 key(s64 cz, s64 cy, s64 cx) {
        // 21-bit fields: coords <= 2^18, /cell keeps them far under 2^21; negatives cannot
        // occur (scroll coords are non-negative)
        return (static_cast<u64>(cz) << 42) | (static_cast<u64>(cy) << 21) | static_cast<u64>(cx);
    }
    void build_grid(f32 c) {
        cell = c;
        grid.reserve(pts.size() / 2);
        for (u32 i = 0; i < pts.size(); ++i) {
            const Vec3f& p = pts[i];
            grid[key(static_cast<s64>(p.z / cell), static_cast<s64>(p.y / cell), static_cast<s64>(p.x / cell))]
                .push_back(i);
        }
    }
    // nearest point within `near` of q; returns index or -1
    [[nodiscard]] s64 nearest(Vec3f q, f32 near_r) const {
        const s64 cz = static_cast<s64>(q.z / cell), cy = static_cast<s64>(q.y / cell),
                  cx = static_cast<s64>(q.x / cell);
        f32 best = near_r * near_r;
        s64 bi = -1;
        for (s64 dz = -1; dz <= 1; ++dz)
            for (s64 dy = -1; dy <= 1; ++dy)
                for (s64 dx = -1; dx <= 1; ++dx) {
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
};

}  // namespace detail

inline Expected<int> run_surf_consist(std::span<const std::string_view> args, Context&) {
    if (args.empty())
        return err(Errc::invalid_argument, "usage: surf-consist <fxsurf...> [k=4000] [near=6] [pts_mb=64]");
    s64 k = 4000, pts_mb = 64;
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

    // Load each mesh into a point cloud (+ per-point stencil normal) at a stride that keeps
    // it under the pts_mb budget; point-to-cloud distance then approximates point-to-surface
    // to within ~spacing/2.
    const usize max_pts = static_cast<usize>(pts_mb) * (1u << 20) / (2 * sizeof(Vec3f));
    std::vector<detail::MeshCloud> clouds;
    for (const auto& mp : paths) {
        auto s = io::read_fxsurf(mp);
        if (!s) return std::unexpected(s.error());
        detail::MeshCloud mc;
        mc.path = mp;
        const f64 cells = static_cast<f64>(s->nu) * static_cast<f64>(s->nv);
        const s64 stride =
            std::max<s64>(1, static_cast<s64>(std::ceil(std::sqrt(cells / static_cast<f64>(max_pts)))));
        mc.spacing = static_cast<f32>(stride) * std::max(s->scale_u, s->scale_v);
        bool first = true;
        s64 co_n = 0, co_ok = 0;  // normal coherence, measured on the same sweep
        for (s64 v = 0; v < s->nv; v += stride)
            for (s64 u = 0; u < s->nu; u += stride) {
                if (!s->is_valid(u, v)) continue;
                const auto nm = detail::stencil_normal(*s, u, v);
                if (!nm) continue;
                const Vec3f p = s->at(u, v);
                mc.pts.push_back(p);
                mc.nrm.push_back(*nm);
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
                if (const auto n2 = detail::stencil_normal(*s, u + stride, v)) {
                    ++co_n;
                    const f32 d = nm->z * n2->z + nm->y * n2->y + nm->x * n2->x;
                    co_ok += std::abs(d) >= 0.7f;
                }
            }
        if (mc.pts.size() < 16) {
            std::printf("surf-consist mesh %s  DEGENERATE (%zu usable points)\n", mp.c_str(), mc.pts.size());
            continue;
        }
        mc.build_grid(static_cast<f32>(near_r));
        std::printf("surf-consist mesh %s  points %zu  spacing %.1f  normal-coherent %.0f%%%s\n",
                    mp.c_str(),
                    mc.pts.size(),
                    static_cast<f64>(mc.spacing),
                    co_n ? 100.0 * static_cast<f64>(co_ok) / static_cast<f64>(co_n) : 0.0,
                    (co_n && static_cast<f64>(co_ok) / static_cast<f64>(co_n) < 0.9) ? "  INCOHERENT" : "");
        clouds.push_back(std::move(mc));
    }

    // Pairwise: for bbox-overlapping pairs, sample A's cloud and measure distance to B.
    int suspicious = 0;
    for (usize a = 0; a < clouds.size(); ++a)
        for (usize b2 = a + 1; b2 < clouds.size(); ++b2) {
            const auto& A = clouds[a];
            const auto& B = clouds[b2];
            bool overlap = true;
            for (int d = 0; d < 3; ++d)
                overlap &= A.lo[d] <= B.hi[d] + static_cast<f32>(near_r) && B.lo[d] <= A.hi[d] + static_cast<f32>(near_r);
            if (!overlap) continue;
            const usize step2 = std::max<usize>(1, A.pts.size() / static_cast<usize>(k));
            std::vector<f32> dists;
            s64 n_query = 0, n_pos = 0, n_signed = 0;
            for (usize i = 0; i < A.pts.size(); i += step2) {
                ++n_query;
                const s64 j = B.nearest(A.pts[i], static_cast<f32>(near_r));
                if (j < 0) continue;
                const Vec3f d = B.pts[static_cast<usize>(j)] - A.pts[i];
                dists.push_back(std::sqrt(d.z * d.z + d.y * d.y + d.x * d.x));
                // side of A the B-point sits on, along A's normal (only meaningful when the
                // surfaces are locally distinct: skip near-coincident points)
                const f32 sgn = d.z * A.nrm[i].z + d.y * A.nrm[i].y + d.x * A.nrm[i].x;
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
            // small distance = the traces CROSS (impossible for real sheets).
            const char* verdict = "OFFSET";
            if (med <= 2.0f)
                verdict = "AGREE";
            if (side_mix > 0.15 && med < 4.0f && n_signed > 50) verdict = "CROSS";
            if (std::string_view(verdict) != "AGREE") ++suspicious;
            std::printf("surf-consist pair %s <-> %s  overlap %.0f%% (n=%zu)  med %.1f  p90 %.1f  "
                        "coincident %.0f%%  side-mix %.2f  %s\n",
                        A.path.c_str(),
                        B.path.c_str(),
                        100.0 * ov_frac,
                        dists.size(),
                        static_cast<f64>(med),
                        static_cast<f64>(p90),
                        100.0 * coinc,
                        side_mix,
                        verdict);
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
