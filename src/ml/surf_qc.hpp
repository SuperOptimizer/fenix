// ml/surf_qc.hpp — `fenix surf-qc`: mesh↔volume alignment QC (torch-free).
// Some corpus meshes' upstream "-on-<volume>" resamples are misaligned (measured on
// PHercParis4: one mesh +13.7 on-sheet brightness, another ~0/negative — training on the
// bad ones feeds label noise that stalls learning at chance). For each mesh: sample K valid
// cells, compare CT at the surface point against CT at ±off voxels along the local surface
// NORMAL (stencil-averaged central differences — single-cell tangents are too noisy).
// Aligned meshes sit on bright papyrus with darker gaps alongside: delta = ct@surface −
// mean(ct@±off) is clearly positive.
//   fenix surf-qc <ct.fxvol|cache@url> <fxsurf...> [k=200] [off=12] [min_delta=5]
//                 [regions=out.trust] [tile=256] [rk=8]
//                 [profile=1] [search=8] [prom=15] [offsets=out.tsv]
// Prints one line per mesh + PASS/FAIL; exit 0 iff all pass (scriptable filter).
//
// regions= (delta mode): REGION-LEVEL QC — the per-mesh scalar throws away a whole mesh for
// one bad lobe. Tiles the uv grid (tile= cells/side), scores the delta PER TILE, and writes
// a trust grid ('P' pass / 'F' fail / '?' insufficient) that the rasterizer consumes via the
// pairs `trust=` token (fail tiles → unlabeled-ignore instead of sheet). Format:
//   fxtrust1 <nu> <nv> <tile>\n  then nv_tiles rows of nu_tiles chars.
#pragma once

#include "core/core.hpp"
#include "core/surface.hpp"
#include "core/vec.hpp"
#include "io/cached_volume.hpp"
#include "io/surface.hpp"
#include "preprocess/aircut.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::ml {

namespace detail {
// Stencil-averaged surface normal at (u,v): central differences over up to ±2 grid steps.
// Single-cell tangent crosses wobble by tens of degrees at mesh resolution — the 2026-07-03
// profile-QC autopsy traced neighbor-wrap capture partly to normals that missed the sheet.
inline std::optional<Vec3f> stencil_normal(const Surface& s, s64 u, s64 v) {
    auto ok = [&](s64 uu, s64 vv) { return uu >= 0 && vv >= 0 && uu < s.nu && vv < s.nv && s.is_valid(uu, vv); };
    s64 du = 2, dv = 2;
    while (du > 0 && !(ok(u + du, v) && ok(u - du, v))) --du;
    while (dv > 0 && !(ok(u, v + dv) && ok(u, v - dv))) --dv;
    if (du == 0 || dv == 0) return std::nullopt;
    const Vec3f tu = s.at(u + du, v) - s.at(u - du, v);
    const Vec3f tv = s.at(u, v + dv) - s.at(u, v - dv);
    Vec3f nm = cross(tu, tv);
    const f32 nn = std::sqrt(nm.z * nm.z + nm.y * nm.y + nm.x * nm.x);
    if (nn < 1e-3f) return std::nullopt;
    return Vec3f{nm.z / nn, nm.y / nn, nm.x / nn};
}

// ALPHA SNAPPING (2026-07-04): where a sheet borders AIR the profile has a sharp gradient
// (a literal 0-vs-nonzero boundary on air-cut/masked data) — localize the face SUB-VOXEL
// as the alpha-crossing between the air run and the adjacent material run. Rules: a face
// needs a contiguous air run (>= min_run samples below thr) abutting a material run
// (>= 3 samples); the material run is the one containing t=0 (or nearest — a mesh
// floating IN the gap snaps to the nearest sheet's near face); among the run's ends, the
// face nearest t=0 within |t| <= search wins. thr <= 0 = per-profile auto:
// floor + 0.35*(peak - floor), requiring span > 40 — a papyrus-contact profile has no
// air and must return nullopt: alpha snapping only moves what it can see.
// out_pair (optional): BOTH faces of the material run — {toward -t, toward +t} — for
// callers that pick a side by physics (recto = air on the umbilicus-outward side,
// taberna air_trace lineage) instead of by proximity.
using AirEdgePair = std::pair<std::optional<f64>, std::optional<f64>>;
inline std::optional<f64> air_edge(std::span<const f32> prof, s64 W, s64 search, f32 thr = 0.0f, s64 min_run = 4,
                                   AirEdgePair* out_pair = nullptr) {
    const s64 n = static_cast<s64>(prof.size());
    if (thr <= 0.0f) {
        f32 lo = prof[0], hi = prof[0];
        for (f32 x : prof) {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
        if (hi - lo < 40.0f) return std::nullopt;  // unimodal: no air in this window
        thr = lo + 0.35f * (hi - lo);
    }
    auto is_air = [&](s64 i) { return prof[static_cast<usize>(i)] < thr; };
    // material run containing (or nearest to) t=0
    s64 c = W;
    if (is_air(c)) {  // floating in air: walk to the nearest material sample
        s64 best_c = -1;
        for (s64 d = 1; d <= search && best_c < 0; ++d) {
            if (c - d >= 0 && !is_air(c - d)) best_c = c - d;
            else if (c + d < n && !is_air(c + d)) best_c = c + d;
        }
        if (best_c < 0) return std::nullopt;
        c = best_c;
    }
    s64 m0 = c, m1 = c;
    while (m0 - 1 >= 0 && !is_air(m0 - 1)) --m0;
    while (m1 + 1 < n && !is_air(m1 + 1)) ++m1;
    if (m1 - m0 + 1 < 3) return std::nullopt;  // speckle, not a sheet
    // candidate faces at the run ends that abut a real air run; subvoxel alpha-crossing.
    // The crossing is searched across the WHOLE descent (last material sample -> end of
    // the air run), not just the first bracketing pair — a hard edge crosses immediately,
    // a soft edge (blur, or a model-probability ramp) crosses several samples in.
    auto face_at = [&](s64 m, s64 dir) -> std::optional<f64> {
        const s64 a0 = m + dir;
        if (a0 < 0 || a0 >= n) return std::nullopt;
        s64 run = 0, aend = a0;
        f64 floor_sum = 0;
        for (s64 i = a0; i >= 0 && i < n && is_air(i); i += dir) {
            floor_sum += prof[static_cast<usize>(i)];
            aend = i;
            ++run;
        }
        if (run < min_run) return std::nullopt;
        const f64 floor_v = floor_sum / static_cast<f64>(run);
        const f64 alpha = floor_v + 0.3 * (prof[static_cast<usize>(m)] - floor_v);
        for (s64 i = m; i != aend; i += dir) {
            const f64 pi = prof[static_cast<usize>(i)], pj = prof[static_cast<usize>(i + dir)];
            if (pi >= alpha && pj <= alpha) {
                const f64 frac = pi - pj != 0.0 ? (pi - alpha) / (pi - pj) : 0.5;
                return static_cast<f64>(i - W) + frac * static_cast<f64>(dir);
            }
        }
        return static_cast<f64>(m - W) + 0.5 * static_cast<f64>(dir);  // degenerate: step edge
    };
    auto lo_face = face_at(m0, -1), hi_face = face_at(m1, +1);
    if (lo_face && std::abs(*lo_face) > static_cast<f64>(search)) lo_face.reset();
    if (hi_face && std::abs(*hi_face) > static_cast<f64>(search)) hi_face.reset();
    if (out_pair) *out_pair = {lo_face, hi_face};
    std::optional<f64> best;
    for (const auto& fc : {lo_face, hi_face}) {
        if (!fc) continue;
        if (!best || std::abs(*fc) < std::abs(*best)) best = fc;
    }
    return best;
}

// Local maxima of `prof` with topographic prominence >= prom, positions as t offsets
// (index - W). Returns the prominent peak NEAREST t=0 within |t| <= search, if any.
// "Global max in the window" was the old bug: near tight winding the window almost always
// contains SOME bright wrap, so a misplaced mesh happily locked onto the neighbor.
inline std::optional<s64> nearest_prominent_peak(std::span<const f32> prof, s64 W, s64 search, f32 prom) {
    const s64 n = static_cast<s64>(prof.size());
    std::optional<s64> best;
    for (s64 i = 1; i + 1 < n; ++i) {
        if (!(prof[static_cast<usize>(i)] >= prof[static_cast<usize>(i - 1)] &&
              prof[static_cast<usize>(i)] >= prof[static_cast<usize>(i + 1)]))
            continue;
        const f32 pv = prof[static_cast<usize>(i)];
        f32 lmin = pv, rmin = pv;
        for (s64 j = i - 1; j >= 0 && prof[static_cast<usize>(j)] <= pv; --j) lmin = std::min(lmin, prof[static_cast<usize>(j)]);
        for (s64 j = i + 1; j < n && prof[static_cast<usize>(j)] <= pv; ++j) rmin = std::min(rmin, prof[static_cast<usize>(j)]);
        if (pv - std::max(lmin, rmin) < prom) continue;
        const s64 t = i - W;
        if (std::abs(t) > search) continue;
        if (!best || std::abs(t) < std::abs(*best)) best = t;
    }
    return best;
}
}  // namespace detail

inline Expected<int> run_surf_qc(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2)
        return err(Errc::invalid_argument,
                   "usage: surf-qc <ct.fxvol|cache@zarr-url> <fxsurf...> [k=200] [off=12] [min_delta=5] "
                   "[regions=out.trust] [tile=256] [rk=8] [profile=1] [search=8] [prom=15] "
                   "[papthr=40|0=auto] [offsets=out.tsv]");
    s64 k = 200, tile = 256, rk = 8, search = 8;
    f64 off = 12, min_delta = 5, prom = 15, papthr = 40;  // 0 = auto (per-mesh Otsu; opt-in — its
    // valley lands between dim-gap and sheet-core in dense regions, a DIFFERENT semantics than
    // the thr-40 "not air" the grade cuts were calibrated on; recalibrate before making default)
    int profile_mode = 0;
    std::string offsets_path, regions_path;
    std::vector<std::string> meshes;
    for (usize i = 1; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("k=", k) || num("off=", off) || num("min_delta=", min_delta) || num("profile=", profile_mode) ||
            num("tile=", tile) || num("rk=", rk) || num("search=", search) || num("prom=", prom) ||
            num("papthr=", papthr))
            continue;
        if (a.starts_with("offsets=")) {
            offsets_path = std::string(a.substr(8));
            continue;
        }
        if (a.starts_with("regions=")) {
            regions_path = std::string(a.substr(8));
            continue;
        }
        meshes.emplace_back(a);
    }
    if (meshes.empty()) return err(Errc::invalid_argument, "surf-qc: no meshes");
    if (!regions_path.empty() && meshes.size() != 1)
        return err(Errc::invalid_argument, "surf-qc: regions= wants exactly one mesh (one trust grid per file)");

    // volume: plain archive or on-demand cache — same duality as the feeder
    std::optional<io::CachedVolume> cached;
    std::optional<codec::VolumeArchive> arch;
    Extent3 dims{};
    const std::string ct(args[0]);
    if (const auto at = ct.find('@'); at != std::string::npos) {
        auto cv = io::CachedVolume::open(ct.substr(0, at), ct.substr(at + 1));
        if (!cv) return std::unexpected(cv.error());
        dims = cv->dims();
        cached = std::move(*cv);
    } else {
        auto a = codec::VolumeArchive::open(ct);
        if (!a) return std::unexpected(a.error());
        dims = a->dims();
        arch = std::move(*a);
    }
    auto sample_box = [&](Index3 org, Extent3 e, u8* out) -> Expected<void> {
        if (cached) return cached->gather_box_u8(org.z, org.y, org.x, e.z, e.y, e.x, out);
        return arch->gather_box_u8(0, org.z, org.y, org.x, e.z, e.y, e.x, out);
    };
    // 3x3x3-mean CT probe at a float position (shared by both modes)
    auto at_vox = [&](Vec3f q) -> Expected<f64> {
        const Index3 org{static_cast<s64>(std::lround(q.z)),
                         static_cast<s64>(std::lround(q.y)),
                         static_cast<s64>(std::lround(q.x))};
        if (org.z < 1 || org.y < 1 || org.x < 1 || org.z + 2 >= dims.z || org.y + 2 >= dims.y || org.x + 2 >= dims.x)
            return err(Errc::invalid_argument, "oob");
        u8 buf[27];
        if (auto r = sample_box(Index3{org.z - 1, org.y - 1, org.x - 1}, Extent3{3, 3, 3}, buf); !r)
            return std::unexpected(r.error());
        f64 m = 0;
        for (u8 b : buf) m += b;
        return m / 27.0;
    };
    // one point's delta statistic: ct@surface − mean(ct@±off along the stencil normal)
    auto point_delta = [&](const Surface& s, s64 u, s64 v) -> std::optional<f64> {
        if (!s.is_valid(u, v)) return std::nullopt;
        const auto nm = detail::stencil_normal(s, u, v);
        if (!nm) return std::nullopt;
        const Vec3f p = s.at(u, v);
        const auto c0 = at_vox(p);
        const auto cp = at_vox(p + *nm * static_cast<f32>(off));
        const auto cm = at_vox(p - *nm * static_cast<f32>(off));
        if (!c0 || !cp || !cm) return std::nullopt;
        return *c0 - (*cp + *cm) / 2.0;
    };

    // PROFILE MODE (profile=1): per-point normal-profile classification + misalignment vector.
    // A correctly-placed surface point must "bump against void": profile classes are
    //   RIDGE    bright at 0, void both sides           (midline trace, correct)
    //   EDGE     bright one side, void the other        (face trace, correct)
    //   EMBEDDED bright everywhere                      (inside material — locally uncheckable)
    //   AIR      dark everywhere                        (floating in void — definitively wrong)
    // The offset is the NEAREST PROMINENT peak within ±search (prominence >= prom gray levels)
    // — not the window's global max, which near tight winding locks onto neighbor wraps
    // (the 2026-07-03 autopsy; that bug parked this mode). Smooth offset fields over uv =
    // systematic transform, repairable; random = damage. offsets=<out.tsv> dumps u v du for
    // the snap-to-ridge repair pipeline. Sampling k points is statistics enough.
    if (profile_mode) {
        for (const auto& mp : meshes) {
            auto s = io::read_fxsurf(mp);
            if (!s) return std::unexpected(s.error());
            const s64 W = static_cast<s64>(off);  // half-window along the normal
            s64 n = 0, n_ridge = 0, n_edge = 0, n_embed = 0, n_air = 0, n_nopeak = 0, n_pap = 0;
            std::vector<f32> pap_centers;  // smoothed CT at each probed surface point
            std::vector<f64> alpha_offs;   // air-edge offsets (M2 dispersion substitute)
            std::vector<f64> fwhms;        // peak full-width half-prominence (M6 sheet thickness)
            std::vector<u8> pap_pop;       // all profile samples (for the Otsu auto threshold)
            std::vector<f64> offs;
            std::FILE* of = offsets_path.empty() ? nullptr : std::fopen(offsets_path.c_str(), "w");
            const s64 stride = std::max<s64>(1, s->nu * s->nv / std::max<s64>(1, k * 4));
            for (s64 c = 0; c < s->nu * s->nv && n < k; c += stride) {
                const s64 u = c % s->nu, v = c / s->nu;
                if (!s->is_valid(u, v)) continue;
                const auto nmo = detail::stencil_normal(*s, u, v);
                if (!nmo) continue;
                const Vec3f p = s->at(u, v), nm = *nmo;
                std::vector<f32> prof(static_cast<usize>(2 * W + 1));
                bool ok = true;
                for (s64 t = -W; t <= W && ok; ++t) {
                    const Vec3f q = p + nm * static_cast<f32>(t);
                    const Index3 o{static_cast<s64>(std::lround(q.z)),
                                   static_cast<s64>(std::lround(q.y)),
                                   static_cast<s64>(std::lround(q.x))};
                    if (o.z < 0 || o.y < 0 || o.x < 0 || o.z >= dims.z || o.y >= dims.y || o.x >= dims.x) {
                        ok = false;
                        break;
                    }
                    u8 b;
                    if (!sample_box(o, Extent3{1, 1, 1}, &b)) {
                        ok = false;
                        break;
                    }
                    prof[static_cast<usize>(t + W)] = static_cast<f32>(b);
                }
                if (!ok) continue;
                ++n;
                // 3-tap smooth before peak-finding (single-voxel speckle makes fake maxima)
                std::vector<f32> sm(prof.size());
                for (usize i2 = 0; i2 < prof.size(); ++i2) {
                    const f32 a = prof[i2 == 0 ? 0 : i2 - 1], b = prof[i2],
                              c2 = prof[i2 + 1 < prof.size() ? i2 + 1 : i2];
                    sm[i2] = (a + b + c2) / 3.0f;
                }
                // raw on-papyrus: collect the SMOOTHED center sample + the whole profile
                // population; tallied after the loop against papthr (or its Otsu auto value).
                // The peak classifier below UNDERCOUNTS in dense low-contrast regions (plateau
                // profile -> "AIR" even with papyrus 1-3 vox away, measured on PHercParis4) —
                // this raw metric is the honest alignment denominator there.
                pap_centers.push_back(sm[static_cast<usize>(W)]);
                for (const f32 pv : prof) pap_pop.push_back(static_cast<u8>(std::clamp(pv, 0.0f, 255.0f)));
                // M2 (gt-metrics-hardening.md): alpha-crossing offset as the dispersion
                // substitute where the peak detector starves (dense regions) — air_edge
                // fires on air->material boundaries, complementary to ridge peaks.
                if (const auto ae = detail::air_edge(sm, W, search)) alpha_offs.push_back(*ae);
                const auto peak = detail::nearest_prominent_peak(sm, W, search, static_cast<f32>(prom));
                if (!peak) {  // no prominent ridge near the point: flat — embedded or air by level
                    f64 m = 0;
                    for (f32 x : prof) m += x;
                    m /= static_cast<f64>(prof.size());
                    (m > 80.0 ? n_embed : n_air)++;
                    ++n_nopeak;
                    continue;
                }
                const s64 pt = *peak;
                const f32 pv = sm[static_cast<usize>(pt + W)];
                // M6 (gt-metrics-hardening.md): peak FWHM = sheet-thickness estimate. Walk
                // from the peak to half-prominence on each side; the width validates the
                // rasterizer's band radius per scroll (band ~ thickness/2, not a constant).
                {
                    const f32 half = pv - 0.5f * static_cast<f32>(prom);
                    s64 lw = pt, rw = pt;
                    while (lw - 1 >= -W && sm[static_cast<usize>(lw - 1 + W)] >= half) --lw;
                    while (rw + 1 <= W && sm[static_cast<usize>(rw + 1 + W)] >= half) ++rw;
                    if (lw > -W && rw < W) fwhms.push_back(static_cast<f64>(rw - lw + 1));
                }
                bool void_left = false, void_right = false;
                for (s64 t = -W; t < pt; ++t) void_left |= sm[static_cast<usize>(t + W)] < pv - static_cast<f32>(prom);
                for (s64 t = pt + 1; t <= W; ++t)
                    void_right |= sm[static_cast<usize>(t + W)] < pv - static_cast<f32>(prom);
                if (void_left && void_right)
                    ++n_ridge;
                else if (void_left || void_right)
                    ++n_edge;
                else
                    ++n_embed;
                offs.push_back(static_cast<f64>(pt));
                if (of)
                    std::fprintf(of,
                                 "%lld\t%lld\t%lld\n",
                                 static_cast<long long>(u),
                                 static_cast<long long>(v),
                                 static_cast<long long>(pt));
            }
            if (of) std::fclose(of);
            std::sort(offs.begin(), offs.end());
            const f64 med = offs.empty() ? 0 : offs[offs.size() / 2];
            // COHERENCE is the discriminator, not ridge presence: a registered mesh's nearest-
            // ridge offsets CLUSTER at the face-trace value (≈ +half sheet thickness) while a
            // misregistered one scatters. IQR + fraction within ±3 of the median.
            // FAIL CLOSED: iqr/coher are only defined over peak-firing points. A mesh where
            // almost no peaks fire must NOT report iqr=0 ("perfectly tight") — emit -1 =
            // unmeasured, and emit n_offs so consumers can gate on the sample size.
            f64 iqr = -1, coher = -1;
            if (offs.size() > 4) {
                iqr = offs[offs.size() * 3 / 4] - offs[offs.size() / 4];
                s64 nc = 0;
                for (f64 o : offs) nc += std::abs(o - med) <= 3.0;
                coher = 100.0 * static_cast<f64>(nc) / static_cast<f64>(offs.size());
            }
            const f64 cert = n ? 100.0 * static_cast<f64>(n_ridge + n_edge) / static_cast<f64>(n) : 0;
            // resolve the pap threshold: explicit papthr=, else per-mesh Otsu over the full
            // profile population (air+papyrus modes) — makes pap transportable across scans
            // (the hardcoded 40 was calibrated on one 2.4um/78keV u8 scan). Degenerate/unimodal
            // valley (population all-papyrus or all-air) falls back to 40.
            f64 pthr = papthr;
            if (pthr <= 0 && pap_pop.size() > 256) {
                const f32 ot = preprocess::otsu_threshold_u8(std::span<const u8>(pap_pop.data(), pap_pop.size()), 1);
                pthr = (ot > 8.0f && ot < 200.0f) ? static_cast<f64>(ot) : 40.0;
            }
            if (pthr <= 0) pthr = 40.0;
            for (const f32 cvv : pap_centers) n_pap += cvv > static_cast<f32>(pthr);
            f64 fwhm_med = -1;
            if (fwhms.size() > 4) {
                std::sort(fwhms.begin(), fwhms.end());
                fwhm_med = fwhms[fwhms.size() / 2];
            }
            f64 alpha_iqr = -1;
            if (alpha_offs.size() > 4) {
                std::sort(alpha_offs.begin(), alpha_offs.end());
                alpha_iqr = alpha_offs[alpha_offs.size() * 3 / 4] - alpha_offs[alpha_offs.size() / 4];
            }
            std::printf("surf-qc-profile %s  n=%lld  n_offs=%zu  on-papyrus %.0f%%  ridge %.0f%%  edge %.0f%%  embedded %.0f%%  AIR %.0f%%  "
                        "no-peak %.0f%%  certified %.0f%%  median-offset %+.0f  offset-IQR %.0f  coherent %.0f%%  "
                        "alpha-IQR %.1f n_alpha=%zu  fwhm %.1f  papthr %.0f\n",
                        mp.c_str(),
                        static_cast<long long>(n),
                        offs.size(),
                        n ? 100.0 * static_cast<f64>(n_pap) / static_cast<f64>(n) : 0,
                        n ? 100.0 * static_cast<f64>(n_ridge) / static_cast<f64>(n) : 0,
                        n ? 100.0 * static_cast<f64>(n_edge) / static_cast<f64>(n) : 0,
                        n ? 100.0 * static_cast<f64>(n_embed) / static_cast<f64>(n) : 0,
                        n ? 100.0 * static_cast<f64>(n_air) / static_cast<f64>(n) : 0,
                        n ? 100.0 * static_cast<f64>(n_nopeak) / static_cast<f64>(n) : 0,
                        cert,
                        med,
                        iqr,
                        coher,
                        alpha_iqr,
                        alpha_offs.size(),
                        fwhm_med,
                        pthr);
        }
        return 0;
    }

    // REGION MODE (regions=): per-uv-tile delta-QC -> trust grid for the rasterizer.
    if (!regions_path.empty()) {
        auto s = io::read_fxsurf(meshes[0]);
        if (!s) return std::unexpected(s.error());
        const s64 tu = (s->nu + tile - 1) / tile, tv = (s->nv + tile - 1) / tile;
        std::vector<char> grid(static_cast<usize>(tu * tv), '?');
        f64 g_delta = 0;
        s64 g_n = 0, n_pass = 0, n_fail = 0, n_unk = 0;
        for (s64 tj = 0; tj < tv; ++tj)
            for (s64 ti = 0; ti < tu; ++ti) {
                f64 dsum = 0;
                s64 dn = 0;
                const s64 u0 = ti * tile, v0 = tj * tile;
                const s64 u1 = std::min(u0 + tile, s->nu), v1 = std::min(v0 + tile, s->nv);
                // up to rk samples on a jittered-free lattice over the tile
                const s64 g = std::max<s64>(1, static_cast<s64>(std::ceil(std::sqrt(static_cast<f64>(rk)))));
                for (s64 jj = 0; jj < g && dn < rk; ++jj)
                    for (s64 ii = 0; ii < g && dn < rk; ++ii) {
                        const s64 u = u0 + (u1 - u0) * (2 * ii + 1) / (2 * g);
                        const s64 v = v0 + (v1 - v0) * (2 * jj + 1) / (2 * g);
                        if (const auto d = point_delta(*s, u, v)) {
                            dsum += *d;
                            ++dn;
                        }
                    }
                char verdict = '?';
                if (dn >= std::max<s64>(3, rk / 2)) verdict = (dsum / static_cast<f64>(dn) >= min_delta) ? 'P' : 'F';
                grid[static_cast<usize>(tj * tu + ti)] = verdict;
                (verdict == 'P' ? n_pass : verdict == 'F' ? n_fail : n_unk)++;
                g_delta += dsum;
                g_n += dn;
            }
        std::FILE* rf = std::fopen(regions_path.c_str(), "w");
        if (!rf) return err(Errc::io_error, "surf-qc: cannot write " + regions_path);
        std::fprintf(rf,
                     "fxtrust1 %lld %lld %lld\n",
                     static_cast<long long>(s->nu),
                     static_cast<long long>(s->nv),
                     static_cast<long long>(tile));
        for (s64 tj = 0; tj < tv; ++tj) {
            std::fwrite(grid.data() + tj * tu, 1, static_cast<usize>(tu), rf);
            std::fputc('\n', rf);
        }
        std::fclose(rf);
        std::printf("surf-qc-regions %s  tiles %lldx%lld (tile=%lld)  P %lld  F %lld  ? %lld  "
                    "overall-delta %+.1f (n=%lld)  -> %s\n",
                    meshes[0].c_str(),
                    static_cast<long long>(tu),
                    static_cast<long long>(tv),
                    static_cast<long long>(tile),
                    static_cast<long long>(n_pass),
                    static_cast<long long>(n_fail),
                    static_cast<long long>(n_unk),
                    g_n ? g_delta / static_cast<f64>(g_n) : 0.0,
                    static_cast<long long>(g_n),
                    regions_path.c_str());
        return 0;
    }

    int failures = 0;
    for (const auto& mp : meshes) {
        auto s = io::read_fxsurf(mp);
        if (!s) return std::unexpected(s.error());
        f64 d_sum = 0;
        s64 n = 0, probes = 0;
        const s64 stride = std::max<s64>(1, s->nu * s->nv / std::max<s64>(1, k * 4));
        for (s64 c = 0; c < s->nu * s->nv && n < k; c += stride) {
            const s64 u = c % s->nu, v = c / s->nu;
            ++probes;
            if (const auto d = point_delta(*s, u, v)) {
                d_sum += *d;
                ++n;
            }
        }
        if (n < k / 4) {
            std::printf("surf-qc %s  INSUFFICIENT (%lld/%lld probes ok)\n",
                        mp.c_str(),
                        static_cast<long long>(n),
                        static_cast<long long>(probes));
            ++failures;
            continue;
        }
        const f64 delta = d_sum / static_cast<f64>(n);
        const bool pass = delta >= min_delta;
        std::printf("surf-qc %s  n=%lld  delta %+.1f (ct@surf vs ct@±%.0f)  %s\n",
                    mp.c_str(),
                    static_cast<long long>(n),
                    delta,
                    off,
                    pass ? "PASS" : "FAIL");
        if (!pass) ++failures;
    }
    if (failures) return err(Errc::invalid_argument, "surf-qc: " + std::to_string(failures) + " mesh(es) failed");
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_surf_qc = ::fenix::register_stage(::fenix::Stage{
    "surf-qc", "mesh<->volume alignment QC (on-sheet brightness vs normal offsets)", ::fenix::ml::run_surf_qc});
}  // namespace
