// ml/surf_repair.hpp — `fenix surf-repair`: snap-to-ridge mesh repair (torch-free).
// Upgrades the mesh-constant `shift=` correction to a smooth per-uv OFFSET FIELD: many
// corpus resamples are not offset by a constant but warped (finding 2026-07-03) — the
// trace drifts off the intensity ridge by different amounts across the segment. Per
// coarse uv lattice point: measure the nearest-prominent-peak offset along the stencil
// normal (same estimator as surf-qc profile mode); reject outliers vs the local median;
// smooth with validity-weighted box passes; bilinearly apply to every vertex, clamped.
// A ±max_shift clamp and outlier rejection keep the repair conservative: where the field
// is unmeasurable the vertex stays put. Validate with surf-qc delta before/after.
//   fenix surf-repair <ct.fxvol|cache@url> <in.fxsurf> <out.fxsurf>
//                     [mode=ridge|alpha] [grid=8] [off=12] [search=8] [prom=15]
//                     [smooth=2] [max_shift=6] [thr=0]
// mode=alpha — ALPHA SNAPPING: snap to the sub-voxel AIR->material face (detail::air_edge)
// instead of the brightness ridge. Corpus meshes trace a FACE, so this is the semantically
// correct target wherever the sheet borders air; papyrus-contact lattice points measure
// nothing and their vertices stay put (the field only diffuses ~2 lattice cells). thr=0 =
// per-profile auto threshold; max_shift stays well below sheet thickness so a recto trace
// can never flip to the verso face.
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "io/cached_volume.hpp"
#include "io/surface.hpp"
#include "ml/surf_qc.hpp"

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

inline Expected<int> run_surf_repair(std::span<const std::string_view> args, Context&) {
    if (args.size() < 3)
        return err(Errc::invalid_argument,
                   "usage: surf-repair <ct.fxvol|cache@zarr-url> <in.fxsurf> <out.fxsurf> "
                   "[grid=8] [off=12] [search=8] [prom=15] [smooth=2] [max_shift=6]");
    s64 grid = 8, search = 8, smooth = 2;
    f64 off = 12, prom = 15, max_shift = 6, thr = 0;
    std::string mode = "ridge";
    for (usize i = 3; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("grid=", grid) || num("off=", off) || num("search=", search) || num("prom=", prom) ||
            num("smooth=", smooth) || num("max_shift=", max_shift) || num("thr=", thr))
            continue;
        if (a.starts_with("mode=")) {
            mode = std::string(a.substr(5));
            continue;
        }
        return err(Errc::invalid_argument, "surf-repair: unknown arg '" + std::string(a) + "'");
    }

    if (mode != "ridge" && mode != "alpha") return err(Errc::invalid_argument, "surf-repair: mode wants ridge|alpha");
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
    auto s = io::read_fxsurf(std::string(args[1]));
    if (!s) return std::unexpected(s.error());

    // 1) measure the offset at a coarse uv lattice
    const s64 gu = (s->nu + grid - 1) / grid, gv = (s->nv + grid - 1) / grid;
    std::vector<f32> field(static_cast<usize>(gu * gv), 0.0f);
    std::vector<u8> has(static_cast<usize>(gu * gv), 0);
    const s64 W = static_cast<s64>(off);
    s64 n_meas = 0, n_try = 0;
    std::vector<f32> prof(static_cast<usize>(2 * W + 1)), sm(prof.size());
    for (s64 gj = 0; gj < gv; ++gj)
        for (s64 gi = 0; gi < gu; ++gi) {
            const s64 u = std::min(gi * grid, s->nu - 1), v = std::min(gj * grid, s->nv - 1);
            if (!s->is_valid(u, v)) continue;
            const auto nm = detail::stencil_normal(*s, u, v);
            if (!nm) continue;
            ++n_try;
            const Vec3f p = s->at(u, v);
            bool ok = true;
            for (s64 t = -W; t <= W && ok; ++t) {
                const Vec3f q = p + *nm * static_cast<f32>(t);
                const Index3 o{static_cast<s64>(std::lround(q.z)),
                               static_cast<s64>(std::lround(q.y)),
                               static_cast<s64>(std::lround(q.x))};
                if (o.z < 0 || o.y < 0 || o.x < 0 || o.z >= dims.z || o.y >= dims.y || o.x >= dims.x) {
                    ok = false;
                    break;
                }
                u8 b;
                Expected<void> g = cached ? cached->gather_box_u8(o.z, o.y, o.x, 1, 1, 1, &b)
                                          : arch->gather_box_u8(0, o.z, o.y, o.x, 1, 1, 1, &b);
                if (!g) {
                    ok = false;
                    break;
                }
                prof[static_cast<usize>(t + W)] = static_cast<f32>(b);
            }
            if (!ok) continue;
            for (usize i2 = 0; i2 < prof.size(); ++i2) {
                const f32 a2 = prof[i2 == 0 ? 0 : i2 - 1], b2 = prof[i2],
                          c2 = prof[i2 + 1 < prof.size() ? i2 + 1 : i2];
                sm[i2] = (a2 + b2 + c2) / 3.0f;
            }
            std::optional<f64> meas;
            if (mode == "alpha") {
                // alpha snapping reads the RAW profile (smoothing blurs the very edge we
                // want sub-voxel); ridge mode keeps the smoothed one (speckle fake-peaks)
                meas = detail::air_edge(prof, W, search, static_cast<f32>(thr));
            } else if (const auto pk = detail::nearest_prominent_peak(sm, W, search, static_cast<f32>(prom))) {
                meas = static_cast<f64>(*pk);
            }
            if (meas) {
                field[static_cast<usize>(gj * gu + gi)] = static_cast<f32>(*meas);
                has[static_cast<usize>(gj * gu + gi)] = 1;
                ++n_meas;
            }
        }
    const s64 min_meas = mode == "alpha" ? 9 : std::max<s64>(9, n_try / 8);
    if (n_meas < min_meas)
        return err(Errc::invalid_argument,
                   "surf-repair: offset field unmeasurable (" + std::to_string(n_meas) + "/" +
                       std::to_string(n_try) + " lattice points) — refusing to modify the mesh");

    // 2) outlier rejection vs the local 5x5 median (a neighbor-wrap lock-on is a point
    // failure; the true field is smooth)
    {
        std::vector<u8> keep = has;
        for (s64 gj = 0; gj < gv; ++gj)
            for (s64 gi = 0; gi < gu; ++gi) {
                if (!has[static_cast<usize>(gj * gu + gi)]) continue;
                std::vector<f32> nb;
                for (s64 dj = -2; dj <= 2; ++dj)
                    for (s64 di = -2; di <= 2; ++di) {
                        const s64 a2 = gi + di, b2 = gj + dj;
                        if (a2 < 0 || b2 < 0 || a2 >= gu || b2 >= gv) continue;
                        if (has[static_cast<usize>(b2 * gu + a2)]) nb.push_back(field[static_cast<usize>(b2 * gu + a2)]);
                    }
                if (nb.size() < 5) continue;
                std::nth_element(nb.begin(), nb.begin() + static_cast<s64>(nb.size()) / 2, nb.end());
                const f32 med = nb[nb.size() / 2];
                if (std::abs(field[static_cast<usize>(gj * gu + gi)] - med) > 3.0f)
                    keep[static_cast<usize>(gj * gu + gi)] = 0;
            }
        has = std::move(keep);
    }

    // 3) validity-weighted box smoothing (also diffuses into unmeasured holes)
    for (s64 pass = 0; pass < std::max<s64>(1, smooth); ++pass) {
        std::vector<f32> nf(field.size(), 0.0f);
        std::vector<f32> nw(field.size(), 0.0f);
        for (s64 gj = 0; gj < gv; ++gj)
            for (s64 gi = 0; gi < gu; ++gi)
                for (s64 dj = -1; dj <= 1; ++dj)
                    for (s64 di = -1; di <= 1; ++di) {
                        const s64 a2 = gi + di, b2 = gj + dj;
                        if (a2 < 0 || b2 < 0 || a2 >= gu || b2 >= gv) continue;
                        if (!has[static_cast<usize>(b2 * gu + a2)]) continue;
                        nf[static_cast<usize>(gj * gu + gi)] += field[static_cast<usize>(b2 * gu + a2)];
                        nw[static_cast<usize>(gj * gu + gi)] += 1.0f;
                    }
        for (usize i2 = 0; i2 < field.size(); ++i2)
            if (nw[i2] > 0) {
                field[i2] = nf[i2] / nw[i2];
                has[i2] = 1;
            }
    }

    // 4) apply: bilinear field sample per vertex, along the vertex stencil normal, clamped
    f64 sum_d = 0, sum_ad = 0;
    s64 n_apply = 0;
    auto field_at = [&](s64 u, s64 v) -> std::optional<f32> {
        const f64 fu = static_cast<f64>(u) / static_cast<f64>(grid), fv = static_cast<f64>(v) / static_cast<f64>(grid);
        const s64 i0 = std::clamp<s64>(static_cast<s64>(fu), 0, gu - 1);
        const s64 j0 = std::clamp<s64>(static_cast<s64>(fv), 0, gv - 1);
        const s64 i1 = std::min(i0 + 1, gu - 1), j1 = std::min(j0 + 1, gv - 1);
        if (!has[static_cast<usize>(j0 * gu + i0)] || !has[static_cast<usize>(j0 * gu + i1)] ||
            !has[static_cast<usize>(j1 * gu + i0)] || !has[static_cast<usize>(j1 * gu + i1)])
            return std::nullopt;
        const f32 a2 = static_cast<f32>(fu - static_cast<f64>(i0)), b2 = static_cast<f32>(fv - static_cast<f64>(j0));
        return field[static_cast<usize>(j0 * gu + i0)] * (1 - a2) * (1 - b2) +
               field[static_cast<usize>(j0 * gu + i1)] * a2 * (1 - b2) +
               field[static_cast<usize>(j1 * gu + i0)] * (1 - a2) * b2 +
               field[static_cast<usize>(j1 * gu + i1)] * a2 * b2;
    };
    // two-phase: compute every shift from the ORIGINAL geometry (mutating in place would
    // feed already-shifted neighbors into later stencil normals), then apply
    std::vector<std::pair<usize, Vec3f>> updates;
    for (s64 v = 0; v < s->nv; ++v)
        for (s64 u = 0; u < s->nu; ++u) {
            if (!s->is_valid(u, v)) continue;
            const auto nm = detail::stencil_normal(*s, u, v);
            if (!nm) continue;
            const auto d = field_at(u, v);
            if (!d) continue;
            const f32 dc = std::clamp(*d, -static_cast<f32>(max_shift), static_cast<f32>(max_shift));
            updates.emplace_back(static_cast<usize>(s->idx(u, v)), s->at(u, v) + *nm * dc);
            sum_d += dc;
            sum_ad += std::abs(dc);
            ++n_apply;
        }
    for (const auto& [ix, c] : updates) s->coord[ix] = c;
    if (auto w = io::write_fxsurf(std::string(args[2]), *s); !w) return std::unexpected(w.error());
    std::printf("surf-repair[%s] %s -> %s  lattice %lldx%lld  measured %lld/%lld  applied %lld verts  "
                "mean-shift %+.2f  mean|shift| %.2f\n",
                mode.c_str(),
                std::string(args[1]).c_str(),
                std::string(args[2]).c_str(),
                static_cast<long long>(gu),
                static_cast<long long>(gv),
                static_cast<long long>(n_meas),
                static_cast<long long>(n_try),
                static_cast<long long>(n_apply),
                sum_d / std::max<s64>(1, n_apply),
                sum_ad / std::max<s64>(1, n_apply));
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_surf_repair = ::fenix::register_stage(::fenix::Stage{
    "surf-repair", "snap-to-ridge mesh repair (smooth per-uv offset field)", ::fenix::ml::run_surf_repair});
}  // namespace
