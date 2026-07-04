// ml/surf_sheet.hpp — `fenix surf-sheet`: contact-sheet visual QC (torch-free). Renders n
// random band crossings of one mesh as a tiled JPEG grid: grayscale CT z-slice crops with
// the mesh's rasterized band overlaid in red and the sample point cross-haired in green.
// Ten seconds of human glance per mesh beats any single statistic — the overlay images are
// what cracked both the bg-shell collapse and the volume-inference wash (finding
// 2026-07-03). This automates them at corpus scale.
//   fenix surf-sheet <ct.fxvol|cache@zarr-url> <fxsurf> <out.jpg>
//                    [n=16] [crop=192] [thickness=2] [quality=90] [seed=7]
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "io/cached_volume.hpp"
#include "io/jpeg.hpp"
#include "io/surface.hpp"
#include "ml/rasterize.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::ml {

inline Expected<int> run_surf_sheet(std::span<const std::string_view> args, Context&) {
    if (args.size() < 3)
        return err(Errc::invalid_argument,
                   "usage: surf-sheet <ct.fxvol|cache@zarr-url> <fxsurf> <out.jpg> "
                   "[n=16] [crop=192] [thickness=2] [quality=90] [seed=7]");
    s64 n = 16, crop = 192, quality = 90;
    u64 seed = 7;
    f32 thickness = 2.0f;
    for (usize i = 3; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("n=", n) || num("crop=", crop) || num("thickness=", thickness) || num("quality=", quality) ||
            num("seed=", seed))
            continue;
        return err(Errc::invalid_argument, "surf-sheet: unknown arg '" + std::string(a) + "'");
    }
    n = std::clamp<s64>(n, 1, 64);

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

    // n sample points spread over the valid uv grid, hash-jittered so reruns with another
    // seed show different crossings
    struct Pick {
        Vec3f p;
    };
    std::vector<Pick> picks;
    const s64 want_scan = n * 32;  // oversample candidates, keep every (candidates/n)-th valid
    const s64 stride = std::max<s64>(1, s->nu * s->nv / want_scan);
    std::vector<Vec3f> valid;
    for (s64 c = hash_value(std::array<u64, 1>{seed}) % static_cast<u64>(stride); c < s->nu * s->nv; c += stride) {
        const s64 u = c % s->nu, v = c / s->nu;
        if (s->is_valid(u, v)) valid.push_back(s->at(u, v));
    }
    if (valid.empty()) return err(Errc::invalid_argument, "surf-sheet: mesh has no valid cells");
    for (s64 i = 0; i < n; ++i) picks.push_back({valid[static_cast<usize>(i * static_cast<s64>(valid.size()) / n)]});

    const int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<f64>(n))));
    const int rows = static_cast<int>((n + cols - 1) / cols);
    const int gap = 2, tile = static_cast<int>(crop);
    io::Image img;
    img.comps = 3;
    img.w = cols * (tile + gap) - gap;
    img.h = rows * (tile + gap) - gap;
    img.px.assign(static_cast<usize>(img.w) * static_cast<usize>(img.h) * 3, 255);

    std::vector<u8> ctbuf(static_cast<usize>(crop * crop));
    for (s64 i = 0; i < n; ++i) {
        const Vec3f p = picks[static_cast<usize>(i)].p;
        const s64 z = std::clamp<s64>(static_cast<s64>(std::lround(p.z)), 0, dims.z - 1);
        const s64 y0 = std::clamp<s64>(static_cast<s64>(std::lround(p.y)) - crop / 2, 0, std::max<s64>(0, dims.y - crop));
        const s64 x0 = std::clamp<s64>(static_cast<s64>(std::lround(p.x)) - crop / 2, 0, std::max<s64>(0, dims.x - crop));
        Expected<void> g = cached ? cached->gather_box_u8(z, y0, x0, 1, crop, crop, ctbuf.data())
                                  : arch->gather_box_u8(0, z, y0, x0, 1, crop, crop, ctbuf.data());
        if (!g) return std::unexpected(g.error());
        Volume<u8> band = rasterize_band(*s, Index3{z, y0, x0}, Extent3{1, crop, crop}, {.thickness = thickness});
        const auto bv = band.view();

        const int ty = static_cast<int>(i) / cols * (tile + gap), tx = static_cast<int>(i) % cols * (tile + gap);
        for (s64 yy = 0; yy < crop; ++yy)
            for (s64 xx = 0; xx < crop; ++xx) {
                const u8 v = ctbuf[static_cast<usize>(yy * crop + xx)];
                u8 r = v, gc = v, b = v;
                if (bv(0, yy, xx) == kLabelSheet) {  // red overlay, 50% blend
                    r = static_cast<u8>((v + 255) / 2);
                    gc = static_cast<u8>(v / 2);
                    b = static_cast<u8>(v / 2);
                }
                img.at(ty + static_cast<int>(yy), tx + static_cast<int>(xx), 0) = r;
                img.at(ty + static_cast<int>(yy), tx + static_cast<int>(xx), 1) = gc;
                img.at(ty + static_cast<int>(yy), tx + static_cast<int>(xx), 2) = b;
            }
        // green cross-hair at the sampled point
        const s64 cy = static_cast<s64>(std::lround(p.y)) - y0, cx = static_cast<s64>(std::lround(p.x)) - x0;
        for (s64 d = -6; d <= 6; ++d) {
            if (cy >= 0 && cy < crop && cx + d >= 0 && cx + d < crop) {
                img.at(ty + static_cast<int>(cy), tx + static_cast<int>(cx + d), 1) = 255;
            }
            if (cx >= 0 && cx < crop && cy + d >= 0 && cy + d < crop) {
                img.at(ty + static_cast<int>(cy + d), tx + static_cast<int>(cx), 1) = 255;
            }
        }
    }
    if (auto r = io::write_jpeg(std::string(args[2]), img, static_cast<int>(quality)); !r)
        return std::unexpected(r.error());
    log(LogLevel::info, "surf-sheet: {} crossings of {} -> {}", n, args[1], args[2]);
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_surf_sheet = ::fenix::register_stage(::fenix::Stage{
    "surf-sheet", "contact-sheet visual QC (CT crops + band overlay grid)", ::fenix::ml::run_surf_sheet});
}  // namespace
