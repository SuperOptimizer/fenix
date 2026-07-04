// render/bake.hpp — `fenix surf-bake`: bake a segment's papyrus texture from the CT
// (torch-free). Composites the volume along the surface normals (view's streaming
// composite: mean/max/alpha/beer_lambert over ±offset layers) into the uv grid, windows
// to u8, writes a grayscale JPEG whose pixels map 1:1 to uv cells — the base-color
// texture `fenix view-surf` drapes over the 3D mesh. This is the render module's
// layer-sampling lineage pointed at REAL-TIME segment display instead of ink stacks.
//   fenix surf-bake <ct.fxvol|cache@zarr-url> <fxsurf> <out.jpg>
//                   [lo=-4] [hi=4] [step=1] [mode=mean|max|beer] [q=92]
// cache@url: the band is ensure()-prefetched first (the composite sampler reads the
// underlying archive directly and must never see absent-as-air).
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "io/cached_volume.hpp"
#include "io/jpeg.hpp"
#include "io/surface.hpp"
#include "view/composite.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::render {

inline Expected<int> run_surf_bake(std::span<const std::string_view> args, Context&) {
    if (args.size() < 3)
        return err(Errc::invalid_argument,
                   "usage: surf-bake <ct.fxvol|cache@zarr-url> <fxsurf> <out.jpg> "
                   "[lo=-4] [hi=4] [step=1] [mode=mean|max|beer] [q=92]");
    f64 lo = -4, hi = 4, step = 1;
    s64 q = 92;
    std::string mode = "mean";
    for (usize i = 3; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("lo=", lo) || num("hi=", hi) || num("step=", step) || num("q=", q)) continue;
        if (a.starts_with("mode=")) {
            mode = std::string(a.substr(5));
            continue;
        }
        return err(Errc::invalid_argument, "surf-bake: unknown arg '" + std::string(a) + "'");
    }
    view::CompositeMode cm = view::CompositeMode::mean;
    if (mode == "max") cm = view::CompositeMode::max;
    else if (mode == "beer") cm = view::CompositeMode::beer_lambert;
    else if (mode != "mean") return err(Errc::invalid_argument, "surf-bake: mode wants mean|max|beer");

    auto s = io::read_fxsurf(std::string(args[1]));
    if (!s) return std::unexpected(s.error());

    std::optional<io::CachedVolume> cached;
    std::optional<codec::VolumeArchive> local;
    const std::string ct(args[0]);
    if (const auto at = ct.find('@'); at != std::string::npos) {
        auto cv = io::CachedVolume::open(ct.substr(0, at), ct.substr(at + 1));
        if (!cv) return std::unexpected(cv.error());
        cached = std::move(*cv);
        // prefetch the band: the composite's BlockSampler reads the archive directly and
        // must never see absent-as-air. ensure() dedups via the in-flight claim set.
        const s64 pad = static_cast<s64>(std::ceil(std::max(std::abs(lo), std::abs(hi)))) + 2;
        for (s64 v = 0; v < s->nv; v += 4)
            for (s64 u = 0; u < s->nu; u += 4) {
                if (!s->is_valid(u, v)) continue;
                const Vec3f p = s->at(u, v);
                (void)cached->ensure(Index3{static_cast<s64>(p.z) - pad,
                                            static_cast<s64>(p.y) - pad,
                                            static_cast<s64>(p.x) - pad},
                                     Extent3{2 * pad, 2 * pad, 2 * pad});
            }
    } else {
        auto a = codec::VolumeArchive::open(ct);
        if (!a) return std::unexpected(a.error());
        local = std::move(*a);
    }
    codec::VolumeArchive& arch = cached ? cached->archive() : *local;
    arch.reserve_cache(u64{2} << 30);

    view::CompositeSpec spec;
    spec.mode = cm;
    spec.lo = static_cast<f32>(lo);
    spec.hi = static_cast<f32>(hi);
    spec.step = static_cast<f32>(step);
    auto img = view::render_surface_composite(arch, *s, spec);
    if (!img) return std::unexpected(img.error());

    // percentile windowing (1..99) over valid pixels -> u8
    std::vector<f32> vals;
    vals.reserve(img->pix.size() / 4);
    for (usize i = 0; i < img->pix.size(); i += 4)
        if (img->valid[i]) vals.push_back(img->pix[i]);
    if (vals.size() < 64) return err(Errc::invalid_argument, "surf-bake: surface produced almost no valid pixels");
    std::sort(vals.begin(), vals.end());
    const f32 wlo = vals[vals.size() / 100], whi = std::max(wlo + 1e-3f, vals[vals.size() * 99 / 100]);

    io::Image out;
    out.w = static_cast<int>(img->width);
    out.h = static_cast<int>(img->height);
    out.comps = 1;
    out.px.assign(static_cast<usize>(out.w) * static_cast<usize>(out.h), 0);
    for (usize i = 0; i < img->pix.size(); ++i)
        if (img->valid[i])
            out.px[i] = static_cast<u8>(std::clamp((img->pix[i] - wlo) / (whi - wlo), 0.0f, 1.0f) * 255.0f + 0.5f);
    if (auto w = io::write_jpeg(std::string(args[2]), out, static_cast<int>(q)); !w) return std::unexpected(w.error());
    log(LogLevel::info,
        "surf-bake: {}x{} texture ({} mode, offsets {}..{}) -> {}",
        img->width,
        img->height,
        mode,
        lo,
        hi,
        args[2]);
    return 0;
}

}  // namespace fenix::render

namespace {
[[maybe_unused]] const int fenix_stage_surf_bake = ::fenix::register_stage(
    ::fenix::Stage{"surf-bake", "bake a segment's papyrus texture from the CT (uv JPEG)", ::fenix::render::run_surf_bake});
}  // namespace
