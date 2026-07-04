// render/render.hpp — render stage. Registers `render`: an end-to-end unroll of a .fxvol
// (umbilicus estimate -> winding field -> unroll image -> NRRD). See render/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include "annotate/umbilicus.hpp"
#include "codec/archive.hpp"
#include "io/nrrd.hpp"
#include "render/bake.hpp"
#include "render/surface_render.hpp"
#include "render/unroll.hpp"
#include "winding/winding_field.hpp"

#include <charconv>
#include <string>

namespace fenix::render {

// `fenix render <in.fxvol> <out.nrrd> [pitch=8] [samp=4]`
inline Expected<int> run(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix render <in.fxvol> <out.nrrd> [pitch=8] [samp=4]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto pf = [](std::string_view s, f32 def) {
        f32 v = def;
        std::from_chars(s.data(), s.data() + s.size(), v);
        return v;
    };
    const f32 pitch = args.size() > 2 ? pf(args[2], 8.0f) : 8.0f;
    const f32 samp = args.size() > 3 ? pf(args[3], 4.0f) : 4.0f;

    auto a = codec::VolumeArchive::open(std::string(args[0]));
    if (!a) return std::unexpected(a.error());
    auto vol = a->read_volume();
    if (!vol) return std::unexpected(vol.error());

    // Material threshold = mean intensity (papyrus brighter than air).
    f64 sum = 0;
    for (f32 v : vol->flat()) sum += static_cast<f64>(v);
    const f32 thresh = static_cast<f32>(sum / static_cast<f64>(vol->size()));

    auto umb = annotate::Umbilicus::estimate(vol->view(), thresh);
    auto field = winding::winding_init(vol->dims(), umb, {.pitch = pitch});
    auto img = unroll(vol->view(), field.view(), {.samp = samp});

    // Output the unrolled texture as a .fxvol (the render/ink contract; 2D image = a 1-deep volume).
    // We never write NRRD.
    const std::string outp(args[1]);
    auto out = codec::VolumeArchive::create(outp, img.dims(), codec::DctParams{});
    if (!out) return std::unexpected(out.error());
    if (auto w = out->write_volume(img.view()); !w) return std::unexpected(w.error());
    if (auto w = out->close(); !w) return std::unexpected(w.error());
    log(LogLevel::info, "unrolled {} -> {} ({}x{} image, pitch={})", args[0], args[1],
        img.dims().y, img.dims().x, pitch);
    return 0;
}

}  // namespace fenix::render

FENIX_REGISTER_STAGE(render, "unroll a .fxvol volume to a flattened NRRD image", ::fenix::render::run)
