// io/io.hpp — IO stage: registers the `ingest` subcommand (NRRD -> .fxvol). See io/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include "codec/archive.hpp"
#include "io/jpeg.hpp"
#include "io/nrrd.hpp"
#include "io/slice.hpp"
#include "io/zarr.hpp"

#include <charconv>
#include <string>

namespace fenix::io {

// `fenix ingest <in.nrrd> <out.fxvol> [q] [levels]` — read a NRRD and transcode to .fxvol.
inline Expected<int> ingest(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix ingest <in.nrrd> <out.fxvol> [q=2] [levels=4]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto parse_f = [](std::string_view s, f32 def) {
        f32 v = def;
        std::from_chars(s.data(), s.data() + s.size(), v);
        return v;
    };
    auto parse_i = [](std::string_view s, int def) {
        int v = def;
        std::from_chars(s.data(), s.data() + s.size(), v);
        return v;
    };

    auto vol = read_nrrd(std::string(args[0]));
    if (!vol) return std::unexpected(vol.error());

    codec::BlockParams bp{.q = args.size() > 2 ? parse_f(args[2], 2.0f) : 2.0f,
                          .levels = args.size() > 3 ? parse_i(args[3], 4) : 4};
    auto a = codec::VolumeArchive::create(std::string(args[1]), vol->dims(), bp);
    if (!a) return std::unexpected(a.error());
    if (auto w = a->write_volume(vol->view()); !w) return std::unexpected(w.error());
    if (auto c = a->close(); !c) return std::unexpected(c.error());

    const Extent3 d = vol->dims();
    log(LogLevel::info, "ingested {} ({}x{}x{}) -> {} (wavelet q={} levels={})", args[0], d.z, d.y,
        d.x, args[1], bp.q, bp.levels);
    return 0;
}

// `fenix ingest-zarr <zarr-root|url> <level> <z0> <y0> <x0> <D> <H> <W> <out.fxvol|.nrrd>`
// Pull a dense [z0:z0+D, y0:y0+H, x0:x0+W] region from an OME-Zarr pyramid level (local path
// or s3://, http(s):// URL) and write it as .fxvol (transcoded) or .nrrd. Missing chunks = air.
inline Expected<int> ingest_zarr(std::span<const std::string_view> args, Context&) {
    if (args.size() < 9) {
        log(LogLevel::error,
            "usage: fenix ingest-zarr <zarr-root|url> <level> <z0> <y0> <x0> <D> <H> <W> <out.fxvol|.nrrd>");
        return err(Errc::invalid_argument, "missing args");
    }
    auto pi = [](std::string_view s) { s64 v = 0; std::from_chars(s.data(), s.data() + s.size(), v); return v; };
    std::string root(args[0]);
    while (!root.empty() && root.back() == '/') root.pop_back();
    const std::string level(args[1]);
    const std::string lroot = root + "/" + level;  // pyramid level dir holds .zarray + chunks
    const Index3 origin{pi(args[2]), pi(args[3]), pi(args[4])};
    const Extent3 extent{pi(args[5]), pi(args[6]), pi(args[7])};
    const std::string outpath(args[8]);

    log(LogLevel::info, "ingest-zarr: {} level {} region z{}:{} y{}:{} x{}:{} -> {}", root, level,
        origin.z, origin.z + extent.z, origin.y, origin.y + extent.y, origin.x, origin.x + extent.x,
        outpath);

    auto vol = read_zarr_region(lroot, origin, extent);
    if (!vol) return std::unexpected(vol.error());

    if (outpath.size() > 5 && outpath.substr(outpath.size() - 5) == ".nrrd") {
        if (auto w = write_nrrd(outpath, vol->view()); !w) return std::unexpected(w.error());
    } else {
        auto a = codec::VolumeArchive::create(outpath, vol->dims(), codec::BlockParams{});
        if (!a) return std::unexpected(a.error());
        if (auto w = a->write_volume(vol->view()); !w) return std::unexpected(w.error());
        if (auto c = a->close(); !c) return std::unexpected(c.error());
    }
    const Extent3 d = vol->dims();
    log(LogLevel::info, "ingest-zarr: wrote {} ({}x{}x{} ZYX)", outpath, d.z, d.y, d.x);
    return 0;
}

}  // namespace fenix::io

FENIX_REGISTER_STAGE(ingest, "ingest a NRRD volume into a .fxvol archive", ::fenix::io::ingest)

namespace {
[[maybe_unused]] const int fenix_stage_ingest_zarr = ::fenix::register_stage(
    ::fenix::Stage{"ingest-zarr", "pull an OME-Zarr region (local/s3/http) into .fxvol/.nrrd",
                   ::fenix::io::ingest_zarr});
}  // namespace
