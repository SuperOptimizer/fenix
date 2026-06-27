// io/io.hpp — IO stage: registers the `ingest` subcommand (NRRD -> .fxvol). See io/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include "codec/archive.hpp"
#include "io/nrrd.hpp"
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

}  // namespace fenix::io

FENIX_REGISTER_STAGE(ingest, "ingest a NRRD volume into a .fxvol archive", ::fenix::io::ingest)
