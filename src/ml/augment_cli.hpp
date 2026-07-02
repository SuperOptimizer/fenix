// ml/augment_cli.hpp — `fenix augment` stage: apply the training-augmentation chain to a .fxvol and
// write the result, for eyeballing / golden tests / generating augmented training crops offline. Torch-
// free (augment.hpp is pure Volume<f32> math), so it is NOT behind the FENIX_ML firewall — always built.
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "ml/augment.hpp"

#include <string>

namespace fenix::ml {

// fenix augment <in.fxvol> <out.fxvol> [seed] [op=all|octa|rot|elastic|intensity|ct|compress] [param]
// op selects a single transform (for inspecting one effect) or the full policy chain (default).
inline Expected<int> run_augment(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2)
        return fenix::err(Errc::invalid_argument,
                          "usage: augment <in.fxvol> <out.fxvol> [seed] [op=all|octa|rot|elastic|intensity|ct|compress] [param]");
    const std::string inpath(args[0]), outpath(args[1]);
    const u64 seed = args.size() >= 3 ? static_cast<u64>(std::stoull(std::string(args[2]))) : 12345ull;
    std::string op = args.size() >= 4 ? std::string(args[3]) : "all";
    const double param = args.size() >= 5 ? std::stod(std::string(args[4])) : 0.0;

    auto a = codec::VolumeArchive::open(inpath);
    if (!a) return std::unexpected(a.error());
    auto v = a->read_volume(0);
    if (!v) return std::unexpected(v.error());
    aug::Sample s{std::move(*v), Volume<u8>()};
    const Extent3 d = s.image.dims();

    if (op == "all") aug::augment(s, seed);
    else if (op == "octa") aug::octahedral(s, static_cast<int>(seed % 48));
    else if (op == "rot") aug::rotate_z(s, param != 0.0 ? static_cast<f32>(param) : 15.0f);
    else if (op == "elastic") aug::elastic(s, seed, param != 0.0 ? static_cast<f32>(param) : 3.0f, 24.0f);
    else if (op == "intensity") aug::intensity(s, seed);
    else if (op == "ct") aug::ct_degrade(s, seed);
    else if (op == "compress") aug::compression(s, seed, param != 0.0 ? static_cast<f32>(param) : 0.6f);
    else return fenix::err(Errc::invalid_argument, "augment: unknown op '" + op + "'");

    fenix::log(LogLevel::info, "augment: {} op={} seed={} -> {}x{}x{}", inpath, op, seed,
               s.image.dims().z, s.image.dims().y, s.image.dims().x);

    auto out = codec::VolumeArchive::create(outpath, s.image.dims(), codec::DctParams{});
    if (!out) return std::unexpected(out.error());
    if (auto r = out->write_volume(s.image.view()); !r) return std::unexpected(r.error());
    if (auto r = out->close(); !r) return std::unexpected(r.error());
    (void)d;
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_augment = ::fenix::register_stage(::fenix::Stage{
    "augment", "apply train-time data augmentation to a .fxvol (octa/rot/elastic/intensity/ct/compress)",
    ::fenix::ml::run_augment});
}  // namespace
