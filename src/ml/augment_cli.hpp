// ml/augment_cli.hpp — `fenix augment` stage: apply the training-augmentation chain to a .fxvol and
// write the result, for eyeballing / golden tests / generating augmented training crops offline. Torch-
// free (augment.hpp is pure Volume<f32> math), so it is NOT behind the FENIX_ML firewall — always built.
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "ml/augment.hpp"

#include <algorithm>
#include <charconv>
#include <string>

namespace fenix::ml {

// fenix augment <in.fxvol> <out.fxvol> [seed] [op=all|octa|rot|elastic|intensity|ct|compress] [param]
// op selects a single transform (for inspecting one effect) or the full policy chain (default).
inline Expected<int> run_augment(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2)
        return fenix::err(Errc::invalid_argument,
                          "usage: augment <in.fxvol> <out.fxvol> [seed] [op=all|octa|rot|elastic|intensity|ct|compress] [param]");
    const std::string inpath(args[0]), outpath(args[1]);
    // std::from_chars, not std::sto* (which throws under -fno-exceptions).
    auto parse = [](std::string_view key, std::string_view tok, auto& v) -> Expected<void> {
        const auto r = std::from_chars(tok.data(), tok.data() + tok.size(), v);
        if (r.ec != std::errc{} || r.ptr != tok.data() + tok.size())
            return err(Errc::invalid_argument,
                       "augment: bad value for " + std::string(key) + ": '" + std::string(tok) + "'");
        return {};
    };
    u64 seed = 12345ull;
    if (args.size() >= 3)
        if (auto r = parse("seed", args[2], seed); !r) return std::unexpected(r.error());
    std::string op = args.size() >= 4 ? std::string(args[3]) : "all";
    double param = 0.0;
    if (args.size() >= 5)
        if (auto r = parse("param", args[4], param); !r) return std::unexpected(r.error());

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

    // Augmented CT stays in the u8 [0,255] domain on disk — round-clamp, never write f32 (hard rule).
    const Extent3 od = s.image.dims();
    Volume<u8> q(od);
    for (s64 i = 0; i < od.count(); ++i)
        q.flat()[static_cast<usize>(i)] =
            static_cast<u8>(std::clamp(s.image.flat()[static_cast<usize>(i)], 0.0f, 255.0f) + 0.5f);
    auto out = codec::VolumeArchive::create(outpath, od, codec::DctParams{});
    if (!out) return std::unexpected(out.error());
    if (auto r = out->template write_volume<u8>(q.view()); !r) return std::unexpected(r.error());
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
