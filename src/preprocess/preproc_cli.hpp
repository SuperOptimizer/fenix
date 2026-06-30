// preprocess/preproc_cli.hpp — CLI subcommands for the fysics preprocessing kernels that live as
// pure library functions in their own headers (deconv, guided denoise). dering ships its own stage
// in dering.hpp. Header-only; self-registering. Load/write helpers shared across the stages here.
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/nrrd.hpp"
#include "preprocess/deconv.hpp"
#include "preprocess/guided.hpp"

#include <span>
#include <string>
#include <string_view>

namespace fenix::preprocess {
namespace cli {

inline Expected<Volume<f32>> load(const std::string& p) {
    if (p.size() > 6 && p.substr(p.size() - 6) == ".fxvol") {
        auto a = codec::VolumeArchive::open(p);
        if (!a) return std::unexpected(a.error());
        return a->read_volume();
    }
    return io::read_nrrd(p);
}
inline Expected<void> write(const std::string& p, VolumeView<const f32> v) {
    if (p.size() > 6 && p.substr(p.size() - 6) == ".fxvol") {
        auto a = codec::VolumeArchive::create(p, v.dims(), codec::DctParams{});
        if (!a) return std::unexpected(a.error());
        if (auto w = a->write_volume(v); !w) return std::unexpected(w.error());
        return a->close();
    }
    return io::write_nrrd(p, v);
}
inline std::string opt(std::span<const std::string_view> args, std::string_view k, std::string_view dflt) {
    for (auto a : args) {
        const auto pos = a.find('=');
        if (pos != std::string_view::npos && a.substr(0, pos) == k) return std::string(a.substr(pos + 1));
    }
    return std::string(dflt);
}

}  // namespace cli

// `fenix deconv <in.nrrd|.fxvol> <out> [sigma=1.0] [reg=0.015]` — Wiener deconvolution of a Gaussian
// PSF (restore the contrast/sharpness the reconstruction low-passed away). Dims must be powers of two.
inline Expected<int> run_deconv(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix deconv <in.nrrd|.fxvol> <out> [sigma=1.0] [reg=0.015]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto v = cli::load(std::string(args[0]));
    if (!v) return std::unexpected(v.error());
    const Extent3 d = v->dims();
    auto pow2 = [](s64 n) { return n > 0 && (n & (n - 1)) == 0; };
    if (!(pow2(d.z) && pow2(d.y) && pow2(d.x)))
        return err(Errc::invalid_argument, "deconv: dims must be powers of two (FFT)");
    const f32 sigma = std::stof(cli::opt(args, "sigma", "1.0"));
    const f32 reg = std::stof(cli::opt(args, "reg", "0.015"));
    Volume<f32> out = wiener_deconvolve(v->view(), sigma, reg);
    log(LogLevel::info, "deconv: wiener sigma={} reg={} dims {}x{}x{}", sigma, reg, d.z, d.y, d.x);
    if (auto w = cli::write(std::string(args[1]), out.view()); !w) return std::unexpected(w.error());
    log(LogLevel::info, "deconv: wrote {}", args[1]);
    return 0;
}

// `fenix denoise <in.nrrd|.fxvol> <out> [r=2] [eps=4.0]` — He-Sun-Tang guided edge-preserving denoise.
// eps ~ (noise_std)^2 in value units; larger => more smoothing.
inline Expected<int> run_denoise(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix denoise <in.nrrd|.fxvol> <out> [r=2] [eps=4.0]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto v = cli::load(std::string(args[0]));
    if (!v) return std::unexpected(v.error());
    const s64 r = std::stoll(cli::opt(args, "r", "2"));
    const f32 eps = std::stof(cli::opt(args, "eps", "4.0"));
    Volume<f32> out = guided_filter(v->view(), r, eps);
    const Extent3 d = v->dims();
    log(LogLevel::info, "denoise: guided r={} eps={} dims {}x{}x{}", r, eps, d.z, d.y, d.x);
    if (auto w = cli::write(std::string(args[1]), out.view()); !w) return std::unexpected(w.error());
    log(LogLevel::info, "denoise: wrote {}", args[1]);
    return 0;
}

}  // namespace fenix::preprocess

FENIX_REGISTER_STAGE(deconv, "wiener deconvolution — restore recon-blurred contrast (fysics)",
                     ::fenix::preprocess::run_deconv)
FENIX_REGISTER_STAGE(denoise, "guided edge-preserving denoise (fysics He-Sun-Tang)",
                     ::fenix::preprocess::run_denoise)
