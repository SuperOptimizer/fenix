// preprocess/preproc_cli.hpp — CLI subcommands for the fysics preprocessing kernels that live as
// pure library functions in their own headers (deconv, guided denoise). dering ships its own stage
// in dering.hpp. Header-only; self-registering. Load/write helpers shared across the stages here.
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/nrrd.hpp"
#include "preprocess/aircut.hpp"
#include "preprocess/deconv.hpp"
#include "preprocess/guided.hpp"
#include "preprocess/musica.hpp"

#include <algorithm>
#include <charconv>
#include <span>
#include <string>
#include <string_view>

namespace fenix::preprocess {
namespace cli {

// std::from_chars-based numeric option parse (no std::sto*, which throws under -fno-exceptions).
// Requires the whole token to be consumed (rejects "1.5x"-style trailing garbage).
template <class T>
[[nodiscard]] inline Expected<T> parse(std::string_view key, std::string_view tok) {
    T v{};
    const auto r = std::from_chars(tok.data(), tok.data() + tok.size(), v);
    if (r.ec != std::errc{} || r.ptr != tok.data() + tok.size())
        return err(Errc::invalid_argument, "bad value for " + std::string(key) + ": '" + std::string(tok) + "'");
    return v;
}

inline Expected<Volume<f32>> load(const std::string& p) {
    if (p.size() > 6 && p.substr(p.size() - 6) == ".fxvol") {
        auto a = codec::VolumeArchive::open(p);
        if (!a) return std::unexpected(a.error());
        return a->read_volume();
    }
    return io::read_nrrd(p);
}
// CT-domain preprocessing output is always a .fxvol. The dense buffer stays u8 (0..255) — round-clamp,
// never widen to f32 — and the archive encoder consumes the u8 source directly. We never write NRRD.
inline Expected<void> write(const std::string& p, VolumeView<const f32> v) {
    const Extent3 d = v.dims();
    Volume<u8> out(d);
    for (s64 i = 0; i < d.count(); ++i)
        out.flat()[static_cast<usize>(i)] = static_cast<u8>(std::clamp(v.flat()[static_cast<usize>(i)], 0.0f, 255.0f) + 0.5f);
    auto a = codec::VolumeArchive::create(p, d, codec::DctParams{});
    if (!a) return std::unexpected(a.error());
    if (auto w = a->template write_volume<u8>(out.view()); !w) return std::unexpected(w.error());
    return a->close();
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
    auto sigma_r = cli::parse<f32>("sigma", cli::opt(args, "sigma", "1.0"));
    if (!sigma_r) return std::unexpected(sigma_r.error());
    auto reg_r = cli::parse<f32>("reg", cli::opt(args, "reg", "0.015"));
    if (!reg_r) return std::unexpected(reg_r.error());
    const f32 sigma = *sigma_r, reg = *reg_r;
    if (!(reg > 0.0f)) return err(Errc::invalid_argument, "deconv: reg must be > 0 (got " + std::to_string(reg) + ")");
    if (!(sigma > 0.0f)) return err(Errc::invalid_argument, "deconv: sigma must be > 0 (got " + std::to_string(sigma) + ")");
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
    auto r_r = cli::parse<s64>("r", cli::opt(args, "r", "2"));
    if (!r_r) return std::unexpected(r_r.error());
    auto eps_r = cli::parse<f32>("eps", cli::opt(args, "eps", "4.0"));
    if (!eps_r) return std::unexpected(eps_r.error());
    const s64 r = *r_r;
    const f32 eps = *eps_r;
    Volume<f32> out = guided_filter(v->view(), r, eps);
    const Extent3 d = v->dims();
    log(LogLevel::info, "denoise: guided r={} eps={} dims {}x{}x{}", r, eps, d.z, d.y, d.x);
    if (auto w = cli::write(std::string(args[1]), out.view()); !w) return std::unexpected(w.error());
    log(LogLevel::info, "denoise: wrote {}", args[1]);
    return 0;
}

// `fenix aircut <in.nrrd|.fxvol> <out> [lo=0] [hi=255]` — zero everything below the Otsu valley
// (separating low-density background from papyrus). NOTE: real data isn't perfectly bimodal, so the
// valley cut zeros some genuine low-value material near the threshold.
inline Expected<int> run_aircut(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix aircut <in.nrrd|.fxvol> <out> [thr=<v>] [lo=0] [hi=255]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto v = cli::load(std::string(args[0]));
    if (!v) return std::unexpected(v.error());
    const std::string thr_s = cli::opt(args, "thr", "");
    const Extent3 d = v->dims();
    f32 thr;
    if (!thr_s.empty()) {  // manual threshold (Otsu is a poor fit when the data isn't bimodal)
        auto thr_r = cli::parse<f32>("thr", thr_s);
        if (!thr_r) return std::unexpected(thr_r.error());
        thr = *thr_r;
        auto vw = v->view();
        parallel_for_z(d, [&](s64 z) {
            for (s64 y = 0; y < d.y; ++y)
                for (s64 x = 0; x < d.x; ++x)
                    if (vw(z, y, x) < thr) vw(z, y, x) = 0.0f;
        });
        log(LogLevel::info, "aircut: manual threshold {:.1f} (dims {}x{}x{})", thr, d.z, d.y, d.x);
    } else {
        auto lo_r = cli::parse<f32>("lo", cli::opt(args, "lo", "0"));
        if (!lo_r) return std::unexpected(lo_r.error());
        auto hi_r = cli::parse<f32>("hi", cli::opt(args, "hi", "255"));
        if (!hi_r) return std::unexpected(hi_r.error());
        const f32 lo = *lo_r, hi = *hi_r;
        thr = air_cut(v->view(), lo, hi);
        log(LogLevel::info, "aircut: Otsu threshold {:.1f} (dims {}x{}x{})", thr, d.z, d.y, d.x);
    }
    if (auto w = cli::write(std::string(args[1]), v->view()); !w) return std::unexpected(w.error());
    log(LogLevel::info, "aircut: wrote {}", args[1]);
    return 0;
}

// `fenix musica <in.nrrd|.fxvol> <out> [levels=4] [p=0.7] [core=0] [vmax=255]` — MUSICA multiscale
// contrast amplification (per z-slice). p<1 lifts faint detail; core soft-cores noise.
inline Expected<int> run_musica(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix musica <in.nrrd|.fxvol> <out> [levels=4] [p=0.7] [core=0] [vmax=255]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto v = cli::load(std::string(args[0]));
    if (!v) return std::unexpected(v.error());
    auto levels_r = cli::parse<s32>("levels", cli::opt(args, "levels", "4"));
    if (!levels_r) return std::unexpected(levels_r.error());
    auto p_r = cli::parse<f32>("p", cli::opt(args, "p", "0.7"));
    if (!p_r) return std::unexpected(p_r.error());
    auto core_r = cli::parse<f32>("core", cli::opt(args, "core", "0"));
    if (!core_r) return std::unexpected(core_r.error());
    auto vmax_r = cli::parse<f32>("vmax", cli::opt(args, "vmax", "255"));
    if (!vmax_r) return std::unexpected(vmax_r.error());
    const s32 levels = *levels_r;
    const f32 p = *p_r, core = *core_r, vmax = *vmax_r;
    musica_inplace(v->view(), levels, p, core, vmax);
    const Extent3 d = v->dims();
    log(LogLevel::info, "musica: levels={} p={} core={} (dims {}x{}x{})", levels, p, core, d.z, d.y, d.x);
    if (auto w = cli::write(std::string(args[1]), v->view()); !w) return std::unexpected(w.error());
    log(LogLevel::info, "musica: wrote {}", args[1]);
    return 0;
}

}  // namespace fenix::preprocess

FENIX_REGISTER_STAGE(deconv, "wiener deconvolution — restore recon-blurred contrast (fysics)",
                     ::fenix::preprocess::run_deconv)
FENIX_REGISTER_STAGE(denoise, "guided edge-preserving denoise (fysics He-Sun-Tang)",
                     ::fenix::preprocess::run_denoise)
FENIX_REGISTER_STAGE(aircut, "Otsu air-cut — zero low-density background (fysics)",
                     ::fenix::preprocess::run_aircut)
FENIX_REGISTER_STAGE(musica, "MUSICA multiscale contrast amplification (fysics)",
                     ::fenix::preprocess::run_musica)
