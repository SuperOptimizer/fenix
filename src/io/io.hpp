// io/io.hpp — IO stage: registers the `ingest` subcommand (NRRD -> .fxvol). See io/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include "codec/archive.hpp"
#include "io/jpeg.hpp"
#include "io/nrrd.hpp"
#include "io/slice.hpp"
#include "io/zarr.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

namespace fenix::io {

inline s64 parse_i(std::string_view s, s64 def) {
    s64 v = def;
    std::from_chars(s.data(), s.data() + s.size(), v);
    return v;
}

// `fenix ingest <in.nrrd> <out.fxvol> [q]` — read a NRRD and transcode to .fxvol (DCT tile codec).
inline Expected<int> ingest(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix ingest <in.nrrd> <out.fxvol> [q=2]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto parse_f = [](std::string_view s, f32 def) {
        f32 v = def;
        std::from_chars(s.data(), s.data() + s.size(), v);
        return v;
    };

    auto vol = read_nrrd(std::string(args[0]));
    if (!vol) return std::unexpected(vol.error());

    codec::DctParams bp{.q = args.size() > 2 ? parse_f(args[2], 2.0f) : 2.0f};
    auto a = codec::VolumeArchive::create(std::string(args[1]), vol->dims(), bp);
    if (!a) return std::unexpected(a.error());
    if (auto w = a->write_volume(vol->view()); !w) return std::unexpected(w.error());
    if (auto c = a->close(); !c) return std::unexpected(c.error());

    const Extent3 d = vol->dims();
    log(LogLevel::info, "ingested {} ({}x{}x{}) -> {} (DCT q={})", args[0], d.z, d.y, d.x, args[1], bp.q);
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

    // `.zarr` output: keep it native — raw-copy chunk bytes (preserves the source dtype, e.g. u8;
    // no f32 widening) and write fresh metadata for the sub-volume. Skips the f32 assemble below.
    if (outpath.size() > 5 && outpath.substr(outpath.size() - 5) == ".zarr") {
        if (auto r = copy_zarr_region_local(lroot, outpath, origin, extent); !r)
            return std::unexpected(r.error());
        log(LogLevel::info, "ingest-zarr: wrote local zarr {} ({}x{}x{} ZYX, raw chunk copy)", outpath,
            extent.z, extent.y, extent.x);
        return 0;
    }

    auto vol = read_zarr_region(lroot, origin, extent);
    if (!vol) return std::unexpected(vol.error());

    if (outpath.size() > 5 && outpath.substr(outpath.size() - 5) == ".nrrd") {
        if (auto w = write_nrrd(outpath, vol->view()); !w) return std::unexpected(w.error());
    } else {
        auto a = codec::VolumeArchive::create(outpath, vol->dims(), codec::DctParams{});
        if (!a) return std::unexpected(a.error());
        if (auto w = a->write_volume(vol->view()); !w) return std::unexpected(w.error());
        if (auto c = a->close(); !c) return std::unexpected(c.error());
    }
    const Extent3 d = vol->dims();
    log(LogLevel::info, "ingest-zarr: wrote {} ({}x{}x{} ZYX)", outpath, d.z, d.y, d.x);
    return 0;
}

// `fenix export <in.fxvol> <out.nrrd> [lod=0]` — decode a LOD level of a .fxvol archive back to NRRD.
inline Expected<int> export_vol(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix export <in.fxvol> <out.nrrd> [lod=0]");
        return err(Errc::invalid_argument, "missing args");
    }
    const s64 lod = args.size() > 2 ? parse_i(args[2], 0) : 0;
    auto a = codec::VolumeArchive::open(std::string(args[0]));
    if (!a) return std::unexpected(a.error());
    if (lod < 0 || lod >= static_cast<s64>(a->nlods()))
        return err(Errc::invalid_argument, "lod out of range");
    auto vol = a->read_volume(lod);
    if (!vol) return std::unexpected(vol.error());
    if (auto w = write_nrrd(std::string(args[1]), vol->view()); !w) return std::unexpected(w.error());
    const Extent3 d = vol->dims();
    log(LogLevel::info, "exported {} lod {} ({}x{}x{}) -> {}", args[0], lod, d.z, d.y, d.x, args[1]);
    return 0;
}

// `fenix finalize <in.fxvol> <out.fxvol>` — repack into the SEALED coarse-first, front-loaded-index form.
inline Expected<int> finalize_vol(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix finalize <in.fxvol> <out.fxvol>");
        return err(Errc::invalid_argument, "missing args");
    }
    auto a = codec::VolumeArchive::open(std::string(args[0]));
    if (!a) return std::unexpected(a.error());
    if (auto r = a->finalize(std::string(args[1])); !r) return std::unexpected(r.error());
    log(LogLevel::info, "finalized {} -> {} (sealed: coarse-first, front-loaded index)", args[0], args[1]);
    return 0;
}

// `fenix fxinfo <in.fxvol>` — inspect dims, LOD pyramid, coverage tri-state counts, size, compression ratio.
inline Expected<int> info_vol(std::span<const std::string_view> args, Context&) {
    if (args.size() < 1) {
        log(LogLevel::error, "usage: fenix fxinfo <in.fxvol>");
        return err(Errc::invalid_argument, "missing args");
    }
    auto a = codec::VolumeArchive::open(std::string(args[0]));
    if (!a) return std::unexpected(a.error());
    const Extent3 d = a->dims();
    std::error_code ec;
    const u64 fsz = std::filesystem::file_size(std::string(args[0]), ec);
    const f64 voxels = static_cast<f64>(d.z) * static_cast<f64>(d.y) * static_cast<f64>(d.x);
    const u64 csz = a->committed_size();  // actual data bytes (the file may be fallocate-padded if not closed)
    std::printf("fxvol %s\n  dims (ZYX) %lldx%lldx%lld   q=%.2f   LODs=%u   %s\n  data %.2f MiB (file %.2f MiB)   ratio8 %.1fx\n",
                std::string(args[0]).c_str(), (long long)d.z, (long long)d.y, (long long)d.x,
                static_cast<double>(a->params().q), a->nlods(), a->data_offset() ? "SEALED" : "live",
                static_cast<double>(csz) / (1024.0 * 1024.0), static_cast<double>(fsz) / (1024.0 * 1024.0),
                voxels / static_cast<f64>(csz ? csz : 1));
    for (s64 lod = 0; lod < static_cast<s64>(a->nlods()); ++lod) {
        const Extent3 ld = a->dims_at(lod);
        const ChunkCoord ce = a->chunk_extent(lod);
        u64 real = 0, zero = 0, absent = 0;
        for (s64 cz = 0; cz < ce.z; ++cz)
            for (s64 cy = 0; cy < ce.y; ++cy)
                for (s64 cx = 0; cx < ce.x; ++cx)
                    switch (a->coverage(lod, {cz, cy, cx})) {
                        case codec::Coverage::Real: ++real; break;
                        case codec::Coverage::Zero: ++zero; break;
                        default: ++absent; break;
                    }
        std::printf("  LOD%-2lld %5lldx%-5lld x%-5lld  chunks %lldx%lldx%lld   real %llu  zero %llu  absent %llu\n",
                    (long long)lod, (long long)ld.z, (long long)ld.y, (long long)ld.x, (long long)ce.z,
                    (long long)ce.y, (long long)ce.x, (unsigned long long)real, (unsigned long long)zero,
                    (unsigned long long)absent);
    }
    return 0;
}

// `fenix compare <a.nrrd> <b.nrrd>` — PSNR / MAE / max-abs-error between two volumes (roundtrip check).
inline Expected<int> compare_vol(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix compare <a.nrrd> <b.nrrd>");
        return err(Errc::invalid_argument, "missing args");
    }
    auto a = read_nrrd(std::string(args[0]));
    if (!a) return std::unexpected(a.error());
    auto b = read_nrrd(std::string(args[1]));
    if (!b) return std::unexpected(b.error());
    if (!(a->dims() == b->dims())) return err(Errc::invalid_argument, "dims mismatch");
    const Extent3 d = a->dims();
    auto av = a->view(), bv = b->view();
    f64 sse = 0, sae = 0, mx = 0;
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const f64 e = std::abs(static_cast<f64>(av(z, y, x)) - bv(z, y, x));
                sse += e * e;
                sae += e;
                mx = std::max(mx, e);
            }
    const f64 n = static_cast<f64>(d.z) * static_cast<f64>(d.y) * static_cast<f64>(d.x);
    const f64 mse = sse / n;
    std::printf("compare %s vs %s (%lldx%lldx%lld)\n  PSNR %.2f dB   MAE %.4f   max-abs %.4f\n",
                std::string(args[0]).c_str(), std::string(args[1]).c_str(), (long long)d.z, (long long)d.y,
                (long long)d.x, mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0, sae / n, mx);
    return 0;
}

}  // namespace fenix::io

FENIX_REGISTER_STAGE(ingest, "ingest a NRRD volume into a .fxvol archive", ::fenix::io::ingest)

namespace {
[[maybe_unused]] const int fenix_stage_ingest_zarr = ::fenix::register_stage(
    ::fenix::Stage{"ingest-zarr", "pull an OME-Zarr region (local/s3/http) into .fxvol/.nrrd",
                   ::fenix::io::ingest_zarr});
[[maybe_unused]] const int fenix_stage_export = ::fenix::register_stage(
    ::fenix::Stage{"export", "decode a .fxvol LOD level back to NRRD", ::fenix::io::export_vol});
[[maybe_unused]] const int fenix_stage_finalize = ::fenix::register_stage(
    ::fenix::Stage{"finalize", "repack a .fxvol into the SEALED coarse-first form", ::fenix::io::finalize_vol});
[[maybe_unused]] const int fenix_stage_fxinfo = ::fenix::register_stage(
    ::fenix::Stage{"fxinfo", "inspect a .fxvol (dims, LODs, coverage, size, ratio)", ::fenix::io::info_vol});
[[maybe_unused]] const int fenix_stage_compare = ::fenix::register_stage(
    ::fenix::Stage{"compare", "PSNR/MAE/max-abs between two NRRD volumes", ::fenix::io::compare_vol});
}  // namespace
