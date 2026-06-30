// io/io.hpp — IO stage: registers the `ingest` subcommand (NRRD -> .fxvol). See io/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include "codec/archive.hpp"
#include "io/jpeg.hpp"
#include "io/nrrd.hpp"
#include "io/slice.hpp"
#include "io/zarr.hpp"

#include <charconv>
#include <chrono>
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

// `fenix export-scroll <zarr-root|url> <level> <out.fxvol> [q=8] [region=256] [z0 y0 x0 D H W]`
// Stream a whole OME-Zarr level into a .fxvol archive OUT-OF-CORE and RESUMABLY. Iterates the volume in
// `region`³ super-blocks (kept in RAM one at a time), reads each via the parallel zarr reader (missing
// chunks = air), writes its 64³ tiles, and COMMITS per region as a crash-safe checkpoint. RESUME: the
// archive's coverage tri-state is the bookmark — a region whose tiles are already present (Real/Zero) is
// skipped; a crash mid-region leaves those tiles ABSENT (COW) so a rerun redoes only the in-flight region.
// The optional [z0 y0 x0 D H W] limits the export to a sub-box of the (full-shape) archive.
inline Expected<int> export_scroll(std::span<const std::string_view> args, Context&) {
    if (args.size() < 3) {
        log(LogLevel::error, "usage: fenix export-scroll <zarr-root|url> <level> <out.fxvol> [q=8] [region=256] [z0 y0 x0 D H W]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto parse_f = [](std::string_view s, f32 d) { f32 v = d; std::from_chars(s.data(), s.data() + s.size(), v); return v; };
    std::string root(args[0]);
    while (!root.empty() && root.back() == '/') root.pop_back();
    const std::string lroot = root + "/" + std::string(args[1]);
    const std::string out(args[2]);
    const f32 q = args.size() > 3 ? parse_f(args[3], 8.0f) : 8.0f;
    s64 R = args.size() > 4 ? parse_i(args[4], 256) : 256;
    R = std::max<s64>(64, (R / 64) * 64);  // region must be a multiple of the 64³ tile

    auto meta = read_zarray(lroot);
    if (!meta) return std::unexpected(meta.error());
    const Extent3 shape = meta->shape;

    // Sub-box to export (default = whole volume). The archive is always created at the FULL shape.
    Index3 b0{0, 0, 0};
    Extent3 bd = shape;
    if (args.size() >= 11) {
        b0 = {parse_i(args[5], 0), parse_i(args[6], 0), parse_i(args[7], 0)};
        bd = {parse_i(args[8], shape.z), parse_i(args[9], shape.y), parse_i(args[10], shape.x)};
    }

    const bool resume = std::filesystem::exists(out);
    Expected<codec::VolumeArchive> ar = resume ? codec::VolumeArchive::open(out, true)
                                               : codec::VolumeArchive::create(out, shape, {.q = q});
    if (!ar) return std::unexpected(ar.error());
    codec::VolumeArchive a = std::move(*ar);
    if (resume && !(a.dims() == shape)) return err(Errc::invalid_argument, "resume: archive dims != zarr shape");

    const s64 T = codec::fxvol_chunk_side;  // 64
    auto ndiv = [](s64 n, s64 d) { return (n + d - 1) / d; };
    const s64 rz0 = b0.z / R, ry0 = b0.y / R, rx0 = b0.x / R;
    const s64 rz1 = ndiv(std::min(b0.z + bd.z, shape.z), R), ry1 = ndiv(std::min(b0.y + bd.y, shape.y), R), rx1 = ndiv(std::min(b0.x + bd.x, shape.x), R);
    const s64 total = (rz1 - rz0) * (ry1 - ry0) * (rx1 - rx0);
    log(LogLevel::info, "export-scroll {} L{} ({}x{}x{}) -> {} q={} region={}  {} regions  ({})", root, std::string(args[1]),
        shape.z, shape.y, shape.x, out, q, R, total, resume ? "RESUME" : "fresh");

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    s64 done = 0, skipped = 0;
    u64 real_tiles = 0, zero_tiles = 0;
    std::vector<f32> block(static_cast<usize>(T * T * T));
    for (s64 rz = rz0; rz < rz1; ++rz)
        for (s64 ry = ry0; ry < ry1; ++ry)
            for (s64 rx = rx0; rx < rx1; ++rx, ++done) {
                const Index3 org{rz * R, ry * R, rx * R};
                const ChunkCoord ft{org.z / T, org.y / T, org.x / T};  // a region is committed atomically →
                if (a.coverage(0, ft) != codec::Coverage::Absent) { ++skipped; continue; }  // its 1st tile says done
                const Extent3 ext{std::min(R, shape.z - org.z), std::min(R, shape.y - org.y), std::min(R, shape.x - org.x)};
                auto reg = read_zarr_region(lroot, org, ext);
                if (!reg) return std::unexpected(reg.error());  // hard fetch fail → abort; rerun resumes
                auto rv = reg->view();
                for (s64 tz = 0; tz < ndiv(ext.z, T); ++tz)
                    for (s64 ty = 0; ty < ndiv(ext.y, T); ++ty)
                        for (s64 tx = 0; tx < ndiv(ext.x, T); ++tx) {
                            for (s64 z = 0; z < T; ++z)  // extract a 64³ tile, edge-replicating at the box edge
                                for (s64 y = 0; y < T; ++y)
                                    for (s64 x = 0; x < T; ++x)
                                        block[static_cast<usize>((z * T + y) * T + x)] =
                                            rv(std::min(tz * T + z, ext.z - 1), std::min(ty * T + y, ext.y - 1), std::min(tx * T + x, ext.x - 1));
                            const ChunkCoord tc{rz * (R / T) + tz, ry * (R / T) + ty, rx * (R / T) + tx};
                            if (auto w = a.write_chunk(0, tc, block); !w) return std::unexpected(w.error());
                            (a.coverage(0, tc) == codec::Coverage::Real ? real_tiles : zero_tiles)++;
                        }
                if (auto c = a.commit(); !c) return std::unexpected(c.error());  // crash-safe checkpoint
                if (done % 200 == 0 || done + 1 == total) {
                    const f64 el = std::chrono::duration<f64>(clk::now() - t0).count();
                    const f64 frac = static_cast<f64>(done + 1) / static_cast<f64>(total ? total : 1);
                    log(LogLevel::info, "  {}/{} regions ({:.1f}%)  {:.0f}s  ETA {:.0f}s  | skipped {}  real {}  zero {}  data {:.1f}MiB",
                        done + 1, total, 100.0 * frac, el, frac > 0 ? el * (1.0 - frac) / frac : 0.0, skipped, real_tiles, zero_tiles,
                        static_cast<f64>(a.committed_size()) / (1024.0 * 1024.0));
                }
            }
    if (auto c = a.close(); !c) return std::unexpected(c.error());
    log(LogLevel::info, "export-scroll done: {} regions ({} skipped), real {} / zero {} tiles, {:.1f} MiB",
        total, skipped, real_tiles, zero_tiles, static_cast<f64>(a.committed_size()) / (1024.0 * 1024.0));
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
[[maybe_unused]] const int fenix_stage_export_scroll = ::fenix::register_stage(
    ::fenix::Stage{"export-scroll", "stream a whole OME-Zarr level into .fxvol (out-of-core, resumable)", ::fenix::io::export_scroll});
[[maybe_unused]] const int fenix_stage_export = ::fenix::register_stage(
    ::fenix::Stage{"export", "decode a .fxvol LOD level back to NRRD", ::fenix::io::export_vol});
[[maybe_unused]] const int fenix_stage_finalize = ::fenix::register_stage(
    ::fenix::Stage{"finalize", "repack a .fxvol into the SEALED coarse-first form", ::fenix::io::finalize_vol});
[[maybe_unused]] const int fenix_stage_fxinfo = ::fenix::register_stage(
    ::fenix::Stage{"fxinfo", "inspect a .fxvol (dims, LODs, coverage, size, ratio)", ::fenix::io::info_vol});
[[maybe_unused]] const int fenix_stage_compare = ::fenix::register_stage(
    ::fenix::Stage{"compare", "PSNR/MAE/max-abs between two NRRD volumes", ::fenix::io::compare_vol});
}  // namespace
