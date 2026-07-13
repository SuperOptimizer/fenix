// io/tifxyz.hpp — importer for VC/villa `tifxyz` surface meshes (the Vesuvius Challenge
// segment GT format): a directory of x.tif / y.tif / z.tif (2D f32 coordinate images,
// XYZ-named — we store ZYX) + meta.json ({"scale": [su, sv], ...} = grid step as a
// fraction of a full-res voxel; 0.05 -> one grid cell every 20 voxels). Invalid cells
// are marked with negative coords (VC uses -1). The directory may be local OR an
// http(s)://s3:// URL (open-data bucket segment dirs) — one fetch_object code path.
// Import-once -> .fxsurf, then fully native (io CLAUDE.md importer policy).
#pragma once

#include "core/core.hpp"
#include "core/surface.hpp"
#include "io/surface.hpp"
#include "io/tiff.hpp"
#include "io/zarr.hpp"

#include <charconv>
#include <span>
#include <string>
#include <string_view>

namespace fenix::io {

namespace detail {
// Pull "scale": [a, b] out of a tifxyz meta.json. Tiny targeted scan, not a JSON parser —
// the files are machine-written by one exporter; anything unparseable falls back to 1.0.
inline void tifxyz_meta_scale(const std::string& text, f32& su, f32& sv) {
    const auto k = text.find("\"scale\"");
    if (k == std::string::npos) return;
    const auto lb = text.find('[', k);
    if (lb == std::string::npos) return;
    f64 vals[2] = {0, 0};
    usize p = lb + 1;
    for (int i = 0; i < 2; ++i) {
        while (p < text.size() &&
               (text[p] == ' ' || text[p] == '\n' || text[p] == '\r' || text[p] == '\t' || text[p] == ','))
            ++p;
        const char* b = text.data() + p;
        const char* e = text.data() + text.size();
        auto r = std::from_chars(b, e, vals[i]);
        if (r.ec != std::errc{}) return;
        p = static_cast<usize>(r.ptr - text.data());
    }
    if (vals[0] > 0 && vals[1] > 0) {
        su = static_cast<f32>(1.0 / vals[0]);  // grid-index -> voxels
        sv = static_cast<f32>(1.0 / vals[1]);
    }
}
}  // namespace detail

// Read a tifxyz directory (local path or remote URL) into a core::Surface
// (coords in LOD-0 voxels, ZYX).
inline Expected<Surface> read_tifxyz(const std::string& dir) {
    auto tif = [&](const char* name) -> Expected<TiffImage> {
        auto b = fetch_object(dir, name);
        if (!b) return std::unexpected(b.error());
        if (!*b) return err(Errc::not_found, "tifxyz: missing " + dir + "/" + name);
        return decode_tiff(**b);
    };
    auto xi = tif("x.tif");
    if (!xi) return std::unexpected(xi.error());
    auto yi = tif("y.tif");
    if (!yi) return std::unexpected(yi.error());
    auto zi = tif("z.tif");
    if (!zi) return std::unexpected(zi.error());
    if (xi->width != yi->width || xi->width != zi->width || xi->height != yi->height || xi->height != zi->height)
        return err(Errc::decode_error, "tifxyz: x/y/z dimension mismatch in " + dir);

    Surface s(xi->width, xi->height);
    // meta.json is optional (scale falls back to 1 grid cell per voxel), but a FAILED
    // fetch of a present object is still a hard error per absent-vs-failed.
    auto mb = fetch_object(dir, "meta.json");
    if (!mb) return std::unexpected(mb.error());
    if (*mb) {
        const std::string meta(reinterpret_cast<const char*>((*mb)->data()), (*mb)->size());
        detail::tifxyz_meta_scale(meta, s.scale_u, s.scale_v);
    }
    const usize n = static_cast<usize>(s.nu * s.nv);
    s64 nvalid = 0;
    for (usize i = 0; i < n; ++i) {
        const f32 x = xi->pix[i], y = yi->pix[i], z = zi->pix[i];
        // VC marks invalid cells with -1 coords; also treat NaN as invalid (fast-math: compare v!=v
        // is unreliable, so test via the negative sentinel first — VC never emits NaN in practice).
        const bool ok = x >= 0.0f && y >= 0.0f && z >= 0.0f;
        s.coord[i] = Vec3f{z, y, x};
        s.valid[i] = ok ? 1u : 0u;
        nvalid += ok;
    }
    if (nvalid == 0) return err(Errc::decode_error, "tifxyz: no valid cells in " + dir);
    return s;
}

// fenix import-tifxyz <tifxyz-dir-or-url> <out.fxsurf> [coordscale=<f>]
// coordscale multiplies every valid coord (and the grid step) — the LOD-k → LOD-0 lift
// for meshes traced on a downscaled grid (e.g. a `tifxyz_normalized` LOD-2 mesh needs
// coordscale=4 to reach the full-res scan frame). Default 1 (coords used verbatim).
inline Expected<int> run_import_tifxyz(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2)
        return err(Errc::invalid_argument, "usage: import-tifxyz <tifxyz-dir-or-url> <out.fxsurf> [coordscale=<f>]");
    const std::string dir(args[0]), out(args[1]);
    f32 cscale = 1.0f;
    for (usize i = 2; i < args.size(); ++i) {
        if (args[i].starts_with("coordscale=")) {
            const auto t = args[i].substr(11);
            if (std::from_chars(t.data(), t.data() + t.size(), cscale).ec != std::errc{} || !(cscale > 0))
                return err(Errc::invalid_argument, "import-tifxyz: bad coordscale");
        } else {
            return err(Errc::invalid_argument, "import-tifxyz: unknown arg '" + std::string(args[i]) + "'");
        }
    }
    auto s = read_tifxyz(dir);
    if (!s) return std::unexpected(s.error());
    if (cscale != 1.0f) {
        for (usize i = 0; i < static_cast<usize>(s->nu * s->nv); ++i)
            if (s->valid[i]) s->coord[i] = s->coord[i] * cscale;
        s->scale_u *= cscale;
        s->scale_v *= cscale;
    }
    if (auto w = write_fxsurf(out, *s); !w) return std::unexpected(w.error());
    log(LogLevel::info,
        "import-tifxyz: {}x{} grid, {} valid cells ({:.1f}%), grid step {:.3g}x{:.3g} vox -> {}",
        s->nu,
        s->nv,
        s->valid_count(),
        100.0 * static_cast<f64>(s->valid_count()) / static_cast<f64>(s->nu * s->nv),
        s->scale_u,
        s->scale_v,
        out);
    return 0;
}

namespace detail {
// Minimal classic-LE TIFF writer: uncompressed f32 grayscale, one strip. Enough for the
// tifxyz x/y/z planes (tifffile/VC tooling read it); NOT a general TIFF encoder.
inline Expected<void> write_tiff_f32(const std::string& path, s64 w, s64 h, std::span<const f32> pix) {
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return err(Errc::io_error, "cannot open " + tmp);
    const u64 data_bytes = static_cast<u64>(w) * static_cast<u64>(h) * 4;
    const u32 ifd_off = static_cast<u32>(8 + data_bytes);
    const u8 hdr[8] = {'I', 'I', 42, 0, static_cast<u8>(ifd_off), static_cast<u8>(ifd_off >> 8),
                       static_cast<u8>(ifd_off >> 16), static_cast<u8>(ifd_off >> 24)};
    f.write(reinterpret_cast<const char*>(hdr), 8);
    f.write(reinterpret_cast<const char*>(pix.data()), static_cast<std::streamsize>(data_bytes));
    auto put16 = [&](u16 v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    auto put32 = [&](u32 v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto entry = [&](u16 tag, u16 type, u32 count, u32 value) { put16(tag); put16(type); put32(count); put32(value); };
    put16(10);                                                       // entry count
    entry(256, 4, 1, static_cast<u32>(w));                           // ImageWidth
    entry(257, 4, 1, static_cast<u32>(h));                           // ImageLength
    entry(258, 3, 1, 32);                                            // BitsPerSample
    entry(259, 3, 1, 1);                                             // Compression: none
    entry(262, 3, 1, 1);                                             // Photometric: BlackIsZero
    entry(273, 4, 1, 8);                                             // StripOffsets
    entry(277, 3, 1, 1);                                             // SamplesPerPixel
    entry(278, 4, 1, static_cast<u32>(h));                           // RowsPerStrip
    entry(279, 4, 1, static_cast<u32>(data_bytes));                  // StripByteCounts
    entry(339, 3, 1, 3);                                             // SampleFormat: IEEE float
    put32(0);                                                        // next IFD
    f.close();
    if (!f) return err(Errc::io_error, "write failed: " + tmp);
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) return err(Errc::io_error, "rename failed: " + path);
    return {};
}
}  // namespace detail

// fenix export-tifxyz <in.fxsurf> <out-dir>
// Inverse of import-tifxyz: .fxsurf coord grid -> x/y/z.tif (f32, invalid = -1) + meta.json
// ({"scale":[su,sv]} in the VC fraction-of-a-voxel convention). Lets tifxyz-based tooling
// (crop_qc et al.) run on fenix-native meshes: OBJ imports, repaired surfaces.
inline Expected<int> run_export_tifxyz(std::span<const std::string_view> args, Context&) {
    if (args.size() != 2)
        return err(Errc::invalid_argument, "usage: export-tifxyz <in.fxsurf> <out-dir>");
    const std::string in(args[0]), dir(args[1]);
    auto s = read_fxsurf(in);
    if (!s) return std::unexpected(s.error());
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return err(Errc::io_error, "cannot create " + dir);
    const usize n = static_cast<usize>(s->nu * s->nv);
    std::vector<f32> plane(n);
    for (int c = 0; c < 3; ++c) {
        for (usize i = 0; i < n; ++i)
            plane[i] = s->valid[i] ? (c == 0 ? s->coord[i].x : c == 1 ? s->coord[i].y : s->coord[i].z) : -1.0f;
        const char* name = c == 0 ? "/x.tif" : c == 1 ? "/y.tif" : "/z.tif";
        if (auto w = detail::write_tiff_f32(dir + name, s->nu, s->nv, plane); !w) return std::unexpected(w.error());
    }
    std::ofstream m(dir + "/meta.json");
    m << "{\"format\":\"tifxyz\",\"type\":\"seg\",\"scale\":[" << (1.0f / s->scale_u) << "," << (1.0f / s->scale_v)
      << "]}\n";
    if (!m) return err(Errc::io_error, "cannot write " + dir + "/meta.json");
    log(LogLevel::info, "export-tifxyz: {} ({}x{}, {} valid) -> {}", in, s->nu, s->nv, s->valid_count(), dir);
    return 0;
}

}  // namespace fenix::io

namespace {
[[maybe_unused]] const int fenix_stage_export_tifxyz =
    ::fenix::register_stage(::fenix::Stage{"export-tifxyz",
                                           "export a .fxsurf coord grid as VC tifxyz (x/y/z.tif + meta.json)",
                                           ::fenix::io::run_export_tifxyz});
[[maybe_unused]] const int fenix_stage_import_tifxyz =
    ::fenix::register_stage(::fenix::Stage{"import-tifxyz",
                                           "import a VC tifxyz surface mesh (x/y/z.tif + meta.json) as .fxsurf",
                                           ::fenix::io::run_import_tifxyz});
}  // namespace
