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

// fenix import-tifxyz <tifxyz-dir-or-url> <out.fxsurf>
inline Expected<int> run_import_tifxyz(std::span<const std::string_view> args, Context&) {
    if (args.size() != 2) return err(Errc::invalid_argument, "usage: import-tifxyz <tifxyz-dir-or-url> <out.fxsurf>");
    const std::string dir(args[0]), out(args[1]);
    auto s = read_tifxyz(dir);
    if (!s) return std::unexpected(s.error());
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

}  // namespace fenix::io

namespace {
[[maybe_unused]] const int fenix_stage_import_tifxyz =
    ::fenix::register_stage(::fenix::Stage{"import-tifxyz",
                                           "import a VC tifxyz surface mesh (x/y/z.tif + meta.json) as .fxsurf",
                                           ::fenix::io::run_import_tifxyz});
}  // namespace
