// ml/surfaces_cli.hpp — `fenix surfaces`: the spatial-index query as a CLI (torch-free).
// "I want a 256³ chunk at (4000,4000,4000) — which surfaces exist there, where, and render
// them as training labels": lists every mesh intersecting the box with its uv extent +
// world bbox inside the box, and optionally rasterizes the union tri-state GT
// (255 sheet / 128 trusted background / 0 unlabeled) to a u8 .fxvol for training/viewing.
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/surface.hpp"
#include "ml/rasterize.hpp"
#include "ml/surface_index.hpp"

#include <charconv>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::ml {

// fenix surfaces <z> <y> <x> <D> <H> <W> <mesh.fxsurf...> [out=<gt.fxvol>] [thickness=2] [shell=12]
inline Expected<int> run_surfaces(std::span<const std::string_view> args, Context&) {
    if (args.size() < 7)
        return err(Errc::invalid_argument,
                   "usage: surfaces <z> <y> <x> <D> <H> <W> <mesh.fxsurf...> [out=gt.fxvol] "
                   "[thickness=2] [shell=12]");
    auto pi = [](std::string_view t, s64& v) {
        return std::from_chars(t.data(), t.data() + t.size(), v).ec == std::errc{};
    };
    Index3 org;
    Extent3 ext;
    if (!pi(args[0], org.z) || !pi(args[1], org.y) || !pi(args[2], org.x) || !pi(args[3], ext.z) ||
        !pi(args[4], ext.y) || !pi(args[5], ext.x))
        return err(Errc::invalid_argument, "surfaces: bad box coords");

    std::string outpath;
    f32 thickness = 2.0f, shell = 12.0f;
    std::vector<std::string> mesh_paths;
    for (usize i = 6; i < args.size(); ++i) {
        const auto a = args[i];
        if (a.starts_with("out="))
            outpath = std::string(a.substr(4));
        else if (a.starts_with("thickness=")) {
            const auto t = a.substr(10);
            std::from_chars(t.data(), t.data() + t.size(), thickness);
        } else if (a.starts_with("shell=")) {
            const auto t = a.substr(6);
            std::from_chars(t.data(), t.data() + t.size(), shell);
        } else
            mesh_paths.emplace_back(a);
    }
    if (mesh_paths.empty()) return err(Errc::invalid_argument, "surfaces: no meshes given");

    std::vector<Surface> surfs;
    surfs.reserve(mesh_paths.size());
    for (const auto& p : mesh_paths) {
        auto s = io::read_fxsurf(p);
        if (!s) return std::unexpected(s.error());
        surfs.push_back(std::move(*s));
    }
    std::vector<const Surface*> ptrs;
    for (auto& s : surfs) ptrs.push_back(&s);
    VolumeSurfaceIndex index(ptrs);

    const geom::Box3f q{static_cast<f32>(org.z),
                        static_cast<f32>(org.z + ext.z),
                        static_cast<f32>(org.y),
                        static_cast<f32>(org.y + ext.y),
                        static_cast<f32>(org.x),
                        static_cast<f32>(org.x + ext.x)};
    const auto hits = index.query(q);
    log(LogLevel::info,
        "surfaces: box z{}:{} y{}:{} x{}:{} -> {} of {} meshes intersect",
        org.z,
        org.z + ext.z,
        org.y,
        org.y + ext.y,
        org.x,
        org.x + ext.x,
        hits.size(),
        surfs.size());
    for (const auto& h : hits) {
        // tight uv + world extent of the hit, from the returned tile rects
        s64 u0 = INT64_MAX, v0 = INT64_MAX, u1 = 0, v1 = 0;
        geom::Box3f wb;
        for (const auto& r : h.rects) {
            u0 = std::min(u0, r.u0);
            v0 = std::min(v0, r.v0);
            u1 = std::max(u1, r.u1);
            v1 = std::max(v1, r.v1);
        }
        const Surface& s = *ptrs[h.mesh];
        for (const auto& r : h.rects)
            for (s64 v = r.v0; v <= std::min(r.v1, s.nv - 1); ++v)
                for (s64 u = r.u0; u <= std::min(r.u1, s.nu - 1); ++u) {
                    if (!s.is_valid(u, v)) continue;
                    const Vec3f c = s.at(u, v);
                    if (c.z < q.zlo || c.z > q.zhi || c.y < q.ylo || c.y > q.yhi || c.x < q.xlo || c.x > q.xhi)
                        continue;
                    wb.expand({c.z, c.z, c.y, c.y, c.x, c.x});
                }
        if (wb.zhi < wb.zlo) continue;  // tiles touched the box but no valid cell landed inside
        log(LogLevel::info,
            "  {}  uv [{}:{}]x[{}:{}]  inside-box z {:.0f}:{:.0f} y {:.0f}:{:.0f} x {:.0f}:{:.0f}",
            mesh_paths[h.mesh],
            u0,
            u1,
            v0,
            v1,
            wb.zlo,
            wb.zhi,
            wb.ylo,
            wb.yhi,
            wb.xlo,
            wb.xhi);
    }

    if (!outpath.empty()) {
        Volume<u8> gt = rasterize_band_multi(ptrs, org, ext, {.thickness = thickness, .shell = shell}, &index);
        s64 sheet = 0, bg = 0;
        for (u8 v : gt.flat()) {
            sheet += v == kLabelSheet;
            bg += v == kLabelBackground;
        }
        auto a = codec::VolumeArchive::create(outpath, ext, codec::DctParams{.q = 0.5f});
        if (!a) return std::unexpected(a.error());
        if (auto r = a->template write_volume<u8>(gt.view()); !r) return std::unexpected(r.error());
        if (auto r = a->close(); !r) return std::unexpected(r.error());
        log(LogLevel::info,
            "surfaces: GT -> {} (sheet {:.2f}%, trusted-bg {:.2f}%, unlabeled {:.2f}%)",
            outpath,
            100.0 * static_cast<f64>(sheet) / static_cast<f64>(ext.count()),
            100.0 * static_cast<f64>(bg) / static_cast<f64>(ext.count()),
            100.0 * static_cast<f64>(ext.count() - sheet - bg) / static_cast<f64>(ext.count()));
    }
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_surfaces = ::fenix::register_stage(
    ::fenix::Stage{"surfaces",
                   "query which surfaces exist in a box (R-tree) + optionally rasterize training GT",
                   ::fenix::ml::run_surfaces});
}  // namespace
