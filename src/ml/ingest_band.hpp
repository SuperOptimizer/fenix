// ml/ingest_band.hpp — band-limited data plane for the teacher sweep (torch-free).
// Whole-bbox sweeps measured 1.75 Pvox over the training corpus (spiral segments' bboxes
// cover most of the scroll); the surface BAND is ~3 orders less. Two subcommands:
//   fenix band-blocks <fxsurf,...> [block=1792] [halo=128] [band_r=640]
//     — plan sub-crops: print `z y x D H W` for every block-grid cell whose band_r-dilated
//       box touches a surface (whale bboxes don't fit predict's dense-decode; sweeps run
//       per block). Halo overlaps neighbors so patch-edge context exists at seams.
//   fenix ingest-band <zarr-url> <level> <out.fxvol> <fxsurf,...> <z0 y0 x0 D H W>
//                     [band_r=640] [q=8]
//     — like ingest-zarr for the crop, but fetches ONLY the 128³ zarr groups within band_r
//       of a surface; the rest stay zero (Zero-coverage chunks — the .fxvol is sparse).
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/surface.hpp"
#include "io/zarr.hpp"
#include "ml/surface_index.hpp"

#include <charconv>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::ml {

namespace detail {

inline Expected<std::vector<Surface>> load_band_surfs(std::string_view list) {
    std::vector<Surface> surfs;
    for (usize p = 0; p < list.size();) {
        const usize c = list.find(',', p);
        const auto one = list.substr(p, c == std::string_view::npos ? std::string_view::npos : c - p);
        if (!one.empty()) {
            auto s = io::read_fxsurf(std::string(one));
            if (!s) return std::unexpected(s.error());
            surfs.push_back(std::move(*s));
        }
        if (c == std::string_view::npos) break;
        p = c + 1;
    }
    if (surfs.empty()) return err(Errc::invalid_argument, "band: no surfaces given");
    return surfs;
}

inline bool band_hit(const VolumeSurfaceIndex& idx, Index3 org, Extent3 ext, f32 r) {
    const geom::Box3f q{static_cast<f32>(org.z) - r,
                        static_cast<f32>(org.z + ext.z) + r,
                        static_cast<f32>(org.y) - r,
                        static_cast<f32>(org.y + ext.y) + r,
                        static_cast<f32>(org.x) - r,
                        static_cast<f32>(org.x + ext.x) + r};
    return !idx.query(q).empty();
}

}  // namespace detail

// fenix band-blocks <fxsurf,...> [block=1792] [halo=128] [band_r=640]
inline Expected<int> run_band_blocks(std::span<const std::string_view> args, Context&) {
    if (args.size() < 1)
        return err(Errc::invalid_argument, "usage: band-blocks <fxsurf,...> [block=1792] [halo=128] [band_r=640]");
    s64 block = 1792, halo = 128;
    f64 band_r = 640;
    for (usize i = 1; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view k, auto& v) {
            if (!a.starts_with(k)) return false;
            const auto t = a.substr(k.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (!(num("block=", block) || num("halo=", halo) || num("band_r=", band_r)))
            return err(Errc::invalid_argument, "band-blocks: unknown arg " + std::string(a));
    }
    auto surfs = detail::load_band_surfs(args[0]);
    if (!surfs) return std::unexpected(surfs.error());
    std::vector<const Surface*> ptrs;
    for (auto& s : *surfs) ptrs.push_back(&s);
    const VolumeSurfaceIndex idx(ptrs);

    // grid over the union bbox: step = block (blocks OVERLAP by halo on each side via extent)
    Vec3f lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (const auto* s : ptrs)
        for (s64 v = 0; v < s->nv; ++v)
            for (s64 u = 0; u < s->nu; ++u) {
                if (!s->is_valid(u, v)) continue;
                const Vec3f c = s->at(u, v);
                lo = Vec3f{std::min(lo.z, c.z), std::min(lo.y, c.y), std::min(lo.x, c.x)};
                hi = Vec3f{std::max(hi.z, c.z), std::max(hi.y, c.y), std::max(hi.x, c.x)};
            }
    if (hi.z < lo.z) return err(Errc::invalid_argument, "band-blocks: no valid cells");
    const f32 r = static_cast<f32>(band_r);
    s64 n = 0;
    for (s64 z = static_cast<s64>(lo.z - r) / block * block; z < static_cast<s64>(hi.z + r); z += block)
        for (s64 y = static_cast<s64>(lo.y - r) / block * block; y < static_cast<s64>(hi.y + r); y += block)
            for (s64 x = static_cast<s64>(lo.x - r) / block * block; x < static_cast<s64>(hi.x + r); x += block) {
                const Index3 org{std::max<s64>(0, z - halo), std::max<s64>(0, y - halo), std::max<s64>(0, x - halo)};
                const Extent3 ext{z + block + halo - org.z, y + block + halo - org.y, x + block + halo - org.x};
                if (!detail::band_hit(idx, org, ext, r)) continue;
                std::printf("%lld %lld %lld %lld %lld %lld\n",
                            static_cast<long long>(org.z),
                            static_cast<long long>(org.y),
                            static_cast<long long>(org.x),
                            static_cast<long long>(ext.z),
                            static_cast<long long>(ext.y),
                            static_cast<long long>(ext.x));
                ++n;
            }
    log(LogLevel::info, "band-blocks: {} blocks (block={} halo={} band_r={})", n, block, halo, band_r);
    return 0;
}

// fenix ingest-band <zarr-url> <level> <out.fxvol> <fxsurf,...> <z0 y0 x0 D H W> [band_r=640] [q=8]
inline Expected<int> run_ingest_band(std::span<const std::string_view> args, Context& ctx) {
    if (args.size() < 10)
        return err(Errc::invalid_argument,
                   "usage: ingest-band <zarr-url> <level> <out.fxvol> <fxsurf,...> "
                   "<z0 y0 x0 D H W> [band_r=640] [q=8]");
    auto pi = [](std::string_view t, s64& v) {
        return std::from_chars(t.data(), t.data() + t.size(), v).ec == std::errc{};
    };
    s64 level = 0;
    Index3 org;
    Extent3 ext;
    if (!pi(args[1], level) || !pi(args[4], org.z) || !pi(args[5], org.y) || !pi(args[6], org.x) ||
        !pi(args[7], ext.z) || !pi(args[8], ext.y) || !pi(args[9], ext.x))
        return err(Errc::invalid_argument, "ingest-band: bad numbers");
    f64 band_r = 640, q = 8;
    for (usize i = 10; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view k, f64& v) {
            if (!a.starts_with(k)) return false;
            const auto t = a.substr(k.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (!(num("band_r=", band_r) || num("q=", q)))
            return err(Errc::invalid_argument, "ingest-band: unknown arg " + std::string(a));
    }
    auto surfs = detail::load_band_surfs(args[3]);
    if (!surfs) return std::unexpected(surfs.error());
    // crop-local coords: the surfaces are global, the output archive is the crop
    for (auto& s : *surfs)
        for (auto& c : s.coord)
            c = c - Vec3f{static_cast<f32>(org.z), static_cast<f32>(org.y), static_cast<f32>(org.x)};
    std::vector<const Surface*> ptrs;
    for (auto& s : *surfs) ptrs.push_back(&s);
    const VolumeSurfaceIndex idx(ptrs);

    const std::string url = std::string(args[0]) + "/" + std::to_string(level);
    auto arch = codec::VolumeArchive::create(std::string(args[2]), ext, codec::DctParams{.q = static_cast<f32>(q)});
    if (!arch) return std::unexpected(arch.error());

    // 128³ fetch groups (one zarr chunk each) that touch the band -> 8 output 64³ chunks
    constexpr s64 G = 128, C = codec::fxvol_chunk_side;
    struct Grp {
        Index3 o;
        Extent3 e;
    };
    std::vector<Grp> grps;
    for (s64 z = 0; z < ext.z; z += G)
        for (s64 y = 0; y < ext.y; y += G)
            for (s64 x = 0; x < ext.x; x += G) {
                const Extent3 ge{std::min(G, ext.z - z), std::min(G, ext.y - y), std::min(G, ext.x - x)};
                if (detail::band_hit(idx, {z, y, x}, ge, static_cast<f32>(band_r))) grps.push_back({{z, y, x}, ge});
            }
    log(LogLevel::info,
        "ingest-band: {} of {} groups in band ({:.1f}%)",
        grps.size(),
        ((ext.z + G - 1) / G) * ((ext.y + G - 1) / G) * ((ext.x + G - 1) / G),
        100.0 * static_cast<f64>(grps.size()) /
            static_cast<f64>(std::max<s64>(1, ((ext.z + G - 1) / G) * ((ext.y + G - 1) / G) * ((ext.x + G - 1) / G))));

    std::mutex mu;  // archive appends serialize
    std::atomic<u64> nerr{0}, ndone{0};
    parallel_for_io(0, static_cast<s64>(grps.size()), 16, [&](s64 gi) {
        const auto& g = grps[static_cast<usize>(gi)];
        auto v = io::read_zarr_region<u8>(url, Index3{org.z + g.o.z, org.y + g.o.y, org.x + g.o.x}, g.e);
        if (!v) {
            ++nerr;
            return;
        }
        auto vv = v->view();
        std::vector<u8> block(static_cast<usize>(C * C * C));
        std::lock_guard<std::mutex> lk(mu);
        for (s64 cz = g.o.z / C; cz <= (g.o.z + g.e.z - 1) / C; ++cz)
            for (s64 cy = g.o.y / C; cy <= (g.o.y + g.e.y - 1) / C; ++cy)
                for (s64 cx = g.o.x / C; cx <= (g.o.x + g.e.x - 1) / C; ++cx) {
                    for (s64 z = 0; z < C; ++z)
                        for (s64 y = 0; y < C; ++y)
                            for (s64 x = 0; x < C; ++x) {
                                const s64 sz = std::min(cz * C + z - g.o.z, g.e.z - 1);
                                const s64 sy = std::min(cy * C + y - g.o.y, g.e.y - 1);
                                const s64 sx = std::min(cx * C + x - g.o.x, g.e.x - 1);
                                block[static_cast<usize>((z * C + y) * C + x)] =
                                    vv(std::max<s64>(0, sz), std::max<s64>(0, sy), std::max<s64>(0, sx));
                            }
                    if (!arch->write_chunk(0, ChunkCoord{cz, cy, cx}, std::span<const u8>(block), u8{0})) ++nerr;
                }
        ++ndone;
    });
    if (nerr) return err(Errc::fetch_failed, "ingest-band: " + std::to_string(nerr.load()) + " group(s) failed");
    if (auto r = arch->close(); !r) return std::unexpected(r.error());
    (void)ctx;
    log(LogLevel::info, "ingest-band: wrote {} ({} band groups)", args[2], ndone.load());
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_band_blocks = ::fenix::register_stage(
    ::fenix::Stage{"band-blocks", "plan band sub-crops for the teacher sweep", ::fenix::ml::run_band_blocks});
[[maybe_unused]] const int fenix_stage_ingest_band = ::fenix::register_stage(
    ::fenix::Stage{"ingest-band", "ingest only the surface-band zarr chunks of a crop", ::fenix::ml::run_ingest_band});
}  // namespace
