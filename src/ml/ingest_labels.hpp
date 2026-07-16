// ml/ingest_labels.hpp — convert a dense surface-label zarr (canonical-model release) into a
// tri-state GT band .fxvol for train-feed's dense-GT pairs.
//
//   fenix ingest-labels <zarr-level-dir> <out.fxvol> [sheet=1] [moat=0] [minkb=6] [q=2]
//
// Input chunks hold surface SHELLS: sheet=<v> on labeled wrap surfaces, 2=ignore, 0=rest.
// Output band encoding matches rasterize_band_multi: 255 sheet / 128 trusted-bg / 0 unlabeled.
// moat=0 trusts label 0 as background outright (releases with ignore=2 discipline: 0139 wrap
// zarrs, 0500P2 [sheet=255], 0343P, MANBp). moat=R (vox) restricts background to within R of a
// labeled sheet — required for the 1667 full-scroll zarr whose fill_value is 0, so a bare 0 is
// ambiguous (unlabeled wraps are also 0 there; measured 2026-07-16).
// Only sheet-bearing 128³ zarr chunks are decoded (candidates pre-filtered by compressed size
// >= minkb) and only sheet-bearing cores are written, so the archive's Real coverage doubles as
// the feeder's origin-sampling occupancy.
#pragma once

#include <atomic>
#include <charconv>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "geom/edt.hpp"
#include "io/zarr.hpp"

namespace fenix::ml {

inline Expected<int> run_ingest_labels(std::span<const std::string_view> args, Context& ctx) {
    if (args.size() < 2)
        return err(Errc::invalid_argument,
                   "usage: ingest-labels <zarr-level-dir> <out.fxvol> [sheet=1] [moat=0] [minkb=6] [q=2]");
    s64 sheet_val = 1, minkb = 6;
    f64 moat = 0, q = 2;
    for (usize i = 2; i < args.size(); ++i) {
        const auto a = args[i];
        auto snum = [&](std::string_view k, s64& v) {
            if (!a.starts_with(k)) return false;
            const auto t = a.substr(k.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        auto fnum = [&](std::string_view k, f64& v) {
            if (!a.starts_with(k)) return false;
            const auto t = a.substr(k.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (!(snum("sheet=", sheet_val) || fnum("moat=", moat) || snum("minkb=", minkb) || fnum("q=", q)))
            return err(Errc::invalid_argument, "ingest-labels: unknown arg " + std::string(a));
    }
    const std::string lroot(args[0]);
    auto meta = io::read_zarray(lroot);
    if (!meta) return std::unexpected(meta.error());
    const Extent3 vol = meta->shape;
    constexpr s64 G = 128;  // canonical release chunk side
    if (meta->chunks.z != G || meta->chunks.y != G || meta->chunks.x != G)
        return err(Errc::invalid_argument, "ingest-labels: expected 128^3 chunks");

    // Candidate chunks from the filesystem: flat "z.y.x" files or nested z/y/x dirs, compressed
    // size >= minkb (all-constant blosc chunks are ~2 KB; sheet-bearing ones are far larger).
    namespace fs = std::filesystem;
    std::vector<Index3> cand;
    auto push = [&](const std::string& name, uintmax_t sz, char sep) {
        if (sz < static_cast<uintmax_t>(minkb) * 1024) return;
        Index3 c{};
        usize p0 = name.find(sep);
        usize p1 = p0 == std::string::npos ? std::string::npos : name.find(sep, p0 + 1);
        if (p1 == std::string::npos) return;
        auto pi = [&](usize b, usize e, s64& v) {
            return std::from_chars(name.data() + b, name.data() + e, v).ec == std::errc{};
        };
        if (!pi(0, p0, c.z) || !pi(p0 + 1, p1, c.y) || !pi(p1 + 1, name.size(), c.x)) return;
        cand.push_back(c);
    };
    std::error_code fec;
    for (const auto& e : fs::directory_iterator(lroot, fec)) {
        if (e.is_regular_file()) {
            const std::string n = e.path().filename().string();
            if (n.front() != '.') push(n, e.file_size(), '.');
        } else if (e.is_directory()) {  // nested layout: <z>/<y>/<x>
            for (const auto& e2 : fs::recursive_directory_iterator(e.path(), fec))
                if (e2.is_regular_file())
                    push(fs::relative(e2.path(), lroot, fec).generic_string(), e2.file_size(), '/');
        }
    }
    if (cand.empty()) return err(Errc::invalid_argument, "ingest-labels: no candidate chunks >= minkb");
    log(LogLevel::info, "ingest-labels: {} candidate chunks (of {}x{}x{} grid), moat={} sheet={}",
        cand.size(), (vol.z + G - 1) / G, (vol.y + G - 1) / G, (vol.x + G - 1) / G, moat, sheet_val);

    auto arch = codec::VolumeArchive::create(std::string(args[1]), vol, codec::DctParams{.q = static_cast<f32>(q)});
    if (!arch) return std::unexpected(arch.error());

    const s64 halo = moat > 0 ? static_cast<s64>(moat) + 2 : 0;
    const f32 moat2 = static_cast<f32>(moat * moat);
    constexpr s64 C = codec::fxvol_chunk_side;
    std::mutex mu;
    std::atomic<u64> nerr{0}, nwritten{0}, nskipped{0};
    parallel_for_io(0, static_cast<s64>(cand.size()), 16, [&](s64 gi) {
        const Index3 c = cand[static_cast<usize>(gi)];
        const Index3 co{c.z * G, c.y * G, c.x * G};
        const Index3 bo{std::max<s64>(0, co.z - halo), std::max<s64>(0, co.y - halo), std::max<s64>(0, co.x - halo)};
        const Extent3 be{std::min(co.z + G + halo, vol.z) - bo.z,
                         std::min(co.y + G + halo, vol.y) - bo.y,
                         std::min(co.x + G + halo, vol.x) - bo.x};
        auto v = io::read_zarr_region<u8>(lroot, bo, be);
        if (!v) {
            ++nerr;
            return;
        }
        auto vv = v->view();
        bool any_sheet = false;
        for (s64 z = 0; z < be.z && !any_sheet; ++z)
            for (s64 y = 0; y < be.y && !any_sheet; ++y)
                for (s64 x = 0; x < be.x; ++x)
                    if (vv(z, y, x) == sheet_val) {
                        any_sheet = true;
                        break;
                    }
        if (!any_sheet) {
            ++nskipped;
            return;
        }
        Volume<f32> dist2;  // squared distance to nearest sheet voxel (moat mode only)
        if (moat > 0) {
            Volume<u8> seed(be);
            auto sv = seed.view();
            for (s64 z = 0; z < be.z; ++z)
                for (s64 y = 0; y < be.y; ++y)
                    for (s64 x = 0; x < be.x; ++x) sv(z, y, x) = vv(z, y, x) == sheet_val;
            dist2 = geom::edt_squared(seed.view());
        }
        auto dv = moat > 0 ? dist2.view() : VolumeView<f32>{};
        const Extent3 ce{std::min(co.z + G, vol.z) - co.z, std::min(co.y + G, vol.y) - co.y,
                         std::min(co.x + G, vol.x) - co.x};
        std::vector<u8> band(static_cast<usize>(ce.z * ce.y * ce.x), 0);
        bool core_sheet = false;
        for (s64 z = 0; z < ce.z; ++z)
            for (s64 y = 0; y < ce.y; ++y)
                for (s64 x = 0; x < ce.x; ++x) {
                    const s64 sz = co.z + z - bo.z, sy = co.y + y - bo.y, sx = co.x + x - bo.x;
                    const u8 lv = vv(sz, sy, sx);
                    u8 o = 0;
                    if (lv == sheet_val) {
                        o = 255;
                        core_sheet = true;
                    } else if (lv == 0) {
                        if (moat <= 0)
                            o = 128;
                        else if (dv(sz, sy, sx) <= moat2)
                            o = 128;
                    }
                    band[static_cast<usize>((z * ce.y + y) * ce.x + x)] = o;
                }
        if (!core_sheet) {
            ++nskipped;
            return;
        }
        std::vector<u8> block(static_cast<usize>(C * C * C));
        std::lock_guard<std::mutex> lk(mu);
        for (s64 cz = co.z / C; cz * C < co.z + ce.z; ++cz)
            for (s64 cy = co.y / C; cy * C < co.y + ce.y; ++cy)
                for (s64 cx = co.x / C; cx * C < co.x + ce.x; ++cx) {
                    for (s64 z = 0; z < C; ++z)
                        for (s64 y = 0; y < C; ++y)
                            for (s64 x = 0; x < C; ++x) {
                                const s64 bz = cz * C + z - co.z, by = cy * C + y - co.y, bx = cx * C + x - co.x;
                                block[static_cast<usize>((z * C + y) * C + x)] =
                                    (bz < ce.z && by < ce.y && bx < ce.x)
                                        ? band[static_cast<usize>((bz * ce.y + by) * ce.x + bx)]
                                        : u8{0};
                            }
                    if (!arch->write_chunk(0, ChunkCoord{cz, cy, cx}, std::span<const u8>(block), u8{0})) ++nerr;
                }
        ++nwritten;
    });
    if (nerr) return err(Errc::fetch_failed, "ingest-labels: " + std::to_string(nerr.load()) + " chunk(s) failed");
    if (auto r = arch->close(); !r) return std::unexpected(r.error());
    (void)ctx;
    log(LogLevel::info, "ingest-labels: wrote {} ({} sheet chunks written, {} candidates skipped)", args[1],
        nwritten.load(), nskipped.load());
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_ingest_labels = ::fenix::register_stage(
    ::fenix::Stage{"ingest-labels", "dense surface-label zarr -> tri-state GT band .fxvol", ::fenix::ml::run_ingest_labels});
}  // namespace
