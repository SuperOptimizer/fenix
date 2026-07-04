// ml/qc_chunk.hpp — `fenix qc-chunk`: export small CT+band chunks for 3D triage
// (torch-free). 2D contact sheets fail exactly where labels do — oblique sheets, tangled
// wraps — so the triage gallery gets rotatable 3D: per sample point, a size³ CT chunk plus
// the mesh's rasterized band, written as a tiny two-volume file that
// tools/labelqc/chunk_viewer.py turns into a self-contained WebGL raycaster page.
// Format (.qcchunk): one JSON header line "{\"size\":N,\"origin\":[z,y,x],...}\n",
// then N³ raw CT u8, then N³ raw band u8 (255 = sheet).
//   fenix qc-chunk <ct.fxvol|cache@zarr-url> <fxsurf> <out_prefix>
//                  [n=4] [size=96] [thickness=2] [seed=7] [others=<dir-of-fxsurf>]
// others=: every OTHER mesh in the dir that crosses the chunk is rasterized at value 128
// (primary stays 255) — without it a chunk can't distinguish "trace off its sheet" from
// "trace on a sheet another segment owns".
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "io/cached_volume.hpp"
#include "io/surface.hpp"
#include "ml/rasterize.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::ml {

inline Expected<int> run_qc_chunk(std::span<const std::string_view> args, Context&) {
    if (args.size() < 3)
        return err(Errc::invalid_argument,
                   "usage: qc-chunk <ct.fxvol|cache@zarr-url> <fxsurf> <out_prefix> "
                   "[n=4] [size=96] [thickness=2] [seed=7]");
    s64 n = 4, size = 96;
    u64 seed = 7;
    f32 thickness = 2.0f;
    std::string others_dir;
    for (usize i = 3; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("n=", n) || num("size=", size) || num("thickness=", thickness) || num("seed=", seed)) continue;
        if (a.starts_with("others=")) {
            others_dir = std::string(a.substr(7));
            continue;
        }
        return err(Errc::invalid_argument, "qc-chunk: unknown arg '" + std::string(a) + "'");
    }
    n = std::clamp<s64>(n, 1, 32);
    size = std::clamp<s64>(size, 16, 320);

    std::optional<io::CachedVolume> cached;
    std::optional<codec::VolumeArchive> arch;
    Extent3 dims{};
    const std::string ct(args[0]);
    if (const auto at = ct.find('@'); at != std::string::npos) {
        auto cv = io::CachedVolume::open(ct.substr(0, at), ct.substr(at + 1));
        if (!cv) return std::unexpected(cv.error());
        dims = cv->dims();
        cached = std::move(*cv);
    } else {
        auto a = codec::VolumeArchive::open(ct);
        if (!a) return std::unexpected(a.error());
        dims = a->dims();
        arch = std::move(*a);
    }
    auto s = io::read_fxsurf(std::string(args[1]));
    if (!s) return std::unexpected(s.error());
    std::vector<std::string> other_paths;
    if (!others_dir.empty()) {
        for (const auto& de : std::filesystem::directory_iterator(others_dir)) {
            const std::string op = de.path().string();
            if (op.size() > 7 && op.ends_with(".fxsurf") && op != std::string(args[1])) other_paths.push_back(op);
        }
        std::sort(other_paths.begin(), other_paths.end());
    }

    // sample points spread over valid uv, hash-jittered by seed (same scheme as surf-sheet)
    std::vector<Vec3f> valid;
    const s64 stride = std::max<s64>(1, s->nu * s->nv / (n * 64));
    for (s64 c = hash_value(std::array<u64, 1>{seed}) % static_cast<u64>(stride); c < s->nu * s->nv; c += stride) {
        const s64 u = c % s->nu, v = c / s->nu;
        if (s->is_valid(u, v)) valid.push_back(s->at(u, v));
    }
    if (valid.empty()) return err(Errc::invalid_argument, "qc-chunk: mesh has no valid cells");

    const u64 tensor = static_cast<u64>(size) * static_cast<u64>(size) * static_cast<u64>(size);
    std::vector<u8> ctbuf(tensor);
    for (s64 i = 0; i < n; ++i) {
        const Vec3f p = valid[static_cast<usize>(i * static_cast<s64>(valid.size()) / n)];
        const Index3 org{
            std::clamp<s64>(static_cast<s64>(std::lround(p.z)) - size / 2, 0, std::max<s64>(0, dims.z - size)),
            std::clamp<s64>(static_cast<s64>(std::lround(p.y)) - size / 2, 0, std::max<s64>(0, dims.y - size)),
            std::clamp<s64>(static_cast<s64>(std::lround(p.x)) - size / 2, 0, std::max<s64>(0, dims.x - size))};
        Expected<void> g = cached ? cached->gather_box_u8(org.z, org.y, org.x, size, size, size, ctbuf.data())
                                  : arch->gather_box_u8(0, org.z, org.y, org.x, size, size, size, ctbuf.data());
        if (!g) return std::unexpected(g.error());
        // primary mesh at 255; every other corpus mesh crossing the chunk at 128 (value
        // precedence: primary stamps last and wins where they touch). Others load one at a
        // time — the whole corpus resident would be GBs.
        Volume<u8> band = Volume<u8>::zeros(Extent3{size, size, size});
        auto bv = band.view();
        const f32 rb = std::max(0.5f, thickness * 0.5f);
        s64 n_others = 0;
        for (const auto& op : other_paths) {
            auto os = io::read_fxsurf(op);
            if (!os) continue;
            const UvRect full{0, 0, os->nu - 1, os->nv - 1};
            detail::raster_pass(*os, org, Extent3{size, size, size}, rb, 128, bv, 0.5f, {&full, 1});
            ++n_others;
        }
        {
            const UvRect full{0, 0, s->nu - 1, s->nv - 1};
            detail::raster_pass(*s, org, Extent3{size, size, size}, rb, 255, bv, 0.5f, {&full, 1});
        }
        const std::string path = std::string(args[2]) + "_" + std::to_string(i) + ".qcchunk";
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return err(Errc::io_error, "qc-chunk: cannot write " + path);
        std::fprintf(f,
                     "{\"size\":%lld,\"origin\":[%lld,%lld,%lld],\"mesh\":\"%s\"}\n",
                     static_cast<long long>(size),
                     static_cast<long long>(org.z),
                     static_cast<long long>(org.y),
                     static_cast<long long>(org.x),
                     std::string(args[1]).c_str());
        std::fwrite(ctbuf.data(), 1, tensor, f);
        std::fwrite(band.flat().data(), 1, tensor, f);
        std::fclose(f);
        log(LogLevel::info, "qc-chunk: {} (origin {} {} {})", path, org.z, org.y, org.x);
    }
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_qc_chunk = ::fenix::register_stage(::fenix::Stage{
    "qc-chunk", "export CT+band chunks for the 3D triage viewer", ::fenix::ml::run_qc_chunk});
}  // namespace
