// io/project.hpp — `fenix project`: z-projection of an .fxvol to a JPEG (torch-free).
// The ink-hunt review image: predict-ink emits a per-voxel probability volume over the
// layer stack — the human looks at its MAX (or mean) projection along z. General-purpose
// beyond ink (any stack -> quicklook).
//   fenix project <in.fxvol> <out.jpg> [mode=max|mean] [z0=0] [z1=-1] [q=92]
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/jpeg.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::io {

inline Expected<int> run_project(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2)
        return err(Errc::invalid_argument, "usage: project <in.fxvol> <out.jpg> [mode=max|mean] [z0=] [z1=] [q=92]");
    std::string mode = "max";
    s64 z0 = 0, z1 = -1, q = 92;
    for (usize i = 2; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("z0=", z0) || num("z1=", z1) || num("q=", q)) continue;
        if (a.starts_with("mode=")) {
            mode = std::string(a.substr(5));
            continue;
        }
        return err(Errc::invalid_argument, "project: unknown arg '" + std::string(a) + "'");
    }
    if (mode != "max" && mode != "mean") return err(Errc::invalid_argument, "project: mode wants max|mean");

    auto arch = codec::VolumeArchive::open(std::string(args[0]));
    if (!arch) return std::unexpected(arch.error());
    arch->reserve_cache(u64{1} << 30);
    const Extent3 d = arch->dims();
    if (z1 < 0 || z1 > d.z) z1 = d.z;
    z0 = std::clamp<s64>(z0, 0, std::max<s64>(0, z1 - 1));

    const usize plane = static_cast<usize>(d.y * d.x);
    std::vector<f32> acc(plane, mode == "max" ? 0.0f : 0.0f);
    std::vector<u8> layer(plane);
    for (s64 z = z0; z < z1; ++z) {
        if (auto g = arch->gather_box_u8(0, z, 0, 0, 1, d.y, d.x, layer.data()); !g) return std::unexpected(g.error());
        if (mode == "max") {
            for (usize i = 0; i < plane; ++i) acc[i] = std::max(acc[i], static_cast<f32>(layer[i]));
        } else {
            for (usize i = 0; i < plane; ++i) acc[i] += static_cast<f32>(layer[i]);
        }
    }
    const f32 div = mode == "mean" ? static_cast<f32>(std::max<s64>(1, z1 - z0)) : 1.0f;
    Image img;
    img.w = static_cast<int>(d.x);
    img.h = static_cast<int>(d.y);
    img.comps = 1;
    img.px.resize(plane);
    for (usize i = 0; i < plane; ++i) img.px[i] = static_cast<u8>(std::clamp(acc[i] / div, 0.0f, 255.0f) + 0.5f);
    if (auto w = write_jpeg(std::string(args[1]), img, static_cast<int>(q)); !w) return std::unexpected(w.error());
    log(LogLevel::info, "project: {} z[{},{}) {} -> {}", mode, z0, z1, args[0], args[1]);
    return 0;
}

}  // namespace fenix::io

namespace {
[[maybe_unused]] const int fenix_stage_project =
    ::fenix::register_stage(::fenix::Stage{"project", "z-projection of an .fxvol to JPEG (max/mean)", ::fenix::io::run_project});
}  // namespace
