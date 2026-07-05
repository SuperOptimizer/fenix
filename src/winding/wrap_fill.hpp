// winding/wrap_fill.hpp — `wrap-fill`: DENSE per-voxel sheet-instance labels. The fitted
// spiral model partitions space (wrap = round(winding_cont)); the CT air-cut says which
// voxels are papyrus. Label u8: 0 = air/bg, 1..k = papyrus of (wrap mod k), 255 = ignore
// (wrap-boundary uncertainty buffer |cont - round| > buffer, or out-of-table winding).
// A chunk where every papyrus voxel carries its wrap identity IS a surface model —
// the volumetric-instance training target (docs/design/multiscale-instance-surface.md §1).
//   fenix wrap-fill ct=<fxvol|cache@zarr> model=<fxmodel> out=<label.fxvol>
//                   [origin=z,y,x] [size=z,y,x] [k=8] [thresh=40] [buffer=0.42]
// origin/size: world box when ct is a cache@zarr (whole local fxvol otherwise).
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/cached_volume.hpp"
#include "winding/model_io.hpp"

#include <charconv>
#include <cmath>
#include <span>
#include <string>
#include <string_view>

namespace fenix::winding {

// The label core, reusable by the feeder: labels `out` (same dims as ct) in place.
inline void wrap_fill_labels(VolumeView<const u8> ct, const SpiralModel& m, Vec3f org, int k,
                             u8 thresh, f32 buffer, VolumeView<u8> out) {
    const Extent3 d = ct.dims();
    parallel_for(0, d.z, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const bool pap = ct(z, y, x) >= thresh;
                if (!pap) {
                    out(z, y, x) = 0;
                    continue;
                }
                const Vec3f p{org.z + static_cast<f32>(z), org.y + static_cast<f32>(y),
                              org.x + static_cast<f32>(x)};
                const f32 w = m.winding_cont(p);
                const f32 r = std::round(w);
                if (!(std::abs(w) < 1e30f) || r < 0 || std::abs(w - r) > buffer) {
                    out(z, y, x) = 255;  // boundary/uncertain -> ignore
                    continue;
                }
                out(z, y, x) = static_cast<u8>(1 + static_cast<s64>(r) % k);
            }
    });
}

inline Expected<int> run_wrap_fill(std::span<const std::string_view> args, Context&) {
    std::string ct_path, model_path, out_path;
    f64 oz = 0, oy = 0, ox = 0, sz = 0, sy = 0, sx = 0, thresh = 40, buffer = 0.42;
    s64 k = 8;
    for (const auto a : args) {
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            using V = std::remove_reference_t<decltype(v)>;
            f64 d2 = 0;
            std::from_chars(t.data(), t.data() + t.size(), d2);
            v = static_cast<V>(d2);
            return true;
        };
        auto triple = [&](std::string_view key, f64& z, f64& y, f64& x) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            const auto c1 = t.find(','), c2 = t.rfind(',');
            if (c1 == std::string_view::npos || c2 == c1) return false;
            std::from_chars(t.data(), t.data() + c1, z);
            std::from_chars(t.data() + c1 + 1, t.data() + c2, y);
            std::from_chars(t.data() + c2 + 1, t.data() + t.size(), x);
            return true;
        };
        if (num("k=", k) || num("thresh=", thresh) || num("buffer=", buffer)) continue;
        if (triple("origin=", oz, oy, ox) || triple("size=", sz, sy, sx)) continue;
        if (a.starts_with("ct=")) { ct_path = std::string(a.substr(3)); continue; }
        if (a.starts_with("model=")) { model_path = std::string(a.substr(6)); continue; }
        if (a.starts_with("out=")) { out_path = std::string(a.substr(4)); continue; }
        return err(Errc::invalid_argument, "wrap-fill: unknown arg '" + std::string(a) + "'");
    }
    if (ct_path.empty() || model_path.empty() || out_path.empty())
        return err(Errc::invalid_argument,
                   "usage: wrap-fill ct=<fxvol|cache@zarr> model=<fxmodel> out=<label.fxvol> "
                   "[origin=z,y,x] [size=z,y,x] [k=8] [thresh=40] [buffer=0.42]");
    auto m = read_fxmodel(model_path);
    if (!m) return std::unexpected(m.error());

    Volume<u8> ct;
    const auto at = ct_path.find('@');
    if (at != std::string::npos) {
        if (sz <= 0) return err(Errc::invalid_argument, "wrap-fill: cache@zarr needs size=z,y,x");
        auto cv = io::CachedVolume::open(ct_path.substr(0, at), ct_path.substr(at + 1));
        if (!cv) return std::unexpected(cv.error());
        ct = Volume<u8>::zeros({static_cast<s64>(sz), static_cast<s64>(sy), static_cast<s64>(sx)});
        if (auto g = cv->gather_box_u8(static_cast<s64>(oz), static_cast<s64>(oy), static_cast<s64>(ox),
                                       ct.dims().z, ct.dims().y, ct.dims().x, ct.view().data());
            !g)
            return std::unexpected(g.error());
    } else {
        auto a2 = codec::VolumeArchive::open(ct_path);
        if (!a2) return std::unexpected(a2.error());
        auto v = a2->read_volume_as<u8>();
        if (!v) return std::unexpected(v.error());
        ct = std::move(*v);
    }

    Volume<u8> lab = Volume<u8>::zeros(ct.dims());
    wrap_fill_labels(ct.view(), *m, Vec3f{static_cast<f32>(oz), static_cast<f32>(oy), static_cast<f32>(ox)},
                     static_cast<int>(k), static_cast<u8>(thresh), static_cast<f32>(buffer), lab.view());
    s64 counts[4] = {0, 0, 0, 0};  // air, colored, ignore, total
    for (u8 v : lab.flat()) {
        ++counts[3];
        if (v == 0) ++counts[0];
        else if (v == 255) ++counts[2];
        else ++counts[1];
    }
    log(LogLevel::info, "wrap-fill: {}x{}x{} k={} — air {:.1f}%, colored {:.1f}%, ignore {:.1f}%",
        lab.dims().z, lab.dims().y, lab.dims().x, k, 100.0 * counts[0] / counts[3],
        100.0 * counts[1] / counts[3], 100.0 * counts[2] / counts[3]);
    auto arch = codec::VolumeArchive::create(out_path, lab.dims(), codec::DctParams{.q = 0.5f});
    if (!arch) return std::unexpected(arch.error());
    if (auto w = arch->write_volume<u8>(lab.view()); !w) return std::unexpected(w.error());
    log(LogLevel::info, "wrap-fill: -> {}", out_path);
    return 0;
}

}  // namespace fenix::winding

namespace {
[[maybe_unused]] const int fenix_stage_wrap_fill = ::fenix::register_stage(::fenix::Stage{
    "wrap-fill", "dense per-voxel sheet-instance labels (model wrap partition x CT air-cut)",
    ::fenix::winding::run_wrap_fill});
}  // namespace
