// bench_vol.hpp — shared volume loaders for the standalone argv-driven benches (test_trace_*, test_grow,
// test_mr, test_multiscale, test_codec_bench, test_transform_probe, test_multiseg_render). fenix reads
// only .fxvol archives (NRRD was removed); a `.zarr` path loads its level-0 region. These replace the old
// io::read_nrrd / read_nrrd_u8 / nrrd_max helpers the benches used to pull real data off disk.
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/zarr.hpp"

#include <algorithm>
#include <string>

namespace fenix::bench {

inline bool is_zarr(const std::string& p) { return p.size() > 5 && p.substr(p.size() - 5) == ".zarr"; }

// Load a volume (.fxvol archive, or a .zarr level-0 region) as dense f32.
inline Expected<Volume<f32>> load_f32(const std::string& path) {
    if (is_zarr(path)) {
        auto mr = io::read_zarray(path + "/0");
        if (!mr) return std::unexpected(mr.error());
        return io::read_zarr_region(path + "/0", {0, 0, 0}, mr->shape);
    }
    auto a = codec::VolumeArchive::open(path);
    if (!a) return std::unexpected(a.error());
    return a->read_volume(0);
}

// Load a volume as u8 = clamp(round(raw*scale)). u8-native .fxvol archives decode without an f32 pass.
inline Expected<Volume<u8>> load_u8(const std::string& path, f32 scale) {
    if (!is_zarr(path)) {
        auto a = codec::VolumeArchive::open(path);
        if (!a) return std::unexpected(a.error());
        if (a->src_dtype() == codec::DType::u8 && scale == 1.0f) return a->template read_volume_as<u8>(0);
    }
    auto vr = load_f32(path);
    if (!vr) return std::unexpected(vr.error());
    Volume<u8> out(vr->dims());
    const auto s = vr->view().flat();
    auto d = out.view().flat();
    for (s64 i = 0; i < vr->size(); ++i) {
        const f32 v = s[static_cast<usize>(i)] * scale;
        d[static_cast<usize>(i)] = static_cast<u8>(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v) + 0.5f);
    }
    return out;
}

// Max scalar value of a volume (for auto-scaling: 0..1 probs vs 0..255 codec scale).
inline Expected<f32> peak(const std::string& path) {
    auto vr = load_f32(path);
    if (!vr) return std::unexpected(vr.error());
    f32 mx = -1e30f;
    const auto s = vr->view().flat();
    for (s64 i = 0; i < vr->size(); ++i) mx = std::max(mx, s[static_cast<usize>(i)]);
    return mx;
}

}  // namespace fenix::bench
