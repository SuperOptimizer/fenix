// preprocess/register_scans.hpp — `fenix register-scans`: cross-scan registration
// (torch-free). The ink hunt needs old-scan segment meshes in the NEW 2.4 µm rescans'
// frames, and upstream ships no transforms for PHercParis3/PHerc0332 — so we compute the
// affine ourselves: same physical scroll, scans standardized in orientation, so the
// transform is (known) uniform scale + (unknown) translation [+ optional z-flip]:
//   1) resample both COARSE volumes onto a common physical pitch (trilinear),
//   2) coarse-align material centroids,
//   3) FFT phase correlation (preprocess/fft) for the residual translation,
//   4) emit a VC-compatible transform.json mapping A-input voxels -> B-input voxels.
// Callers pass coarse pyramid LEVELS (whole-volume residency); the mesh pipeline then
// composes to full res via import-obj pre_scale (1/level_a) and post_scale (level_b).
// Self-validation downstream: surf-qc delta on the imported mesh (bright-sheet delta
// >= +5 == the frame is right).
//   fenix register-scans <a.fxvol|cache@url> <b.fxvol|cache@url> <out_transform.json>
//                        um_a=<um/vox of A input> um_b=<um/vox of B input>
//                        [grid=128] [zflip=auto|0|1] [thr=auto]
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/cached_volume.hpp"
#include "preprocess/aircut.hpp"
#include "preprocess/phasecorr.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::preprocess {

namespace detail {

struct RegVolume {
    std::optional<io::CachedVolume> cached;
    std::optional<codec::VolumeArchive> arch;
    Extent3 dims{};
    Expected<void> gather(Index3 o, Extent3 e, u8* out) {
        if (cached) return cached->gather_box_u8(o.z, o.y, o.x, e.z, e.y, e.x, out);
        return arch->gather_box_u8(0, o.z, o.y, o.x, e.z, e.y, e.x, out);
    }
    static Expected<RegVolume> open(const std::string& src) {
        RegVolume rv;
        if (const auto at = src.find('@'); at != std::string::npos) {
            auto cv = io::CachedVolume::open(src.substr(0, at), src.substr(at + 1));
            if (!cv) return std::unexpected(cv.error());
            rv.dims = cv->dims();
            rv.cached = std::move(*cv);
        } else {
            auto a = codec::VolumeArchive::open(src);
            if (!a) return std::unexpected(a.error());
            rv.dims = a->dims();
            rv.arch = std::move(*a);
        }
        return rv;
    }
};

// whole volume -> grid^3 f32 cube at `pitch_vox` voxels per grid cell (trilinear from a
// resident copy — coarse levels are small by contract), zflip mirrors z
inline Expected<std::vector<f32>> resample_cube(RegVolume& v, s64 grid, f64 pitch_vox, bool zflip) {
    std::vector<u8> whole(static_cast<usize>(v.dims.z * v.dims.y * v.dims.x));
    if (auto g = v.gather(Index3{0, 0, 0}, v.dims, whole.data()); !g) return std::unexpected(g.error());
    auto at = [&](s64 z, s64 y, s64 x) -> f32 {
        z = std::clamp<s64>(z, 0, v.dims.z - 1);
        y = std::clamp<s64>(y, 0, v.dims.y - 1);
        x = std::clamp<s64>(x, 0, v.dims.x - 1);
        return static_cast<f32>(whole[static_cast<usize>((z * v.dims.y + y) * v.dims.x + x)]);
    };
    std::vector<f32> cube(static_cast<usize>(grid * grid * grid), 0.0f);
    for (s64 z = 0; z < grid; ++z)
        for (s64 y = 0; y < grid; ++y)
            for (s64 x = 0; x < grid; ++x) {
                const f64 sz0 = (zflip ? static_cast<f64>(v.dims.z - 1) - static_cast<f64>(z) * pitch_vox
                                       : static_cast<f64>(z) * pitch_vox);
                const f64 sy0 = static_cast<f64>(y) * pitch_vox, sx0 = static_cast<f64>(x) * pitch_vox;
                if (sz0 < 0 || sz0 > static_cast<f64>(v.dims.z - 1) || sy0 > static_cast<f64>(v.dims.y - 1) ||
                    sx0 > static_cast<f64>(v.dims.x - 1))
                    continue;
                const s64 iz = static_cast<s64>(sz0), iy = static_cast<s64>(sy0), ix = static_cast<s64>(sx0);
                const f32 fz = static_cast<f32>(sz0 - static_cast<f64>(iz));
                const f32 fy = static_cast<f32>(sy0 - static_cast<f64>(iy));
                const f32 fx = static_cast<f32>(sx0 - static_cast<f64>(ix));
                const f32 c00 = at(iz, iy, ix) * (1 - fx) + at(iz, iy, ix + 1) * fx;
                const f32 c01 = at(iz, iy + 1, ix) * (1 - fx) + at(iz, iy + 1, ix + 1) * fx;
                const f32 c10 = at(iz + 1, iy, ix) * (1 - fx) + at(iz + 1, iy, ix + 1) * fx;
                const f32 c11 = at(iz + 1, iy + 1, ix) * (1 - fx) + at(iz + 1, iy + 1, ix + 1) * fx;
                cube[static_cast<usize>((z * grid + y) * grid + x)] =
                    (c00 * (1 - fy) + c01 * fy) * (1 - fz) + (c10 * (1 - fy) + c11 * fy) * fz;
            }
    return cube;
}

// wrap the fysics phase-correlation primitive: returns {tz,ty,tx,confidence} in grid
// units with b = a + t (peak of conj(FA)*FB / |.|)
inline std::array<f64, 4> correlate_cubes(std::span<const f32> a, std::span<const f32> b, s64 g) {
    Volume<f32> va({g, g, g}), vb({g, g, g});
    std::copy(a.begin(), a.end(), va.flat().begin());
    std::copy(b.begin(), b.end(), vb.flat().begin());
    f32 conf = 0;
    const Vec3f t = phase_correlate(va.view(), vb.view(), &conf);
    return {static_cast<f64>(t.z), static_cast<f64>(t.y), static_cast<f64>(t.x), static_cast<f64>(conf)};
}

}  // namespace detail

inline Expected<int> run_register_scans(std::span<const std::string_view> args, Context&) {
    if (args.size() < 3)
        return err(Errc::invalid_argument,
                   "usage: register-scans <a> <b> <out_transform.json> um_a=<v> um_b=<v> "
                   "[grid=128] [zflip=auto|0|1]");
    f64 um_a = 0, um_b = 0;
    s64 grid = 128;
    std::string zflip = "auto";
    for (usize i = 3; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("um_a=", um_a) || num("um_b=", um_b) || num("grid=", grid)) continue;
        if (a.starts_with("zflip=")) {
            zflip = std::string(a.substr(6));
            continue;
        }
        return err(Errc::invalid_argument, "register-scans: unknown arg '" + std::string(a) + "'");
    }
    if (!(um_a > 0) || !(um_b > 0)) return err(Errc::invalid_argument, "register-scans: um_a/um_b required");
    // power-of-2 grid for the radix-2 FFT
    s64 g = 64;
    while (g < grid) g <<= 1;
    grid = g;

    auto va = detail::RegVolume::open(std::string(args[0]));
    if (!va) return std::unexpected(va.error());
    auto vb = detail::RegVolume::open(std::string(args[1]));
    if (!vb) return std::unexpected(vb.error());

    // common physical pitch: both cubes cover their volume's full extent
    const f64 ext_a = std::max({static_cast<f64>(va->dims.z), static_cast<f64>(va->dims.y),
                                static_cast<f64>(va->dims.x)}) *
                      um_a;
    const f64 ext_b = std::max({static_cast<f64>(vb->dims.z), static_cast<f64>(vb->dims.y),
                                static_cast<f64>(vb->dims.x)}) *
                      um_b;
    const f64 pitch_um = std::max(ext_a, ext_b) / static_cast<f64>(grid - 1);

    auto cube_a = detail::resample_cube(*va, grid, pitch_um / um_a, false);
    if (!cube_a) return std::unexpected(cube_a.error());

    auto run_case = [&](bool zf) -> Expected<std::array<f64, 4>> {
        auto cube_b = detail::resample_cube(*vb, grid, pitch_um / um_b, zf);
        if (!cube_b) return std::unexpected(cube_b.error());
        return detail::correlate_cubes(*cube_a, *cube_b, grid);  // s: p_b = p_a + s (phasecorr convention)
    };
    bool use_zflip = zflip == "1";
    auto best = run_case(use_zflip);
    if (!best) return std::unexpected(best.error());
    if (zflip == "auto") {
        auto flipped = run_case(true);
        if (flipped && (*flipped)[3] > (*best)[3]) {
            best = flipped;
            use_zflip = true;
        }
    }
    const auto [tz_g, ty_g, tx_g, conf] = *best;

    // compose A-input-voxel -> B-input-voxel: p_b = s*p_a + t  (s = um_a/um_b);
    // grid-space: gb = ga + t_g (with optional z mirror on b's sampling); grid cell =
    // pitch_um. b_vox = gb*pitch/um_b (unflipped) or mirror.
    const f64 s = um_a / um_b;
    f64 t_vox[3] = {tz_g * pitch_um / um_b, ty_g * pitch_um / um_b, tx_g * pitch_um / um_b};
    std::FILE* f = std::fopen(std::string(args[2]).c_str(), "w");
    if (!f) return err(Errc::io_error, "register-scans: cannot write " + std::string(args[2]));
    if (!use_zflip) {
        std::fprintf(f,
                     "{\n  \"transformation_matrix\": [\n"
                     "    [%.8f, 0.0, 0.0, %.4f],\n"
                     "    [0.0, %.8f, 0.0, %.4f],\n"
                     "    [0.0, 0.0, %.8f, %.4f],\n"
                     "    [0.0, 0.0, 0.0, 1.0]\n  ],\n"
                     "  \"source\": \"%s\",\n  \"target\": \"%s\",\n  \"confidence\": %.4f,\n"
                     "  \"generator\": \"fenix register-scans (phase correlation)\"\n}\n",
                     s, t_vox[2], s, t_vox[1], s, t_vox[0],
                     std::string(args[0]).c_str(), std::string(args[1]).c_str(), conf);
    } else {
        // z mirrored: z_b = -s*z_a + (Zb-1 + t_z)
        const f64 zoff = static_cast<f64>(vb->dims.z - 1) - t_vox[0];  // mirrored-grid sign
        std::fprintf(f,
                     "{\n  \"transformation_matrix\": [\n"
                     "    [%.8f, 0.0, 0.0, %.4f],\n"
                     "    [0.0, %.8f, 0.0, %.4f],\n"
                     "    [0.0, 0.0, %.8f, %.4f],\n"
                     "    [0.0, 0.0, 0.0, 1.0]\n  ],\n"
                     "  \"source\": \"%s\",\n  \"target\": \"%s\",\n  \"confidence\": %.4f, \"zflip\": true,\n"
                     "  \"generator\": \"fenix register-scans (phase correlation)\"\n}\n",
                     s, t_vox[2], s, t_vox[1], -s, zoff,
                     std::string(args[0]).c_str(), std::string(args[1]).c_str(), conf);
    }
    std::fclose(f);
    std::printf("register-scans: scale %.5f  t(zyx vox_b) %.1f %.1f %.1f  zflip %d  confidence %.3f -> %s\n",
                s, t_vox[0], t_vox[1], t_vox[2], use_zflip ? 1 : 0, std::string(args[2]).c_str());
    return 0;
}

}  // namespace fenix::preprocess

namespace {
[[maybe_unused]] const int fenix_stage_register_scans = ::fenix::register_stage(::fenix::Stage{
    "register-scans", "cross-scan registration (phase correlation) -> transform.json", ::fenix::preprocess::run_register_scans});
}  // namespace
