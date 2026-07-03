// ml/surf_qc.hpp — `fenix surf-qc`: mesh↔volume alignment QC (torch-free).
// Some corpus meshes' upstream "-on-<volume>" resamples are misaligned (measured on
// PHercParis4: one mesh +13.7 on-sheet brightness, another ~0/negative — training on the
// bad ones feeds label noise that stalls learning at chance). For each mesh: sample K valid
// cells, compare CT at the surface point against CT at ±off voxels along the local surface
// NORMAL (from the uv tangent cross product). Aligned meshes sit on bright papyrus with
// darker gaps alongside: delta = ct@surface − mean(ct@±off) is clearly positive.
//   fenix surf-qc <ct.fxvol|cache@url> <fxsurf...> [k=200] [off=12] [min_delta=5]
// Prints one line per mesh + PASS/FAIL; exit 0 iff all pass (scriptable filter).
#pragma once

#include "core/core.hpp"
#include "core/surface.hpp"
#include "core/vec.hpp"
#include "io/cached_volume.hpp"
#include "io/surface.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::ml {

inline Expected<int> run_surf_qc(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2)
        return err(Errc::invalid_argument,
                   "usage: surf-qc <ct.fxvol|cache@zarr-url> <fxsurf...> [k=200] [off=12] [min_delta=5]");
    s64 k = 200;
    f64 off = 12, min_delta = 5;
    std::vector<std::string> meshes;
    for (usize i = 1; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("k=", k) || num("off=", off) || num("min_delta=", min_delta)) continue;
        meshes.emplace_back(a);
    }
    if (meshes.empty()) return err(Errc::invalid_argument, "surf-qc: no meshes");

    // volume: plain archive or on-demand cache — same duality as the feeder
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
    auto sample_box = [&](Index3 org, Extent3 e, u8* out) -> Expected<void> {
        if (cached) return cached->gather_box_u8(org.z, org.y, org.x, e.z, e.y, e.x, out);
        return arch->gather_box_u8(0, org.z, org.y, org.x, e.z, e.y, e.x, out);
    };

    int failures = 0;
    for (const auto& mp : meshes) {
        auto s = io::read_fxsurf(mp);
        if (!s) return std::unexpected(s.error());
        f64 d_on = 0, d_off = 0;
        s64 n = 0, probes = 0;
        const s64 stride = std::max<s64>(1, s->nu * s->nv / std::max<s64>(1, k * 4));
        for (s64 c = 0; c < s->nu * s->nv && n < k; c += stride) {
            const s64 u = c % s->nu, v = c / s->nu;
            if (!s->is_valid(u, v) || u + 1 >= s->nu || v + 1 >= s->nv || !s->is_valid(u + 1, v) ||
                !s->is_valid(u, v + 1))
                continue;
            const Vec3f p = s->at(u, v);
            // local normal = normalized cross of the uv tangents (ZYX; direction sign irrelevant)
            const Vec3f tu = s->at(u + 1, v) - p, tv = s->at(u, v + 1) - p;
            Vec3f nm = cross(tu, tv);
            const f32 nn = std::sqrt(nm.z * nm.z + nm.y * nm.y + nm.x * nm.x);
            if (nn < 1e-3f) continue;
            nm = Vec3f{nm.z / nn, nm.y / nn, nm.x / nn};
            auto at_vox = [&](Vec3f q) -> Expected<f64> {
                const Index3 org{static_cast<s64>(std::lround(q.z)),
                                 static_cast<s64>(std::lround(q.y)),
                                 static_cast<s64>(std::lround(q.x))};
                if (org.z < 1 || org.y < 1 || org.x < 1 || org.z + 2 >= dims.z || org.y + 2 >= dims.y ||
                    org.x + 2 >= dims.x)
                    return err(Errc::invalid_argument, "oob");
                u8 buf[27];
                if (auto r = sample_box(Index3{org.z - 1, org.y - 1, org.x - 1}, Extent3{3, 3, 3}, buf); !r)
                    return std::unexpected(r.error());
                f64 m = 0;
                for (u8 b : buf) m += b;
                return m / 27.0;
            };
            const auto c0 = at_vox(p);
            const auto cp = at_vox(p + nm * static_cast<f32>(off));
            const auto cm = at_vox(p - nm * static_cast<f32>(off));
            ++probes;
            if (!c0 || !cp || !cm) continue;
            d_on += *c0;
            d_off += (*cp + *cm) / 2.0;
            ++n;
        }
        if (n < k / 4) {
            std::printf("surf-qc %s  INSUFFICIENT (%lld/%lld probes ok)\n",
                        mp.c_str(),
                        static_cast<long long>(n),
                        static_cast<long long>(probes));
            ++failures;
            continue;
        }
        const f64 delta = (d_on - d_off) / static_cast<f64>(n);
        const bool pass = delta >= min_delta;
        std::printf("surf-qc %s  n=%lld  ct@surf %.1f  ct@±%.0f %.1f  delta %+.1f  %s\n",
                    mp.c_str(),
                    static_cast<long long>(n),
                    d_on / static_cast<f64>(n),
                    off,
                    d_off / static_cast<f64>(n),
                    delta,
                    pass ? "PASS" : "FAIL");
        if (!pass) ++failures;
    }
    if (failures) return err(Errc::invalid_argument, "surf-qc: " + std::to_string(failures) + " mesh(es) failed");
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_surf_qc = ::fenix::register_stage(::fenix::Stage{
    "surf-qc", "mesh<->volume alignment QC (on-sheet brightness vs normal offsets)", ::fenix::ml::run_surf_qc});
}  // namespace
