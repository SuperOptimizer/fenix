// view/composite.hpp — layered surface rendering straight off the archive: for each
// surface cell, march ±offsets along the across-sheet normal and reduce the samples with
// a composite mode (VC3D's Compositing lineage: mean/max/min/alpha/beer-Lambert). This
// is the "surface pane" of the viewer and the streamed successor of render_surface
// (which needs the whole volume in RAM). Values come back raw (u8 archives: [0,255]);
// alpha/beer normalize internally via value_scale.
#pragma once

#include "codec/archive.hpp"
#include "codec/source.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "view/sampler.hpp"

#include <atomic>
#include <cmath>
#include <vector>

namespace fenix::view {

enum class CompositeMode : u8 { mean, max, min, alpha, beer_lambert };

struct CompositeSpec {
    CompositeMode mode = CompositeMode::max;
    f32 lo = -8.0f, hi = 8.0f, step = 1.0f;  // normal offsets, LOD-0 voxels
    s64 lod = 0;
    f32 value_scale = 1.0f / 255.0f;  // value -> [0,1] for alpha/beer accumulation
    f32 absorption = 4.0f;            // beer-Lambert extinction per unit optical depth
    // best-effort/adaptive: never block on the network — sample the finest LOCAL level per
    // voxel, schedule misses in the background, set SurfaceImage::incomplete for redraw.
    bool best_effort = false;
};

struct SurfaceImage {
    s64 width = 0, height = 0;  // = surface (nu, nv)
    bool incomplete = false;    // best-effort renders: some samples used coarse/no data
    std::vector<f32> pix;       // height*width
    std::vector<u8> valid;      // 1 where the surface cell was rendered
};

// Render the whole (u,v) grid of `surf` at one value per cell. Cells without a stored
// normal use the coord-grid tangent cross product; degenerate cells come back invalid.
inline Expected<SurfaceImage> render_surface_composite(codec::VolumeSource& src, const Surface& surf,
                                                       CompositeSpec spec = {}) {
    if (surf.nu <= 0 || surf.nv <= 0) return err(Errc::invalid_argument, "empty surface");
    if (!(spec.step > 0) || spec.hi < spec.lo) return err(Errc::invalid_argument, "bad offset range");
    if (spec.lod < 0 || spec.lod >= static_cast<s64>(src.nlods())) return err(Errc::invalid_argument, "bad lod");
    const s64 nlayer = static_cast<s64>(std::floor((spec.hi - spec.lo) / spec.step)) + 1;

    SurfaceImage img;
    img.width = surf.nu;
    img.height = surf.nv;
    img.pix.assign(static_cast<usize>(surf.nu * surf.nv), 0.0f);
    img.valid.assign(img.pix.size(), 0);
    const f32 inv = 1.0f / static_cast<f32>(1 << spec.lod);
    std::atomic<bool> failed{false}, incomplete{false};

    parallel_for(0, surf.nv, [&](s64 v) {
        BlockSampler smp(src, spec.lod, spec.best_effort);
        for (s64 u = 0; u < surf.nu; ++u) {
            const usize i = surf.idx(u, v);
            if (!surf.valid[i]) continue;
            Vec3f n;
            if (!surf.normal.empty()) {
                n = surf.normal[i];
            } else {
                const Vec3f tu = surf.at(std::min(u + 1, surf.nu - 1), v) - surf.at(std::max<s64>(u - 1, 0), v);
                const Vec3f tv = surf.at(u, std::min(v + 1, surf.nv - 1)) - surf.at(u, std::max<s64>(v - 1, 0));
                n = cross(tu, tv);
            }
            const f32 len = norm(n);
            if (!(len > 1e-6f)) continue;
            n = n * (1.0f / len);
            const Vec3f base = surf.coord[i];

            f32 acc = 0, aux = 0;
            bool first = true;
            for (s64 k = 0; k < nlayer; ++k) {
                const f32 t = spec.lo + static_cast<f32>(k) * spec.step;
                const Vec3f p = base + n * t;
                const f32 val = smp.trilinear(p.z * inv, p.y * inv, p.x * inv);
                switch (spec.mode) {
                    case CompositeMode::mean: acc += val; break;
                    case CompositeMode::max: acc = first ? val : std::max(acc, val); break;
                    case CompositeMode::min: acc = first ? val : std::min(acc, val); break;
                    case CompositeMode::alpha: {  // front-to-back; aux = transmittance
                        if (first) aux = 1.0f;
                        const f32 a = std::clamp(val * spec.value_scale, 0.0f, 1.0f);
                        acc += aux * a * val;
                        aux *= (1.0f - a);
                        break;
                    }
                    case CompositeMode::beer_lambert: {  // aux = optical depth so far
                        acc += val * std::exp(-spec.absorption * aux);
                        aux += val * spec.value_scale * spec.step;
                        break;
                    }
                }
                first = false;
                if (spec.mode == CompositeMode::alpha && aux < 0.01f) break;
            }
            if (spec.mode == CompositeMode::mean) acc /= static_cast<f32>(nlayer);
            img.pix[i] = acc;
            img.valid[i] = 1;
        }
        if (smp.failed()) failed.store(true, std::memory_order_relaxed);
        if (smp.incomplete()) incomplete.store(true, std::memory_order_relaxed);
    });
    if (failed.load()) return err(Errc::decode_error, "surface composite: chunk decode failed");
    img.incomplete = incomplete.load();
    return img;
}

inline Expected<SurfaceImage> render_surface_composite(codec::VolumeArchive& arch, const Surface& surf,
                                                       CompositeSpec spec = {}) {
    codec::ArchiveSource src(arch);
    return render_surface_composite(src, surf, spec);
}

}  // namespace fenix::view
