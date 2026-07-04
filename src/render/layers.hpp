// render/layers.hpp — `fenix render-layers`: surface layer-stack rendering (torch-free).
// The ink-detection input contract: for every uv cell, sample the CT at N offsets along
// the surface normal -> an (nlayers, nv, nu) u8 volume ("surface volume"). This is the
// render module's core job (villa mesh_to_surface lineage) pointed at the ink hunt.
//   fenix render-layers <ct.fxvol|cache@zarr-url> <fxsurf> <out.fxvol>
//                       [layers=65] [step=1] [q=8]
// step is in THIS volume's voxels: upstream ink models were trained on 7.91 um scans at
// 1-voxel spacing — rendering from the 2.4 um rescans for those models wants
// step=3.296 (same PHYSICAL spacing) — while fenix-native models use step=1.
// cache@url inputs get an ensure() band prefetch (the sampler must never see
// absent-as-air).
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "io/cached_volume.hpp"
#include "io/surface.hpp"
#include "ml/surf_qc.hpp"
#include "view/sampler.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::render {

inline Expected<int> run_render_layers(std::span<const std::string_view> args, Context&) {
    if (args.size() < 3)
        return err(Errc::invalid_argument,
                   "usage: render-layers <ct.fxvol|cache@zarr-url> <fxsurf> <out.fxvol> "
                   "[layers=65] [step=1] [q=8]");
    s64 layers = 65;
    f64 step = 1.0, q = 8.0;
    for (usize i = 3; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("layers=", layers) || num("step=", step) || num("q=", q)) continue;
        return err(Errc::invalid_argument, "render-layers: unknown arg '" + std::string(a) + "'");
    }
    layers = std::clamp<s64>(layers, 3, 257);
    const s64 half = (layers - 1) / 2;

    auto s = io::read_fxsurf(std::string(args[1]));
    if (!s) return std::unexpected(s.error());

    std::optional<io::CachedVolume> cached;
    std::optional<codec::VolumeArchive> local;
    const std::string ct(args[0]);
    if (const auto at = ct.find('@'); at != std::string::npos) {
        auto cv = io::CachedVolume::open(ct.substr(0, at), ct.substr(at + 1));
        if (!cv) return std::unexpected(cv.error());
        cached = std::move(*cv);
        const s64 pad = static_cast<s64>(std::ceil(static_cast<f64>(half) * step)) + 2;
        for (s64 v = 0; v < s->nv; v += 4)
            for (s64 u = 0; u < s->nu; u += 4) {
                if (!s->is_valid(u, v)) continue;
                const Vec3f p = s->at(u, v);
                (void)cached->ensure(Index3{static_cast<s64>(p.z) - pad,
                                            static_cast<s64>(p.y) - pad,
                                            static_cast<s64>(p.x) - pad},
                                     Extent3{2 * pad, 2 * pad, 2 * pad});
            }
    } else {
        auto a = codec::VolumeArchive::open(ct);
        if (!a) return std::unexpected(a.error());
        local = std::move(*a);
    }
    codec::VolumeArchive& arch = cached ? cached->archive() : *local;
    arch.reserve_cache(u64{4} << 30);

    // (nlayers, nv, nu): layer index is the Z axis of the output volume — the shape the
    // upstream ink stacks use (a TIFF per layer, stacked). z is REPLICATE-PADDED to the
    // DCT block (16): a hard step to zero padding rings 6-8 counts into the outer REAL
    // layers (measured on a synthetic gradient stack) — replication removes the step.
    const s64 pad_z = (layers + 15) / 16 * 16;
    const Extent3 od{pad_z, s->nv, s->nu};
    Volume<u8> out = Volume<u8>::zeros(od);
    auto ov = out.view();
    std::atomic<bool> failed{false};
    parallel_for(0, s->nv, [&](s64 v) {
        view::BlockSampler smp(arch, /*lod=*/0);
        for (s64 u = 0; u < s->nu; ++u) {
            if (!s->is_valid(u, v)) continue;
            const auto nm = ml::detail::stencil_normal(*s, u, v);
            if (!nm) continue;
            const Vec3f p = s->at(u, v);
            for (s64 L = 0; L < layers; ++L) {
                const f32 off = static_cast<f32>((L - half) * step);
                const Vec3f qp = p + *nm * off;
                const f32 val = smp.trilinear(qp.z, qp.y, qp.x);
                ov(L, v, u) = static_cast<u8>(std::clamp(val, 0.0f, 255.0f) + 0.5f);
            }
        }
        if (smp.failed()) failed.store(true, std::memory_order_relaxed);
        for (s64 L = layers; L < pad_z; ++L)
            for (s64 u = 0; u < s->nu; ++u) ov(L, v, u) = ov(layers - 1, v, u);
    });
    if (failed.load()) return err(Errc::io_error, "render-layers: chunk decode/fetch failed (never silent air)");

    auto oa = codec::VolumeArchive::create(std::string(args[2]), od, codec::DctParams{.q = static_cast<f32>(q)});
    if (!oa) return std::unexpected(oa.error());
    if (auto w = oa->write_volume<u8>(out.view()); !w) return std::unexpected(w.error());
    if (auto c = oa->close(); !c) return std::unexpected(c.error());
    log(LogLevel::info,
        "render-layers: {} layers (z padded to {}) x {}x{} (step {} vox) -> {}",
        layers,
        pad_z,
        s->nv,
        s->nu,
        step,
        args[2]);
    return 0;
}

}  // namespace fenix::render

namespace {
[[maybe_unused]] const int fenix_stage_render_layers = ::fenix::register_stage(::fenix::Stage{
    "render-layers", "surface layer-stack rendering (the ink-detection input)", ::fenix::render::run_render_layers});
}  // namespace
