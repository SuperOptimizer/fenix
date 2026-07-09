// segment/segment.hpp — STUB. See segment/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "segment/affinity.hpp"
#include "segment/ced.hpp"
#include "segment/hessian.hpp"
#include "segment/partition.hpp"
#include "segment/ridge.hpp"
#include "segment/structure_tensor.hpp"
#include "segment/tracer.hpp"
#include "segment/trace_eval.hpp"
#include "segment/trace_surface.hpp"

namespace fenix::segment {

// Stage entry point (stub). Real implementation per segment/CLAUDE.md.
// `fenix sheetness <ct.fxvol> <out.fxvol> [sigma_grad=1] [sigma_tensor=2]` — per-voxel
// structure-tensor planarity (l0-l1)/l0 in [0,1]: "is there an actual planar SHEET here",
// tiled/haloed (O(tile^3) RAM). A much more specific GT-alignment oracle than raw brightness
// (a bright blob is not a sheet). Feeds the sheetness axis of GT grading.
inline Expected<int> run_sheetness(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2)
        return err(Errc::invalid_argument,
                   "usage: fenix sheetness <ct.fxvol> <out.fxvol> [sigma_grad=1] [sigma_tensor=2]");
    StParams p;
    for (usize i = 2; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view pfx, f32& dst) {
            if (a.starts_with(pfx)) {
                const auto t = a.substr(pfx.size());
                std::from_chars(t.data(), t.data() + t.size(), dst);
            }
        };
        num("sigma_grad=", p.sigma_grad);
        num("sigma_tensor=", p.sigma_tensor);
    }
    auto arch = codec::VolumeArchive::open(std::string(args[0]));
    if (!arch) return std::unexpected(arch.error());
    auto vol = arch->read_volume_as<u8>(0);
    if (!vol) return std::unexpected(vol.error());
    Volume<u8> sheet = structure_tensor_sheetness<u8, u8>(vol->view(), p);
    auto out = codec::VolumeArchive::create(std::string(args[1]), sheet.dims(), codec::DctParams{.q = 2.0f});
    if (!out) return std::unexpected(out.error());
    if (auto w = out->template write_volume<u8>(sheet.view()); !w) return std::unexpected(w.error());
    if (auto c = out->close(); !c) return std::unexpected(c.error());
    log(LogLevel::info, "sheetness {} -> {} ({}x{}x{}, sg={} st={})", args[0], args[1],
        sheet.dims().z, sheet.dims().y, sheet.dims().x, p.sigma_grad, p.sigma_tensor);
    return 0;
}

inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("segment");
}

}  // namespace fenix::segment

FENIX_REGISTER_STAGE(segment, "segment stage (stub)", ::fenix::segment::run)

namespace {
[[maybe_unused]] const int fenix_stage_sheetness = ::fenix::register_stage(::fenix::Stage{
    "sheetness", "per-voxel structure-tensor planarity (is there a real sheet here) -> .fxvol",
    ::fenix::segment::run_sheetness});
}  // namespace
