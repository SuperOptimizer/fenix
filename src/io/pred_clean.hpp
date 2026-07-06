#pragma once

// `fenix pred-clean <pred.fxvol> <ct.fxvol>` — zero prediction chunks with no real CT under them.
//
// Inference on a masked scroll can emit structured garbage where the CT is masked out (constant
// input → z-score noise amplification); predict-scroll now skips constant tiles at the source,
// but archives predicted before that fix (or with other tools) carry the garbage. Any pred chunk
// whose CT chunk is not Real has NO evidence under it — rewrite it as ZERO. Page-table-driven:
// touches only the offending chunks, decodes nothing.

#include <string>
#include <vector>

#include "codec/archive.hpp"
#include "core/core.hpp"

namespace fenix::io {

inline Expected<int> pred_clean_cmd(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix pred-clean <pred.fxvol> <ct.fxvol>");
        return err(Errc::invalid_argument, "missing args");
    }
    auto pred = codec::VolumeArchive::open(std::string(args[0]), /*writable=*/true);
    if (!pred) return std::unexpected(pred.error());
    auto ct = codec::VolumeArchive::open(std::string(args[1]));
    if (!ct) return std::unexpected(ct.error());
    if (!(pred->dims() == ct->dims())) return err(Errc::invalid_argument, "pred-clean: pred dims != ct dims");

    const ChunkCoord ce = pred->chunk_extent(0);
    const std::vector<u8> zeros(static_cast<usize>(codec::fxvol_chunk_side * codec::fxvol_chunk_side *
                                                   codec::fxvol_chunk_side),
                                0);
    s64 cleaned = 0, kept = 0;
    for (s64 cz = 0; cz < ce.z; ++cz)
        for (s64 cy = 0; cy < ce.y; ++cy)
            for (s64 cx = 0; cx < ce.x; ++cx) {
                const ChunkCoord c{cz, cy, cx};
                if (pred->coverage(0, c) != codec::Coverage::Real) continue;
                if (ct->coverage(0, c) == codec::Coverage::Real) { ++kept; continue; }
                if (auto w = pred->write_chunk(0, c, std::span<const u8>(zeros), u8{0}); !w)
                    return std::unexpected(w.error());
                ++cleaned;
            }
    if (auto c = pred->commit(); !c) return std::unexpected(c.error());
    log(LogLevel::info, "pred-clean: {} — zeroed {} garbage chunks (kept {} with real CT)", args[0], cleaned,
        kept);
    return 0;
}

}  // namespace fenix::io

namespace {
[[maybe_unused]] const int fenix_stage_pred_clean = ::fenix::register_stage(
    ::fenix::Stage{"pred-clean", "zero prediction chunks with no real CT under them (masked-region garbage)",
                   ::fenix::io::pred_clean_cmd});
}  // namespace
