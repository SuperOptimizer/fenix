// segment/trace_stream.hpp — out-of-core STREAMING form of the tiled tracer. Identical tiling, growth,
// clipping and stitch as trace_volume_tiled (it shares detail::trace_one_tile), but each tile's
// (tile + 2*halo) pred/CT block is FETCHED from a zarr store via io::read_zarr_region (only the chunks
// covering the tile are read) instead of cropped from a resident volume. The full multi-TB volume is
// therefore NEVER resident: peak RAM is one process-region block (a few MiB) + the accumulating
// fragments. This is the project's out-of-core hard rule applied to the tracer — the same fragments the
// in-core tiler makes, re-stitched downstream by the patch graph / Eulerian winding field.
#pragma once

#include "io/zarr.hpp"
#include "segment/grow.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace fenix::segment {

// Fetch [porg, porg+pe) from a zarr root into a contiguous u8 block (value = clamp(zarr_f32 * scale)).
// Missing chunks read as the zarr fill (air) inside read_zarr_region; a hard fetch error yields zeros
// here — production callers that must honour "absent != fetch-failed" should thread the Expected through.
inline Volume<u8> stream_tile_u8(const std::string& root, Index3 porg, Extent3 pe, f32 scale) {
    Volume<u8> out(pe);
    auto r = io::read_zarr_region(root, porg, pe);
    if (!r) return out;
    VolumeView<const f32> sv = r->view();
    VolumeView<u8> ov = out.view();
    parallel_for_z(pe, [&](s64 z) {
        for (s64 y = 0; y < pe.y; ++y)
            for (s64 x = 0; x < pe.x; ++x)
                ov(z, y, x) = static_cast<u8>(std::clamp(sv(z, y, x) * scale, 0.0f, 255.0f));
    });
    return out;
}

// Stream-trace the whole volume tile by tile from zarr stores (`pred_root`, optional `ct_root`).
// `full` is the volume extent (from the zarr `.zarray` shape). All other knobs match trace_volume_tiled.
// `pred_scale`/`ct_scale` map stored f32 -> u8 (e.g. 255 for 0..1 predictions, 1 for already-0..255).
inline VolumeResult trace_volume_streamed(const std::string& pred_root, const std::string& ct_root,
                                          Extent3 full, GrowParams p, int max_sheets, s64 min_valid,
                                          int seed_stride, f32 seed_thresh, int tile_core, int halo,
                                          int overlap = 0, int nf_ds = 8, f32 pred_scale = 1.0f,
                                          f32 ct_scale = 1.0f) {
    const bool has_ct = !ct_root.empty();
    VolumeResult R;
    for (s64 tz = 0; tz < full.z; tz += tile_core)
        for (s64 ty = 0; ty < full.y; ty += tile_core)
            for (s64 tx = 0; tx < full.x; tx += tile_core) {
                if (static_cast<int>(R.sheets.size()) >= max_sheets) return R;
                const Index3 porg{std::max<s64>(0, tz - halo), std::max<s64>(0, ty - halo), std::max<s64>(0, tx - halo)};
                const Extent3 pe{std::min<s64>(full.z, tz + tile_core + halo) - porg.z,
                                 std::min<s64>(full.y, ty + tile_core + halo) - porg.y,
                                 std::min<s64>(full.x, tx + tile_core + halo) - porg.x};
                if (pe.z < 16 || pe.y < 16 || pe.x < 16) continue;
                const Volume<u8> tf = stream_tile_u8(pred_root, porg, pe, pred_scale);  // <-- the OOC fetch
                Volume<u8> tc;
                if (has_ct) tc = stream_tile_u8(ct_root, porg, pe, ct_scale);
                const Index3 clo{tz - porg.z, ty - porg.y, tx - porg.x};
                const Index3 chi{std::min<s64>(pe.z, clo.z + tile_core), std::min<s64>(pe.y, clo.y + tile_core), std::min<s64>(pe.x, clo.x + tile_core)};
                std::vector<Surface> frags = detail::trace_one_tile<u8>(
                    tf.view(), has_ct ? tc.view() : VolumeView<const u8>{}, porg, clo, chi, p, min_valid, seed_stride, seed_thresh, nf_ds, overlap);
                for (Surface& s : frags) {
                    if (static_cast<int>(R.sheets.size()) >= max_sheets) break;
                    R.sheets.push_back(std::move(s));
                }
            }
    return R;
}

}  // namespace fenix::segment
