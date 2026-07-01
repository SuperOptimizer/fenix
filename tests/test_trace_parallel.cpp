// test_trace_parallel.cpp — the parallel-tiled tracer must produce EXACTLY the same result as the
// serial one. Tiles are disjoint cores, so parallelism only changes *which thread* grows a tile, never
// the outcome: the merge iterates per-tile slots in tile-index order, so fragment order/ids/cells are
// bit-identical. g_parallel_serial forces the whole tiled trace serial for a clean A/B in one process.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "segment/grow.hpp"

#include <cmath>
#include <vector>

using namespace fenix;

namespace {
// A stack of gently-WAVY, mildly-noisy sheets spanning several tiles in every axis. The waviness +
// noise matter: a perfectly-flat axis-aligned plane makes degenerate grid quads that the grower's
// sliver/component cleanup discards, so it wouldn't produce fragments (real papyrus is never that
// clean). Deterministic (fixed-seed LCG) so serial and parallel see identical input.
Volume<u8> make_wavy_sheets(s64 s) {
    Volume<u8> v = Volume<u8>::zeros({s, s, s});
    auto vv = v.view();
    u32 st = 0x9e3779b9u;
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return static_cast<f32>(st >> 8) / 16777216.0f; };
    const f32 planes[] = {30.f, 70.f, 110.f};
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x) {
                f32 b = 0;
                for (f32 p : planes) {
                    const f32 px = p + 3.0f * std::sin(static_cast<f32>(y) * 0.06f) + 2.5f * std::cos(static_cast<f32>(z) * 0.05f);
                    b = std::max(b, std::exp(-0.5f * std::pow((static_cast<f32>(x) - px) / 2.5f, 2.0f)));
                }
                vv(z, y, x) = static_cast<u8>(std::clamp((b + (rnd() - 0.5f) * 0.10f) * 255.0f + 0.5f, 0.0f, 255.0f));
            }
    return v;
}

segment::VolumeResult trace(VolumeView<const u8> pred, bool serial, int tile_core) {
    segment::GrowParams gp;
    gp.step = 2.f;
    gp.surf_thresh = 0.3f * 255.f;
    gp.snap_radius = gp.step * 1.5f;
    gp.fold_thresh = 6;
    gp.arap_tol = 0.15f;
    const f32 seed_thresh = 0.55f * 255.f;
    g_parallel_serial = serial;
    // nf_ds=4 (finer normal field for these synthetic sheets); no CT (pred-only).
    auto R = segment::trace_volume_tiled<u8>(pred, VolumeView<const u8>{}, gp, 10000, 200, 8, seed_thresh, tile_core, 16, 0, 4);
    g_parallel_serial = false;
    return R;
}
s64 total_cells(const segment::VolumeResult& R) {
    s64 n = 0;
    for (const Surface& s : R.sheets) n += s.valid_count();
    return n;
}
}  // namespace

TEST(parallel_tiled_trace_equals_serial) {
    const Volume<u8> pred = make_wavy_sheets(160);
    const segment::VolumeResult par = trace(pred.view(), /*serial=*/false, /*tile_core=*/64);
    const segment::VolumeResult ser = trace(pred.view(), /*serial=*/true, /*tile_core=*/64);

    // Non-vacuous: the trace actually found sheets across multiple tiles.
    REQUIRE(ser.sheets.size() >= 8);
    REQUIRE(total_cells(ser) > 5000);

    // Bit-identical: same fragment count, same total cells, same per-fragment valid counts (in order).
    REQUIRE(par.sheets.size() == ser.sheets.size());
    CHECK(total_cells(par) == total_cells(ser));
    for (usize i = 0; i < ser.sheets.size(); ++i) CHECK(par.sheets[i].valid_count() == ser.sheets[i].valid_count());
}

TEST(parallel_tiled_trace_equals_serial_core128) {
    const Volume<u8> pred = make_wavy_sheets(200);
    const segment::VolumeResult par = trace(pred.view(), /*serial=*/false, /*tile_core=*/128);
    const segment::VolumeResult ser = trace(pred.view(), /*serial=*/true, /*tile_core=*/128);
    REQUIRE(par.sheets.size() >= 4);
    REQUIRE(par.sheets.size() == ser.sheets.size());
    CHECK(total_cells(par) == total_cells(ser));
}
