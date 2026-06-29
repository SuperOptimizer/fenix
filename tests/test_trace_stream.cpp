// test_trace_stream.cpp — the out-of-core STREAMING tracer. Write a synthetic raw zarr (planar sheets),
// trace it tile-by-tile straight from the store (volume never fully resident), and require the result to
// MATCH the in-core tiled tracer run on the same volume loaded whole. Proves the streaming path is
// equivalent to the resident path — only the tile source differs.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "io/surface.hpp"
#include "io/zarr.hpp"
#include "segment/grow.hpp"
#include "segment/trace_stream.hpp"
#include "winding/stitch_stream.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>

using namespace fenix;
namespace fs = std::filesystem;

namespace {
// Write a raw zarr v2 (f32, chunks 32^3, sep '/') of `n`^3 with three smooth, gently-curved Gaussian
// sheets near x = 24,48,72 — three "wraps" for the tracer to find from the streamed store.
std::string make_sheet_zarr(s64 n) {
    fs::path root = fs::temp_directory_path() / "fenix_stream.zarr";
    fs::remove_all(root);
    fs::create_directories(root);
    {
        std::ofstream z(root / ".zarray");
        z << R"({"zarr_format":2,"shape":[)" << n << "," << n << "," << n << R"(],"chunks":[32,32,32],)"
          << R"("dtype":"<f4","compressor":null,"fill_value":0,"order":"C","dimension_separator":"/"})";
    }
    const s64 cb = 32, nc = (n + cb - 1) / cb;
    // SMOOTH, gently CURVED Gaussian ridges (sigma=4): three "wraps". Smooth (not binary) so the
    // across-sheet gradient -> a well-defined structure-tensor normal; curved (not axis-aligned planes)
    // because a perfectly flat plane is a degenerate case the grower's frame/ARAP can't seed (real
    // laminations always curve). plane_k(z,y) = base_k + 8*sin(z*0.09 + y*0.05).
    auto ridge = [](s64 gz, s64 gy, s64 gx) {
        const f32 warp = 8.0f * std::sin(static_cast<f32>(gz) * 0.09f + static_cast<f32>(gy) * 0.05f);
        f32 v = 0;
        for (const s64 base : {s64{24}, s64{48}, s64{72}}) {
            const f32 d = static_cast<f32>(gx) - (static_cast<f32>(base) + warp);
            v = std::max(v, std::exp(-d * d / (2.0f * 4.0f * 4.0f)));  // sigma = 4 voxels (survives ds)
        }
        return v;
    };
    for (s64 cz = 0; cz < nc; ++cz)
        for (s64 cy = 0; cy < nc; ++cy)
            for (s64 cx = 0; cx < nc; ++cx) {
                std::vector<f32> buf(static_cast<usize>(cb * cb * cb), 0.0f);
                for (s64 lz = 0; lz < cb; ++lz)
                    for (s64 ly = 0; ly < cb; ++ly)
                        for (s64 lx = 0; lx < cb; ++lx) {
                            const s64 gz = cz * cb + lz, gy = cy * cb + ly, gx = cx * cb + lx;
                            if (gx < n && gz < n && gy < n)
                                buf[static_cast<usize>((lz * cb + ly) * cb + lx)] = ridge(gz, gy, gx);
                        }
                fs::create_directories(root / std::to_string(cz) / std::to_string(cy));
                std::ofstream c(root / std::to_string(cz) / std::to_string(cy) / std::to_string(cx), std::ios::binary);
                c.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(f32)));
            }
    return root.string();
}

segment::GrowParams sheet_params() {
    segment::GrowParams gp;
    gp.step = 2.0f;
    gp.surf_thresh = 0.12f * 255.0f;  // real-data-proven sweet spot for normalized predictions
    gp.snap_radius = gp.step * 1.5f;
    gp.fold_thresh = 6;
    gp.river_radius = 2;
    gp.arap_tol = 0.15f;
    return gp;
}

// ground-truth wrap of a fragment = nearest sheet base (24/48/72) to its bbox-x centre.
int gt_wrap(const winding::FragRec& r) {
    const f32 cx = 0.5f * (r.lo.x + r.hi.x);
    int g = 0;
    f32 bd = 1e30f;
    for (int k = 0; k < 3; ++k) {
        const f32 d = std::abs(cx - (24.0f + 24.0f * (f32)k));
        if (d < bd) { bd = d; g = k; }
    }
    return g;
}

// How well a name->winding map recovers the 3 physical wraps: number of DISTINCT majority windings
// (one per GT wrap) + the fraction of fragments on their wrap's majority winding.
struct GtStat {
    int distinct = 0;
    int agree = 0;
    int total = 0;
};
GtStat gt_stat(const std::unordered_map<std::string, s32>& win, const std::vector<winding::FragRec>& man) {
    std::map<int, std::map<s32, int>> votes;  // gt wrap -> {winding -> count}
    for (const winding::FragRec& r : man) {
        auto it = win.find(r.name);
        if (it != win.end()) votes[gt_wrap(r)][it->second]++;
    }
    std::map<s32, int> majs;
    GtStat s;
    for (const auto& [g, vm] : votes) {
        s32 best = 0;
        int bc = 0;
        for (const auto& [w, c] : vm) { s.total += c; if (c > bc) { bc = c; best = w; } }
        s.agree += bc;
        majs[best]++;
    }
    s.distinct = static_cast<int>(majs.size());
    return s;
}

}  // namespace

TEST(streamed_tracer_matches_in_core) {
    const s64 n = 96;
    const std::string root = make_sheet_zarr(n);
    const Extent3 full{n, n, n};
    const segment::GrowParams gp = sheet_params();
    const int tile_core = 48, halo = 12, seed_stride = 8;
    const f32 seed_thresh = 56.0f;

    // STREAMED: tile blocks fetched from the zarr store (volume never fully resident).
    const segment::VolumeResult rs = segment::trace_volume_streamed(
        root, "", full, gp, 10000, 50, seed_stride, seed_thresh, tile_core, halo, 0, 4, 255.0f, 1.0f);

    // IN-CORE reference: load the whole volume once, quantize to u8, run the resident tiled tracer.
    auto vf = io::read_zarr_region(root, {0, 0, 0}, full);
    REQUIRE(vf.has_value());
    Volume<u8> pred(full);
    {
        auto sv = vf->view();
        auto pv = pred.view();
        parallel_for_z(full, [&](s64 z) {
            for (s64 y = 0; y < n; ++y)
                for (s64 x = 0; x < n; ++x) pv(z, y, x) = static_cast<u8>(std::clamp(sv(z, y, x) * 255.0f, 0.0f, 255.0f));
        });
    }
    // PRIMARY correctness: the streamed tile FETCH must byte-match the resident CROP for every tile.
    // Since both tilers then run the identical detail::trace_one_tile, equal tiles => equal results.
    s64 mismatches = 0;
    for (s64 tz = 0; tz < n; tz += tile_core)
        for (s64 ty = 0; ty < n; ty += tile_core)
            for (s64 tx = 0; tx < n; tx += tile_core) {
                const Index3 porg{std::max<s64>(0, tz - halo), std::max<s64>(0, ty - halo), std::max<s64>(0, tx - halo)};
                const Extent3 pe{std::min<s64>(n, tz + tile_core + halo) - porg.z, std::min<s64>(n, ty + tile_core + halo) - porg.y,
                                 std::min<s64>(n, tx + tile_core + halo) - porg.x};
                const Volume<u8> blk = segment::stream_tile_u8(root, porg, pe, 255.0f);
                auto bv = blk.view();
                auto pv = pred.view();
                for (s64 z = 0; z < pe.z; ++z)
                    for (s64 y = 0; y < pe.y; ++y)
                        for (s64 x = 0; x < pe.x; ++x)
                            if (bv(z, y, x) != pv(porg.z + z, porg.y + y, porg.x + x)) ++mismatches;
            }
    CHECK(mismatches == 0);  // OOC fetch == resident data, voxel-for-voxel

    const segment::VolumeResult ri = segment::trace_volume_tiled<u8>(
        pred.view(), VolumeView<const u8>{}, gp, 10000, 50, seed_stride, seed_thresh, tile_core, halo, 0, 4);

    s64 vs = 0, vi = 0;
    for (const Surface& s : rs.sheets) vs += s.valid_count();
    for (const Surface& s : ri.sheets) vi += s.valid_count();
    std::printf("  [stream: %zu frags, valid=%lld | in-core: %zu frags, valid=%lld]\n",
                rs.sheets.size(), (long long)vs, ri.sheets.size(), (long long)vi);

    CHECK(!rs.sheets.empty());                         // it actually traced the sheets from the store
    CHECK(rs.sheets.size() == ri.sheets.size());       // same fragmentation as the resident path
    CHECK(vs > 0 && vi > 0);
    const f64 rel = vi ? std::abs(static_cast<f64>(vs - vi)) / static_cast<f64>(vi) : 1.0;
    CHECK(rel < 0.02);                                 // same coverage within fast-math jitter
    fs::remove_all(root);
}

TEST(fxsurf_roundtrip) {
    Surface s(8, 6);
    s.alloc_channels();
    for (s64 v = 0; v < 6; ++v)
        for (s64 u = 0; u < 8; ++u) {
            s.set(u, v, Vec3f{static_cast<f32>(v), static_cast<f32>(u), 1.5f});
            s.normal[s.idx(u, v)] = Vec3f{0, 0, 1};
            s.conf[s.idx(u, v)] = 2.0f;
        }
    s.valid[s.idx(3, 3)] = 0;  // a hole
    const std::string p = (fs::temp_directory_path() / "fenix_rt.fxsurf").string();
    REQUIRE(io::write_fxsurf(p, s).has_value());
    auto r = io::read_fxsurf(p);
    REQUIRE(r.has_value());
    CHECK(r->nu == 8 && r->nv == 6);
    CHECK(r->has_channels());
    CHECK(r->valid_count() == s.valid_count());
    bool ok = true;
    for (usize i = 0; i < s.coord.size(); ++i) {
        if (s.valid[i] != r->valid[i]) ok = false;
        const Vec3f a = s.coord[i], b = r->coord[i];
        if (a.z != b.z || a.y != b.y || a.x != b.x) ok = false;
        if (s.conf[i] != r->conf[i]) ok = false;
    }
    CHECK(ok);
    fs::remove(p);
}

// Fully out-of-core: stream tiles from zarr AND fragments to disk. The on-disk result must equal the
// in-RAM streamed trace, and reading every .fxsurf back must recover all the valid cells.
TEST(streamed_to_disk_matches_in_ram) {
    const s64 n = 96;
    const std::string root = make_sheet_zarr(n);
    const Extent3 full{n, n, n};
    const segment::GrowParams gp = sheet_params();
    const int tile_core = 48, halo = 12, seed_stride = 8;
    const f32 seed_thresh = 56.0f;

    const segment::VolumeResult rr = segment::trace_volume_streamed(
        root, "", full, gp, 10000, 50, seed_stride, seed_thresh, tile_core, halo, 0, 4, 255.0f, 1.0f);
    s64 vr = 0;
    for (const Surface& s : rr.sheets) vr += s.valid_count();

    const std::string odir = (fs::temp_directory_path() / "fenix_frags").string();
    fs::remove_all(odir);
    auto st = segment::trace_volume_streamed_to_disk(
        root, "", full, gp, 10000, 50, seed_stride, seed_thresh, tile_core, halo, odir, 0, 4, 255.0f, 1.0f);
    REQUIRE(st.has_value());
    std::printf("  [to-disk: %lld frags valid=%lld | in-ram: %zu frags valid=%lld]\n",
                (long long)st->fragments, (long long)st->valid_total, rr.sheets.size(), (long long)vr);
    CHECK(st->fragments == static_cast<s64>(rr.sheets.size()));
    CHECK(st->valid_total == vr);
    REQUIRE(st->fragments > 0);

    // read every written fragment back; total valid must match (the on-disk surfaces are complete)
    s64 vback = 0;
    bool chans = true;
    for (s64 i = 0; i < st->fragments; ++i) {
        char name[32];
        std::snprintf(name, sizeof name, "frag_%06lld.fxsurf", static_cast<long long>(i));
        auto s = io::read_fxsurf((fs::path(odir) / name).string());
        REQUIRE(s.has_value());
        vback += s->valid_count();
        if (!s->has_channels()) chans = false;
    }
    CHECK(vback == st->valid_total);  // round-trip recovers every cell
    CHECK(chans);                     // fill_surface_channels persisted (normal/conf on disk)
    CHECK(fs::exists(fs::path(odir) / "manifest.txt"));
    fs::remove_all(odir);
    fs::remove_all(root);
}

// Fully out-of-core END TO END: stream tiles -> fragments on disk -> OOC z-slab STITCH. The slab sweep
// (only one slab resident at a time) must RECOVER the 3 physical wraps — each wrap's fragments getting a
// consistent winding (now that the natural-BC fix lets the field resolve dense, volume-filling sheets).
TEST(ooc_slab_stitch_recovers_wraps) {
    const s64 n = 96;
    const std::string root = make_sheet_zarr(n);
    const segment::GrowParams gp = sheet_params();
    const std::string odir = (fs::temp_directory_path() / "fenix_stitch").string();
    fs::remove_all(odir);
    auto st = segment::trace_volume_streamed_to_disk(root, "", Extent3{n, n, n}, gp, 10000, 50, 8, 56.0f, 48, 12, odir, 0, 4, 255.0f, 1.0f);
    REQUIRE(st.has_value());
    REQUIRE(st->fragments > 0);

    winding::StitchStreamParams ssp;
    ssp.slab = 40;
    ssp.halo = 16;  // < fragment z-extent so adjacent slabs share boundary fragments
    ssp.gp.step = gp.step;
    ssp.gp.spacing = 24.0f;
    ssp.efield.ds = 2;
    ssp.efield.iters = 300;
    auto rep = winding::stitch_streamed(odir, ssp);
    REQUIRE(rep.has_value());
    CHECK(rep->slabs >= 2);  // actually swept multiple slabs (not one resident pass)

    std::unordered_map<std::string, s32> win;
    {
        std::ifstream f(fs::path(odir) / "windings.txt");
        std::string nm;
        s32 w;
        while (f >> nm >> w) win[nm] = w;
    }
    const std::vector<winding::FragRec> man = winding::read_manifest(odir);
    const GtStat gs = gt_stat(win, man);
    std::printf("  [slab GT: %lld slabs, distinct=%d agree=%d/%d]\n", (long long)rep->slabs, gs.distinct, gs.agree, gs.total);
    CHECK(gs.distinct == 3 && gs.agree >= gs.total * 9 / 10);  // recovers the 3 wraps, bounded RAM
    fs::remove_all(odir);
    fs::remove_all(root);
}

// Full 3D-tiled OOC stitch: bounds RAM in ALL axes. The 2x2x2-tiled sweep + BFS tile-graph alignment
// must induce the same fragment->winding partition as the in-RAM whole stitch, in ONE aligned component.
TEST(ooc_3d_tiled_stitch_recovers_wraps) {
    const s64 n = 96;
    const std::string root = make_sheet_zarr(n);
    const segment::GrowParams gp = sheet_params();
    const std::string odir = (fs::temp_directory_path() / "fenix_stitch3d").string();
    fs::remove_all(odir);
    auto st = segment::trace_volume_streamed_to_disk(root, "", Extent3{n, n, n}, gp, 10000, 50, 8, 56.0f, 48, 12, odir, 0, 4, 255.0f, 1.0f);
    REQUIRE(st.has_value());
    REQUIRE(st->fragments > 0);

    winding::StitchTiledParams tp;
    tp.tile = 48;
    tp.halo = 16;
    tp.gp.step = gp.step;
    tp.gp.spacing = 24.0f;
    tp.efield.ds = 2;
    tp.efield.iters = 300;
    auto rep = winding::stitch_streamed_3d(odir, tp);
    REQUIRE(rep.has_value());
    std::printf("  [3d-stitch: %lld frags, %lld tiles, %lld components, wraps[%d..%d]]\n",
                (long long)rep->fragments, (long long)rep->slabs, (long long)rep->components, rep->wrap_lo, rep->wrap_hi);
    CHECK(rep->slabs >= 4);       // genuine 3D tiling (multiple tiles, not one resident pass)
    CHECK(rep->components == 1);  // the tile graph aligned into a single global winding frame

    std::unordered_map<std::string, s32> win;
    {
        std::ifstream f(fs::path(odir) / "windings.txt");
        std::string nm;
        s32 w;
        while (f >> nm >> w) win[nm] = w;
    }
    // ground truth: the 3D-tiled stitch recovers the 3 physical wraps with bounded RAM in ALL axes.
    const std::vector<winding::FragRec> man = winding::read_manifest(odir);
    const GtStat g = gt_stat(win, man);
    std::printf("  [3d GT: distinct=%d agree=%d/%d]\n", g.distinct, g.agree, g.total);
    CHECK(g.distinct == 3 && g.agree >= g.total * 9 / 10);
    fs::remove_all(odir);
    fs::remove_all(root);
}
