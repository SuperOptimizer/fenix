// winding/stitch_stream.hpp — out-of-core winding STITCH over on-disk fragments (the .fxsurf + manifest
// written by segment::trace_volume_streamed_to_disk). Sweeps z-slabs: per slab, load ONLY the fragments
// whose bbox intersects the slab (+halo), run the in-RAM patch graph + band-Eulerian winding locally
// (translated to a slab-local origin so the field is slab-sized), then align the slab's windings to the
// previous slab via the fragments they SHARE in the overlap (median winding offset, with flip
// detection). Peak RAM is one slab's fragments + one slab-sized field — not the whole surface set.
// Writes <dir>/windings.txt (one `name winding` per fragment). NOTE: z-slabs bound the z dimension only;
// a scroll whose y,x cross-section doesn't fit RAM needs full 3D tiling (the next step — same pattern).
#pragma once

#include "annotate/umbilicus.hpp"
#include "core/core.hpp"
#include "io/surface.hpp"
#include "segment/patch_graph.hpp"
#include "winding/patch_field.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fenix::winding {

struct StitchStreamParams {
    s64 slab = 256;                  // z-slab thickness (full-res voxels)
    s64 halo = 64;                   // z overlap so adjacent slabs SHARE boundary fragments (winding align)
    segment::PatchGraphParams gp{};  // patch-graph params (orientation + spacing)
    FieldParams efield{};            // Eulerian field params (band auto-set from spacing if <=0)
};

struct StitchStreamReport {
    s64 fragments = 0;
    s64 slabs = 0;        // z-slabs (stitch_streamed) or 3D tiles (stitch_streamed_3d)
    s64 components = 0;   // disconnected tile-graph components (1 = fully aligned; >1 = islands)
    s32 wrap_lo = 0, wrap_hi = 0;
};

struct StitchTiledParams {
    s64 tile = 256;                  // cubic 3D tile size (full-res voxels) — bounds RAM in ALL axes
    s64 halo = 64;                   // tile overlap so adjacent tiles SHARE fragments (for alignment)
    segment::PatchGraphParams gp{};
    FieldParams efield{};
};

struct FragRec {
    std::string name;
    s64 valid = 0;
    Vec3f lo{0, 0, 0}, hi{0, 0, 0};
};

// Parse the manifest written by trace_volume_streamed_to_disk: `name valid bz by bx Bz By Bx` per line.
inline std::vector<FragRec> read_manifest(const std::string& dir) {
    std::vector<FragRec> out;
    std::ifstream f(std::filesystem::path(dir) / "manifest.txt");
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        FragRec r;
        if (ss >> r.name >> r.valid >> r.lo.z >> r.lo.y >> r.lo.x >> r.hi.z >> r.hi.y >> r.hi.x && !r.name.empty())
            out.push_back(r);
    }
    return out;
}

namespace detail {
// Stitch ONE loaded tile's fragments locally: translate to a tile-local origin (so the Eulerian field is
// tile-sized, not volume-sized), run the patch graph + merge + band-Eulerian winding, and return each
// fragment's integer winding (per input Surface, in order) in the tile's own gauge. Shared by the z-slab
// and 3D tilers — both then ALIGN tiles into a global frame via fragments shared across tile overlaps.
inline std::vector<s32> stitch_tile_windings(std::vector<Surface>& sheets, segment::PatchGraphParams gp,
                                             FieldParams ef) {
    Vec3f blo{1e30f, 1e30f, 1e30f}, bhi{-1e30f, -1e30f, -1e30f};
    for (const Surface& s : sheets)
        for (usize i = 0; i < s.valid.size(); ++i)
            if (s.valid[i]) {
                const Vec3f c = s.coord[i];
                blo = Vec3f{std::min(blo.z, c.z), std::min(blo.y, c.y), std::min(blo.x, c.x)};
                bhi = Vec3f{std::max(bhi.z, c.z), std::max(bhi.y, c.y), std::max(bhi.x, c.x)};
            }
    const Vec3f org{std::floor(blo.z) - 2, std::floor(blo.y) - 2, std::floor(blo.x) - 2};
    for (Surface& s : sheets)
        for (usize i = 0; i < s.valid.size(); ++i)
            if (s.valid[i]) s.coord[i] = s.coord[i] - org;
    const Extent3 fe{static_cast<s64>(bhi.z - org.z) + 3, static_cast<s64>(bhi.y - org.y) + 3, static_cast<s64>(bhi.x - org.x) + 3};
    annotate::Umbilicus umb;  // dummy (make_patch keeps raw normals; SignDSU orients)
    umb.z = {0.0f, static_cast<f32>(fe.z)};
    umb.y = {static_cast<f32>(fe.y) * 0.5f, static_cast<f32>(fe.y) * 0.5f};
    umb.x = {static_cast<f32>(fe.x) * 0.5f, static_cast<f32>(fe.x) * 0.5f};
    segment::PatchGraph g = segment::build_patch_graph(sheets, umb, gp);
    segment::merge_same_sheet(g);
    const f32 spacing = g.spacing;
    if (ef.band <= 0) ef.band = std::max(2, static_cast<int>(std::lround(static_cast<double>(spacing) / std::max(1, ef.ds))) + 2);
    const WindingField wf = build_eulerian_winding_field(g.patches, fe, spacing, ef);
    assign_windings_from_field(g, wf);
    std::vector<s32> w(g.patches.size());
    for (usize i = 0; i < g.patches.size(); ++i) w[i] = g.patches[i].wrap;
    return w;
}
// median + variance of an integer vector (for picking the winding offset / flip in tile alignment).
inline std::pair<s32, f64> median_var(std::vector<s32> v) {
    std::sort(v.begin(), v.end());
    const s32 m = v[v.size() / 2];
    f64 var = 0;
    for (s32 x : v) var += static_cast<f64>(x - m) * static_cast<f64>(x - m);
    return {m, var / static_cast<f64>(v.size())};
}
}  // namespace detail

inline Expected<StitchStreamReport> stitch_streamed(const std::string& dir, StitchStreamParams sp) {
    namespace fs = std::filesystem;
    const std::vector<FragRec> manifest = read_manifest(dir);
    if (manifest.empty()) return err(Errc::not_found, "no manifest fragments in " + dir);
    f32 zmin = 1e30f, zmax = -1e30f;
    for (const FragRec& r : manifest) { zmin = std::min(zmin, r.lo.z); zmax = std::max(zmax, r.hi.z); }

    StitchStreamReport rep;
    std::unordered_map<std::string, s32> global;  // fragment name -> global winding
    bool have_prev = false;

    for (s64 z0 = static_cast<s64>(std::floor(zmin)); z0 < static_cast<s64>(std::ceil(zmax)); z0 += sp.slab) {
        const f32 lo = static_cast<f32>(z0 - sp.halo), hi = static_cast<f32>(z0 + sp.slab + sp.halo);
        std::vector<std::string> names;
        std::vector<Surface> sheets;
        for (const FragRec& r : manifest)
            if (r.hi.z >= lo && r.lo.z < hi) {  // bbox-z intersects this slab (+halo)
                auto s = io::read_fxsurf((fs::path(dir) / r.name).string());
                if (!s) return std::unexpected(s.error());
                names.push_back(r.name);
                sheets.push_back(std::move(*s));
            }
        if (sheets.empty()) continue;

        const std::vector<s32> tw = detail::stitch_tile_windings(sheets, sp.gp, sp.efield);
        std::vector<s32> local = tw;  // mutable (may be negated if this slab's winding axis is flipped)

        // align this slab's local windings to the previous slab via shared fragments.
        s32 offset = 0;
        if (have_prev) {
            std::vector<s32> deltas, sums;
            for (usize i = 0; i < names.size(); ++i) {
                auto it = global.find(names[i]);
                if (it == global.end()) continue;
                deltas.push_back(it->second - local[i]);
                sums.push_back(it->second + local[i]);
            }
            if (!deltas.empty()) {
                const auto [md, vd] = detail::median_var(deltas);
                const auto [ms, vs] = detail::median_var(sums);
                if (vs < vd) {  // this slab's winding axis is flipped vs the previous -> negate, use sums
                    for (s32& w : local) w = -w;
                    offset = ms;
                } else {
                    offset = md;
                }
            }
        }
        for (usize i = 0; i < names.size(); ++i) {
            const s32 gw = local[i] + offset;
            if (global.find(names[i]) == global.end()) global[names[i]] = gw;  // first slab to see it wins
        }
        have_prev = true;
        ++rep.slabs;
    }

    // gauge to 0-based and write windings.txt in manifest order.
    s32 gmin = std::numeric_limits<s32>::max();
    for (const auto& kv : global) gmin = std::min(gmin, kv.second);
    if (gmin == std::numeric_limits<s32>::max()) gmin = 0;
    std::ofstream w(fs::path(dir) / "windings.txt");
    if (!w) return err(Errc::io_error, "cannot write windings.txt in " + dir);
    rep.wrap_lo = std::numeric_limits<s32>::max();
    rep.wrap_hi = std::numeric_limits<s32>::min();
    for (const FragRec& r : manifest) {
        const auto it = global.find(r.name);
        const s32 gw = (it != global.end() ? it->second : gmin) - gmin;
        w << r.name << ' ' << gw << '\n';
        ++rep.fragments;
        rep.wrap_lo = std::min(rep.wrap_lo, gw);
        rep.wrap_hi = std::max(rep.wrap_hi, gw);
    }
    return rep;
}

// Full 3D-tiled OOC winding stitch: bounds RAM in ALL axes (z-slabs bound only z; a scroll whose y,x
// cross-section doesn't fit RAM needs this). PHASE 1 — stitch each 3D tile locally (load only the
// fragments intersecting tile+halo, run the patch graph + band-Eulerian winding, record per-fragment
// TILE-LOCAL windings; geometry freed per tile). PHASE 2 — winding consistency is now a 3D graph, not a
// 1D sweep: build the tile-adjacency graph (tiles that SHARE a fragment in their overlap) and BFS from a
// seed tile, giving each tile a global SIGN (orientation may flip per tile) + OFFSET from the shared
// fragments (lower-variance of prev±local). PHASE 3 — each fragment's global winding = sign·local +
// offset of the first tile that owns it. Only one tile's geometry is ever resident; the cross-tile align
// is integer-only (the fragment index, not the surfaces). Writes <dir>/windings.txt.
inline Expected<StitchStreamReport> stitch_streamed_3d(const std::string& dir, StitchTiledParams tp) {
    namespace fs = std::filesystem;
    const std::vector<FragRec> manifest = read_manifest(dir);
    if (manifest.empty()) return err(Errc::not_found, "no manifest fragments in " + dir);
    Vec3f glo{1e30f, 1e30f, 1e30f}, ghi{-1e30f, -1e30f, -1e30f};
    for (const FragRec& r : manifest) {
        glo = Vec3f{std::min(glo.z, r.lo.z), std::min(glo.y, r.lo.y), std::min(glo.x, r.lo.x)};
        ghi = Vec3f{std::max(ghi.z, r.hi.z), std::max(ghi.y, r.hi.y), std::max(ghi.x, r.hi.x)};
    }

    // PHASE 1 — per-tile local windings (only this tile's fragments resident at a time).
    struct TileRec {
        std::vector<s32> fidx;  // manifest index per fragment in this tile
        std::vector<s32> wind;  // its tile-local winding
    };
    std::vector<TileRec> tiles;
    for (s64 tz = static_cast<s64>(std::floor(glo.z)); tz < static_cast<s64>(std::ceil(ghi.z)); tz += tp.tile)
        for (s64 ty = static_cast<s64>(std::floor(glo.y)); ty < static_cast<s64>(std::ceil(ghi.y)); ty += tp.tile)
            for (s64 tx = static_cast<s64>(std::floor(glo.x)); tx < static_cast<s64>(std::ceil(ghi.x)); tx += tp.tile) {
                const f32 lz = static_cast<f32>(tz - tp.halo), hz = static_cast<f32>(tz + tp.tile + tp.halo);
                const f32 ly = static_cast<f32>(ty - tp.halo), hy = static_cast<f32>(ty + tp.tile + tp.halo);
                const f32 lx = static_cast<f32>(tx - tp.halo), hx = static_cast<f32>(tx + tp.tile + tp.halo);
                std::vector<Surface> sheets;
                std::vector<s32> fidx;
                for (s32 fi = 0; fi < static_cast<s32>(manifest.size()); ++fi) {
                    const FragRec& r = manifest[static_cast<usize>(fi)];
                    if (r.hi.z >= lz && r.lo.z < hz && r.hi.y >= ly && r.lo.y < hy && r.hi.x >= lx && r.lo.x < hx) {
                        auto s = io::read_fxsurf((fs::path(dir) / r.name).string());
                        if (!s) return std::unexpected(s.error());
                        sheets.push_back(std::move(*s));
                        fidx.push_back(fi);
                    }
                }
                if (sheets.empty()) continue;
                tiles.push_back({std::move(fidx), detail::stitch_tile_windings(sheets, tp.gp, tp.efield)});
            }
    const s32 nT = static_cast<s32>(tiles.size());

    // PHASE 2 — tile adjacency from shared fragments + BFS alignment (global sign + offset per tile).
    std::vector<std::vector<std::pair<s32, s32>>> frag_in(manifest.size());  // fidx -> [(tile, local winding)]
    for (s32 t = 0; t < nT; ++t)
        for (usize k = 0; k < tiles[static_cast<usize>(t)].fidx.size(); ++k)
            frag_in[static_cast<usize>(tiles[static_cast<usize>(t)].fidx[k])].push_back({t, tiles[static_cast<usize>(t)].wind[k]});
    std::unordered_map<s64, std::vector<std::pair<s32, s32>>> edges;  // key a*nT+b (a<b) -> [(wa, wb)]
    std::vector<std::vector<s32>> adj(static_cast<usize>(std::max(1, nT)));
    for (const auto& fl : frag_in)
        for (usize i = 0; i < fl.size(); ++i)
            for (usize j = i + 1; j < fl.size(); ++j) {
                s32 a = fl[i].first, b = fl[j].first, wa = fl[i].second, wb = fl[j].second;
                if (a == b) continue;
                if (a > b) { std::swap(a, b); std::swap(wa, wb); }
                edges[static_cast<s64>(a) * nT + b].push_back({wa, wb});
            }
    for (const auto& kv : edges) {
        const s32 a = static_cast<s32>(kv.first / nT), b = static_cast<s32>(kv.first % nT);
        adj[static_cast<usize>(a)].push_back(b);
        adj[static_cast<usize>(b)].push_back(a);
    }
    std::vector<s32> gsign(static_cast<usize>(std::max(1, nT)), 1), goff(static_cast<usize>(std::max(1, nT)), 0);
    std::vector<u8> done(static_cast<usize>(std::max(1, nT)), 0);
    StitchStreamReport rep;
    rep.slabs = nT;
    for (s32 seed = 0; seed < nT; ++seed) {
        if (done[static_cast<usize>(seed)]) continue;
        ++rep.components;
        done[static_cast<usize>(seed)] = 1;
        gsign[static_cast<usize>(seed)] = 1;
        goff[static_cast<usize>(seed)] = 0;
        std::vector<s32> q{seed};
        for (usize qi = 0; qi < q.size(); ++qi) {
            const s32 P = q[qi];
            for (s32 C : adj[static_cast<usize>(P)]) {
                if (done[static_cast<usize>(C)]) continue;
                s32 a = P, b = C;
                bool sw = false;
                if (a > b) { std::swap(a, b); sw = true; }
                const auto& pairs = edges[static_cast<s64>(a) * nT + b];
                std::vector<s32> dpos, dneg;
                for (const auto& pr : pairs) {
                    const s32 wP = sw ? pr.second : pr.first, wC = sw ? pr.first : pr.second;
                    const s32 gwP = gsign[static_cast<usize>(P)] * wP + goff[static_cast<usize>(P)];
                    dpos.push_back(gwP - wC);  // sign_C = +1
                    dneg.push_back(gwP + wC);  // sign_C = -1
                }
                if (dpos.empty()) continue;
                const auto [op, vp] = detail::median_var(dpos);
                const auto [on, vn] = detail::median_var(dneg);
                if (vn < vp) { gsign[static_cast<usize>(C)] = -1; goff[static_cast<usize>(C)] = on; }
                else { gsign[static_cast<usize>(C)] = 1; goff[static_cast<usize>(C)] = op; }
                done[static_cast<usize>(C)] = 1;
                q.push_back(C);
            }
        }
    }

    // PHASE 3 — per-fragment global winding (first owning tile wins); gauge 0-based; write.
    std::unordered_map<s32, s32> global;
    for (s32 t = 0; t < nT; ++t)
        for (usize k = 0; k < tiles[static_cast<usize>(t)].fidx.size(); ++k) {
            const s32 fi = tiles[static_cast<usize>(t)].fidx[k];
            if (global.find(fi) != global.end()) continue;
            global[fi] = gsign[static_cast<usize>(t)] * tiles[static_cast<usize>(t)].wind[k] + goff[static_cast<usize>(t)];
        }
    s32 gmin = std::numeric_limits<s32>::max();
    for (const auto& kv : global) gmin = std::min(gmin, kv.second);
    if (gmin == std::numeric_limits<s32>::max()) gmin = 0;
    std::ofstream w(fs::path(dir) / "windings.txt");
    if (!w) return err(Errc::io_error, "cannot write windings.txt in " + dir);
    rep.wrap_lo = std::numeric_limits<s32>::max();
    rep.wrap_hi = std::numeric_limits<s32>::min();
    for (s32 fi = 0; fi < static_cast<s32>(manifest.size()); ++fi) {
        const auto it = global.find(fi);
        const s32 gw = (it != global.end() ? it->second : gmin) - gmin;
        w << manifest[static_cast<usize>(fi)].name << ' ' << gw << '\n';
        ++rep.fragments;
        rep.wrap_lo = std::min(rep.wrap_lo, gw);
        rep.wrap_hi = std::max(rep.wrap_hi, gw);
    }
    return rep;
}

}  // namespace fenix::winding
