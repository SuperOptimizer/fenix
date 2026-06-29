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
    s64 slabs = 0;
    s32 wrap_lo = 0, wrap_hi = 0;
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

inline Expected<StitchStreamReport> stitch_streamed(const std::string& dir, StitchStreamParams sp) {
    namespace fs = std::filesystem;
    const std::vector<FragRec> manifest = read_manifest(dir);
    if (manifest.empty()) return err(Errc::not_found, "no manifest fragments in " + dir);
    f32 zmin = 1e30f, zmax = -1e30f;
    for (const FragRec& r : manifest) { zmin = std::min(zmin, r.lo.z); zmax = std::max(zmax, r.hi.z); }

    StitchStreamReport rep;
    std::unordered_map<std::string, s32> global;  // fragment name -> global winding
    bool have_prev = false;
    auto median_var = [](std::vector<s32> v) {
        std::sort(v.begin(), v.end());
        const s32 m = v[v.size() / 2];
        f64 var = 0;
        for (s32 x : v) var += static_cast<f64>(x - m) * static_cast<f64>(x - m);
        return std::pair<s32, f64>{m, var / static_cast<f64>(v.size())};
    };

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

        // translate the slab's patches to a local origin so the Eulerian field is slab-sized (bounded).
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
        annotate::Umbilicus umb;  // dummy (make_patch keeps raw normals; SignDSU orients) — slab centre
        umb.z = {0.0f, static_cast<f32>(fe.z)};
        umb.y = {static_cast<f32>(fe.y) * 0.5f, static_cast<f32>(fe.y) * 0.5f};
        umb.x = {static_cast<f32>(fe.x) * 0.5f, static_cast<f32>(fe.x) * 0.5f};

        segment::PatchGraph g = segment::build_patch_graph(sheets, umb, sp.gp);
        segment::merge_same_sheet(g);
        const f32 spacing = g.spacing;
        FieldParams ef = sp.efield;
        if (ef.band <= 0) ef.band = std::max(2, static_cast<int>(std::lround(static_cast<double>(spacing) / std::max(1, ef.ds))) + 2);
        const WindingField wf = build_eulerian_winding_field(g.patches, fe, spacing, ef);
        assign_windings_from_field(g, wf);

        // align this slab's local windings to the previous slab via shared fragments.
        s32 offset = 0;
        if (have_prev) {
            std::vector<s32> deltas, sums;
            for (usize i = 0; i < names.size(); ++i) {
                auto it = global.find(names[i]);
                if (it == global.end()) continue;
                deltas.push_back(it->second - g.patches[i].wrap);
                sums.push_back(it->second + g.patches[i].wrap);
            }
            if (!deltas.empty()) {
                const auto [md, vd] = median_var(deltas);
                const auto [ms, vs] = median_var(sums);
                if (vs < vd) {  // this slab's winding axis is flipped vs the previous -> negate, use sums
                    for (segment::Patch& pp : g.patches) pp.wrap = -pp.wrap;
                    offset = ms;
                } else {
                    offset = md;
                }
            }
        }
        for (usize i = 0; i < names.size(); ++i) {
            const s32 gw = g.patches[i].wrap + offset;
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

}  // namespace fenix::winding
