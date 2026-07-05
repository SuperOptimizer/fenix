// segment/trace_surface.hpp — turn a surface-probability volume into a renderable sheet mesh,
// with the ink prediction composited onto it. Marching-cubes the surface field at an isolevel,
// optionally splits sheets into connected-component instances, samples the ink (and/or CT)
// volume onto each vertex, and writes a colored PLY/OBJ — the "trace the surface predictions
// and render them with ink overlaid" view. A first, viewable rung of the unroll ladder (the
// full per-sheet flatten/unroll fit lives in winding/). Registers `trace-surface`.
#pragma once

#include "core/core.hpp"
#include "core/eig.hpp"
#include "core/sampling.hpp"
#include "core/surface.hpp"
#include "codec/archive.hpp"
#include "geom/connected_components.hpp"
#include "geom/marching.hpp"
#include "geom/mesh.hpp"
#include "io/jpeg.hpp"

#include <array>
#include <cmath>
#include <string>

namespace fenix::segment {

namespace detail {
inline Expected<Volume<f32>> load_vol(const std::string& p) {
    if (!(p.size() > 6 && p.substr(p.size() - 6) == ".fxvol"))
        return err(Errc::unsupported, "expected a .fxvol volume, got " + p);
    auto a = codec::VolumeArchive::open(p);
    if (!a) return std::unexpected(a.error());
    return a->read_volume();
}
// Stride-downsample a volume by `s` (nearest); keeps marching-cubes mesh sizes tractable.
inline Volume<f32> downsample(VolumeView<const f32> v, int s) {
    const Extent3 d = v.dims();
    if (s <= 1) {
        Volume<f32> o(d);
        std::copy(v.data(), v.data() + d.count(), o.data());
        return o;
    }
    Extent3 nd{d.z / s, d.y / s, d.x / s};
    Volume<f32> o(nd);
    auto ov = o.view();
    for (s64 z = 0; z < nd.z; ++z)
        for (s64 y = 0; y < nd.y; ++y)
            for (s64 x = 0; x < nd.x; ++x) ov(z, y, x) = v(z * s, y * s, x * s);
    return o;
}
// id -> distinct RGB via golden-ratio hue (instance coloring).
inline std::array<u8, 3> palette(s32 id) {
    const double h = std::fmod(id * 0.6180339887, 1.0) * 6.0;
    const double f = h - std::floor(h);
    const double q = 1.0 - f, t = f;
    double r = 0, g = 0, b = 0;
    switch (static_cast<int>(h)) {
        case 0: r = 1; g = t; break;
        case 1: r = q; g = 1; break;
        case 2: g = 1; b = t; break;
        case 3: g = q; b = 1; break;
        case 4: r = t; b = 1; break;
        default: r = 1; b = q; break;
    }
    auto c = [](double v) { return static_cast<u8>(60 + 195 * v); };  // keep bright
    return {c(r), c(g), c(b)};
}
}  // namespace detail

// `fenix trace-surface <surface.fxvol> <out.ply> [ink=<vol>] [ct=<vol>] [iso=0.5]
//                       [step=1] [mode=ink|sheet|ct] [ctmax=255]`
inline Expected<int> trace_surface(std::span<const std::string_view> args, Context&) {
    auto opt = [&](std::string_view k, std::string d) {
        for (auto a : args) { auto e = a.find('='); if (e != std::string_view::npos && a.substr(0, e) == k) return std::string(a.substr(e + 1)); }
        return d;
    };
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix trace-surface <surface.fxvol> <out.ply> "
                             "[ink=<vol>] [ct=<vol>] [iso=0.5] [step=1] [mode=ink|sheet|ct] [ctmax=255]");
        return err(Errc::invalid_argument, "missing args");
    }
    const std::string surfpath(args[0]), outpath(args[1]);
    const f32 iso = std::stof(opt("iso", "0.5"));
    const int step = std::max(1, std::stoi(opt("step", "1")));
    const std::string mode = opt("mode", "ink");
    const f32 ctmax = std::stof(opt("ctmax", "255"));

    auto surf = detail::load_vol(surfpath);
    if (!surf) return std::unexpected(surf.error());
    Volume<f32> sv = detail::downsample(surf->view(), step);
    const Extent3 d = sv.dims();
    log(LogLevel::info, "trace-surface: field {}x{}x{} (step {}), iso {}", d.z, d.y, d.x, step, iso);

    geom::Mesh mesh = geom::marching_tetrahedra(sv.view(), iso);
    log(LogLevel::info, "trace-surface: marching-cubes -> {} verts, {} tris",
        mesh.vertex_count(), mesh.tri_count());
    if (mesh.vertices.empty()) return err(Errc::internal, "empty isosurface (check iso level)");

    // Optional ink / CT volumes (downsampled to match), for compositing onto vertices.
    Volume<f32> inkv, ctv;
    const std::string inkpath = opt("ink", ""), ctpath = opt("ct", "");
    if (!inkpath.empty()) { auto v = detail::load_vol(inkpath); if (!v) return std::unexpected(v.error()); inkv = detail::downsample(v->view(), step); }
    if (!ctpath.empty()) { auto v = detail::load_vol(ctpath); if (!v) return std::unexpected(v.error()); ctv = detail::downsample(v->view(), step); }

    // Sheet-instance labels (connected components of the thresholded surface) for mode=sheet.
    geom::CcResult cc;
    if (mode == "sheet") {
        Volume<u8> mask(d);
        auto mv = mask.view();
        for (s64 z = 0; z < d.z; ++z) for (s64 y = 0; y < d.y; ++y) for (s64 x = 0; x < d.x; ++x)
            mv(z, y, x) = sv.view()(z, y, x) > iso ? 1 : 0;
        cc = geom::connected_components(mask.view(), geom::Conn::TwentySix);
        log(LogLevel::info, "trace-surface: {} sheet components", cc.count);
    }

    mesh.colors.resize(mesh.vertices.size());
    const bool have_ink = inkv.dims().count() > 0, have_ct = ctv.dims().count() > 0;
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const Vec3f p = mesh.vertices[i];  // (z,y,x) voxel coords in the (downsampled) field
        std::array<u8, 3> col{180, 180, 180};
        if (mode == "sheet") {
            const s32 lbl = cc.labels.view().at_clamped(static_cast<s64>(p.z + 0.5f), static_cast<s64>(p.y + 0.5f), static_cast<s64>(p.x + 0.5f));
            col = lbl > 0 ? detail::palette(lbl) : std::array<u8, 3>{90, 90, 90};
        } else if (mode == "ct" && have_ct) {
            const u8 g = static_cast<u8>(std::clamp(sample_trilinear(ctv.view(), p) / ctmax, 0.0f, 1.0f) * 255.0f);
            col = {g, g, g};
        } else {  // ink composite: gray base (CT if given) tinted green by ink probability
            f32 base = have_ct ? std::clamp(sample_trilinear(ctv.view(), p) / ctmax, 0.0f, 1.0f) : 0.55f;
            const u8 g8 = static_cast<u8>(base * 255.0f);
            f32 ink = have_ink ? std::clamp(sample_trilinear(inkv.view(), p), 0.0f, 1.0f) : 0.0f;
            col = {static_cast<u8>(g8 * (1 - ink)), static_cast<u8>(g8 + (255 - g8) * ink), static_cast<u8>(g8 * (1 - ink))};
        }
        // scale vertices back to full-res world coords
        mesh.vertices[i] = Vec3f{p.z * step, p.y * step, p.x * step};
        mesh.colors[i] = col;
    }

    if (auto w = geom::write_ply(outpath, mesh); !w) return std::unexpected(w.error());
    log(LogLevel::info, "trace-surface: wrote {} ({} verts, {} tris, mode={})", outpath,
        mesh.vertex_count(), mesh.tri_count(), mode);
    return 0;
}

// ---- sheet-face tracer/renderer (vc3d-style: a quad Surface grid -> flat texture) ----------
namespace detail {
inline Vec3f field_grad(VolumeView<const f32> s, Vec3f p, f32 h = 1.0f) {
    return Vec3f{(sample_trilinear(s, {p.z + h, p.y, p.x}) - sample_trilinear(s, {p.z - h, p.y, p.x})) * 0.5f,
                 (sample_trilinear(s, {p.z, p.y + h, p.x}) - sample_trilinear(s, {p.z, p.y - h, p.x})) * 0.5f,
                 (sample_trilinear(s, {p.z, p.y, p.x + h}) - sample_trilinear(s, {p.z, p.y, p.x - h})) * 0.5f};
}
// Gradient ascent onto the surface-probability ridge (snap a point onto the sheet) — used only
// for the initial seed (it moves laterally, so it must NOT be used while walking).
inline Vec3f ascend_to_ridge(VolumeView<const f32> s, Vec3f p, int steps = 16, f32 lr = 0.6f) {
    for (int i = 0; i < steps; ++i) {
        Vec3f g = field_grad(s, p);
        f32 n = norm(g);
        if (n < 1e-5f) break;
        p = p + g * (lr / n);
    }
    return p;
}
// Snap a point onto the sheet by a 1-D line search ALONG the across-sheet normal only (keeps the
// in-plane position fixed, so it corrects the across-sheet offset without undoing the walk).
inline Vec3f snap_along_normal(VolumeView<const f32> s, Vec3f p, Vec3f n, f32 range = 4.0f, f32 step = 0.25f) {
    f32 best = sample_trilinear(s, p), bt = 0;
    for (f32 t = -range; t <= range; t += step) {
        f32 v = sample_trilinear(s, p + n * t);
        if (v > best) { best = v; bt = t; }
    }
    return p + n * bt;
}
// Across-sheet normal = Hessian eigenvector of most-negative curvature (sharpest ridge).
inline Vec3f sheet_normal(VolumeView<const f32> s, Vec3f p, f32 h = 1.0f) {
    auto S = [&](f32 dz, f32 dy, f32 dx) { return sample_trilinear(s, {p.z + dz, p.y + dy, p.x + dx}); };
    const f32 c = S(0, 0, 0);
    const f32 hzz = S(h, 0, 0) - 2 * c + S(-h, 0, 0), hyy = S(0, h, 0) - 2 * c + S(0, -h, 0), hxx = S(0, 0, h) - 2 * c + S(0, 0, -h);
    const f32 hzy = (S(h, h, 0) - S(h, -h, 0) - S(-h, h, 0) + S(-h, -h, 0)) * 0.25f;
    const f32 hzx = (S(h, 0, h) - S(h, 0, -h) - S(-h, 0, h) + S(-h, 0, -h)) * 0.25f;
    const f32 hyx = (S(0, h, h) - S(0, h, -h) - S(0, -h, h) + S(0, -h, -h)) * 0.25f;
    auto e = sym_eig3<f32>(hzz, hyy, hxx, hzy, hzx, hyx);
    return normalized(e.vectors[2]);  // most-negative eigenvalue (descending order)
}
inline Vec3f reorthonormal(Vec3f dir, Vec3f n) {
    Vec3f d = dir - n * dot(dir, n);
    f32 m = norm(d);
    return m > 1e-6f ? d / m : dir;
}
}  // namespace detail

// `fenix render-sheet <surface.fxvol> <out.jpg> [ct=<vol>] [ink=<vol>] [seed=z,y,x]
//   [nu=800] [nv=800] [spacing=1.0] [ctmax=255] [alpha=0.8]`
// Trace ONE sheet from `seed` as a u×v Surface (walk along the sheet, re-projecting each grid
// point onto the surface-prob ridge), then render its face = CT sampled on the grid, ink in green.
inline Expected<int> render_sheet(std::span<const std::string_view> args, Context&) {
    auto opt = [&](std::string_view k, std::string d) {
        for (auto a : args) { auto e = a.find('='); if (e != std::string_view::npos && a.substr(0, e) == k) return std::string(a.substr(e + 1)); }
        return d;
    };
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix render-sheet <surface.fxvol> <out.jpg> [ct=<vol>] "
                             "[ink=<vol>] [seed=z,y,x] [nu=800] [nv=800] [spacing=1.0] [ctmax=255] [alpha=0.8]");
        return err(Errc::invalid_argument, "missing args");
    }
    const std::string outpath(args[1]);
    auto surf = detail::load_vol(std::string(args[0]));
    if (!surf) return std::unexpected(surf.error());
    const auto sv = surf->view();
    const Extent3 d = sv.dims();
    const s64 nu = std::stoll(opt("nu", "800")), nv = std::stoll(opt("nv", "800"));
    const f32 spacing = std::stof(opt("spacing", "1.0")), ctmax = std::stof(opt("ctmax", "255")), alpha = std::stof(opt("alpha", "0.8"));

    // Seed: parse "z,y,x" or auto-pick the strongest surface voxel in the central region.
    Vec3f seed;
    const std::string ss = opt("seed", "");
    if (!ss.empty()) {
        f32 z, y, x; std::sscanf(ss.c_str(), "%f,%f,%f", &z, &y, &x); seed = {z, y, x};
    } else {
        f32 best = -1; seed = {d.z / 2.0f, d.y / 2.0f, d.x / 2.0f};
        for (s64 z = d.z / 4; z < 3 * d.z / 4; z += 4) for (s64 y = d.y / 4; y < 3 * d.y / 4; y += 4) for (s64 x = d.x / 4; x < 3 * d.x / 4; x += 4)
            if (sv(z, y, x) > best) { best = sv(z, y, x); seed = {static_cast<f32>(z), static_cast<f32>(y), static_cast<f32>(x)}; }
    }
    seed = detail::ascend_to_ridge(sv, seed);
    log(LogLevel::info, "render-sheet: seed ({:.0f},{:.0f},{:.0f}) prob {:.3f}, grid {}x{}",
        seed.z, seed.y, seed.x, sample_trilinear(sv, seed), nu, nv);

    // Initial in-plane tangents from the seed's across-sheet normal.
    Vec3f n0 = detail::sheet_normal(sv, seed);
    Vec3f ref = (std::abs(n0.x) < 0.9f) ? Vec3f{0, 0, 1} : Vec3f{1, 0, 0};
    Vec3f tu0 = normalized(cross(n0, ref)), tv0 = normalized(cross(n0, tu0));

    Surface surface(nu, nv);
    const s64 cu = nu / 2, cv = nv / 2;
    // Center column along v: walk +/- from seed, re-projecting onto the ridge each step.
    surface.set(cu, cv, seed);
    // build center column
    auto stepwalk = [&](Vec3f p, Vec3f dir, auto setfn) {
        Vec3f dcur = dir;
        for (;;) {
            Vec3f pred = p + dcur * spacing;
            if (pred.z < 1 || pred.z >= d.z - 1 || pred.y < 1 || pred.y >= d.y - 1 || pred.x < 1 || pred.x >= d.x - 1) break;
            Vec3f n = detail::sheet_normal(sv, pred);
            Vec3f s2 = detail::snap_along_normal(sv, pred, n);  // across-sheet snap only
            dcur = detail::reorthonormal(dcur, n);              // keep walking in the tangent plane
            if (!setfn(s2)) break;
            p = s2;
        }
    };
    {  // center column along v
        s64 v = cv;
        stepwalk(seed, tv0, [&](Vec3f s2) { if (++v >= nv) return false; surface.set(cu, v, s2); return true; });
        v = cv;
        stepwalk(seed, tv0 * -1.0f, [&](Vec3f s2) { if (--v < 0) return false; surface.set(cu, v, s2); return true; });
    }
    for (s64 v = 0; v < nv; ++v) {  // each row: walk +/- u from the center-column point
        if (!surface.is_valid(cu, v)) continue;
        Vec3f center = surface.at(cu, v);
        Vec3f du = detail::reorthonormal(tu0, detail::sheet_normal(sv, center));
        s64 u = cu;
        stepwalk(center, du, [&](Vec3f s2) { if (++u >= nu) return false; surface.set(u, v, s2); return true; });
        u = cu;
        stepwalk(center, du * -1.0f, [&](Vec3f s2) { if (--u < 0) return false; surface.set(u, v, s2); return true; });
    }
    log(LogLevel::info, "render-sheet: traced {} / {} grid points valid", surface.valid_count(), nu * nv);
    {
        Vec3f lo{1e9f, 1e9f, 1e9f}, hi{-1e9f, -1e9f, -1e9f};
        for (s64 i = 0; i < nu * nv; ++i) if (surface.valid[static_cast<usize>(i)]) {
            const Vec3f& p = surface.coord[static_cast<usize>(i)];
            lo = {std::min(lo.z, p.z), std::min(lo.y, p.y), std::min(lo.x, p.x)};
            hi = {std::max(hi.z, p.z), std::max(hi.y, p.y), std::max(hi.x, p.x)};
        }
        log(LogLevel::info, "render-sheet: surface bbox z[{:.0f},{:.0f}] y[{:.0f},{:.0f}] x[{:.0f},{:.0f}]",
            lo.z, hi.z, lo.y, hi.y, lo.x, hi.x);
    }

    // Render the face: CT (gray) on the grid, ink (green) composited.
    Volume<f32> ctv, inkv;
    const std::string ctp = opt("ct", ""), inkp = opt("ink", "");
    if (!ctp.empty()) { auto v = detail::load_vol(ctp); if (!v) return std::unexpected(v.error()); ctv = std::move(*v); }
    if (!inkp.empty()) { auto v = detail::load_vol(inkp); if (!v) return std::unexpected(v.error()); inkv = std::move(*v); }
    const bool have_ct = ctv.dims().count() > 0, have_ink = inkv.dims().count() > 0;

    io::Image im;
    im.w = static_cast<int>(nu); im.h = static_cast<int>(nv); im.comps = 3;
    im.px.assign(static_cast<usize>(nu) * static_cast<usize>(nv) * 3, 0);
    for (s64 v = 0; v < nv; ++v)
        for (s64 u = 0; u < nu; ++u) {
            if (!surface.is_valid(u, v)) continue;
            const Vec3f p = surface.at(u, v);
            f32 base = have_ct ? std::clamp(sample_trilinear(ctv.view(), p) / ctmax, 0.0f, 1.0f) : std::clamp(sample_trilinear(sv, p), 0.0f, 1.0f);
            const f32 g = base * 255.0f;
            f32 ink = have_ink ? std::clamp(sample_trilinear(inkv.view(), p), 0.0f, 1.0f) * alpha : 0.0f;
            im.at(static_cast<int>(v), static_cast<int>(u), 0) = static_cast<u8>(g * (1 - ink));
            im.at(static_cast<int>(v), static_cast<int>(u), 1) = static_cast<u8>(g + (255 - g) * ink);
            im.at(static_cast<int>(v), static_cast<int>(u), 2) = static_cast<u8>(g * (1 - ink));
        }
    if (auto w = io::write_jpeg(outpath, im, 92); !w) return std::unexpected(w.error());
    log(LogLevel::info, "render-sheet: wrote {} ({}x{} sheet face{})", outpath, nu, nv, have_ink ? " +ink" : "");
    return 0;
}

}  // namespace fenix::segment

namespace {
[[maybe_unused]] const int fenix_stage_trace_surface = ::fenix::register_stage(
    ::fenix::Stage{"trace-surface", "marching-cubes a surface-prob volume into an ink-composited mesh",
                   ::fenix::segment::trace_surface});
[[maybe_unused]] const int fenix_stage_render_sheet = ::fenix::register_stage(
    ::fenix::Stage{"render-sheet", "trace one sheet (quad Surface) and render its flattened face (CT + ink)",
                   ::fenix::segment::render_sheet});
}  // namespace
