// io/import_obj.hpp — `fenix import-obj`: VC segment OBJ -> .fxsurf uv grid (torch-free).
// The ink-hunt entry point (docs: ink pipeline, 2026-07-04): upstream segment meshes are
// uv-textured triangle OBJs in the coordinates of the volume they were traced on. This
// rasterizes the triangles in uv space onto a regular grid (barycentric 3D coords per
// covered cell) — the fxsurf every downstream tool speaks (render/bake, surf-qc,
// rasterize, view-surf). An optional affine maps old-scan coords into a NEW scan's frame
// (the 2.4 µm rescans; upstream ships no transforms for Paris3/0332 — we register and
// validate with surf-qc delta ourselves).
//   fenix import-obj <in.obj> <out.fxsurf> [grid=8]
//                    [affine=a00,a01,a02,t0,a10,a11,a12,t1,a20,a21,a22,t2]  (ZYX rows)
//                    [transform=<transform.json>] [pre_scale=1] [post_scale=1]
// grid: target world voxels per uv cell (fxsurf scale_u/scale_v).
// transform=: villa volume-cartographer transform.json ({"transformation_matrix": 3x4 or
// 4x4, XYZ row order, bottom row 0 0 0 1) — applied as p_new = post_scale * (M * (p_old *
// pre_scale)), exactly VC's transformSurfacePoints(scaleBefore, M, scaleAfter). The XYZ
// matrix is remapped into our ZYX rows at load.
#pragma once

#include "core/core.hpp"
#include "core/surface.hpp"
#include "geom/mesh.hpp"
#include "io/surface.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::io {

inline Expected<int> run_import_obj(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2)
        return err(Errc::invalid_argument,
                   "usage: import-obj <in.obj> <out.fxsurf> [grid=8] [affine=12 comma floats, ZYX rows]");
    f64 grid = 8, pre_scale = 1, post_scale = 1;
    std::array<f64, 12> A{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};  // identity | zero translation
    bool have_affine = false;
    std::string transform_path;
    for (usize i = 2; i < args.size(); ++i) {
        const auto a = args[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("grid=", grid) || num("pre_scale=", pre_scale) || num("post_scale=", post_scale)) continue;
        if (a.starts_with("transform=")) {
            transform_path = std::string(a.substr(10));
            continue;
        }
        if (a.starts_with("affine=")) {
            usize p = 7;
            for (int k = 0; k < 12 && p <= a.size(); ++k) {
                const auto [q, ec] = std::from_chars(a.data() + p, a.data() + a.size(), A[static_cast<usize>(k)]);
                if (ec != std::errc()) return err(Errc::invalid_argument, "import-obj: bad affine");
                p = static_cast<usize>(q - a.data()) + 1;
            }
            have_affine = true;
            continue;
        }
        return err(Errc::invalid_argument, "import-obj: unknown arg '" + std::string(a) + "'");
    }

    if (!transform_path.empty()) {
        // minimal fixed-format reader for VC transform.json: locate "transformation_matrix"
        // and pull the next 12 (3x4) or 16 (4x4) numbers; validate the homogeneous row
        std::ifstream tf(transform_path);
        if (!tf) return err(Errc::not_found, "import-obj: cannot open " + transform_path);
        std::string body((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
        const auto key = body.find("transformation_matrix");
        if (key == std::string::npos)
            return err(Errc::invalid_argument, "import-obj: no transformation_matrix in " + transform_path);
        std::vector<f64> nums;
        for (usize i = key; i < body.size() && nums.size() < 16;) {
            const char c = body[i];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
                f64 v = 0;
                const auto [q, ec] = std::from_chars(body.data() + i, body.data() + body.size(), v);
                if (ec == std::errc()) {
                    nums.push_back(v);
                    i = static_cast<usize>(q - body.data());
                    continue;
                }
            }
            ++i;
        }
        if (nums.size() < 12) return err(Errc::invalid_argument, "import-obj: short transformation_matrix");
        if (nums.size() >= 16 &&
            (std::abs(nums[12]) > 1e-9 || std::abs(nums[13]) > 1e-9 || std::abs(nums[14]) > 1e-9 ||
             std::abs(nums[15] - 1.0) > 1e-9))
            return err(Errc::invalid_argument, "import-obj: transform bottom row must be 0 0 0 1");
        // VC rows are XYZ: new_x = M00 x + M01 y + M02 z + M03 ... -> remap into ZYX rows
        const f64* M = nums.data();
        A = {M[10], M[9], M[8], M[11],   // z row (z,y,x cols | t)
             M[6],  M[5], M[4], M[7],    // y row
             M[2],  M[1], M[0], M[3]};   // x row
        have_affine = true;
    }

    auto m = geom::read_obj(std::string(args[0]));
    if (!m) return std::unexpected(m.error());
    if (m->uvs.size() != m->vertices.size())
        return err(Errc::invalid_argument,
                   "import-obj: OBJ has no per-vertex texcoords (" + std::to_string(m->uvs.size()) + " vt vs " +
                       std::to_string(m->vertices.size()) + " v) — not a VC segment mesh?");
    if (m->tris.empty()) return err(Errc::invalid_argument, "import-obj: no faces");

    auto xf = [&](Vec3f p) -> Vec3f {
        if (!have_affine && pre_scale == 1.0 && post_scale == 1.0) return p;
        const f64 z = static_cast<f64>(p.z) * pre_scale, y = static_cast<f64>(p.y) * pre_scale,
                  x = static_cast<f64>(p.x) * pre_scale;
        return Vec3f{static_cast<f32>((A[0] * z + A[1] * y + A[2] * x + A[3]) * post_scale),
                     static_cast<f32>((A[4] * z + A[5] * y + A[6] * x + A[7]) * post_scale),
                     static_cast<f32>((A[8] * z + A[9] * y + A[10] * x + A[11]) * post_scale)};
    };

    // uv range + world-per-uv estimate -> grid dims such that one cell ~ `grid` voxels.
    f32 u0 = 1e30f, u1 = -1e30f, v0 = 1e30f, v1 = -1e30f;
    for (const auto& t : m->uvs) {
        u0 = std::min(u0, t[0]);
        u1 = std::max(u1, t[0]);
        v0 = std::min(v0, t[1]);
        v1 = std::max(v1, t[1]);
    }
    if (!(u1 > u0) || !(v1 > v0)) return err(Errc::invalid_argument, "import-obj: degenerate uv range");
    f64 du_sum = 0, dv_sum = 0;
    s64 dn = 0;
    for (usize i = 0; i < m->tris.size(); i += std::max<usize>(1, m->tris.size() / 4096)) {
        const auto& t = m->tris[i];
        const Vec3f pa = xf(m->vertices[static_cast<usize>(t[0])]);
        const Vec3f pb = xf(m->vertices[static_cast<usize>(t[1])]);
        const Vec3f pc = xf(m->vertices[static_cast<usize>(t[2])]);
        const auto &ua = m->uvs[static_cast<usize>(t[0])], &ub = m->uvs[static_cast<usize>(t[1])],
                   &uc = m->uvs[static_cast<usize>(t[2])];
        const f64 eab = norm(pb - pa), eac = norm(pc - pa);
        const f64 uab = std::hypot(ub[0] - ua[0], ub[1] - ua[1]);
        const f64 uac = std::hypot(uc[0] - ua[0], uc[1] - ua[1]);
        if (uab > 1e-9 && uac > 1e-9) {
            du_sum += (eab / uab + eac / uac) / 2.0;  // world units per uv unit (isotropy assumed)
            dv_sum += (eab / uab + eac / uac) / 2.0;
            ++dn;
        }
    }
    if (dn == 0) return err(Errc::invalid_argument, "import-obj: cannot estimate uv scale");
    const f64 world_per_uv = du_sum / static_cast<f64>(dn);
    const s64 nu = std::clamp<s64>(static_cast<s64>((u1 - u0) * world_per_uv / grid) + 1, 2, 60000);
    const s64 nv = std::clamp<s64>(static_cast<s64>((v1 - v0) * world_per_uv / grid) + 1, 2, 60000);

    Surface s(nu, nv);
    s.scale_u = static_cast<f32>(grid);
    s.scale_v = static_cast<f32>(grid);
    // rasterize each triangle over the uv grid: barycentric 3D interp at covered cells
    auto gu = [&](f32 u) { return (u - u0) / (u1 - u0) * static_cast<f32>(nu - 1); };
    auto gv = [&](f32 v) { return (v - v0) / (v1 - v0) * static_cast<f32>(nv - 1); };
    for (const auto& t : m->tris) {
        const auto &ua = m->uvs[static_cast<usize>(t[0])], &ub = m->uvs[static_cast<usize>(t[1])],
                   &uc = m->uvs[static_cast<usize>(t[2])];
        const Vec3f pa = xf(m->vertices[static_cast<usize>(t[0])]);
        const Vec3f pb = xf(m->vertices[static_cast<usize>(t[1])]);
        const Vec3f pc = xf(m->vertices[static_cast<usize>(t[2])]);
        const f32 ax = gu(ua[0]), ay = gv(ua[1]);
        const f32 bx = gu(ub[0]), by = gv(ub[1]);
        const f32 cx = gu(uc[0]), cy = gv(uc[1]);
        const s64 x0 = std::max<s64>(0, static_cast<s64>(std::floor(std::min({ax, bx, cx}))));
        const s64 x1 = std::min<s64>(nu - 1, static_cast<s64>(std::ceil(std::max({ax, bx, cx}))));
        const s64 y0 = std::max<s64>(0, static_cast<s64>(std::floor(std::min({ay, by, cy}))));
        const s64 y1 = std::min<s64>(nv - 1, static_cast<s64>(std::ceil(std::max({ay, by, cy}))));
        const f32 den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
        if (std::abs(den) < 1e-12f) continue;
        for (s64 y = y0; y <= y1; ++y)
            for (s64 x = x0; x <= x1; ++x) {
                const f32 l0 = ((by - cy) * (static_cast<f32>(x) - cx) + (cx - bx) * (static_cast<f32>(y) - cy)) / den;
                const f32 l1 = ((cy - ay) * (static_cast<f32>(x) - cx) + (ax - cx) * (static_cast<f32>(y) - cy)) / den;
                const f32 l2 = 1.0f - l0 - l1;
                if (l0 < -1e-4f || l1 < -1e-4f || l2 < -1e-4f) continue;
                s.set(x, y, pa * l0 + pb * l1 + pc * l2);
            }
    }
    if (s.valid_count() < 16) return err(Errc::invalid_argument, "import-obj: rasterization produced no cells");
    s.src = std::string(args[0]);
    s.coordscale = 1.0f;  // OBJ imports register via transform.json, not a LOD lift
    s.imported_at = std::format("{:%F}", std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now()));
    if (auto w = write_fxsurf(std::string(args[1]), s); !w) return std::unexpected(w.error());
    log(LogLevel::info,
        "import-obj: {} -> {} ({}x{} grid @ {} vox/cell, {} valid cells, affine {})",
        args[0],
        args[1],
        nu,
        nv,
        grid,
        s.valid_count(),
        have_affine ? "yes" : "no");
    return 0;
}

}  // namespace fenix::io

namespace {
[[maybe_unused]] const int fenix_stage_import_obj = ::fenix::register_stage(::fenix::Stage{
    "import-obj", "VC segment OBJ -> .fxsurf uv grid (optional cross-scan affine)", ::fenix::io::run_import_obj});
}  // namespace
