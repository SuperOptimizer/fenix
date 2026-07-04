// annotate/umbilicus_fit.hpp — the CURVED umbilicus: estimate the scroll axis from
// segment meshes + versioned TOML IO + the `umbilicus` stage. At every z the meshes wrap
// AROUND the core and their across-sheet normals point radially at it, so per z-band the
// axis is the least-squares intersection of the cells' normal lines (a 2x2 solve in
// (y,x)), IRLS-reweighted (Tukey) against curvature/edge outliers, then smoothed over z.
// No annotation needed — the corpus measures its own axis. A straight axis over a 70k-z
// scroll is the dominant winding-gauge error; this replaces it (winding umb=<toml>).
//   fenix umbilicus surf=<fxsurf>... out=<umb.toml> [band=1024] [stride=4] [irls=3]
#pragma once

#include "annotate/umbilicus.hpp"
#include "core/config.hpp"
#include "core/core.hpp"
#include "io/surface.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::annotate {

inline constexpr s64 kUmbilicusVersion = 1;

namespace detail {

// per-cell across-sheet normal from the uv-grid tangents (corpus meshes rarely persist a
// normal channel); zero when a grid neighbour is missing
inline Vec3f grid_normal(const Surface& s, s64 u, s64 v) {
    const s64 u1 = std::min(u + 1, s.nu - 1), u0 = std::max<s64>(u - 1, 0);
    const s64 v1 = std::min(v + 1, s.nv - 1), v0 = std::max<s64>(v - 1, 0);
    if (!s.is_valid(u0, v) || !s.is_valid(u1, v) || !s.is_valid(u, v0) || !s.is_valid(u, v1))
        return {0, 0, 0};
    const Vec3f tu = s.at(u1, v) - s.at(u0, v);
    const Vec3f tv = s.at(u, v1) - s.at(u, v0);
    const Vec3f n = cross(tu, tv);
    const f32 l = norm(n);
    return l > 1e-6f ? n / l : Vec3f{0, 0, 0};
}

}  // namespace detail

struct UmbilicusFitParams {
    f32 band = 1024.0f;  // z bin height (voxels) per control point
    int stride = 4;      // uv subsample of each mesh
    int irls = 3;        // Tukey reweight rounds per band
    f32 tukey_c = 200.0f;// residual scale (voxels): lines missing the axis by >> this are dropped
};

// Least-squares axis from segment meshes. For each z-band: minimize the sum of squared
// point-to-line distances over all (cell, radial-normal-line) pairs — normal-projector
// normal equations in (y,x) — with `irls` Tukey reweights. Bands with too little support
// inherit their neighbour. Returns control points at band centres.
inline Umbilicus umbilicus_from_sheets(std::span<const Surface> sheets, UmbilicusFitParams p = {}) {
    struct Ray {
        f32 z, y, x, ny, nx;
    };
    std::vector<Ray> rays;
    f32 z_lo = 1e30f, z_hi = -1e30f;
    for (const Surface& s : sheets)
        for (s64 v = 0; v < s.nv; v += p.stride)
            for (s64 u = 0; u < s.nu; u += p.stride) {
                if (!s.is_valid(u, v)) continue;
                const Vec3f n = detail::grid_normal(s, u, v);
                const f32 nyx = std::sqrt(n.y * n.y + n.x * n.x);
                if (nyx < 0.5f) continue;  // normal too axial — its (y,x) line is unstable
                const Vec3f q = s.at(u, v);
                rays.push_back({q.z, q.y, q.x, n.y / nyx, n.x / nyx});
                z_lo = std::min(z_lo, q.z);
                z_hi = std::max(z_hi, q.z);
            }
    Umbilicus out;
    if (rays.empty() || z_hi <= z_lo) return out;
    const s64 nb = std::max<s64>(1, static_cast<s64>((z_hi - z_lo) / p.band) + 1);
    std::vector<std::vector<Ray>> bands(static_cast<usize>(nb));
    for (const Ray& r : rays)
        bands[static_cast<usize>(std::min<s64>(nb - 1, static_cast<s64>((r.z - z_lo) / p.band)))].push_back(r);

    std::vector<f32> cy(static_cast<usize>(nb), -1e30f), cx(static_cast<usize>(nb), -1e30f);
    for (s64 b = 0; b < nb; ++b) {
        const auto& br = bands[static_cast<usize>(b)];
        if (br.size() < 32) continue;
        // seed: centroid; then IRLS on the point-to-line normal equations. For a line
        // through q with unit direction n, the residual projector is P = I - n n^T:
        // sum_i w_i P_i c = sum_i w_i P_i q_i.
        f64 sy = 0, sx = 0;
        for (const Ray& r : br) {
            sy += static_cast<f64>(r.y);
            sx += static_cast<f64>(r.x);
        }
        f64 y = sy / static_cast<f64>(br.size()), x = sx / static_cast<f64>(br.size());
        for (int it = 0; it <= p.irls; ++it) {
            f64 a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
            for (const Ray& r : br) {
                const f64 p00 = 1.0 - static_cast<f64>(r.ny) * r.ny, p01 = -static_cast<f64>(r.ny) * r.nx,
                          p11 = 1.0 - static_cast<f64>(r.nx) * r.nx;
                f64 w = 1.0;
                if (it > 0) {
                    const f64 dy = y - r.y, dx = x - r.x;
                    const f64 ry = p00 * dy + p01 * dx, rx = p01 * dy + p11 * dx;
                    const f64 d = std::sqrt(ry * ry + rx * rx) / static_cast<f64>(p.tukey_c);
                    w = d < 1.0 ? (1.0 - d * d) * (1.0 - d * d) : 0.0;
                }
                a00 += w * p00;
                a01 += w * p01;
                a11 += w * p11;
                b0 += w * (p00 * r.y + p01 * r.x);
                b1 += w * (p01 * r.y + p11 * r.x);
            }
            const f64 det = a00 * a11 - a01 * a01;
            if (std::abs(det) < 1e-9) break;
            y = (a11 * b0 - a01 * b1) / det;
            x = (a00 * b1 - a01 * b0) / det;
        }
        cy[static_cast<usize>(b)] = static_cast<f32>(y);
        cx[static_cast<usize>(b)] = static_cast<f32>(x);
    }
    // fill unsupported bands from the nearest solved one, then 3-tap smooth
    s64 first = -1;
    for (s64 b = 0; b < nb; ++b)
        if (cy[static_cast<usize>(b)] > -1e29f) {
            if (first < 0) first = b;
            for (s64 k = (first == b ? 0 : b - 1); k >= 0 && cy[static_cast<usize>(k)] < -1e29f; --k) {
                cy[static_cast<usize>(k)] = cy[static_cast<usize>(b)];
                cx[static_cast<usize>(k)] = cx[static_cast<usize>(b)];
            }
        }
    if (first < 0) return out;
    for (s64 b = 1; b < nb; ++b)
        if (cy[static_cast<usize>(b)] < -1e29f) {
            cy[static_cast<usize>(b)] = cy[static_cast<usize>(b - 1)];
            cx[static_cast<usize>(b)] = cx[static_cast<usize>(b - 1)];
        }
    for (s64 b = 0; b < nb; ++b) {
        const usize i0 = static_cast<usize>(std::max<s64>(0, b - 1)), i1 = static_cast<usize>(b),
                    i2 = static_cast<usize>(std::min(nb - 1, b + 1));
        out.z.push_back(z_lo + (static_cast<f32>(b) + 0.5f) * p.band);
        out.y.push_back((cy[i0] + cy[i1] + cy[i2]) / 3.0f);
        out.x.push_back((cx[i0] + cx[i1] + cx[i2]) / 3.0f);
    }
    return out;
}

inline Expected<void> save_umbilicus(const Umbilicus& u, const std::string& path) {
    std::string out = std::format("version = {}\n\n[umbilicus]\n", kUmbilicusVersion);
    auto arr = [&](const char* key, const std::vector<f32>& v) {
        out += key;
        out += " = [";
        for (usize i = 0; i < v.size(); ++i) {
            if (i) out += ", ";
            out += std::format("{}", v[i]);
        }
        out += "]\n";
    };
    arr("z", u.z);
    arr("y", u.y);
    arr("x", u.x);
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return err(Errc::io_error, "cannot open " + tmp);
    f << out;
    f.close();
    if (!f) return err(Errc::io_error, "write failed: " + tmp);
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) return err(Errc::io_error, "rename failed: " + path);
    return {};
}

inline Expected<Umbilicus> load_umbilicus(const std::string& path) {
    auto c = Config::load(path);
    if (!c) return std::unexpected(c.error());
    const auto ver = c->get_int("version");
    if (!ver || *ver != kUmbilicusVersion)
        return err(Errc::unsupported, "umbilicus TOML version mismatch: " + path);
    Umbilicus u;
    auto arr = [&](const char* key, std::vector<f32>& v) {
        for (const std::string& s : c->get_array(std::string("umbilicus.") + key)) {
            f32 x = 0;
            std::from_chars(s.data(), s.data() + s.size(), x);
            v.push_back(x);
        }
    };
    arr("z", u.z);
    arr("y", u.y);
    arr("x", u.x);
    if (u.z.empty() || u.z.size() != u.y.size() || u.z.size() != u.x.size())
        return err(Errc::decode_error, "umbilicus TOML: bad/mismatched z/y/x arrays");
    return u;
}

inline Expected<int> run_umbilicus(std::span<const std::string_view> args, Context&) {
    UmbilicusFitParams p;
    std::string out_path;
    std::vector<std::string> surf_paths;
    for (const auto a : args) {
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            using V = std::remove_reference_t<decltype(v)>;
            if constexpr (std::is_floating_point_v<V>) {
                f64 d = 0;
                std::from_chars(t.data(), t.data() + t.size(), d);
                v = static_cast<V>(d);
            } else {
                std::from_chars(t.data(), t.data() + t.size(), v);
            }
            return true;
        };
        if (num("band=", p.band) || num("stride=", p.stride) || num("irls=", p.irls) ||
            num("tukey=", p.tukey_c))
            continue;
        if (a.starts_with("surf=")) {
            surf_paths.emplace_back(a.substr(5));
            continue;
        }
        if (a.starts_with("out=")) {
            out_path = std::string(a.substr(4));
            continue;
        }
        return err(Errc::invalid_argument, "umbilicus: unknown arg '" + std::string(a) + "'");
    }
    if (surf_paths.empty() || out_path.empty())
        return err(Errc::invalid_argument,
                   "usage: umbilicus surf=<fxsurf>... out=<umb.toml> [band=1024] [stride=4] "
                   "[irls=3] [tukey=200]");
    std::vector<Surface> sheets;
    for (const auto& sp : surf_paths) {
        auto s = io::read_fxsurf(sp);
        if (!s) return std::unexpected(s.error());
        sheets.push_back(std::move(*s));
    }
    const Umbilicus u = umbilicus_from_sheets(sheets, p);
    if (u.empty()) return err(Errc::invalid_argument, "umbilicus: no bands had enough support");
    f32 dy = 0, dx = 0;
    for (usize i = 1; i < u.z.size(); ++i) {
        dy = std::max(dy, std::abs(u.y[i] - u.y[0]));
        dx = std::max(dx, std::abs(u.x[i] - u.x[0]));
    }
    log(LogLevel::info, "umbilicus: {} control points, z [{:.0f},{:.0f}], drift y {:.0f} x {:.0f} vox",
        u.z.size(), u.z.front(), u.z.back(), dy, dx);
    if (auto w = save_umbilicus(u, out_path); !w) return std::unexpected(w.error());
    log(LogLevel::info, "umbilicus: -> {}", out_path);
    return 0;
}

}  // namespace fenix::annotate

FENIX_REGISTER_STAGE(umbilicus, "estimate the curved scroll axis from segment meshes (-> TOML)",
                     ::fenix::annotate::run_umbilicus)
