// flatten/flatten.hpp — the `flatten` stage: fitted .fxmodel -> per-wrap parametric
// surfaces (.fxsurf), the P6 unroll step. Each integer winding W is extracted in CLOSED
// FORM: in canonical space the wrap is the Archimedean spiral
//   r_ideal(theta) = dr * (W - winding_offset + theta/2pi)   (>= 0, else no wrap there)
//   r_canon = gap.forward(r_ideal);  q = {z, r sin(theta), r cos(theta)};  p = to_scroll(q)
// — no ray-marching, no dense winding volume. The output grids (u = theta, v = z) feed
// surf-bake / render-layers / view-surf directly and slim.hpp for distortion-minimizing UVs.
//   fenix flatten model=<fxmodel> out=<prefix> [wraps=lo:hi] [nu=1024] [zstep=4]
//                 [z=lo:hi] [rmax=1e9]
// wraps default: every W whose spiral has r_ideal >= 0 somewhere in [0,2pi) and
// r <= rmax, up to gap-table extent + 2. z defaults to the model umbilicus span.
#pragma once

#include "core/core.hpp"

#include "flatten/extract_wrap.hpp"
#include "io/surface.hpp"
#include "winding/model_io.hpp"

#include <charconv>
#include <cmath>
#include <numbers>
#include <string>
#include <string_view>

namespace fenix::flatten {

// One wrap of the fitted spiral as a parametric Surface: u = theta in [0,2pi), v = z.
// Cells with r_ideal < 0 (before the spiral starts) or r_canon > rmax are invalid.
inline Surface wrap_surface(const winding::SpiralModel& m, f32 wrap, s64 nu, f32 z_lo, f32 z_hi,
                            f32 zstep, f32 rmax) {
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    const s64 nv = std::max<s64>(2, static_cast<s64>((z_hi - z_lo) / zstep) + 1);
    Surface s(nu, nv);
    s.scale_u = two_pi / static_cast<f32>(nu);
    s.scale_v = zstep;
    for (s64 v = 0; v < nv; ++v) {
        const f32 z = z_lo + static_cast<f32>(v) * zstep;
        for (s64 u = 0; u < nu; ++u) {
            // half-bin centering keeps samples off the theta = ±pi branch cut, where the
            // Archimedean winding readout steps to the next wrap
            const f32 theta =
                two_pi * (static_cast<f32>(u) + 0.5f) / static_cast<f32>(nu) - std::numbers::pi_v<f32>;
            const f32 r_ideal = m.dr_per_winding * (wrap - m.winding_offset + theta / two_pi);
            if (r_ideal < 0) continue;
            const f32 r = m.gap.forward(r_ideal);
            if (r > rmax) continue;
            const Vec3f q{z, r * std::sin(theta), r * std::cos(theta)};
            s.set(u, v, m.to_scroll(q));
        }
    }
    return s;
}

inline Expected<int> run(std::span<const std::string_view> args, Context&) {
    std::string model_path, out_prefix;
    s64 nu = 1024;
    f64 zstep = 4, z_lo = -1, z_hi = -1, rmax = 1e9;
    s64 w_lo = 1, w_hi = -1;
    for (const auto a : args) {
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        auto range = [&](std::string_view key, auto& lo, auto& hi) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            const auto colon = t.find(':');
            if (colon == std::string_view::npos) return false;
            std::from_chars(t.data(), t.data() + colon, lo);
            std::from_chars(t.data() + colon + 1, t.data() + t.size(), hi);
            return true;
        };
        if (num("nu=", nu) || num("zstep=", zstep) || num("rmax=", rmax) || range("wraps=", w_lo, w_hi) ||
            range("z=", z_lo, z_hi))
            continue;
        if (a.starts_with("model=")) {
            model_path = std::string(a.substr(6));
            continue;
        }
        if (a.starts_with("out=")) {
            out_prefix = std::string(a.substr(4));
            continue;
        }
        return err(Errc::invalid_argument, "flatten: unknown arg '" + std::string(a) + "'");
    }
    if (model_path.empty() || out_prefix.empty())
        return err(Errc::invalid_argument,
                   "usage: flatten model=<fxmodel> out=<prefix> [wraps=lo:hi] [nu=1024] "
                   "[zstep=4] [z=lo:hi] [rmax=]");

    auto m = winding::read_fxmodel(model_path);
    if (!m) return std::unexpected(m.error());
    if (m->umbilicus.empty()) return err(Errc::invalid_argument, "flatten: model has no umbilicus");
    if (z_lo < 0) z_lo = m->umbilicus.z.front();
    if (z_hi < 0) z_hi = m->umbilicus.z.back();
    if (w_hi < 0) {
        // last wrap whose innermost radius still fits the gap table (+2 for the tail) / rmax
        const f32 r_lim = std::min(static_cast<f32>(rmax),
                                   m->gap.forward(m->dr_per_winding *
                                                  (static_cast<f32>(m->gap.logits.size()) + 2.0f)));
        w_hi = w_lo;
        while (m->gap.forward(m->dr_per_winding * (static_cast<f32>(w_hi + 1) - m->winding_offset)) <= r_lim &&
               w_hi - w_lo < 4096)
            ++w_hi;
    }
    log(LogLevel::info, "flatten: {} wraps [{},{}], nu {}, z [{:.0f},{:.0f}] step {:.1f}, dr {:.2f}",
        w_hi - w_lo + 1, w_lo, w_hi, nu, z_lo, z_hi, zstep, m->dr_per_winding);

    s64 written = 0;
    for (s64 w = w_lo; w <= w_hi; ++w) {
        const Surface s = wrap_surface(*m, static_cast<f32>(w), nu, static_cast<f32>(z_lo),
                                       static_cast<f32>(z_hi), static_cast<f32>(zstep),
                                       static_cast<f32>(rmax));
        if (s.valid_count() < 16) continue;
        const std::string path = out_prefix + "_w" + std::to_string(w) + ".fxsurf";
        if (auto r = io::write_fxsurf(path, s); !r) return std::unexpected(r.error());
        ++written;
        log(LogLevel::info, "flatten: wrap {} -> {} ({}x{}, {:.1f}% valid)", w, path, s.nu, s.nv,
            100.0 * static_cast<f64>(s.valid_count()) / static_cast<f64>(s.nu * s.nv));
    }
    if (written == 0) return err(Errc::invalid_argument, "flatten: no wraps produced any surface");
    log(LogLevel::info, "flatten: {} wrap surfaces written", written);
    return 0;
}

}  // namespace fenix::flatten

FENIX_REGISTER_STAGE(flatten, "unroll: fitted .fxmodel -> per-wrap .fxsurf surfaces",
                     ::fenix::flatten::run)
