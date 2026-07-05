// winding/wrap_label.hpp — `wrap-label`: stamp every corpus-mesh cell with its ABSOLUTE
// wrap index under the fitted spiral model (the instance-label source for multiscale
// multi-instance surface training; docs/design/multiscale-instance-surface.md P1).
// Per mesh: unwrap the angular turn per component (corpus_bridge machinery), gauge the
// component against the model (g = median(winding_cont - turn)), wrap(cell) =
// round(g + turn). Cells where the model and the gauged mesh disagree by > conf_tol
// windings are masked (the model may be wrong near the umbilicus) — masked cells
// rasterize as ignore. Output: a SIDECAR '<mesh>.wrapcolor' (fxwcol1: u16 wrap per cell,
// 0xFFFF = masked) next to trust grids in the pairs file — no fxsurf format bump, and
// the mod-k coloring stays a FEED-time choice (re-k without re-labelling).
//   fenix wrap-label model=<fxmodel> surf=<fxsurf>... [outdir=<dir>] [conf_tol=0.35]
#pragma once

#include "core/core.hpp"
#include "io/surface.hpp"
#include "winding/corpus_bridge.hpp"
#include "winding/model_io.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::winding {

inline Expected<void> write_wrapcolor(const std::string& path, s64 nu, s64 nv,
                                      const std::vector<u16>& wrap) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return err(Errc::io_error, "wrap-label: cannot open " + tmp);
    std::fwrite("FXWCOL1\0", 1, 8, f);
    std::fwrite(&nu, sizeof nu, 1, f);
    std::fwrite(&nv, sizeof nv, 1, f);
    std::fwrite(wrap.data(), sizeof(u16), wrap.size(), f);
    if (std::fclose(f) != 0) return err(Errc::io_error, "wrap-label: write failed " + tmp);
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) return err(Errc::io_error, "wrap-label: rename failed " + path);
    return {};
}

inline Expected<int> run_wrap_label(std::span<const std::string_view> args, Context&) {
    std::string model_path, outdir;
    std::vector<std::string> surf_paths;
    f64 conf_tol = 0.35;
    for (const auto a : args) {
        if (a.starts_with("model=")) { model_path = std::string(a.substr(6)); continue; }
        if (a.starts_with("surf=")) { surf_paths.emplace_back(a.substr(5)); continue; }
        if (a.starts_with("outdir=")) { outdir = std::string(a.substr(7)); continue; }
        if (a.starts_with("conf_tol=")) {
            const auto t = a.substr(9);
            std::from_chars(t.data(), t.data() + t.size(), conf_tol);
            continue;
        }
        return err(Errc::invalid_argument, "wrap-label: unknown arg '" + std::string(a) + "'");
    }
    if (model_path.empty() || surf_paths.empty())
        return err(Errc::invalid_argument,
                   "usage: wrap-label model=<fxmodel> surf=<fxsurf>... [outdir=] [conf_tol=0.35]");
    auto m = read_fxmodel(model_path);
    if (!m) return std::unexpected(m.error());

    for (const auto& sp : surf_paths) {
        auto s = io::read_fxsurf(sp);
        if (!s) return std::unexpected(s.error());
        const usize N = s->coord.size();
        std::vector<u16> wrap(N, 0xFFFF);
        std::vector<u8> state(s->valid.begin(), s->valid.end());
        std::vector<f32> turn(N, 0.0f);
        s64 labeled = 0, masked = 0;
        s32 comps = 0;
        f32 w_lo = 1e30f, w_hi = -1e30f;
        for (usize seed = 0; seed < N; ++seed) {
            if (state[seed] != 1) continue;
            std::vector<u8> before = state;
            const s64 n = detail::unwrap_component(*s, m->umbilicus, seed, turn, state);
            if (n < 64) continue;
            ++comps;
            // gauge this component against the MODEL (not r/spacing): g aligns the
            // mesh's own turn to the model's continuous winding
            std::vector<f32> res;
            res.reserve(static_cast<usize>(n));
            for (usize j = 0; j < N; ++j)
                if (state[j] == 2 && before[j] == 1) {
                    const f32 w = m->winding_cont(s->coord[j]);
                    if (detail::finite_fm(w)) res.push_back(w - turn[j]);
                }
            if (res.size() < 8) continue;
            const f32 g = detail::median_of(res);
            for (usize j = 0; j < N; ++j) {
                if (!(state[j] == 2 && before[j] == 1)) continue;
                const f32 cont = g + turn[j];
                const f32 w = m->winding_cont(s->coord[j]);
                const f32 rw = std::round(cont);
                if (!detail::finite_fm(w) || std::abs(cont - w) > static_cast<f32>(conf_tol) || rw < 0 ||
                    rw > 60000.0f) {
                    ++masked;
                    continue;
                }
                wrap[j] = static_cast<u16>(rw);
                w_lo = std::min(w_lo, rw);
                w_hi = std::max(w_hi, rw);
                ++labeled;
            }
        }
        std::string base = sp;
        if (const auto sl = base.find_last_of('/'); !outdir.empty() && sl != std::string::npos)
            base = outdir + base.substr(sl);
        const std::string out = base + ".wrapcolor";
        if (auto w = write_wrapcolor(out, s->nu, s->nv, wrap); !w) return std::unexpected(w.error());
        log(LogLevel::info,
            "wrap-label: {} — {} components, wraps [{:.0f},{:.0f}], {} labeled, {} masked -> {}", sp,
            comps, w_lo, w_hi, labeled, masked, out);
    }
    return 0;
}

}  // namespace fenix::winding

namespace {
[[maybe_unused]] const int fenix_stage_wrap_label = ::fenix::register_stage(::fenix::Stage{
    "wrap-label", "stamp mesh cells with fitted-model wrap indices (instance-label sidecars)",
    ::fenix::winding::run_wrap_label});
}  // namespace
