// preprocess/dering.hpp — DETECT-then-subtract residual ring removal (fysics lineage).
//
// Residual ring artifacts (detector-defect stripes surviving the sinogram dering) are concentric
// circles centered on the ROTATION AXIS. The confounder is the papyrus winding, which is locally
// concentric too — so a blind radial high-pass would eat real sheet structure. Instead we DETECT
// rings by an angular-sector sign-consistency vote and subtract ONLY the detected component:
//   - accumulate per (z-slab, sector, radius-bin) intensity sums (pass 1)
//   - per slab+sector: angular-mean radial profile, box high-pass
//   - a radius is a RING only if every valid sector agrees in sign (a true ring sits at the SAME
//     radius in all sectors; a spiral wrap drifts in radius with angle and fails the vote)
//   - ring estimate = median over sectors, clamped; everything else is 0
//   - subtract ring[slab][rbin(y,x)] per voxel (pass 2), a LOCAL op
// Center: BM18 recon places the axis at the slice center, so cy/cx < 0 default to (Y-1)/2,(X-1)/2.
// The empirical gates (hp_win≈15, min_amp 0.5, max_amp 6.0 in value units) are kept verbatim from
// fysics (tuned on 90+ cubes / 18 volumes — see preprocess/CLAUDE.md).
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "io/nrrd.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <numbers>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::preprocess {
namespace detail {

// std::from_chars-based numeric option parse (no std::sto*, which throws under -fno-exceptions).
// Requires the whole token to be consumed (rejects "1.5x"-style trailing garbage).
template <class T>
[[nodiscard]] inline Expected<T> parse_opt(std::string_view key, std::string_view tok) {
    T v{};
    const auto r = std::from_chars(tok.data(), tok.data() + tok.size(), v);
    if (r.ec != std::errc{} || r.ptr != tok.data() + tok.size())
        return err(Errc::invalid_argument, "bad value for " + std::string(key) + ": '" + std::string(tok) + "'");
    return v;
}

}  // namespace detail

struct DeringParams {
    f64 cy = -1.0, cx = -1.0;  // ring center (volume coords); <0 => slice center (Y-1)/2,(X-1)/2
    s32 slab_z = 512;          // z-slab size (rings vary slowly along z)
    s32 ns = 8;                // angular sectors for the consistency vote
    s32 hp_win = 15;           // radial high-pass window (voxels; ~ ring width)
    u32 min_cnt = 16;          // min samples for a (sector,radius) bin to vote
    f64 min_amp = 0.5;         // detection floor (value units)
    f64 max_amp = 6.0;         // amplitude clamp (value units)
    s32 ss = 1;                // spatial subsample stride (y,x) for accumulation
    s32 vote_slack = 2;        // sectors allowed missing: need >= ns-vote_slack valid (higher = looser)
    s32 vote_dissent = 0;      // sectors allowed to disagree in sign (higher = looser; 0 = unanimous)
};

// Intrinsic ring-removal metric (no ground truth needed): how much of the signal is the
// angularly-consistent radial (ring) component. `ring_frac` = ring energy / total signal variance
// energy — run before vs after dering to get "% ring energy removed" (fysics quotes ~93% on PHerc0139).
struct DeringStats {
    long n_rings = 0;     // detected ring radii (summed over slabs)
    f64 ring_energy = 0;  // Σ ring[r]^2 · n_voxels(r)  (the energy subtracted)
    f64 signal_var = 0;   // Σ (v-mean)^2  over the accumulated voxels
    f64 ring_frac = 0;    // ring_energy / signal_var  (fraction of the signal that is rings)
    f64 mean_amp = 0;     // mean |ring amplitude| over detected radii (value units)
};

// Detect + subtract residual concentric rings in place on a ZYX volume. Returns the number of
// detected ring radii (summed over slabs); leaves the volume untouched if < 2 are found. Voxels
// with value <= 0 are treated as masked (skipped in both detect and subtract).
template <class T>
inline DeringStats dering_inplace(VolumeView<T> vol, const DeringParams& pin = {}) {
    const Extent3 d = vol.dims();
    const f64 cy = pin.cy >= 0 ? pin.cy : static_cast<f64>(d.y - 1) / 2.0;
    const f64 cx = pin.cx >= 0 ? pin.cx : static_cast<f64>(d.x - 1) / 2.0;
    const s32 slab_z = pin.slab_z > 0 ? pin.slab_z : 512;
    const s32 nslab = static_cast<s32>((d.z + slab_z - 1) / slab_z);
    const s32 ns = pin.ns > 0 ? pin.ns : 8;
    const f64 ry = std::max(cy, static_cast<f64>(d.y - 1) - cy);
    const f64 rx = std::max(cx, static_cast<f64>(d.x - 1) - cx);
    const s32 nr = static_cast<s32>(std::sqrt(ry * ry + rx * rx)) + 2;
    const s32 hw = (pin.hp_win < 3 ? 15 : pin.hp_win) / 2;
    const f64 min_amp = pin.min_amp > 0 ? pin.min_amp : 0.5;
    const f64 max_amp = pin.max_amp > 0 ? pin.max_amp : 6.0;
    const s32 ss = pin.ss < 1 ? 1 : pin.ss;
    const f64 inv_sect = static_cast<f64>(ns) / (2.0 * std::numbers::pi_v<f64>);
    const usize NR = static_cast<usize>(nr), NS = static_cast<usize>(ns);
    auto ix = [NR](s32 qq, s32 rr) -> usize { return static_cast<usize>(qq) * NR + static_cast<usize>(rr); };

    // pass 1 — accumulate per (slab, sector, radius) intensity sum + count. The (y,x)->(sector,
    // radius) map depends only on (y,x); build it once per row-of-x and reuse across z is implicit
    // since we recompute per (y,x) but the inner z loop is outermost here for slab locality.
    const usize nb = static_cast<usize>(nslab) * NS * NR;
    std::vector<f64> sum(nb, 0.0);
    std::vector<u32> cnt(nb, 0);
    f64 gsum = 0.0, gsum2 = 0.0;  // global accumulators for the signal-variance denominator
    u64 gcnt = 0;
    for (s64 y = 0; y < d.y; y += ss) {
        const f64 gy = static_cast<f64>(y) - cy, gy2 = gy * gy;
        for (s64 x = 0; x < d.x; x += ss) {
            const f64 gx = static_cast<f64>(x) - cx;
            const s32 ri = static_cast<s32>(std::sqrt(gy2 + gx * gx) + 0.5);
            if (ri >= nr) continue;
            s32 q = static_cast<s32>((std::atan2(gy, gx) + std::numbers::pi_v<f64>) * inv_sect);
            if (q >= ns) q = ns - 1;
            const usize qr = ix(q, ri);
            for (s64 z = 0; z < d.z; ++z) {
                const f32 v = static_cast<f32>(vol(z, y, x));
                if (v <= 0) continue;
                const usize b = static_cast<usize>(z / slab_z) * NS * NR + qr;
                sum[b] += static_cast<f64>(v);
                cnt[b]++;
                gsum += static_cast<f64>(v);
                gsum2 += static_cast<f64>(v) * static_cast<f64>(v);
                ++gcnt;
            }
        }
    }

    // finalize — per slab: per-sector mean profile -> box high-pass over valid bins -> per-radius
    // sector sign-consistency vote -> ring[slab][r] (median over sectors, clamped).
    std::vector<f32> ring(static_cast<usize>(nslab) * NR, 0.0f);
    std::vector<f32> prof(NS * NR), hp(NS * NR);
    std::vector<u8> valid(NS * NR);
    std::vector<f32> med(NS);
    long detected = 0;
    for (s32 s = 0; s < nslab; ++s) {
        const usize soff = static_cast<usize>(s) * NS * NR;
        for (s32 q = 0; q < ns; ++q) {
            for (s32 r = 0; r < nr; ++r) {
                const usize b = soff + ix(q, r), i = ix(q, r);
                valid[i] = cnt[b] >= pin.min_cnt ? u8{1} : u8{0};
                prof[i] = valid[i] ? static_cast<f32>(sum[b] / cnt[b]) : 0.0f;
            }
            for (s32 r = 0; r < nr; ++r) {
                const usize i = ix(q, r);
                if (!valid[i]) { hp[i] = 0.0f; continue; }
                f64 acc = 0.0;
                s32 c = 0;
                const s32 lo = std::max(0, r - hw), hi = std::min(nr - 1, r + hw);
                for (s32 j = lo; j <= hi; ++j) {
                    const usize ij = ix(q, j);
                    if (valid[ij]) { acc += static_cast<f64>(prof[ij]); ++c; }
                }
                if (c >= hw) hp[i] = prof[i] - static_cast<f32>(acc / c);
                else { hp[i] = 0.0f; valid[i] = 0; }
            }
        }
        f32* rg = ring.data() + static_cast<usize>(s) * NR;
        for (s32 r = 0; r < nr; ++r) {
            s32 nv = 0, pos = 0, neg = 0;
            for (s32 q = 0; q < ns; ++q) {
                const usize i = ix(q, r);
                if (!valid[i]) continue;
                const f32 v = hp[i];
                med[static_cast<usize>(nv++)] = v;
                if (v > 0) ++pos; else if (v < 0) ++neg;
            }
            // vote: enough sectors covered (>= ns-vote_slack) AND the minority sign within vote_dissent.
            if (nv < ns - pin.vote_slack || std::min(pos, neg) > pin.vote_dissent) { rg[r] = 0.0f; continue; }
            std::sort(med.begin(), med.begin() + nv);
            const f32 m = (nv & 1) ? med[static_cast<usize>(nv / 2)]
                                   : 0.5f * (med[static_cast<usize>(nv / 2 - 1)] + med[static_cast<usize>(nv / 2)]);
            if (std::fabs(m) < static_cast<f32>(min_amp)) { rg[r] = 0.0f; continue; }
            rg[r] = std::clamp(m, static_cast<f32>(-max_amp), static_cast<f32>(max_amp));
            ++detected;
        }
    }
    // intrinsic metric — ring energy (Σ ring^2 · n_voxels) vs total signal-variance energy.
    DeringStats st;
    st.n_rings = detected;
    f64 amp_acc = 0.0;
    for (s32 s = 0; s < nslab; ++s) {
        const usize soff = static_cast<usize>(s) * NS * NR;
        const f32* rg = ring.data() + static_cast<usize>(s) * NR;
        for (s32 r = 0; r < nr; ++r) {
            if (rg[r] == 0.0f) continue;
            u64 nvox = 0;
            for (s32 q = 0; q < ns; ++q) nvox += cnt[soff + ix(q, r)];
            st.ring_energy += static_cast<f64>(rg[r]) * static_cast<f64>(rg[r]) * static_cast<f64>(nvox);
            amp_acc += static_cast<f64>(std::fabs(rg[r]));
        }
    }
    const f64 gmean = gcnt ? gsum / static_cast<f64>(gcnt) : 0.0;
    st.signal_var = gsum2 - gsum * gmean;  // Σ(v-mean)^2 over the accumulated voxels
    st.ring_frac = st.signal_var > 0.0 ? st.ring_energy / st.signal_var : 0.0;
    st.mean_amp = detected ? amp_acc / static_cast<f64>(detected) : 0.0;
    if (detected < 2) return st;  // nothing to subtract

    // pass 2 — subtract ring[slab][rbin(y,x)] per voxel, clamp at 0, skip masked.
    for (s64 z = 0; z < d.z; ++z) {
        const s32 slab = std::clamp(static_cast<s32>(z / slab_z), 0, nslab - 1);
        const f32* rg = ring.data() + static_cast<usize>(slab) * nr;
        for (s64 y = 0; y < d.y; ++y) {
            const f64 gy = static_cast<f64>(y) - cy, gy2 = gy * gy;
            for (s64 x = 0; x < d.x; ++x) {
                const f32 v = static_cast<f32>(vol(z, y, x));
                if (v <= 0) continue;
                const f64 gx = static_cast<f64>(x) - cx;
                const s32 ri = static_cast<s32>(std::sqrt(gy2 + gx * gx) + 0.5);
                if (ri >= nr) continue;
                const f32 nvv = v - rg[ri];
                vol(z, y, x) = static_cast<T>(nvv > 0.0f ? nvv : 0.0f);
            }
        }
    }
    return st;
}

// `fenix dering <in.nrrd|.fxvol> <out.nrrd|.fxvol> [cy=] [cx=] [slab=512] [ns=8] [hp=15]
//               [min_amp=0.5] [max_amp=6] [ss=1] [vote_slack=2] [vote_dissent=0]
//               [iters=1] [until=0]`
// Reports a ring-energy metric per pass; with iters>1 / until>0 it re-runs detect+subtract until a
// pass removes < `until` fraction of the signal variance (diminishing returns).
inline Expected<int> run_dering(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error,
            "usage: fenix dering <in.nrrd|.fxvol> <out.nrrd|.fxvol> [cy=] [cx=] [slab=512] "
            "[ns=8] [hp=15] [min_amp=0.5] [max_amp=6] [ss=1] [vote_slack=2] [vote_dissent=0] "
            "[iters=1] [until=0]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto opt = [&](std::string_view k, std::string_view dflt) -> std::string {
        for (auto a : args) {
            const auto pos = a.find('=');
            if (pos != std::string_view::npos && a.substr(0, pos) == k) return std::string(a.substr(pos + 1));
        }
        return std::string(dflt);
    };
    const std::string inpath(args[0]), outpath(args[1]);
    auto load = [&](const std::string& p) -> Expected<Volume<f32>> {
        if (p.size() > 6 && p.substr(p.size() - 6) == ".fxvol") {
            auto a = codec::VolumeArchive::open(p);
            if (!a) return std::unexpected(a.error());
            return a->read_volume();
        }
        return io::read_nrrd(p);
    };
    auto vol = load(inpath);
    if (!vol) return std::unexpected(vol.error());

    DeringParams p;
#define FENIX_DERING_PARSE(field, key, dflt)                                     \
    do {                                                                         \
        auto pr = detail::parse_opt<decltype(field)>(key, opt(key, dflt));       \
        if (!pr) return std::unexpected(pr.error());                            \
        field = *pr;                                                            \
    } while (0)
    FENIX_DERING_PARSE(p.cy, "cy", "-1");
    FENIX_DERING_PARSE(p.cx, "cx", "-1");
    FENIX_DERING_PARSE(p.slab_z, "slab", "512");
    FENIX_DERING_PARSE(p.ns, "ns", "8");
    FENIX_DERING_PARSE(p.hp_win, "hp", "15");
    FENIX_DERING_PARSE(p.min_amp, "min_amp", "0.5");
    FENIX_DERING_PARSE(p.max_amp, "max_amp", "6");
    FENIX_DERING_PARSE(p.ss, "ss", "1");
    FENIX_DERING_PARSE(p.vote_slack, "vote_slack", "2");
    FENIX_DERING_PARSE(p.vote_dissent, "vote_dissent", "0");
    int iters = 1;
    FENIX_DERING_PARSE(iters, "iters", "1");
    iters = std::max(1, iters);
    f64 until = 0.0;
    FENIX_DERING_PARSE(until, "until", "0");
#undef FENIX_DERING_PARSE

    const Extent3 d = vol->dims();
    // Iterate detect+subtract until a pass removes < `until` of the signal variance (diminishing
    // returns), or `iters` passes are done, or nothing is detected.
    long total_rings = 0;
    f64 cum_frac = 0.0;
    for (int it = 0; it < iters; ++it) {
        const DeringStats st = dering_inplace<f32>(vol->view(), p);
        total_rings += st.n_rings;
        cum_frac += st.ring_frac;
        log(LogLevel::info,
            "dering pass {}/{}: {} rings, ring energy = {:.3f}% of signal variance, mean|amp|={:.2f}",
            it + 1, iters, st.n_rings, st.ring_frac * 100.0, st.mean_amp);
        if (st.n_rings < 2) break;                  // nothing more to remove
        if (until > 0.0 && st.ring_frac < until) break;  // diminishing returns
    }
    log(LogLevel::info,
        "dering: {} ring radii removed over the run (center {:.1f},{:.1f}; cumulative {:.3f}% signal; "
        "{} sectors, vote_slack={}, vote_dissent={}; dims {}x{}x{})",
        total_rings, p.cy >= 0 ? p.cy : (d.y - 1) / 2.0, p.cx >= 0 ? p.cx : (d.x - 1) / 2.0,
        cum_frac * 100.0, p.ns, p.vote_slack, p.vote_dissent, d.z, d.y, d.x);

    // Always .fxvol — CT-domain, so round-clamp to u8 (never widen/keep f32 on disk); the archive
    // encoder consumes the u8 source directly. We never write NRRD.
    Volume<u8> out8(d);
    for (s64 i = 0; i < d.count(); ++i)
        out8.flat()[static_cast<usize>(i)] = static_cast<u8>(std::clamp(vol->flat()[static_cast<usize>(i)], 0.0f, 255.0f) + 0.5f);
    auto a = codec::VolumeArchive::create(outpath, d, codec::DctParams{});
    if (!a) return std::unexpected(a.error());
    if (auto w = a->template write_volume<u8>(out8.view()); !w) return std::unexpected(w.error());
    if (auto w = a->close(); !w) return std::unexpected(w.error());
    log(LogLevel::info, "dering: wrote {}", outpath);
    return 0;
}

}  // namespace fenix::preprocess

FENIX_REGISTER_STAGE(dering, "detect-then-subtract residual ring removal (fysics)",
                     ::fenix::preprocess::run_dering)
