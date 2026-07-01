// ml/infer.hpp — sliding-window 3D inference for the surface/ink segmentation nets. Tiles a
// volume into patches (default 256^3) with overlap, z-score-normalizes each patch, runs the
// net, softmaxes, and blends overlapping patches with a Gaussian importance window (nnU-Net
// style). Out-of-core-ready in shape (per-patch streaming); the accumulators are dense here.
// Only compiled under FENIX_ML.
#pragma once

#include "core/core.hpp"
#include "ml/tiling.hpp"
#include "ml/torch_env.hpp"
#include "ml/nets/resenc_unet.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace fenix::ml {

enum class Norm { zscore, pct_minmax };  // surface: zscore; ink: percentile (0.5/99.5) min-max

struct InferOptions {
    int patch = 256;          // must be divisible by 2^(n_stages-1) = 64
    double overlap = 0.5;     // fraction; step = patch*(1-overlap)
    bool half = true;         // fp16 forward (tolerance-only project; 256^3 needs >8 GB VRAM)
    int channel = 1;          // softmax channel to emit (surface=1); ignored when sigmoid
    bool sigmoid = false;     // ink head: 1-channel logit -> sigmoid (vs 2-channel softmax)
    Norm norm = Norm::zscore;
    int tta = 0;              // TTA: average the first N of the 48 octahedral symmetries (<=1 off, 8 flips, 48 full)
    // Multi-scale TTA / base-rescale: predict at each scale (resample input ×s → predict → resample prob
    // back → MEAN-fuse). Empty = single native scale (byte-identical to before). One entry = a base rescale
    // (e.g. {1.2} to hit the model's ~2µm training grid); several = scale-ensemble TTA (best ridge
    // localization, measured). Stacks with octahedral `tta`. See ADR/notes + fenix-tta-multiscale memory.
    std::vector<double> scales{};
};

namespace detail {
// Percentile [plo,phi] of a patch via a 1024-bin histogram over [min,max] (data is ~0..255).
inline void pct_bounds(const std::vector<float>& v, double plo, double phi, float& lo, float& hi) {
    float mn = v[0], mx = v[0];
    for (float x : v) { mn = std::min(mn, x); mx = std::max(mx, x); }
    if (mx <= mn) { lo = mn; hi = mn + 1.0f; return; }
    constexpr int B = 1024;
    std::array<s64, B> h{};
    const float inv = (B - 1) / (mx - mn);
    for (float x : v) ++h[static_cast<usize>((x - mn) * inv)];
    const s64 n = static_cast<s64>(v.size());
    const s64 tlo = static_cast<s64>(plo / 100.0 * n), thi = static_cast<s64>(phi / 100.0 * n);
    s64 c = 0; int blo = 0, bhi = B - 1;
    for (int b = 0; b < B; ++b) { c += h[static_cast<usize>(b)]; if (c <= tlo) blo = b; if (c < thi) bhi = b; }
    lo = mn + blo / inv;
    hi = mn + (bhi + 1) / inv;
    if (hi <= lo) hi = lo + 1.0f;
}
}  // namespace detail

// Run sliding-window segmentation inference. Returns a probability volume (same dims as `in`).
// Core sliding-window predict, templated on a PATCH FILLER: `void fill(s64 z0,s64 y0,s64 x0,int P,float* out)`
// which writes a P³ ZYX-contiguous patch at (z0,y0,x0), edge-clamped, into `out`. This lets the input come
// from a dense view OR stream from a native-u8 .fxvol block cache (block-batched — no dense f32 volume). The
// filler owns its own parallelism/locking; the net forward + Gaussian-blended accumulate are unchanged.
template <class Filler>
inline Expected<Volume<f32>> predict_surface_filled(Extent3 d, Filler&& fill, nets::ResEncUNet& net,
                                                    torch::Device dev, const InferOptions& opt = {}) {
    if (opt.patch % 64 != 0) return fenix::err(Errc::invalid_argument, "patch must be divisible by 64");
    const int P = opt.patch;

    Volume<f32> prob = Volume<f32>::zeros(d);
    std::vector<float> wacc(static_cast<std::size_t>(d.count()), 0.0f);
    auto pv = prob.view();

    const auto gz = gaussian1d(P), gy = gaussian1d(P), gx = gaussian1d(P);
    const auto zs = tile_starts(d.z, P, opt.overlap);
    const auto ys = tile_starts(d.y, P, opt.overlap);
    const auto xs = tile_starts(d.x, P, opt.overlap);

    torch::NoGradGuard ng;
    const auto fdt = opt.half ? torch::kFloat16 : torch::kFloat32;
    if (opt.half) net->to(torch::kFloat16);

    std::vector<float> patch(static_cast<std::size_t>(P) * P * P);
    long n_tiles = 0, total = static_cast<long>(zs.size() * ys.size() * xs.size());
    for (s64 z0 : zs) for (s64 y0 : ys) for (s64 x0 : xs) {
        // Fill the patch (the filler parallelizes + widens block-batched), then reduce for the z-score.
        fill(z0, y0, x0, P, patch.data());
        std::vector<double> psum(static_cast<std::size_t>(P), 0.0), psq(static_cast<std::size_t>(P), 0.0);
        parallel_for(0, P, [&](s64 z) {
            double ls = 0.0, lq = 0.0;
            const float* row = patch.data() + static_cast<std::size_t>(z) * P * P;
            for (int i = 0; i < P * P; ++i) { const double v = row[i]; ls += v; lq += v * v; }
            psum[static_cast<std::size_t>(z)] = ls; psq[static_cast<std::size_t>(z)] = lq;
        });
        double sum = 0.0, sq = 0.0;
        for (int z = 0; z < P; ++z) { sum += psum[static_cast<std::size_t>(z)]; sq += psq[static_cast<std::size_t>(z)]; }
        const double n = static_cast<double>(P) * P * P;
        if (opt.norm == Norm::zscore) {
            const double mean = sum / n;
            const double std = std::sqrt(std::max(sq / n - mean * mean, 1e-12));
            for (auto& v : patch) v = static_cast<float>((v - mean) / std);
        } else {  // percentile (0.5/99.5) min-max -> [0,1]
            float lo, hi;
            detail::pct_bounds(patch, 0.5, 99.5, lo, hi);
            const float inv = 1.0f / (hi - lo);
            for (auto& v : patch) v = std::clamp((v - lo) * inv, 0.0f, 1.0f);
        }

        auto xin = torch::from_blob(patch.data(), {1, 1, P, P, P}, torch::kFloat32).clone().to(dev).to(fdt);
        // Test-time augmentation over the 48-element octahedral symmetry group (axis-aligned; valid for
        // isotropic volumes). Ordered flips-first: opt.tta=8 == the 8 mirror-flips (identity permutation),
        // opt.tta=48 == the full group (6 axis-permutations × 8 flips). opt.tta<=1 => one forward,
        // byte-for-byte the prior path. Each augmented prob is mapped back to the original frame before
        // accumulating (un-flip on the permuted-frame axes, then inverse-permute), then averaged.
        static const int kPerms[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
        const int ne = opt.tta <= 1 ? 1 : (opt.tta < 48 ? opt.tta : 48);
        torch::Tensor acc;
        for (int n = 0; n < ne; ++n) {
            const int pi = n / 8, fm = n % 8;
            const int* pp = kPerms[pi];
            auto xf = (pi == 0) ? xin : xin.permute({0, 1, 2 + pp[0], 2 + pp[1], 2 + pp[2]});
            std::vector<int64_t> fd;
            if (fm & 1) fd.push_back(2);
            if (fm & 2) fd.push_back(3);
            if (fm & 4) fd.push_back(4);
            if (!fd.empty()) xf = torch::flip(xf, fd);
            auto logits = net->forward(xf.contiguous()).to(torch::kFloat32);  // [1,C,P,P,P]
            auto pr = opt.sigmoid ? torch::sigmoid(logits.index({0, 0}))
                                  : torch::softmax(logits, 1).index({0, opt.channel});  // [P,P,P]
            std::vector<int64_t> pf;
            if (fm & 1) pf.push_back(0);
            if (fm & 2) pf.push_back(1);
            if (fm & 4) pf.push_back(2);
            if (!pf.empty()) pr = torch::flip(pr, pf);
            if (pi != 0) {
                int q[3]; q[pp[0]] = 0; q[pp[1]] = 1; q[pp[2]] = 2;  // inverse permutation
                pr = pr.permute({q[0], q[1], q[2]});
            }
            auto prc = pr.contiguous();
            acc = n == 0 ? prc : acc + prc;
        }
        if (ne > 1) acc = acc / static_cast<float>(ne);
        auto surf = acc.to(torch::kCPU).contiguous();          // [P,P,P]
        const float* sp = surf.data_ptr<float>();

        for (int z = 0; z < P; ++z) {
            const s64 oz = z0 + z; if (oz >= d.z) break;
            for (int y = 0; y < P; ++y) {
                const s64 oy = y0 + y; if (oy >= d.y) break;
                const float wzy = gz[static_cast<std::size_t>(z)] * gy[static_cast<std::size_t>(y)];
                for (int x = 0; x < P; ++x) {
                    const s64 ox = x0 + x; if (ox >= d.x) break;
                    const float w = wzy * gx[static_cast<std::size_t>(x)];
                    pv(oz, oy, ox) += w * sp[(static_cast<std::size_t>(z) * P + y) * P + x];
                    wacc[static_cast<std::size_t>(pv.offset(oz, oy, ox))] += w;
                }
            }
        }
        if (++n_tiles % 25 == 0 || n_tiles == total)
            fenix::log(LogLevel::info, "predict-surface: tile {}/{}", n_tiles, total);
    }

    for (s64 z = 0; z < d.z; ++z) for (s64 y = 0; y < d.y; ++y) for (s64 x = 0; x < d.x; ++x) {
        const float w = wacc[static_cast<std::size_t>(pv.offset(z, y, x))];
        if (w > 0.0f) pv(z, y, x) /= w;
    }
    return prob;
}

// Dense-view entry point (NRRD inputs, tests): fill each patch from RAM with edge clamp, parallel over z.
inline Expected<Volume<f32>> predict_surface(VolumeView<const f32> in, nets::ResEncUNet& net,
                                             torch::Device dev, const InferOptions& opt = {}) {
    return predict_surface_filled(
        in.dims(),
        [in](s64 z0, s64 y0, s64 x0, int P, float* out) {
            parallel_for(0, P, [&](s64 z) {
                for (int y = 0; y < P; ++y)
                    for (int x = 0; x < P; ++x)
                        out[(static_cast<std::size_t>(z) * P + y) * P + x] = in.at_clamped(z0 + z, y0 + y, x0 + x);
            });
        },
        net, dev, opt);
}

// Trilinear-resample an f32 volume to `out` dims (torch, on `dev`). Used by multi-scale TTA to rescale the
// input to the model's preferred grid and to map the prediction back to native resolution.
inline Volume<f32> resample_f32(VolumeView<const f32> in, Extent3 out, torch::Device dev) {
    const Extent3 d = in.dims();
    Volume<f32> src(d);  // contiguous copy of the (possibly strided) view
    { auto sv = src.view(); parallel_for(0, d.z, [&](s64 z) { for (s64 y = 0; y < d.y; ++y) for (s64 x = 0; x < d.x; ++x) sv(z, y, x) = in(z, y, x); }); }
    torch::NoGradGuard ng;
    auto t = torch::from_blob(src.data(), {1, 1, d.z, d.y, d.x}, torch::kFloat32).clone().to(dev);
    auto o = torch::nn::functional::interpolate(
                 t, torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{out.z, out.y, out.x})
                        .mode(torch::kTrilinear)
                        .align_corners(false))
                 .to(torch::kCPU).contiguous();
    Volume<f32> vo(out);
    const float* op = o.data_ptr<float>();
    std::span<f32> vf = vo.flat();
    for (s64 i = 0; i < out.count(); ++i) vf[static_cast<usize>(i)] = op[i];
    return vo;
}

// Multi-scale TTA: predict at each scale in `opt.scales` (resample input ×s → predict → resample the prob
// back to native) and MEAN-fuse. Best ridge LOCALIZATION (measured — mean beats max, which unions the
// scale-shifted bands and thickens). Stacks with octahedral `opt.tta` (the inner per-scale predict keeps
// it). Empty scales => plain single-scale predict_surface (byte-identical).
inline Expected<Volume<f32>> predict_surface_scales(VolumeView<const f32> in, nets::ResEncUNet& net,
                                                    torch::Device dev, const InferOptions& opt) {
    if (opt.scales.empty()) return predict_surface(in, net, dev, opt);
    const Extent3 d = in.dims();
    Volume<f32> acc = Volume<f32>::zeros(d);
    auto av = acc.view();
    InferOptions inner = opt;
    inner.scales.clear();  // inner predict is single-scale (still octahedral-TTA'd via inner.tta)
    int used = 0;
    for (double s : opt.scales) {
        Expected<Volume<f32>> prob = fenix::err(Errc::unsupported, "unset");
        if (s == 1.0) {
            prob = predict_surface(in, net, dev, inner);
        } else {
            const Extent3 sd{std::max<s64>(64, std::llround(static_cast<double>(d.z) * s)),
                             std::max<s64>(64, std::llround(static_cast<double>(d.y) * s)),
                             std::max<s64>(64, std::llround(static_cast<double>(d.x) * s))};
            Volume<f32> in_s = resample_f32(in, sd, dev);
            auto p = predict_surface(in_s.view(), net, dev, inner);
            if (!p) return std::unexpected(p.error());
            prob = resample_f32(p->view(), d, dev);  // map the prob back to native resolution
        }
        if (!prob) return std::unexpected(prob.error());
        auto pv = prob->view();
        for (s64 z = 0; z < d.z; ++z) for (s64 y = 0; y < d.y; ++y) for (s64 x = 0; x < d.x; ++x) av(z, y, x) += pv(z, y, x);
        ++used;
        fenix::log(LogLevel::info, "multiscale: scale {:.3g} done ({}/{})", s, used, static_cast<int>(opt.scales.size()));
    }
    const float inv = 1.0f / static_cast<float>(used);
    for (s64 z = 0; z < d.z; ++z) for (s64 y = 0; y < d.y; ++y) for (s64 x = 0; x < d.x; ++x) av(z, y, x) *= inv;
    return acc;
}

}  // namespace fenix::ml
