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
inline Expected<Volume<f32>> predict_surface(VolumeView<const f32> in, nets::ResEncUNet& net,
                                             torch::Device dev, const InferOptions& opt = {}) {
    if (opt.patch % 64 != 0) return fenix::err(Errc::invalid_argument, "patch must be divisible by 64");
    const Extent3 d = in.dims();
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
        // Gather patch with edge clamp (volumes >= patch in practice; clamp covers small dims).
        double sum = 0.0, sq = 0.0;
        for (int z = 0; z < P; ++z) for (int y = 0; y < P; ++y) for (int x = 0; x < P; ++x) {
            const f32 v = in.at_clamped(z0 + z, y0 + y, x0 + x);
            patch[(static_cast<std::size_t>(z) * P + y) * P + x] = v;
            sum += v; sq += static_cast<double>(v) * v;
        }
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
        auto logits = net->forward(xin).to(torch::kFloat32);   // [1,C,P,P,P]
        auto pr = opt.sigmoid ? torch::sigmoid(logits.index({0, 0}))
                              : torch::softmax(logits, 1).index({0, opt.channel});
        auto surf = pr.to(torch::kCPU).contiguous();           // [P,P,P]
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

}  // namespace fenix::ml
