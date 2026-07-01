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
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
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
    int batch = 3;            // patches per GPU forward (tta<=1). Sweet spot on a 32 GB card at patch=256:
                             // measured ms/patch 338(b1)/218(b2)/198(b3)/305(b4 — VRAM-pressure regression);
                             // b3 ≈ 28 GB. Lower it for smaller cards or larger patches.
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
    auto pv = prob.view();

    const auto gz = gaussian1d(P), gy = gaussian1d(P), gx = gaussian1d(P);
    const auto zs = tile_starts(d.z, P, opt.overlap);
    const auto ys = tile_starts(d.y, P, opt.overlap);
    const auto xs = tile_starts(d.x, P, opt.overlap);

    // The tile set is the full Cartesian product zs×ys×xs, so the accumulated Gaussian weight
    // factorizes exactly: wacc(z,y,x) = Wz(z)·Wy(y)·Wx(x). Three 1D profiles replace the dense
    // per-voxel weight volume (4 bytes/voxel — 4.3 GB at 1024³) and halve the scatter writes.
    // Mirrors the scatter's edge guards (skip out-of-bounds tail of edge tiles).
    auto weight_profile = [P](const std::vector<float>& g, const std::vector<s64>& starts, s64 dim) {
        std::vector<float> w(static_cast<std::size_t>(dim), 0.0f);
        for (s64 s0 : starts)
            for (int i = 0; i < P && s0 + i < dim; ++i) w[static_cast<std::size_t>(s0 + i)] += g[static_cast<std::size_t>(i)];
        return w;
    };
    const std::vector<float> Wz = weight_profile(gz, zs, d.z), Wy = weight_profile(gy, ys, d.y),
                             Wx = weight_profile(gx, xs, d.x);
    auto normalize = [&] {
        parallel_for_z(d, [&](s64 z) {
            const float wz = Wz[static_cast<std::size_t>(z)];
            for (s64 y = 0; y < d.y; ++y) {
                const float wzy = wz * Wy[static_cast<std::size_t>(y)];
                for (s64 x = 0; x < d.x; ++x) {
                    const float w = wzy * Wx[static_cast<std::size_t>(x)];
                    if (w > 0.0f) pv(z, y, x) /= w;
                }
            }
        });
    };

    torch::NoGradGuard ng;
    if (opt.half) net->to(torch::kFloat16);

    // Flat tile list so we can process B patches per GPU forward.
    struct Tile { s64 z0, y0, x0; };
    std::vector<Tile> tiles;
    tiles.reserve(zs.size() * ys.size() * xs.size());
    for (s64 z0 : zs) for (s64 y0 : ys) for (s64 x0 : xs) tiles.push_back({z0, y0, x0});
    const long total = static_cast<long>(tiles.size());
    const std::size_t PN = static_cast<std::size_t>(P) * P * P;
    // Batching only when tta<=1 (the common path). tta>1 keeps B=1 (the TTA machinery is per-patch).
    const int B = (opt.tta <= 1 && opt.batch > 1) ? opt.batch : 1;

    // Gather + normalize one tile into `out` (P³ contiguous f32). CPU-bound; parallel over z internally.
    auto prep = [&](const Tile& t, float* out) {
        fill(t.z0, t.y0, t.x0, P, out);
        std::vector<double> psum(static_cast<std::size_t>(P), 0.0), psq(static_cast<std::size_t>(P), 0.0);
        parallel_for(0, P, [&](s64 z) {
            double ls = 0.0, lq = 0.0;
            const float* row = out + static_cast<std::size_t>(z) * P * P;
            for (int i = 0; i < P * P; ++i) { const double v = row[i]; ls += v; lq += v * v; }
            psum[static_cast<std::size_t>(z)] = ls; psq[static_cast<std::size_t>(z)] = lq;
        });
        double sum = 0.0, sq = 0.0;
        for (int z = 0; z < P; ++z) { sum += psum[static_cast<std::size_t>(z)]; sq += psq[static_cast<std::size_t>(z)]; }
        const double n = static_cast<double>(PN);
        if (opt.norm == Norm::zscore) {
            const double mean = sum / n, sd = std::sqrt(std::max(sq / n - mean * mean, 1e-12));
            for (std::size_t i = 0; i < PN; ++i) out[i] = static_cast<float>((out[i] - mean) / sd);
        } else {
            std::vector<float> tmp(out, out + PN);
            float lo, hi; detail::pct_bounds(tmp, 0.5, 99.5, lo, hi);
            const float inv = 1.0f / (hi - lo);
            for (std::size_t i = 0; i < PN; ++i) out[i] = std::clamp((out[i] - lo) * inv, 0.0f, 1.0f);
        }
    };
    // Scatter a [P,P,P] result (CPU ptr sp) for tile t into the Gaussian-blended accumulator.
    auto scatter = [&](const Tile& t, const float* sp) {
        parallel_for(0, P, [&](s64 z) {
            const s64 oz = t.z0 + z; if (oz >= d.z) return;
            for (int y = 0; y < P; ++y) {
                const s64 oy = t.y0 + y; if (oy >= d.y) break;
                const float wzy = gz[static_cast<std::size_t>(z)] * gy[static_cast<std::size_t>(y)];
                for (int x = 0; x < P; ++x) {
                    const s64 ox = t.x0 + x; if (ox >= d.x) break;
                    pv(oz, oy, ox) += wzy * gx[static_cast<std::size_t>(x)] * sp[(static_cast<std::size_t>(z) * P + y) * P + x];
                }
            }
        });
    };

    static const int kPerms[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    const bool prof = std::getenv("FENIX_INFER_PROFILE") != nullptr;
    double t_prep = 0, t_fwd = 0, t_scat = 0;
    auto clk = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    long n_tiles = 0;

    // ---- PIPELINED batched path (B>1, tta<=1): a producer thread preps batch i+1 into a 2-slot ring while
    // the main thread runs batch i's GPU forward + scatter. Results are byte-identical to the serial path;
    // this just overlaps the CPU prep (~16% of wall time) with the GPU forward. Deadlock-proof: mutex + two
    // condvars over a bounded ring, with a `filled` count and a `stop` flag. ----
    if (B > 1) {
        constexpr int kSlots = 2;
        std::vector<float> ring[kSlots] = {std::vector<float>(PN * static_cast<std::size_t>(B)),
                                           std::vector<float>(PN * static_cast<std::size_t>(B))};
        int slot_i0[kSlots] = {0, 0}, slot_nb[kSlots] = {0, 0};
        std::mutex m;
        std::condition_variable cv_filled, cv_free;
        int head = 0, tail = 0, filled = 0;  // ring indices + count of prepped-but-unconsumed slots
        bool prod_done = false;
        std::thread producer([&] {
            for (long i0 = 0; i0 < total; i0 += B) {
                const int nb = static_cast<int>(std::min<long>(B, total - i0));
                std::unique_lock<std::mutex> lk(m);
                cv_free.wait(lk, [&] { return filled < kSlots; });
                const int s = tail;
                lk.unlock();
                for (int b = 0; b < nb; ++b) prep(tiles[static_cast<std::size_t>(i0 + b)], ring[s].data() + PN * static_cast<std::size_t>(b));
                lk.lock();
                slot_i0[s] = static_cast<int>(i0); slot_nb[s] = nb;
                tail = (tail + 1) % kSlots; ++filled;
                cv_filled.notify_one();
            }
            std::lock_guard<std::mutex> lk(m);
            prod_done = true; cv_filled.notify_one();
        });
        for (;;) {
            std::unique_lock<std::mutex> lk(m);
            cv_filled.wait(lk, [&] { return filled > 0 || prod_done; });
            if (filled == 0 && prod_done) break;
            const int s = head;
            const int i0 = slot_i0[s], nb = slot_nb[s];
            lk.unlock();
            double tp = prof ? clk() : 0;
            // Cast to the forward dtype ON CPU (f32→f16 is the same round-to-nearest on CPU and GPU), then
            // upload — halves the H2D bytes and skips the old fp32 clone+device-cast. The cast itself copies
            // out of the ring slot, so the slot is free for the producer before the upload even starts.
            auto blob = torch::from_blob(ring[s].data(), {nb, 1, P, P, P}, torch::kFloat32);
            auto xhost = opt.half ? blob.to(torch::kFloat16) : blob.clone();
            lk.lock(); head = (head + 1) % kSlots; --filled; cv_free.notify_one(); lk.unlock();
            auto xin = xhost.to(dev);
            // softmax/sigmoid on the fp16 logits (softmax internally upcasts its reduction to fp32 → same
            // result), SELECT the single output channel, download it in the forward dtype (fp16 D2H = half
            // the bytes), and upcast to f32 on the CPU (fp16→f32 is exact). Avoids materializing the full
            // fp32 logits [nb,C,P³] on the GPU and shrinks both transfers.
            auto logits = net->forward(xin);
            auto pr = opt.sigmoid ? torch::sigmoid(logits.index({torch::indexing::Slice(), 0}))
                                  : torch::softmax(logits, 1).index({torch::indexing::Slice(), opt.channel});
            auto surf = pr.contiguous().to(torch::kCPU).to(torch::kFloat32).contiguous();  // [nb,P,P,P]
            if (prof) { t_fwd += clk() - tp; tp = clk(); }
            const float* base = surf.data_ptr<float>();
            for (int b = 0; b < nb; ++b) scatter(tiles[static_cast<std::size_t>(i0 + b)], base + PN * static_cast<std::size_t>(b));
            if (prof) t_scat += clk() - tp;
            n_tiles += nb;
            if (n_tiles % 25 < nb || n_tiles == total)
                fenix::log(LogLevel::info, "predict-surface: tile {}/{}", n_tiles, total);
        }
        producer.join();
        if (prof)
            fenix::log(LogLevel::info, "profile: fwd(gpu)={:.1f}s scatter(cpu)={:.1f}s (prep overlapped)", t_fwd, t_scat);
        normalize();
        return prob;
    }

    // ---- SERIAL single-patch path (B==1, also the ONLY path when tta>1) ----
    std::vector<float> batchbuf(PN);
    for (long i0 = 0; i0 < total; i0 += B) {
        const int nb = static_cast<int>(std::min<long>(B, total - i0));
        if (B == 1) {
            // Single-patch path (also the ONLY path when tta>1) — preserves the exact TTA behavior.
            prep(tiles[static_cast<std::size_t>(i0)], batchbuf.data());
            auto xb = torch::from_blob(batchbuf.data(), {1, 1, P, P, P}, torch::kFloat32);
            auto xin = (opt.half ? xb.to(torch::kFloat16) : xb.clone()).to(dev);
            const int ne = opt.tta <= 1 ? 1 : (opt.tta < 48 ? opt.tta : 48);
            torch::Tensor acc;
            for (int nn = 0; nn < ne; ++nn) {
                const int pi = nn / 8, fm = nn % 8;
                const int* pp = kPerms[pi];
                auto xf = (pi == 0) ? xin : xin.permute({0, 1, 2 + pp[0], 2 + pp[1], 2 + pp[2]});
                std::vector<int64_t> fd;
                if (fm & 1) fd.push_back(2);
                if (fm & 2) fd.push_back(3);
                if (fm & 4) fd.push_back(4);
                if (!fd.empty()) xf = torch::flip(xf, fd);
                auto logits = net->forward(xf.contiguous()).to(torch::kFloat32);
                auto pr = opt.sigmoid ? torch::sigmoid(logits.index({0, 0}))
                                      : torch::softmax(logits, 1).index({0, opt.channel});
                std::vector<int64_t> pf;
                if (fm & 1) pf.push_back(0);
                if (fm & 2) pf.push_back(1);
                if (fm & 4) pf.push_back(2);
                if (!pf.empty()) pr = torch::flip(pr, pf);
                if (pi != 0) { int q[3]; q[pp[0]] = 0; q[pp[1]] = 1; q[pp[2]] = 2; pr = pr.permute({q[0], q[1], q[2]}); }
                auto prc = pr.contiguous();
                acc = nn == 0 ? prc : acc + prc;
            }
            if (ne > 1) acc = acc / static_cast<float>(ne);
            auto surf = acc.to(torch::kCPU).contiguous();
            scatter(tiles[static_cast<std::size_t>(i0)], surf.data_ptr<float>());
            n_tiles += 1;
        }
        if (n_tiles % 25 < nb || n_tiles == total)
            fenix::log(LogLevel::info, "predict-surface: tile {}/{}", n_tiles, total);
    }
    (void)t_prep;

    normalize();
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
