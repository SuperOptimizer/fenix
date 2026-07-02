// ml/infer.hpp — sliding-window 3D inference for the surface/ink segmentation nets. Tiles a
// volume into patches (default 256^3) with overlap, z-score-normalizes each patch, runs the
// net, softmaxes, and blends overlapping patches with a Gaussian importance window (nnU-Net
// style). Out-of-core-ready in shape (per-patch streaming); the accumulators are dense here.
// Only compiled under FENIX_ML.
#pragma once

#include "core/core.hpp"
#include "core/hash.hpp"
#include "core/rng.hpp"
#include "ml/tiling.hpp"
#include "ml/torch_env.hpp"
#include "ml/nets/resenc_unet.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
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
    // Arbitrary-angle TTA: rotations about the z (scroll) axis in DEGREES. For each angle: rotate the
    // input (GPU trilinear), predict, rotate the prob back, validity-weighted mean-fuse (corners clipped
    // by the rotation get zero weight). 0 = the exact unrotated member (no resample). Samples orientations
    // the 48-element octahedral group can't reach (it only has 90° steps). Stacks outside scales and tta.
    std::vector<double> rots{};
    // Noise-injection TTA: number of extra members, each with i.i.d. Gaussian noise (sigma =
    // noise_sigma, in z-scored units since it's added post-normalize) added to every patch before the
    // forward, results averaged in. Samples the model's sensitivity to CT noise — the intensity analog
    // of geometric TTA, with NO interpolation blur. 0 = off. Member 0 is always the clean pass.
    int noise = 0;
    double noise_sigma = 0.1;
    // Tile-offset TTA: shift the whole sliding-window grid origin by a fraction of the step and re-fuse,
    // averaging over grid alignments to remove the fixed-origin seam bias. 0 = off (single origin).
    int offsets = 0;
    // --- internal per-member state set by the ensemble wrapper (not user-facing) ---
    bool noise_apply = false;      // this member injects noise (member 0 stays clean)
    u64 noise_member_seed = 0;     // distinct per noise member
    s64 tile_shift = 0;            // origin shift (voxels) applied to all three tiling axes
    int batch = 3;            // patches per GPU forward (tta<=1). Sweet spot on a 32 GB card at patch=256:
                             // measured ms/patch 338(b1)/218(b2)/198(b3)/305(b4 — VRAM-pressure regression);
                             // b3 ≈ 28 GB. Lower it for smaller cards or larger patches.
    // Resumable inference: if non-empty, checkpoint the un-normalized accumulator + completed-tile count
    // to this path every `ckpt_every` tiles (atomic temp+rename), and RESUME from it on start if it exists
    // and its header matches (dims/patch/overlap/tta/tile_shift AND identity — see below). A crashed/killed
    // run picks up mid-window instead of restarting. Empty = no checkpointing (byte-identical to before).
    // Deleted on success.
    std::string ckpt_path{};
    long ckpt_every = 128;    // flush cadence in tiles (a whole-volume flush of `prob` is O(voxels))
    // Checkpoint IDENTITY: the caller (run_predict) fills these so a resume can never silently blend a
    // different model/input/task into the accumulator. weights_hash = hash64 over the .fxweights file
    // bytes (content, so re-downloads/copies with churned mtimes don't spuriously invalidate a multi-hour
    // checkpoint); input_hash = hash64 over the resolved input path. 0/0 = no identity available (e.g. the
    // dense-view predict_surface() entry point used by tests/NRRD) — a checkpoint with a real identity will
    // then correctly refuse to match it, and one with 0/0 stored will match only another 0/0 run, which is
    // the conservative direction (never worse than before this fix, at worst refuses a resume it used to
    // wrongly accept).
    u64 weights_hash = 0;
    u64 input_hash = 0;
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

// --- inference checkpoint (resumable sliding-window) ------------------------------------------------
// On-disk layout: a small header identifying the run config (so we never resume onto a mismatched grid),
// then the completed-tile count, then the raw un-normalized accumulator (f32, d.count() values). Tiles
// are processed in a FIXED order, so `n_done` alone says where to continue. Written atomically
// (temp+rename) so a crash mid-write can't corrupt a good checkpoint.
//
// magic byte 8 is a format-version tag ('3'): besides dims/patch/overlap/tta/tile_shift, the header also
// identifies WHICH model/input/task this accumulator belongs to (weights content hash, input path hash,
// norm/sigmoid/channel) — resuming onto a checkpoint from a different weights file, input volume, or task
// (predict-surface vs predict-ink to the same output path) would otherwise silently blend two runs'
// predictions with no warning. Field order groups same-size members (no internal padding holes); NOTE
// `has_unique_object_representations_v` still reports false for this struct because of the `double`
// fields (NaN has multiple bit patterns), not because of padding — `CkptHeader h{}` (all NSDMIs) zero-
// inits every byte including any trailing pad before every fwrite, which is the actual fix for
// "uninitialized stack bytes written to disk" (the MSan-visible issue).
struct CkptHeader {
    char magic[8] = {'F','X','I','C','K','P','T','3'};
    s64 dz = 0, dy = 0, dx = 0;
    s64 tile_shift = 0;
    long total_tiles = 0;
    long n_done = 0;       // tiles completed (and reflected in the accumulator below)
    u64 weights_hash = 0;  // hash64 over the .fxweights file bytes (content, not path/mtime)
    u64 input_hash = 0;    // hash64 over the resolved input path (cheap identity; not volume content)
    double overlap = 0;
    double acc_scale = 0;  // accumulator stored as u16 = round(acc / acc_scale·65535); 0 ⇒ all-zero
    int patch = 0;
    int tta = 0;
    int channel = 0;
    int sigmoid = 0;  // 0/1, stored as int (not bool) to keep the struct padding-free/portable
    int norm = 0;      // Norm enum value
};

// Bytes a well-formed checkpoint file must contain for header `h` (payload present only when
// acc_scale > 0 — an all-zero accumulator writes header-only). Used to reject short/truncated
// files up front, before any byte is written into the live accumulator.
inline s64 ckpt_expected_size(const CkptHeader& h, s64 nvox) {
    return static_cast<s64>(sizeof(CkptHeader)) + (h.acc_scale > 0.0 ? nvox * static_cast<s64>(sizeof(u16)) : 0);
}

// Quantize `acc` (nvox floats) into `buf` (nvox u16s) at scale `inv`, in parallel. Snapshot step for
// ckpt_save — runs on the consumer thread (the only mutator of `acc`), so the accumulator can't be
// torn; the resulting `buf` is then handed to a background writer that no longer touches `acc`.
inline void ckpt_quantize(const float* acc, s64 nvox, float inv, u16* buf) {
    parallel_for(0, nvox, [&](s64 i) {
        buf[static_cast<usize>(i)] = static_cast<u16>(acc[static_cast<usize>(i)] * inv + 0.5f);
    });
}

// Parallel max-reduction over the accumulator (the scale for u16 quantization).
inline float ckpt_max(const float* acc, s64 nvox) {
    constexpr s64 kChunk = 1 << 16;
    const s64 nchunks = (nvox + kChunk - 1) / kChunk;
    std::vector<float> partial(static_cast<usize>(nchunks), 0.0f);
    parallel_for(0, nchunks, [&](s64 c) {
        const s64 i0 = c * kChunk, i1 = std::min(nvox, i0 + kChunk);
        float mx = 0.0f;
        for (s64 i = i0; i < i1; ++i) mx = std::max(mx, acc[static_cast<usize>(i)]);
        partial[static_cast<usize>(c)] = mx;
    });
    float mx = 0.0f;
    for (float p : partial) mx = std::max(mx, p);
    return mx;
}

// Write header + u16 payload (already-quantized snapshot, NOT the live accumulator — safe to call
// from a background thread) atomically (temp+rename). Best-effort: a failed checkpoint must never
// abort the run.
inline void ckpt_write(const std::string& path, const CkptHeader& h, const u16* buf, s64 nvox) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return;
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1;
    if (ok && h.acc_scale > 0.0) {
        constexpr s64 kChunk = 1 << 20;
        for (s64 i = 0; i < nvox && ok; i += kChunk) {
            const s64 m = std::min(kChunk, nvox - i);
            ok = std::fwrite(buf + i, sizeof(u16), static_cast<std::size_t>(m), f) == static_cast<std::size_t>(m);
        }
    }
    ok = (std::fflush(f) == 0) && ok;
    std::fclose(f);
    if (ok) std::rename(tmp.c_str(), path.c_str());
    else std::remove(tmp.c_str());
}

// Double-buffered async checkpoint writer: `save()` snapshots the (parallel) max-scan + quantize
// inline on the calling (consumer/GPU) thread — that part must be synchronous, since the accumulator
// is only safe to read while the caller isn't concurrently scattering into it — then hands the
// snapshot to a background thread for the slow fwrite, so the GPU pipeline resumes immediately after
// the quantize instead of blocking on multi-GB disk IO. At most one write is in flight; a save
// requested while the previous one is still writing is DROPPED (the next cadence tick will catch up)
// rather than blocking — checkpointing is best-effort, never a stall source.
struct CkptWriter {
    std::thread worker;
    std::mutex m;
    bool busy = false;
    std::vector<u16> buf;  // snapshot buffer, reused across saves (allocated lazily on first use)

    ~CkptWriter() { join(); }

    void join() {
        std::unique_lock<std::mutex> lk(m);
        if (worker.joinable()) { lk.unlock(); worker.join(); lk.lock(); }
    }

    // Snapshot + quantize `acc` inline (parallel, must run before this returns — `acc` is about to be
    // mutated again by the caller), then write it out on a background thread. If a previous write is
    // still in flight, this save is skipped (best-effort cadence, never blocks the pipeline).
    void save(const std::string& path, CkptHeader h, const float* acc, s64 nvox) {
        std::unique_lock<std::mutex> lk(m);
        if (busy) return;  // previous checkpoint still writing — skip this tick, catch up next cadence
        if (worker.joinable()) { lk.unlock(); worker.join(); lk.lock(); }
        lk.unlock();

        const float mx = ckpt_max(acc, nvox);
        h.acc_scale = mx > 0.0f ? static_cast<double>(mx) : 0.0;
        if (h.acc_scale > 0.0) {
            if (buf.size() != static_cast<usize>(nvox)) buf.assign(static_cast<usize>(nvox), 0);
            const float inv = static_cast<float>(65535.0 / h.acc_scale);
            ckpt_quantize(acc, nvox, inv, buf.data());
        }

        lk.lock();
        busy = true;
        lk.unlock();
        worker = std::thread([this, path, h, nvox] {
            ckpt_write(path, h, buf.data(), nvox);
            std::lock_guard<std::mutex> lk2(m);
            busy = false;
        });
    }
};

// Try to load a checkpoint that MATCHES the current run config (dims/patch/overlap/tta/tile_shift AND
// identity: weights/input/task — see CkptHeader). Returns n_done and fills `acc` on match; on ANY
// failure (absent file, mismatch, short/corrupt read) `acc` is left FULLY ZEROED — never partially
// populated — and -1 is returned, so the caller can safely treat -1 as "start fresh from a clean
// accumulator" without checking a separate flag.
inline long ckpt_load(const std::string& path, const CkptHeader& want, float* acc, s64 nvox) {
    auto zero_and_fail = [&] {
        parallel_for(0, nvox, [&](s64 i) { acc[static_cast<usize>(i)] = 0.0f; });
        return -1L;
    };
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return -1;  // no checkpoint — acc is already zeroed by the caller (Volume::zeros); nothing to do
    CkptHeader h;
    const bool header_ok = std::fread(&h, sizeof h, 1, f) == 1;
    const bool matches = header_ok &&
        std::memcmp(h.magic, want.magic, 8) == 0 && h.dz == want.dz && h.dy == want.dy && h.dx == want.dx &&
        h.patch == want.patch && h.overlap == want.overlap && h.tta == want.tta &&
        h.tile_shift == want.tile_shift && h.total_tiles == want.total_tiles &&
        h.weights_hash == want.weights_hash && h.input_hash == want.input_hash &&
        h.channel == want.channel && h.sigmoid == want.sigmoid && h.norm == want.norm &&
        h.n_done >= 0 && h.n_done <= h.total_tiles;
    if (!matches) {
        std::fclose(f);
        if (header_ok && std::memcmp(h.magic, want.magic, 8) != 0)
            fenix::log(LogLevel::warn, "predict: checkpoint magic/version mismatch — starting fresh: {}", path);
        else if (header_ok)
            fenix::log(LogLevel::warn,
                       "predict: checkpoint found but doesn't match this run (dims/weights/input/task) — "
                       "starting fresh: {}", path);
        return zero_and_fail();  // acc is caller's Volume<f32>::zeros — but zero explicitly, don't assume
    }
    // Validate on-disk size BEFORE writing a single value into `acc` — a short/truncated file (partial
    // fsync, disk-full, crash mid-write despite the temp+rename) must never partially pollute acc.
    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return zero_and_fail(); }
    const long fsize = std::ftell(f);
    if (fsize < 0 || static_cast<s64>(fsize) < ckpt_expected_size(h, nvox)) {
        std::fclose(f);
        fenix::log(LogLevel::warn, "predict: checkpoint is truncated/corrupt — starting fresh: {}", path);
        return zero_and_fail();
    }
    std::fseek(f, sizeof(CkptHeader), SEEK_SET);

    long n = -1;
    if (h.acc_scale <= 0.0) {  // all-zero accumulator (very early checkpoint) — size check already passed
        parallel_for(0, nvox, [&](s64 i) { acc[static_cast<usize>(i)] = 0.0f; });
        n = h.n_done;
    } else {
        const float sc = static_cast<float>(h.acc_scale / 65535.0);
        std::vector<u16> buf(static_cast<usize>(nvox));
        const bool ok = std::fread(buf.data(), sizeof(u16), static_cast<std::size_t>(nvox), f) ==
                        static_cast<std::size_t>(nvox);
        if (ok) {
            parallel_for(0, nvox, [&](s64 i) { acc[static_cast<usize>(i)] = static_cast<float>(buf[static_cast<usize>(i)]) * sc; });
            n = h.n_done;
        } else {
            fenix::log(LogLevel::warn, "predict: checkpoint payload read failed — starting fresh: {}", path);
        }
    }
    std::fclose(f);
    return n >= 0 ? n : zero_and_fail();
}
}  // namespace detail

// Run sliding-window segmentation inference. Returns a probability volume (same dims as `in`).
// Core sliding-window predict, templated on a PATCH FILLER: `void fill(s64 z0,s64 y0,s64 x0,int P,float* out)`
// which writes a P³ ZYX-contiguous patch at (z0,y0,x0), edge-clamped, into `out`. This lets the input come
// from a dense view OR stream from a native-u8 .fxvol block cache (block-batched — no dense f32 volume). The
// filler owns its own parallelism/locking; the net forward + Gaussian-blended accumulate are unchanged.
template <class Filler, class Net>
inline Expected<Volume<f32>> predict_surface_filled(Extent3 d, Filler&& fill, Net& net,
                                                    torch::Device dev, const InferOptions& opt = {}) {
    if (opt.patch % 64 != 0) return fenix::err(Errc::invalid_argument, "patch must be divisible by 64");
    const int P = opt.patch;

    Volume<f32> prob = Volume<f32>::zeros(d);
    auto pv = prob.view();

    const auto gz = gaussian1d(P), gy = gaussian1d(P), gx = gaussian1d(P);
    // Tile-offset TTA prepends a shifted origin tile (0 → tile_shift) so this grid's seams fall in
    // different places than the unshifted grid; the shifted-origin tile's read is edge-clamped by the
    // filler, and the Gaussian window down-weights the clamped border. The far edge stays covered
    // (tile_starts always clamps its last start to dim-P).
    auto shifted_starts = [&](s64 dim) {
        auto s = tile_starts(dim, P, opt.overlap);
        if (opt.tile_shift > 0 && dim > P) {
            const s64 sh = std::min<s64>(opt.tile_shift, dim - P);
            if (std::find(s.begin(), s.end(), sh) == s.end()) s.insert(s.begin(), sh);
        }
        return s;
    };
    const auto zs = shifted_starts(d.z);
    const auto ys = shifted_starts(d.y);
    const auto xs = shifted_starts(d.x);

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

    // Resume: if a matching checkpoint exists, preload the accumulator and start after its completed tiles.
    detail::CkptHeader ckpt_hdr;
    ckpt_hdr.dz = d.z; ckpt_hdr.dy = d.y; ckpt_hdr.dx = d.x; ckpt_hdr.patch = P;
    ckpt_hdr.overlap = opt.overlap; ckpt_hdr.tta = opt.tta; ckpt_hdr.tile_shift = opt.tile_shift;
    ckpt_hdr.total_tiles = total;
    ckpt_hdr.weights_hash = opt.weights_hash; ckpt_hdr.input_hash = opt.input_hash;
    ckpt_hdr.channel = opt.channel; ckpt_hdr.sigmoid = opt.sigmoid ? 1 : 0; ckpt_hdr.norm = static_cast<int>(opt.norm);
    long resume_from = 0;
    if (!opt.ckpt_path.empty()) {
        const long got = detail::ckpt_load(opt.ckpt_path, ckpt_hdr, prob.data(), d.count());
        if (got > 0) {
            resume_from = got;  // the first `got` tiles are already in `prob`
            fenix::log(LogLevel::info, "predict: RESUMING from checkpoint — {}/{} tiles done", got, total);
        }
    }
    const bool ckpt_on = !opt.ckpt_path.empty();
    detail::CkptWriter ckpt_writer;  // owns the background write thread; joined on scope exit / explicit join()
    auto save_ckpt = [&](long n_done) {
        ckpt_hdr.n_done = n_done;
        // Snapshot (max-scan + quantize) runs inline here, parallelized, on the consumer thread that owns
        // `prob` — it must finish before this call returns since the caller resumes scattering into `prob`
        // right after. Only the slow fwrite is backgrounded.
        ckpt_writer.save(opt.ckpt_path, ckpt_hdr, prob.data(), d.count());
    };

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
        // Noise-injection TTA: add i.i.d. Gaussian noise post-normalize (so sigma is in z-scored units).
        // Seeded from the member seed + tile position → deterministic/reproducible, distinct per patch.
        // Box-Muller from two uniforms; sigma is small so a light [-6,6]-clamped log is safe.
        if (opt.noise_apply && opt.noise_sigma > 0.0) {
            const u64 seed = opt.noise_member_seed ^ hash_value(static_cast<u64>(t.z0) * 0x9E3779B97F4A7C15ull +
                             static_cast<u64>(t.y0) * 0xC2B2AE3D27D4EB4Full + static_cast<u64>(t.x0));
            Pcg32 rng(seed);
            const float sg = static_cast<float>(opt.noise_sigma);
            for (std::size_t i = 0; i < PN; ++i) {
                const float u1 = std::max(rng.next_f32(), 1e-7f), u2 = rng.next_f32();
                const float g = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
                out[i] += sg * g;
            }
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
        // `stop` aborts BOTH threads on any error (producer-side alloc/prep throw, or consumer-side torch
        // throw e.g. CUDA OOM). libtorch is compiled WITH exceptions in FENIX_ML builds (see ml/CLAUDE.md
        // build firewall); this loop is the exception/thread-lifetime boundary — nothing may escape past
        // here uncaught, since an exception unwinding through a joinable std::thread is std::terminate.
        bool stop = false;
        std::string err_msg;
        auto set_error = [&](const std::string& msg) {
            std::lock_guard<std::mutex> lk(m);
            if (!stop) { stop = true; err_msg = msg; }
            cv_free.notify_one();  // wake the producer if it's waiting on a free slot
        };
        std::thread producer([&] {
            try {
                for (long i0 = resume_from; i0 < total; i0 += B) {
                    const int nb = static_cast<int>(std::min<long>(B, total - i0));
                    std::unique_lock<std::mutex> lk(m);
                    cv_free.wait(lk, [&] { return filled < kSlots || stop; });
                    if (stop) return;
                    const int s = tail;
                    lk.unlock();
                    for (int b = 0; b < nb; ++b) prep(tiles[static_cast<std::size_t>(i0 + b)], ring[s].data() + PN * static_cast<std::size_t>(b));
                    lk.lock();
                    if (stop) return;
                    slot_i0[s] = static_cast<int>(i0); slot_nb[s] = nb;
                    tail = (tail + 1) % kSlots; ++filled;
                    cv_filled.notify_one();
                }
            } catch (const std::exception& e) {
                set_error(std::string("predict: producer (patch prep) failed: ") + e.what());
                cv_filled.notify_one();  // wake the consumer so it observes stop
                return;
            } catch (...) {
                set_error("predict: producer (patch prep) failed: unknown exception");
                cv_filled.notify_one();
                return;
            }
            std::lock_guard<std::mutex> lk(m);
            prod_done = true; cv_filled.notify_one();
        });
        // Drain the producer/join it cleanly on any exit path (success, early break on stop, or an
        // exception caught below) — never let `producer` unwind while still joinable.
        auto drain_and_join = [&] {
            { std::lock_guard<std::mutex> lk(m); stop = true; }
            cv_free.notify_one();
            producer.join();
        };
        try {
            for (;;) {
                std::unique_lock<std::mutex> lk(m);
                cv_filled.wait(lk, [&] { return filled > 0 || prod_done || stop; });
                if (stop) break;
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
                // fp16 forward can overflow (|x|>65504) into inf logits -> softmax(inf-inf) = NaN; scrub on
                // the GPU (libtorch's own ops are IEEE-safe regardless of this project's -ffast-math) before
                // it Gaussian-scatters into every voxel of this tile AND every overlapping neighbor.
                pr = torch::nan_to_num(pr, /*nan=*/0.0, /*posinf=*/1.0, /*neginf=*/0.0);
                auto surf = pr.contiguous().to(torch::kCPU).to(torch::kFloat32).contiguous();  // [nb,P,P,P]
                if (prof) { t_fwd += clk() - tp; tp = clk(); }
                const float* base = surf.template data_ptr<float>();
                for (int b = 0; b < nb; ++b) scatter(tiles[static_cast<std::size_t>(i0 + b)], base + PN * static_cast<std::size_t>(b));
                if (prof) t_scat += clk() - tp;
                n_tiles += nb;
                const long done = resume_from + n_tiles;
                // Checkpoint after this batch's scatter is fully applied (consumer scatters in i0 order, so
                // `done` tiles are exactly the accumulator's contents). Save cadence, not per-tile (O(voxels)).
                // On error below we also force a save so already-scattered tiles survive without waiting
                // for the next cadence tick.
                if (ckpt_on && (n_tiles % opt.ckpt_every < nb)) save_ckpt(done);
                if (n_tiles % 25 < nb || done == total)
                    fenix::log(LogLevel::info, "predict-surface: tile {}/{}", done, total);
            }
        } catch (const std::exception& e) {
            set_error(std::string("predict: forward pass failed: ") + e.what());
        } catch (...) {
            set_error("predict: forward pass failed: unknown exception");
        }
        drain_and_join();
        // drain_and_join sets `stop` unconditionally (that's the producer's shutdown signal), so
        // `stop` does NOT discriminate failure — a non-empty err_msg (set_error ran) does.
        if (!err_msg.empty()) {
            // Preserve whatever work made it into the accumulator before the failure — a resumable
            // checkpoint means the run can pick back up instead of losing everything since the last
            // cadence save.
            if (ckpt_on) save_ckpt(resume_from + n_tiles);
            return fenix::err(Errc::internal, err_msg);
        }
        if (prof)
            fenix::log(LogLevel::info, "profile: fwd(gpu)={:.1f}s scatter(cpu)={:.1f}s (prep overlapped)", t_fwd, t_scat);
        normalize();
        // Join any in-flight background write BEFORE removing the file, or the writer can rename its
        // snapshot into existence right after the remove (a stale checkpoint reappearing post-success).
        if (ckpt_on) { ckpt_writer.join(); std::remove(opt.ckpt_path.c_str()); }
        return prob;
    }

    // ---- SERIAL single-patch path (B==1, also the ONLY path when tta>1) ----
    // No producer thread here, but torch calls (forward/permute/flip/softmax) still throw on e.g. CUDA
    // OOM; uncaught, that unwinds into -fno-exceptions driver frames -> std::terminate regardless of the
    // lack of a joinable thread. Same exception boundary as the batched path above.
    std::vector<float> batchbuf(PN);
    n_tiles = resume_from;
    try {
        for (long i0 = resume_from; i0 < total; i0 += B) {
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
                    // See the batched path's comment: scrub fp16-overflow NaN/inf before it fuses into acc.
                    pr = torch::nan_to_num(pr, /*nan=*/0.0, /*posinf=*/1.0, /*neginf=*/0.0);
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
            // Checkpoint after this tile's scatter (serial → n_tiles is exactly the accumulator's contents).
            if (ckpt_on && ((n_tiles - resume_from) % opt.ckpt_every == 0)) save_ckpt(n_tiles);
            if (n_tiles % 25 < nb || n_tiles == total)
                fenix::log(LogLevel::info, "predict-surface: tile {}/{}", n_tiles, total);
        }
    } catch (const std::exception& e) {
        if (ckpt_on) save_ckpt(n_tiles);  // preserve progress so far — resumable on the next run
        return fenix::err(Errc::internal, std::string("predict: forward pass failed: ") + e.what());
    } catch (...) {
        if (ckpt_on) save_ckpt(n_tiles);
        return fenix::err(Errc::internal, "predict: forward pass failed: unknown exception");
    }
    (void)t_prep;

    normalize();
    if (ckpt_on) { ckpt_writer.join(); std::remove(opt.ckpt_path.c_str()); }
    return prob;
}

// Dense-view entry point (NRRD inputs, tests): fill each patch from RAM with edge clamp, parallel over z.
template <class Net>
inline Expected<Volume<f32>> predict_surface(VolumeView<const f32> in, Net& net,
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
template <class Net>
inline Expected<Volume<f32>> predict_surface_scales(VolumeView<const f32> in, Net& net,
                                                    torch::Device dev, const InferOptions& opt) {
    if (opt.scales.empty()) return predict_surface(in, net, dev, opt);
    const Extent3 d = in.dims();
    Volume<f32> acc = Volume<f32>::zeros(d);
    auto av = acc.view();
    InferOptions inner = opt;
    inner.scales.clear();  // inner predict is single-scale (still octahedral-TTA'd via inner.tta)
    inner.ckpt_path.clear();  // per-member calls don't checkpoint (resume applies to the whole ensemble, not implemented for it)
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

// Rotate an f32 volume about the z axis by `deg` degrees (GPU trilinear grid_sample, zero-fill outside,
// same dims). torch grid coords are (x,y,z)-ordered in the grid's last dim; a z-axis rotation is a
// rotation in the (x,y) plane. theta maps OUTPUT normalized coords to INPUT coords, so pass -deg to
// undo a +deg rotation.
inline Volume<f32> rotate_z_f32(VolumeView<const f32> in, double deg, torch::Device dev) {
    const Extent3 d = in.dims();
    Volume<f32> src(d);
    { auto sv = src.view(); parallel_for(0, d.z, [&](s64 z) { for (s64 y = 0; y < d.y; ++y) for (s64 x = 0; x < d.x; ++x) sv(z, y, x) = in(z, y, x); }); }
    torch::NoGradGuard ng;
    const double r = deg * 3.14159265358979323846 / 180.0;
    const float c = static_cast<float>(std::cos(r)), s = static_cast<float>(std::sin(r));
    auto t = torch::from_blob(src.data(), {1, 1, d.z, d.y, d.x}, torch::kFloat32).to(dev);
    auto theta = torch::tensor({c, -s, 0.f, 0.f,
                                s, c, 0.f, 0.f,
                                0.f, 0.f, 1.f, 0.f},
                               torch::TensorOptions().device(dev))
                     .reshape({1, 3, 4});
    auto grid = torch::nn::functional::affine_grid(theta, {1, 1, d.z, d.y, d.x}, /*align_corners=*/false);
    auto o = torch::nn::functional::grid_sample(
                 t, grid,
                 torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false))
                 .to(torch::kCPU).contiguous();
    Volume<f32> vo(d);
    const float* op = o.data_ptr<float>();
    std::span<f32> vf = vo.flat();
    parallel_for(0, d.count(), [&](s64 i) { vf[static_cast<usize>(i)] = op[i]; });
    return vo;
}

// Arbitrary-rotation TTA about z: for each angle predict on the rotated input, rotate the prob back,
// and validity-weighted mean-fuse (validity = ones pushed through the same rotate+unrotate, so the
// corner regions the rotation clips don't dilute the mean). Angle 0 contributes the exact unrotated
// member with weight 1. Wraps predict_surface_scales, so `scales` and octahedral `tta` still apply
// per member. Empty rots => plain scales path (byte-identical).
template <class Net>
inline Expected<Volume<f32>> predict_surface_rots(VolumeView<const f32> in, Net& net,
                                                  torch::Device dev, const InferOptions& opt) {
    if (opt.rots.empty()) return predict_surface_scales(in, net, dev, opt);
    const Extent3 d = in.dims();
    Volume<f32> acc = Volume<f32>::zeros(d);
    std::vector<float> wsum(static_cast<usize>(d.count()), 0.0f);
    auto av = acc.view();
    InferOptions inner = opt;
    inner.rots.clear();
    inner.ckpt_path.clear();
    // Every non-zero angle's validity mask is ~0 in the corners the rotation clips; without an
    // unrotated (angle 0) member those corners have wsum≈0 and would silently normalize to
    // probability 0 — fabricated "no surface" rather than the absent-vs-failed distinction this
    // project requires. Inject 0 if the caller didn't include it (weight-1 exact member covers
    // every voxel, so corners always get a real vote).
    std::vector<double> rots = opt.rots;
    if (std::find(rots.begin(), rots.end(), 0.0) == rots.end()) rots.insert(rots.begin(), 0.0);
    int done = 0;
    for (double a : rots) {
        Expected<Volume<f32>> prob = fenix::err(Errc::unsupported, "unset");
        Volume<f32> valid;
        if (a == 0.0) {
            prob = predict_surface_scales(in, net, dev, inner);
        } else {
            Volume<f32> in_r = rotate_z_f32(in, a, dev);
            auto p = predict_surface_scales(in_r.view(), net, dev, inner);
            if (!p) return std::unexpected(p.error());
            prob = rotate_z_f32(p->view(), -a, dev);
            Volume<f32> ones(d);
            { std::span<f32> of = ones.flat(); parallel_for(0, d.count(), [&](s64 i) { of[static_cast<usize>(i)] = 1.0f; }); }
            valid = rotate_z_f32(rotate_z_f32(ones.view(), a, dev).view(), -a, dev);
        }
        if (!prob) return std::unexpected(prob.error());
        auto pv = prob->view();
        if (a == 0.0) {
            parallel_for_z(d, [&](s64 z) {
                for (s64 y = 0; y < d.y; ++y)
                    for (s64 x = 0; x < d.x; ++x) {
                        av(z, y, x) += pv(z, y, x);
                        wsum[static_cast<usize>(av.offset(z, y, x))] += 1.0f;
                    }
            });
        } else {
            auto vv = valid.view();
            parallel_for_z(d, [&](s64 z) {
                for (s64 y = 0; y < d.y; ++y)
                    for (s64 x = 0; x < d.x; ++x) {
                        const float w = vv(z, y, x);
                        av(z, y, x) += w * pv(z, y, x);
                        wsum[static_cast<usize>(av.offset(z, y, x))] += w;
                    }
            });
        }
        ++done;
        fenix::log(LogLevel::info, "rot-tta: angle {:.4g}deg done ({}/{})", a, done, static_cast<int>(rots.size()));
    }
    parallel_for_z(d, [&](s64 z) {
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const float w = wsum[static_cast<usize>(av.offset(z, y, x))];
                if (w > 1e-6f) av(z, y, x) /= w;
            }
    });
    return acc;
}

// Top-level TTA ensemble: geometric (rots/scales/octahedral via predict_surface_rots) × NOISE members
// × TILE-OFFSET members. Noise and tile-offset produce full-volume probs (no clipped corners), so they
// mean-fuse plainly. Member 0 is always the clean, unshifted geometric prediction. The step used for
// offset shifts is patch*(1-overlap). Empty noise+offsets => plain predict_surface_rots (byte-identical).
template <class Net>
inline Expected<Volume<f32>> predict_surface_tta(VolumeView<const f32> in, Net& net,
                                                 torch::Device dev, const InferOptions& opt) {
    if (opt.noise <= 0 && opt.offsets <= 0) return predict_surface_rots(in, net, dev, opt);
    const Extent3 d = in.dims();
    Volume<f32> acc = Volume<f32>::zeros(d);
    auto av = acc.view();
    int members = 0;

    auto add = [&](const Volume<f32>& p) {
        auto pv = p.view();
        parallel_for_z(d, [&](s64 z) { for (s64 y = 0; y < d.y; ++y) for (s64 x = 0; x < d.x; ++x) av(z, y, x) += pv(z, y, x); });
        ++members;
    };

    // Member 0: clean geometric prediction.
    // Ensemble members don't checkpoint individually (resume is per-whole-prediction, not implemented
    // across the noise/offset ensemble) — clear the path on every inner call.
    InferOptions base = opt; base.ckpt_path.clear();
    { auto p = predict_surface_rots(in, net, dev, base); if (!p) return std::unexpected(p.error()); add(*p); }

    // Noise members: same input, distinct noise seed each.
    for (int k = 0; k < opt.noise; ++k) {
        InferOptions o = base; o.noise = 0; o.offsets = 0;
        o.noise_apply = true;
        o.noise_member_seed = hash_value(static_cast<u64>(0x51ED270B + k));
        auto p = predict_surface_rots(in, net, dev, o);
        if (!p) return std::unexpected(p.error());
        add(*p);
        fenix::log(LogLevel::info, "noise-tta: member {}/{} done", k + 1, opt.noise);
    }

    // Tile-offset members: shift the grid origin by fractions of the step.
    const s64 step = std::max<s64>(1, static_cast<s64>(opt.patch * (1.0 - opt.overlap)));
    for (int k = 0; k < opt.offsets; ++k) {
        InferOptions o = base; o.noise = 0; o.offsets = 0;
        o.tile_shift = step * (k + 1) / (opt.offsets + 1);  // spread shifts across (0, step)
        if (o.tile_shift == 0) continue;
        auto p = predict_surface_rots(in, net, dev, o);
        if (!p) return std::unexpected(p.error());
        add(*p);
        fenix::log(LogLevel::info, "offset-tta: member {}/{} (shift {}) done", k + 1, opt.offsets, o.tile_shift);
    }

    const float inv = 1.0f / static_cast<float>(members);
    parallel_for_z(d, [&](s64 z) { for (s64 y = 0; y < d.y; ++y) for (s64 x = 0; x < d.x; ++x) av(z, y, x) *= inv; });
    return acc;
}

}  // namespace fenix::ml
