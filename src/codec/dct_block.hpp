// codec/dct_block.hpp — the DCT-16 lossy codec, fenix's SOLE transform codec (the CDF 9/7 wavelet was
// retired, ADR 0005). Pipeline per 16³ block: widen→f32 → subtract block DC → separable float DCT-16 →
// frequency-weighted dead-zone quant → magnitude-category rANS with a causal neighbour-magnitude-sum
// context + raw mantissa/sign bits + end-of-block. Blocks are coded in TILES of bpa³ (a 64³ tile =
// 4³=64 blocks) that SHARE one set of rANS tables (the big ratio win). A rewrite of matter-compressor's
// mc_codec_float (the freq-weighted step, dead-zone, centroid dequant are theirs; the entropy stage is
// fenix's rANS — SIMD/GPU-amenable). All-float compute. See codec/CLAUDE.md.
#pragma once

#include "codec/dct.hpp"
#include "codec/entropy.hpp"  // detail::{encode_plane,decode_plane,BitWriter,BitReader,put_u32,get_u32,put_var,get_var}
#include "codec/dtype.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <span>
#include <utility>
#include <vector>

namespace fenix::codec {

struct DctParams {
    f32 q = 8.0f;         // base quantization step (smaller = higher fidelity / larger)
    f32 hf_exp = 0.65f;   // frequency weighting: step(cz,cy,cx) = q·(1+cz+cy+cx)^hf_exp (coarser HF)
    f32 dz_frac = 0.80f;  // dead-zone width as a fraction of step (truncate-to-zero below dz)
    bool rdoq = true;     // rate-distortion optimized quantization (encoder-only; bitstream unchanged)
    f32 rdoq_lambda = 0.15f;  // RDOQ Lagrange multiplier: λ = rdoq_lambda·step² (calibrated: RD-positive,
                              // no near-lossless regression, on both datasets; 0.20+ over-trims at low q)
    // Decode-time anti-blocking (NON-NORMATIVE — derived from the quant step, no bitstream change).
    bool deblock = true;       // quant-gated 3D deblock of the 16³ block-boundary seams (decode-only)
    f32 deblock_beta = 2.5f;   // flatness gate: filter a seam only if its 2nd-difference budget < beta·q
    f32 deblock_tc = 0.5f;     // clamp: move a boundary sample by at most tc·q (so only quant seams move)
    bool dc_predict = true;    // predict each block's DC level from its causal neighbour blocks in the tile
};

// DCT significance context, clustered. A coefficient's RAW context is (frequency-band bucket) ×
// (neighbour-magnitude-sum bucket) — richer than a bare sum (a level-9 neighbour predicts a large
// coefficient far better than a significance count, and low/high frequency bands have very different
// stats). Per tile the raw contexts are greedily MERGED (context-map clustering, JPEG XL / Brotli) into
// <= kDctMaxClusters rANS tables, with the table-signalling cost IN the merge objective — so adding
// contexts can't fragment: sparse contexts auto-collapse, well-populated ones stay split. The
// kDctRawCtx-byte context map is stored per tile (negligible, amortized over 64 blocks).
inline constexpr u8 kDctMagCap = 3;        // neighbour magnitude cap (the sum-context input)
inline constexpr int kDctSumBuckets = 6;   // neighbour-magnitude-sum buckets
inline constexpr int kDctBandBuckets = 4;  // frequency-band buckets (from cz+cy+cx)
inline constexpr int kDctRawCtx = kDctSumBuckets * kDctBandBuckets;  // raw contexts before clustering
inline constexpr int kDctMaxClusters = 8;  // rANS tables per tile after clustering
inline constexpr int kDctCat = 18;         // magnitude-category alphabet (bit_width 0..~16 + margin)

namespace detail {
inline f32 dct_step(const DctParams& p, int cz, int cy, int cx) {
    return p.q * std::pow(1.0f + static_cast<f32>(cz + cy + cx), p.hf_exp);
}

// Frequency-ascending scan over a kDctN³ block: coefficient indices sorted by (cz+cy+cx), so the
// low-frequency (high-energy) coefficients come first and the trailing high-frequency zeros come last
// — enabling an end-of-block cutoff (code only up to the last nonzero, skip the rest). Stable within a
// frequency shell. Key property: a coefficient's 3D causal neighbours (z-1,y-1,x-1) each have
// frequency-sum one LESS, so they're always earlier in this order — the spatial significance context
// stays causal exactly as in raster order. Each entry bakes in the flat index, the frequency sum
// (to index a per-block step table, avoiding a per-coefficient std::pow), and the causal-neighbour
// validity mask (avoiding per-coefficient z/y/x division).
struct ScanEntry { u16 idx; u8 fsum; u8 edge; };  // edge bits: 1=z>0, 2=y>0, 4=x>0
inline const std::array<ScanEntry, kDctN * kDctN * kDctN>& dct_scan_order() {
    static const std::array<ScanEntry, kDctN * kDctN * kDctN> s = [] {
        std::array<ScanEntry, kDctN * kDctN * kDctN> a{};
        for (u32 i = 0; i < a.size(); ++i) {
            const int z = static_cast<int>(i) / (kDctN * kDctN), y = (static_cast<int>(i) / kDctN) % kDctN, x = static_cast<int>(i) % kDctN;
            a[i] = {static_cast<u16>(i), static_cast<u8>(z + y + x),
                    static_cast<u8>((z > 0 ? 1 : 0) | (y > 0 ? 2 : 0) | (x > 0 ? 4 : 0))};
        }
        std::stable_sort(a.begin(), a.end(), [](const ScanEntry& p, const ScanEntry& q) { return p.fsum < q.fsum; });
        return a;
    }();
    return s;
}
// Per-block step table: step(cz,cy,cx) depends only on the frequency sum f=cz+cy+cx (0..3(N-1)), so we
// compute the (few) std::pow values once per block instead of once per coefficient.
inline void dct_step_table(const DctParams& p, f32* stepv) {
    for (int f = 0; f < 3 * kDctN; ++f) stepv[f] = p.q * std::pow(1.0f + static_cast<f32>(f), p.hf_exp);
}

// Predict a per-block side-info value (DC level, or significant-coefficient count nsig) from the mean of
// its already-coded causal neighbour blocks within the tile. The volume is smooth, so both block means
// and block "busyness" vary slowly → coding the residual shrinks the DC + nsig sections (which the
// payload breakdown showed grow to several % each at high q). Causal in the raster block order
// b = (bz·bpa+by)·bpa+bx, so the decoder reproduces the same prediction; no cross-tile reference, so
// tiles stay independently decodable (random access preserved).
template <class V>
inline s32 causal_block_pred(const V* a, s64 bz, s64 by, s64 bx, s64 bpa) {
    auto at = [&](s64 z, s64 y, s64 x) { return static_cast<s64>(a[(z * bpa + y) * bpa + x]); };
    s64 sum = 0;
    int n = 0;
    if (bx > 0) { sum += at(bz, by, bx - 1); ++n; }
    if (by > 0) { sum += at(bz, by - 1, bx); ++n; }
    if (bz > 0) { sum += at(bz - 1, by, bx); ++n; }
    return n ? static_cast<s32>(sum / n) : 0;
}

// Raw context of a coefficient: frequency band (from the scan entry's fsum) × neighbour-magnitude-sum.
inline int dct_raw_ctx(u8 fsum, int nsum) {
    const int band = std::min(kDctBandBuckets - 1, static_cast<int>(fsum) / 4);
    return band * kDctSumBuckets + std::min(kDctSumBuckets - 1, nsum);
}

// Gather + DC-remove + DCT + dead-zone quantize one block into `lev` (raster order), and stash the DCT
// coefficients into `coef` (for RDOQ's distortion term). Returns (dc_level, nsig = last-nonzero+1 in
// frequency-scan order). No entropy — that's a tile-level pass.
template <class T>
inline std::pair<s32, u32> quant_one_block(std::span<const T> src, s64 side, s64 oz, s64 oy, s64 ox,
                                           const DctParams& params, const f32* stepv, s32* lev, f32* coef) {
    constexpr s64 N = kDctN, V = N * N * N;
    std::vector<T> blk(static_cast<usize>(V));
    for (s64 z = 0; z < N; ++z)
        for (s64 y = 0; y < N; ++y)
            for (s64 x = 0; x < N; ++x)
                blk[static_cast<usize>((z * N + y) * N + x)] =
                    src[static_cast<usize>(((oz + z) * side + (oy + y)) * side + (ox + x))];
    std::vector<f32> c = to_f32<T>(blk);
    f64 msum = 0;
    for (f32 v : c) msum += static_cast<f64>(v);
    const f32 dc_step = params.q * 0.0625f;  // DC quantized 16x finer than the lowest AC step
    const s32 dc_level = static_cast<s32>(std::lround(static_cast<f32>(msum / static_cast<f64>(V)) / dc_step));
    for (f32& v : c) v -= static_cast<f32>(dc_level) * dc_step;
    dct16_3d_fwd(c);
    for (s64 z = 0; z < N; ++z)
        for (s64 y = 0; y < N; ++y)
            for (s64 x = 0; x < N; ++x) {
                const usize i = static_cast<usize>((z * N + y) * N + x);
                const f32 step = stepv[z + y + x], a0 = std::abs(c[i]);
                coef[i] = c[i];
                s32 level = 0;
                if (a0 >= params.dz_frac * step)
                    level = (c[i] < 0 ? -1 : 1) * static_cast<s32>(std::floor(a0 / step + (1.0f - params.dz_frac)));
                lev[i] = std::clamp(level, -32768, 32767);
            }
    const auto& scan = dct_scan_order();
    u32 nsig = 0;
    for (usize pos = static_cast<usize>(V); pos-- > 0;)
        if (lev[scan[pos].idx] != 0) { nsig = static_cast<u32>(pos + 1); break; }
    return {dc_level, nsig};
}

// Greedy context-map clustering: merge raw-context category histograms (with the per-table signalling
// cost IN the objective) down to <= kDctMaxClusters. Fills cluster_of[raw] (0..K-1), returns K. Dead
// (empty) raw contexts map to 0 — their slot is never read at decode (the same coefficients reproduce
// the same raw contexts), so the choice is don't-care.
inline int cluster_contexts(const std::array<std::array<u32, kDctCat>, kDctRawCtx>& hist,
                            std::array<u8, kDctRawCtx>& cluster_of) {
    auto cost = [](const std::array<u32, kDctCat>& h) -> f64 {
        f64 n = 0;
        int nused = 0;
        for (u32 c : h) { n += c; if (c) ++nused; }
        if (n == 0) return 0.0;
        f64 ent = 0;
        for (u32 c : h) if (c) ent += static_cast<f64>(c) * std::log2(n / static_cast<f64>(c));
        return ent + static_cast<f64>(nused) * 16.0 + 24.0;  // ~table bits: 2 B/used-symbol + size fields
    };
    std::array<std::array<u32, kDctCat>, kDctRawCtx> ch = hist;
    std::array<f64, kDctRawCtx> ccost{};
    std::array<bool, kDctRawCtx> alive{};
    int nactive = 0;
    for (int r = 0; r < kDctRawCtx; ++r) {
        cluster_of[r] = static_cast<u8>(r);
        f64 tot = 0;
        for (u32 c : ch[r]) tot += c;
        if (tot > 0) { alive[r] = true; ccost[r] = cost(ch[r]); ++nactive; }
    }
    if (nactive == 0) { cluster_of.fill(0); return 1; }
    while (nactive > 1) {
        const bool force = nactive > kDctMaxClusters;
        f64 best = 0;
        int bi = -1, bj = -1;
        for (int i = 0; i < kDctRawCtx; ++i) {
            if (!alive[i]) continue;
            for (int j = i + 1; j < kDctRawCtx; ++j) {
                if (!alive[j]) continue;
                std::array<u32, kDctCat> m;
                for (int s = 0; s < kDctCat; ++s) m[static_cast<usize>(s)] = ch[i][static_cast<usize>(s)] + ch[j][static_cast<usize>(s)];
                const f64 d = cost(m) - ccost[i] - ccost[j];
                if (bi < 0 || d < best) { best = d; bi = i; bj = j; }
            }
        }
        if (bi < 0 || (!force && best >= 0)) break;  // no beneficial merge AND within the cluster cap
        for (int s = 0; s < kDctCat; ++s) ch[bi][static_cast<usize>(s)] += ch[bj][static_cast<usize>(s)];
        ccost[bi] = cost(ch[bi]);
        alive[bj] = false;
        for (int r = 0; r < kDctRawCtx; ++r) if (cluster_of[r] == bj) cluster_of[r] = static_cast<u8>(bi);
        --nactive;
    }
    std::array<int, kDctRawCtx> remap;
    remap.fill(-1);
    int K = 0;
    for (int r = 0; r < kDctRawCtx; ++r)
        if (alive[cluster_of[r]] && remap[cluster_of[r]] < 0) remap[cluster_of[r]] = K++;
    if (K == 0) K = 1;
    for (int r = 0; r < kDctRawCtx; ++r)
        cluster_of[r] = static_cast<u8>(alive[cluster_of[r]] ? remap[cluster_of[r]] : 0);
    return K;
}

// Quant-gated 3D deblocking (HEVC DBF lineage, Norkin TCSVT'12), decode-time and NON-NORMATIVE: it only
// smooths the seams that independent-block quantization leaves at the 16³ block boundaries, gated by a
// flatness test + a quant-derived threshold so real edges/texture stay untouched. Operates on a fully
// reconstructed `side³` tile across its INTERIOR block boundaries (multiples of kDctN, excluding the
// outer tile faces — those need a neighbour halo and are a container-level concern). One weak filter
// (first tap only) per boundary line, applied separably along z then y then x. Thresholds scale with the
// base step `q`: a coarser quantization leaves bigger seams and is deblocked harder. No bitstream change.
inline void deblock_tile(std::span<f32> t, s64 side, const DctParams& p) {
    const f32 beta = p.deblock_beta * p.q;  // 2nd-difference budget: above this it's structure, not a seam
    const f32 tc = p.deblock_tc * p.q;      // max sample correction (so only quantization seams move)
    if (tc <= 0.0f || side <= kDctN) return;
    // Smooth across one boundary along a line: stencil [p2 p1 p0 | q0 q1 q2]; only p0,q0 are moved.
    auto smooth = [&](f32 p2, f32 p1, f32& p0, f32& q0, f32 q1, f32 q2) {
        const f32 dp = std::abs(p2 - 2.0f * p1 + p0), dq = std::abs(q2 - 2.0f * q1 + q0);
        if (dp + dq >= beta) return;  // not flat on both sides → a real feature crosses here, leave it
        f32 delta = (9.0f * (q0 - p0) - 3.0f * (q1 - p1)) * (1.0f / 16.0f);  // HEVC weak-filter delta
        if (std::abs(delta) >= 10.0f * tc) return;  // jump far larger than the step → a real edge, not ring
        delta = std::clamp(delta, -tc, tc);
        p0 += delta;
        q0 -= delta;
    };
    auto at = [&](s64 z, s64 y, s64 x) -> f32& { return t[static_cast<usize>((z * side + y) * side + x)]; };
    for (s64 b = kDctN; b < side; b += kDctN) {  // interior boundaries only
        for (s64 i = 0; i < side; ++i)
            for (s64 j = 0; j < side; ++j) {
                smooth(at(b - 3, i, j), at(b - 2, i, j), at(b - 1, i, j), at(b, i, j), at(b + 1, i, j), at(b + 2, i, j));  // z-edges
                smooth(at(i, b - 3, j), at(i, b - 2, j), at(i, b - 1, j), at(i, b, j), at(i, b + 1, j), at(i, b + 2, j));  // y-edges
                smooth(at(i, j, b - 3), at(i, j, b - 2), at(i, j, b - 1), at(i, j, b), at(i, j, b + 1), at(i, j, b + 2));  // x-edges
            }
    }
}
}  // namespace detail

// Encode a TILE of bpa³ DCT blocks from a contiguous (bpa·kDctN)³ region of dtype T. The rANS category
// tables are SHARED across all blocks in the tile (JPEG XL "group" model) AND clustered (see kDctRawCtx).
// Two passes: (A) quantize every block (cache the levels) + accumulate raw-context histograms → cluster;
// (B) emit category symbols into the K clustered streams + the shared mantissa/sign bits. Per tile:
// DC[]+nsig[]+K+context-map+K category tables+bits. Aligns with the .fxvol 64³ IO unit (bpa=4) so random
// access is unaffected. bpa=1 ⇒ the single-block codec.
template <class T>
inline std::vector<u8> encode_tile_dct(std::span<const T> tile, s64 bpa, DctParams params = {}) {
    constexpr s64 N = kDctN, sz = N * N, V = N * N * N;
    const s64 side = bpa * N, nblk = bpa * bpa * bpa;
    f32 stepv[3 * kDctN];
    detail::dct_step_table(params, stepv);
    const auto& scan = detail::dct_scan_order();

    // Phase A: quantize each block (cache levels + coefficients) + accumulate raw-context histograms.
    std::vector<s32> levall(static_cast<usize>(nblk * V));
    std::vector<f32> coefall(static_cast<usize>(nblk * V));  // DCT coefficients, for RDOQ's distortion term
    std::vector<s32> dclev(static_cast<usize>(nblk));
    std::vector<u32> nsigs(static_cast<usize>(nblk));
    std::array<std::array<u32, kDctCat>, kDctRawCtx> hist{};
    std::vector<u8> sig(static_cast<usize>(V));
    for (s64 b = 0; b < nblk; ++b) {
        const s64 bz = b / (bpa * bpa), by = (b / bpa) % bpa, bx = b % bpa;
        s32* lev = levall.data() + b * V;
        const auto [dl, ns] = detail::quant_one_block<T>(tile, side, bz * N, by * N, bx * N, params, stepv, lev, coefall.data() + b * V);
        dclev[static_cast<usize>(b)] = dl;
        nsigs[static_cast<usize>(b)] = ns;
        std::fill_n(sig.data(), static_cast<usize>(V), u8{0});
        for (u32 pos = 0; pos < ns; ++pos) {
            const detail::ScanEntry e = scan[pos];
            const usize i = e.idx;
            const int nsum = ((e.edge & 1) ? sig[i - static_cast<usize>(sz)] : 0) +
                             ((e.edge & 2) ? sig[i - static_cast<usize>(N)] : 0) + ((e.edge & 4) ? sig[i - 1] : 0);
            const u32 a = static_cast<u32>(lev[i] < 0 ? -lev[i] : lev[i]);
            const int cat = a ? std::bit_width(a) : 0;
            hist[static_cast<usize>(detail::dct_raw_ctx(e.fsum, nsum))][static_cast<usize>(cat)]++;
            sig[i] = static_cast<u8>(std::min<u32>(a, kDctMagCap));
        }
    }
    std::array<u8, kDctRawCtx> cmap;
    const int K = detail::cluster_contexts(hist, cmap);

    // RDOQ (encoder-only, decoder/bitstream unchanged): now that the rate model comes from the tile-
    // shared CLUSTERED histograms (trustworthy, unlike the old noisy per-block tables — the reason RDOQ
    // failed before), re-decide each significant coefficient's level among {0, L-1, L} to minimize
    // D + λ·R: D=(|coef|−dequant)² (exact voxel-MSE by Parseval), R=clustered rANS bits + mantissa+sign,
    // λ=rdoq_lambda·step². Trims over-coded coefficients + shortens the EOB; processed in scan order so
    // the causal context reflects the decisions (matches what phase B / decode recompute).
    if (params.rdoq) {
        std::array<std::array<f32, kDctCat>, kDctMaxClusters> bc{};
        std::array<std::array<f64, kDctCat>, kDctMaxClusters> chist{};
        for (int r = 0; r < kDctRawCtx; ++r)
            for (int cat = 0; cat < kDctCat; ++cat) chist[cmap[static_cast<usize>(r)]][static_cast<usize>(cat)] += hist[static_cast<usize>(r)][static_cast<usize>(cat)];
        for (int cl = 0; cl < K; ++cl) {
            f64 tot = 0;
            for (f64 v : chist[static_cast<usize>(cl)]) tot += v;
            for (int cat = 0; cat < kDctCat; ++cat)
                bc[static_cast<usize>(cl)][static_cast<usize>(cat)] =
                    static_cast<f32>(-std::log2((chist[static_cast<usize>(cl)][static_cast<usize>(cat)] + 0.5) / (tot + 0.5 * kDctCat)));
        }
        for (s64 b = 0; b < nblk; ++b) {
            s32* lev = levall.data() + b * V;
            const f32* coef = coefall.data() + b * V;
            std::fill_n(sig.data(), static_cast<usize>(V), u8{0});
            for (u32 pos = 0; pos < nsigs[static_cast<usize>(b)]; ++pos) {
                const detail::ScanEntry e = scan[pos];
                const usize i = e.idx;
                const u32 m0 = static_cast<u32>(lev[i] < 0 ? -lev[i] : lev[i]);
                if (m0 == 0) { sig[i] = 0; continue; }
                const int nsum = ((e.edge & 1) ? sig[i - static_cast<usize>(sz)] : 0) +
                                 ((e.edge & 2) ? sig[i - static_cast<usize>(N)] : 0) + ((e.edge & 4) ? sig[i - 1] : 0);
                const usize cl = cmap[static_cast<usize>(detail::dct_raw_ctx(e.fsum, nsum))];
                const f32 s = stepv[e.fsum], lam = params.rdoq_lambda * s * s, ac = std::abs(coef[i]);
                auto rd = [&](u32 m) {
                    const f32 dq = m ? s * (static_cast<f32>(m) - 1.0f + params.dz_frac + 0.40f) : 0.0f;
                    const int cat = m ? std::bit_width(m) : 0;
                    return (ac - dq) * (ac - dq) + lam * (bc[cl][static_cast<usize>(cat)] + static_cast<f32>(m ? cat : 0));
                };
                u32 bm = m0;
                f32 best = rd(m0);
                if (const f32 c0 = rd(0); c0 < best) { best = c0; bm = 0; }
                if (m0 >= 2)
                    if (const f32 cl1 = rd(m0 - 1); cl1 < best) { best = cl1; bm = m0 - 1; }
                lev[i] = (lev[i] < 0) ? -static_cast<s32>(bm) : static_cast<s32>(bm);
                sig[i] = static_cast<u8>(std::min<u32>(bm, kDctMagCap));
            }
            u32 n2 = 0;
            for (usize pos = nsigs[static_cast<usize>(b)]; pos-- > 0;)
                if (lev[scan[pos].idx] != 0) { n2 = static_cast<u32>(pos + 1); break; }
            nsigs[static_cast<usize>(b)] = n2;
        }
    }

    // Phase B: emit category symbols into the K clustered streams + the shared mantissa/sign bits.
    std::vector<std::vector<u8>> cats(static_cast<usize>(K));
    detail::BitWriter bits;
    for (s64 b = 0; b < nblk; ++b) {
        const s32* lev = levall.data() + b * V;
        std::fill_n(sig.data(), static_cast<usize>(V), u8{0});
        for (u32 pos = 0; pos < nsigs[static_cast<usize>(b)]; ++pos) {
            const detail::ScanEntry e = scan[pos];
            const usize i = e.idx;
            const int nsum = ((e.edge & 1) ? sig[i - static_cast<usize>(sz)] : 0) +
                             ((e.edge & 2) ? sig[i - static_cast<usize>(N)] : 0) + ((e.edge & 4) ? sig[i - 1] : 0);
            const int cl = cmap[static_cast<usize>(detail::dct_raw_ctx(e.fsum, nsum))];
            const s32 level = lev[i];
            const u32 a = static_cast<u32>(level < 0 ? -level : level);
            const int cat = a ? std::bit_width(a) : 0;
            cats[static_cast<usize>(cl)].push_back(static_cast<u8>(cat));
            sig[i] = static_cast<u8>(std::min<u32>(a, kDctMagCap));
            if (cat > 0) {
                bits.put(a - (1u << (cat - 1)), cat - 1);
                bits.put(level < 0 ? 1u : 0u, 1);
            }
        }
    }

    std::vector<u8> out;
    for (s64 b = 0; b < nblk; ++b) {
        const s64 bz = b / (bpa * bpa), by = (b / bpa) % bpa, bx = b % bpa;
        const s32 pred = params.dc_predict ? detail::causal_block_pred(dclev.data(), bz, by, bx, bpa) : 0;
        const s32 res = dclev[static_cast<usize>(b)] - pred;
        detail::put_var(out, (static_cast<u32>(res) << 1) ^ static_cast<u32>(res >> 31));
    }
    const usize dc_end = out.size();
    for (s64 b = 0; b < nblk; ++b) {
        const s64 bz = b / (bpa * bpa), by = (b / bpa) % bpa, bx = b % bpa;
        const s32 pred = params.dc_predict ? detail::causal_block_pred(nsigs.data(), bz, by, bx, bpa) : 0;
        const s32 res = static_cast<s32>(nsigs[static_cast<usize>(b)]) - pred;
        detail::put_var(out, (static_cast<u32>(res) << 1) ^ static_cast<u32>(res >> 31));
    }
    const usize nsig_end = out.size();
    out.push_back(static_cast<u8>(K));
    for (int r = 0; r < kDctRawCtx; ++r) out.push_back(cmap[static_cast<usize>(r)]);  // context map
    detail::g_dc_bytes.fetch_add(dc_end, std::memory_order_relaxed);
    detail::g_nsig_bytes.fetch_add(nsig_end - dc_end, std::memory_order_relaxed);
    detail::g_map_bytes.fetch_add(out.size() - nsig_end, std::memory_order_relaxed);
    for (int cl = 0; cl < K; ++cl) detail::encode_plane(out, cats[static_cast<usize>(cl)]);
    bits.flush();
    detail::put_var(out, static_cast<u32>(bits.bytes.size()));
    out.insert(out.end(), bits.bytes.begin(), bits.bytes.end());
    detail::g_bits_bytes.fetch_add(bits.bytes.size() + 2, std::memory_order_relaxed);
    return out;
}

// Decode an encode_tile_dct payload back to the contiguous (bpa·kDctN)³ region of dtype T.
template <class T>
inline std::vector<T> decode_tile_dct(std::span<const u8> payload, s64 bpa, DctParams params = {}) {
    constexpr s64 N = kDctN, sz = N * N, V = N * N * N;
    const s64 side = bpa * N, nblk = bpa * bpa * bpa;
    usize p = 0;
    std::vector<s32> dclev(static_cast<usize>(nblk));
    std::vector<u32> nsigs(static_cast<usize>(nblk));
    for (s64 b = 0; b < nblk; ++b) {
        const u32 z = detail::get_var(payload, p);
        const s32 res = static_cast<s32>(z >> 1) ^ -static_cast<s32>(z & 1u);
        const s64 bz = b / (bpa * bpa), by = (b / bpa) % bpa, bx = b % bpa;
        const s32 pred = params.dc_predict ? detail::causal_block_pred(dclev.data(), bz, by, bx, bpa) : 0;
        dclev[static_cast<usize>(b)] = res + pred;
    }
    for (s64 b = 0; b < nblk; ++b) {
        const u32 z = detail::get_var(payload, p);
        const s32 res = static_cast<s32>(z >> 1) ^ -static_cast<s32>(z & 1u);
        const s64 bz = b / (bpa * bpa), by = (b / bpa) % bpa, bx = b % bpa;
        const s32 pred = params.dc_predict ? detail::causal_block_pred(nsigs.data(), bz, by, bx, bpa) : 0;
        nsigs[static_cast<usize>(b)] = static_cast<u32>(res + pred);
    }
    const int K = payload[p++];
    std::array<u8, kDctRawCtx> cmap;
    for (int r = 0; r < kDctRawCtx; ++r) cmap[static_cast<usize>(r)] = payload[p++];
    std::vector<std::vector<u8>> cats(static_cast<usize>(K));
    std::vector<usize> cur(static_cast<usize>(K), 0);
    for (int cl = 0; cl < K; ++cl) cats[static_cast<usize>(cl)] = detail::decode_plane(payload, p);
    const u32 nb = detail::get_var(payload, p);
    detail::BitReader bits{payload.data() + p, nb};
    p += nb;
    f32 stepv[3 * kDctN];
    detail::dct_step_table(params, stepv);
    const auto& scan = detail::dct_scan_order();
    std::vector<T> out(static_cast<usize>(side * side * side));
    std::vector<f32> recon(static_cast<usize>(side * side * side));  // full f32 tile, for seam deblocking
    std::vector<f32> c(static_cast<usize>(V));
    std::vector<u8> sig(static_cast<usize>(V));
    for (s64 b = 0; b < nblk; ++b) {
        const s64 bz = b / (bpa * bpa), by = (b / bpa) % bpa, bx = b % bpa;
        std::fill_n(c.data(), static_cast<usize>(V), 0.0f);
        std::fill_n(sig.data(), static_cast<usize>(V), u8{0});
        const f32 dc = static_cast<f32>(dclev[static_cast<usize>(b)]) * (params.q * 0.0625f);
        for (u32 pos = 0; pos < nsigs[static_cast<usize>(b)]; ++pos) {
            const detail::ScanEntry e = scan[pos];
            const usize i = e.idx;
            const int nsum = ((e.edge & 1) ? sig[i - static_cast<usize>(sz)] : 0) +
                             ((e.edge & 2) ? sig[i - static_cast<usize>(N)] : 0) + ((e.edge & 4) ? sig[i - 1] : 0);
            const int cl = cmap[static_cast<usize>(detail::dct_raw_ctx(e.fsum, nsum))];
            const int cat = cats[static_cast<usize>(cl)][cur[static_cast<usize>(cl)]++];
            if (cat > 0) {
                const u32 a = (1u << (cat - 1)) + bits.get(cat - 1);
                const bool neg = bits.get(1) != 0;
                const f32 mag = stepv[e.fsum] * (static_cast<f32>(a) - 1.0f + params.dz_frac + 0.40f);
                c[i] = neg ? -mag : mag;
                sig[i] = static_cast<u8>(std::min<u32>(a, kDctMagCap));
            }
        }
        dct16_3d_inv(c);
        for (s64 z = 0; z < N; ++z)
            for (s64 y = 0; y < N; ++y)
                for (s64 x = 0; x < N; ++x)
                    recon[static_cast<usize>(((bz * N + z) * side + (by * N + y)) * side + (bx * N + x))] =
                        c[static_cast<usize>((z * N + y) * N + x)] + dc;
    }
    if (params.deblock) detail::deblock_tile(recon, side, params);
    for (usize i = 0; i < recon.size(); ++i) out[i] = from_f32_one<T>(recon[i]);
    return out;
}

// Single-block codec = a tile of one block (byte-compatible with the per-block format).
template <class T>
inline std::vector<u8> encode_block_dct(std::span<const T> block, DctParams params = {}) {
    return encode_tile_dct<T>(block, 1, params);
}
template <class T>
inline std::vector<T> decode_block_dct_to(std::span<const u8> payload, DctParams params = {}) {
    return decode_tile_dct<T>(payload, 1, params);
}

// ---- split-build codec firewall (ADR 0008) ---------------------------------------------------------------
// encode/decode_tile_dct are the heaviest backend in the tree (~350 lines of DCT butterfly + two-pass
// clustered rANS), and every module TU that touches the archive (io, segment, preprocess, render) re-optimizes
// them at -O3. They are far too large for the inliner to inline into callers, so an `extern template` makes
// those TUs emit a call while a SINGLE TU (codec.cpp, which defines FENIX_CODEC_INSTANTIATE) provides the one
// optimized copy — the split fenix binary links them together. Only u8/f32 reach the archive; other dtypes
// still instantiate normally where used. Gated on FENIX_SPLIT: the unity build and the per-file test binaries
// (no FENIX_SPLIT, and tests round-trip all 8 dtypes) are entirely unchanged.
#ifdef FENIX_SPLIT
#ifdef FENIX_CODEC_INSTANTIATE
#define FENIX_CODEC_EXTERN  // codec.cpp: emit the explicit instantiation DEFINITION
#else
#define FENIX_CODEC_EXTERN extern  // every other split TU: suppress the implicit instantiation
#endif
FENIX_CODEC_EXTERN template std::vector<u8> encode_tile_dct<u8>(std::span<const u8>, s64, DctParams);
FENIX_CODEC_EXTERN template std::vector<u8> encode_tile_dct<f32>(std::span<const f32>, s64, DctParams);
FENIX_CODEC_EXTERN template std::vector<u8> decode_tile_dct<u8>(std::span<const u8>, s64, DctParams);
FENIX_CODEC_EXTERN template std::vector<f32> decode_tile_dct<f32>(std::span<const u8>, s64, DctParams);
#undef FENIX_CODEC_EXTERN
#endif

}  // namespace fenix::codec
