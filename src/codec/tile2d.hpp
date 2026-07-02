// codec/tile2d.hpp — generic lossy 2D field codec: 64×64 DCT tiles + dead-zone quantization
// + zigzag/nsig + rANS (codec/lossless), the 2D sibling of the volume tile codec. Correctness
// is tolerance-only and VERIFIED AT ENCODE TIME: every tile is decoded back and its max error
// on valid cells checked against tau; violators halve q (up to 4×) and finally fall back to a
// raw-quantized tile (error ≤ tau/4 by construction). Three front-ends:
//   encode_field2d   — one scalar channel (texture layer, height, confidence)
//   encode_rgb2d     — 3-channel color, YCoCg-decorrelated
//   encode_coords2d  — 3-channel ZYX volume-sampling coords (surfaces): per-tile affine fit +
//                      tangent-frame projection turns 3 correlated channels into 1 height
//                      field + 2 near-zero remainders; per-channel tau/sqrt(3) gives an exact
//                      3D max-error bound (the frame is orthonormal).
// All decode paths treat input as UNTRUSTED: every count/length is validated.
#pragma once

#include "codec/dct64.hpp"
#include "codec/lossless.hpp"
#include "core/core.hpp"
#include "core/vec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace fenix::codec {

namespace detail {

inline constexpr int kT2N = kDct64N;              // 64
inline constexpr int kT2NN = kT2N * kT2N;         // 4096
inline constexpr int kT2MaxQShift = 4;            // q, q/2, ... q/16, then raw
inline constexpr u8 kT2ModeDct = 0, kT2ModeRaw = 1;

// Diagonal zigzag scan order for a 64×64 tile (low frequencies first).
inline const std::array<u16, kT2NN>& t2_zigzag() {
    static const std::array<u16, kT2NN> order = [] {
        std::array<u16, kT2NN> o{};
        usize w = 0;
        for (int d = 0; d <= 2 * (kT2N - 1); ++d)
            for (int v = std::max(0, d - (kT2N - 1)); v <= std::min(d, kT2N - 1); ++v)
                o[w++] = static_cast<u16>(v * kT2N + (d - v));
        return o;
    }();
    return order;
}

inline u32 t2_zz32(s32 v) { return (static_cast<u32>(v) << 1) ^ static_cast<u32>(v >> 31); }
inline s32 t2_unzz32(u32 v) { return static_cast<s32>((v >> 1) ^ (~(v & 1) + 1)); }

inline void t2_put_blob(std::vector<u8>& out, const std::vector<u8>& b) {
    const u64 len = b.size();
    const usize at = out.size();
    out.resize(at + 8 + b.size());
    std::memcpy(out.data() + at, &len, 8);
    std::memcpy(out.data() + at + 8, b.data(), b.size());
}

inline Expected<std::span<const u8>> t2_get_blob(std::span<const u8> in, usize& cur) {
    if (cur > in.size() || in.size() - cur < 8) return err(Errc::decode_error, "tile2d: truncated blob header");
    u64 len = 0;
    std::memcpy(&len, in.data() + cur, 8);
    cur += 8;
    if (len > in.size() - cur) return err(Errc::decode_error, "tile2d: blob length out of range");
    auto s = in.subspan(cur, static_cast<usize>(len));
    cur += static_cast<usize>(len);
    return s;
}

struct T2Header {
    u32 w = 0, h = 0;
    f32 q = 0, tau = 0;
};

// One scalar field's encoded streams.
struct T2Streams {
    std::vector<u8> mode;    // per tile
    std::vector<u8> qshift;  // per tile (dct mode)
    std::vector<u16> nsig;   // per tile (dct mode; raw tiles: 0)
    std::vector<u32> coeff;  // concatenated: dct = nsig zigzagged coeffs; raw = kT2NN zigzagged values
};

// Gather one 64² tile from the field; cells outside w×h or with valid==0 count as invalid and
// take the mean of the tile's valid cells (a flat fill keeps the DCT sparse). Returns #valid.
inline int t2_gather(std::span<const f32> f, s64 w, s64 h, std::span<const u8> valid, s64 tx, s64 ty,
                     f32* tile, u8* tvalid) {
    f64 sum = 0;
    int nvalid = 0;
    for (int v = 0; v < kT2N; ++v)
        for (int u = 0; u < kT2N; ++u) {
            const s64 gu = tx * kT2N + u, gv = ty * kT2N + v;
            const int i = v * kT2N + u;
            const bool in = gu < w && gv < h;
            const usize gi = in ? static_cast<usize>(gv * w + gu) : 0;
            const bool ok = in && (valid.empty() || valid[gi] != 0);
            tvalid[i] = ok ? 1u : 0u;
            tile[i] = ok ? f[gi] : 0.0f;
            if (ok) {
                sum += tile[i];
                ++nvalid;
            }
        }
    const f32 fill = nvalid ? static_cast<f32>(sum / nvalid) : 0.0f;
    for (int i = 0; i < kT2NN; ++i)
        if (!tvalid[i]) tile[i] = fill;
    return nvalid;
}

// Encode one tile into st; verified: max |err| over valid cells <= tau.
inline void t2_encode_tile(const f32* tile, const u8* tvalid, f32 q, f32 tau, T2Streams& st) {
    const auto& zz = t2_zigzag();
    f32 coef[kT2NN], recon[kT2NN];
    s32 qi[kT2NN];
    for (int shift = 0; shift <= kT2MaxQShift; ++shift) {
        const f32 qt = q / static_cast<f32>(1 << shift);
        std::memcpy(coef, tile, sizeof coef);
        dct64_fwd_tile(coef);
        int nsig = 0;
        for (int k = 0; k < kT2NN; ++k) {
            const f32 c = coef[zz[k]];
            qi[k] = static_cast<s32>(std::lround(std::clamp(c / qt, -2.1e9f, 2.1e9f)));
            if (qi[k] != 0) nsig = k + 1;
        }
        for (int k = 0; k < kT2NN; ++k) recon[zz[k]] = static_cast<f32>(qi[k]) * qt;
        dct64_inv_tile(recon);
        f32 maxe = 0;
        for (int i = 0; i < kT2NN; ++i)
            if (tvalid[i]) maxe = std::max(maxe, std::abs(recon[i] - tile[i]));
        if (maxe <= tau) {
            st.mode.push_back(kT2ModeDct);
            st.qshift.push_back(static_cast<u8>(shift));
            st.nsig.push_back(static_cast<u16>(nsig));
            for (int k = 0; k < nsig; ++k) st.coeff.push_back(t2_zz32(qi[k]));
            return;
        }
    }
    // Raw fallback: quantize the samples themselves at tau/2 (error <= tau/4 < tau).
    const f32 rq = tau * 0.5f;
    st.mode.push_back(kT2ModeRaw);
    st.qshift.push_back(0);
    st.nsig.push_back(0);
    for (int i = 0; i < kT2NN; ++i)
        st.coeff.push_back(t2_zz32(static_cast<s32>(std::lround(std::clamp(tile[i] / rq, -2.1e9f, 2.1e9f)))));
}

inline void t2_decode_tile(u8 mode, u8 qshift, u16 nsig, std::span<const u32> coeff, f32 q, f32 tau,
                           f32* tile) {
    const auto& zz = t2_zigzag();
    if (mode == kT2ModeRaw) {
        const f32 rq = tau * 0.5f;
        for (int i = 0; i < kT2NN; ++i) tile[i] = static_cast<f32>(t2_unzz32(coeff[static_cast<usize>(i)])) * rq;
        return;
    }
    const f32 qt = q / static_cast<f32>(1 << qshift);
    for (int i = 0; i < kT2NN; ++i) tile[i] = 0;
    for (u16 k = 0; k < nsig; ++k) tile[zz[k]] = static_cast<f32>(t2_unzz32(coeff[k])) * qt;
    dct64_inv_tile(tile);
}

}  // namespace detail

// ---- scalar field ----

// Encode a w×h f32 field with max error <= tau on valid cells (valid empty = all valid).
// `q` is the starting DCT quant step; tau <= q is typical (q = tau works, larger q lets
// smooth tiles use coarser steps and rely on the verify loop).
inline std::vector<u8> encode_field2d(std::span<const f32> f, s64 w, s64 h, f32 q, f32 tau,
                                      std::span<const u8> valid = {}) {
    using namespace detail;
    const s64 ntx = (w + kT2N - 1) / kT2N, nty = (h + kT2N - 1) / kT2N;
    T2Streams st;
    st.mode.reserve(static_cast<usize>(ntx * nty));
    f32 tile[kT2NN];
    u8 tvalid[kT2NN];
    for (s64 ty = 0; ty < nty; ++ty)
        for (s64 tx = 0; tx < ntx; ++tx) {
            t2_gather(f, w, h, valid, tx, ty, tile, tvalid);
            t2_encode_tile(tile, tvalid, q, tau, st);
        }
    std::vector<u8> out;
    T2Header hd{static_cast<u32>(w), static_cast<u32>(h), q, tau};
    out.resize(sizeof hd);
    std::memcpy(out.data(), &hd, sizeof hd);
    t2_put_blob(out, lossless_encode<u8>(st.mode));
    t2_put_blob(out, lossless_encode<u8>(st.qshift));
    t2_put_blob(out, lossless_encode<u16>(st.nsig));
    t2_put_blob(out, lossless_encode<u32>(st.coeff));
    return out;
}

inline Expected<std::vector<f32>> decode_field2d(std::span<const u8> in, s64 want_w = 0, s64 want_h = 0) {
    using namespace detail;
    if (in.size() < sizeof(T2Header)) return err(Errc::decode_error, "tile2d: truncated header");
    T2Header hd;
    std::memcpy(&hd, in.data(), sizeof hd);
    const s64 w = hd.w, h = hd.h;
    if (w <= 0 || h <= 0 || w > (1 << 20) || h > (1 << 20)) return err(Errc::decode_error, "tile2d: bad dims");
    if ((want_w && want_w != w) || (want_h && want_h != h)) return err(Errc::decode_error, "tile2d: dims mismatch");
    if (!(hd.q > 0) || !(hd.tau > 0)) return err(Errc::decode_error, "tile2d: bad quant params");
    const s64 ntx = (w + kT2N - 1) / kT2N, nty = (h + kT2N - 1) / kT2N;
    const usize ntiles = static_cast<usize>(ntx * nty);

    usize cur = sizeof hd;
    auto b_mode = t2_get_blob(in, cur);
    if (!b_mode) return std::unexpected(b_mode.error());
    auto b_qs = t2_get_blob(in, cur);
    if (!b_qs) return std::unexpected(b_qs.error());
    auto b_ns = t2_get_blob(in, cur);
    if (!b_ns) return std::unexpected(b_ns.error());
    auto b_cf = t2_get_blob(in, cur);
    if (!b_cf) return std::unexpected(b_cf.error());
    auto mode = lossless_decode<u8>(*b_mode, ntiles);
    if (!mode) return std::unexpected(mode.error());
    auto qs = lossless_decode<u8>(*b_qs, ntiles);
    if (!qs) return std::unexpected(qs.error());
    auto ns = lossless_decode<u16>(*b_ns, ntiles);
    if (!ns) return std::unexpected(ns.error());
    u64 need = 0;
    for (usize t = 0; t < ntiles; ++t) {
        if ((*mode)[t] > kT2ModeRaw || (*qs)[t] > kT2MaxQShift || (*ns)[t] > kT2NN)
            return err(Errc::decode_error, "tile2d: corrupt tile table");
        need += (*mode)[t] == kT2ModeRaw ? static_cast<u64>(kT2NN) : (*ns)[t];
    }
    auto cf = lossless_decode<u32>(*b_cf, static_cast<usize>(need));
    if (!cf) return std::unexpected(cf.error());

    std::vector<f32> out(static_cast<usize>(w * h));
    f32 tile[kT2NN];
    usize coff = 0;
    for (s64 ty = 0; ty < nty; ++ty)
        for (s64 tx = 0; tx < ntx; ++tx) {
            const usize t = static_cast<usize>(ty * ntx + tx);
            const usize n = (*mode)[t] == kT2ModeRaw ? static_cast<usize>(kT2NN) : (*ns)[t];
            t2_decode_tile((*mode)[t], (*qs)[t], (*ns)[t], std::span<const u32>(*cf).subspan(coff, n), hd.q,
                           hd.tau, tile);
            coff += n;
            for (int v = 0; v < kT2N; ++v) {
                const s64 gv = ty * kT2N + v;
                if (gv >= h) break;
                const s64 gu0 = tx * kT2N;
                const s64 cols = std::min<s64>(kT2N, w - gu0);
                std::memcpy(out.data() + gv * w + gu0, tile + v * kT2N, static_cast<usize>(cols) * 4);
            }
        }
    return out;
}

// ---- RGB (YCoCg decorrelation) ----

// Max per-channel RGB error <= tau: the YCoCg inverse amplifies channel errors by <= 2,
// so each transformed channel is coded at tau/2.
inline std::vector<u8> encode_rgb2d(std::span<const f32> r, std::span<const f32> g, std::span<const f32> b,
                                    s64 w, s64 h, f32 tau, std::span<const u8> valid = {}) {
    const usize n = static_cast<usize>(w * h);
    std::vector<f32> Y(n), Co(n), Cg(n);
    for (usize i = 0; i < n; ++i) {
        Y[i] = 0.25f * r[i] + 0.5f * g[i] + 0.25f * b[i];
        Co[i] = r[i] - b[i];
        Cg[i] = g[i] - 0.5f * (r[i] + b[i]);
    }
    const f32 ct = tau * 0.5f;
    std::vector<u8> out;
    detail::t2_put_blob(out, encode_field2d(Y, w, h, ct, ct, valid));
    detail::t2_put_blob(out, encode_field2d(Co, w, h, ct, ct, valid));
    detail::t2_put_blob(out, encode_field2d(Cg, w, h, ct, ct, valid));
    return out;
}

struct Rgb2d {
    std::vector<f32> r, g, b;
};

inline Expected<Rgb2d> decode_rgb2d(std::span<const u8> in, s64 w, s64 h) {
    usize cur = 0;
    auto by = detail::t2_get_blob(in, cur);
    if (!by) return std::unexpected(by.error());
    auto bo = detail::t2_get_blob(in, cur);
    if (!bo) return std::unexpected(bo.error());
    auto bg = detail::t2_get_blob(in, cur);
    if (!bg) return std::unexpected(bg.error());
    auto Y = decode_field2d(*by, w, h);
    if (!Y) return std::unexpected(Y.error());
    auto Co = decode_field2d(*bo, w, h);
    if (!Co) return std::unexpected(Co.error());
    auto Cg = decode_field2d(*bg, w, h);
    if (!Cg) return std::unexpected(Cg.error());
    Rgb2d out;
    const usize n = Y->size();
    out.r.resize(n);
    out.g.resize(n);
    out.b.resize(n);
    for (usize i = 0; i < n; ++i) {
        const f32 rb2 = (*Y)[i] - 0.5f * (*Cg)[i];  // (r+b)/2
        out.g[i] = (*Cg)[i] + rb2;
        out.r[i] = rb2 + 0.5f * (*Co)[i];
        out.b[i] = rb2 - 0.5f * (*Co)[i];
    }
    return out;
}

// ---- ZYX coordinate surfaces (the 3 volume-sampling components) ----

namespace detail {

struct T2Affine {
    Vec3f base, du, dv;  // coord(u,v) ~= base + u_local*du + v_local*dv (local to the tile)
};

// Orthonormal tangent frame derived DETERMINISTICALLY from the (quantized) affine so encoder
// and decoder agree exactly. Degenerate tiles get canonical axes.
inline void t2_frame(const T2Affine& a, Vec3f& tu, Vec3f& tv, Vec3f& n) {
    const f32 lu = norm(a.du);
    Vec3f e1 = lu > 1e-6f ? a.du * (1.0f / lu) : Vec3f{0, 0, 1};
    Vec3f e2 = a.dv - e1 * dot(a.dv, e1);
    const f32 l2 = norm(e2);
    e2 = l2 > 1e-6f ? e2 * (1.0f / l2) : (std::abs(e1.y) < 0.9f ? Vec3f{0, 1, 0} - e1 * e1.y
                                                                : Vec3f{1, 0, 0} - e1 * e1.z);
    const f32 l2b = norm(e2);
    if (l2b > 1e-6f) e2 = e2 * (1.0f / l2b);
    tu = e1;
    tv = e2;
    n = cross(e2, e1);  // consistent handedness either way — both sides use the same formula
}

inline s32 t2_qcoord(f32 v) { return static_cast<s32>(std::lround(std::clamp(v * 16.0f, -2.1e9f, 2.1e9f))); }
inline f32 t2_dqcoord(s32 v) { return static_cast<f32>(v) / 16.0f; }

}  // namespace detail

// Encode nu×nv ZYX coords with 3D max error <= tau on valid cells.
inline std::vector<u8> encode_coords2d(std::span<const Vec3f> coord, std::span<const u8> valid, s64 nu,
                                       s64 nv, f32 tau) {
    using namespace detail;
    const s64 ntx = (nu + kT2N - 1) / kT2N, nty = (nv + kT2N - 1) / kT2N;
    const usize ntiles = static_cast<usize>(ntx * nty);

    // Per-tile least-squares affine over valid cells, stored quantized (1/16 voxel).
    std::vector<s32> aff_q(ntiles * 9, 0);
    std::vector<T2Affine> aff(ntiles);
    for (s64 ty = 0; ty < nty; ++ty)
        for (s64 tx = 0; tx < ntx; ++tx) {
            // LSQ fit of c ~ b + u*du + v*dv (u,v local, centered to reduce conditioning).
            f64 su = 0, sv = 0, suu = 0, svv = 0, suv = 0, n = 0;
            Vec3<f64> sc{0, 0, 0}, scu{0, 0, 0}, scv{0, 0, 0};
            for (int v = 0; v < kT2N; ++v)
                for (int u = 0; u < kT2N; ++u) {
                    const s64 gu = tx * kT2N + u, gv = ty * kT2N + v;
                    if (gu >= nu || gv >= nv) continue;
                    const usize gi = static_cast<usize>(gv * nu + gu);
                    if (!valid.empty() && valid[gi] == 0) continue;
                    const Vec3f c = coord[gi];
                    const f64 fu = u, fv = v;
                    su += fu; sv += fv; suu += fu * fu; svv += fv * fv; suv += fu * fv; n += 1;
                    sc.z += c.z; sc.y += c.y; sc.x += c.x;
                    scu.z += c.z * fu; scu.y += c.y * fu; scu.x += c.x * fu;
                    scv.z += c.z * fv; scv.y += c.y * fv; scv.x += c.x * fv;
                }
            T2Affine a{};
            if (n >= 3) {
                // Solve the 3x3 normal equations [n su sv; su suu suv; sv suv svv] per component.
                const f64 A[3][3] = {{n, su, sv}, {su, suu, suv}, {sv, suv, svv}};
                const f64 det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                                A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
                if (std::abs(det) > 1e-9) {
                    auto solve = [&](f64 b0, f64 b1, f64 b2, f32& o0, f32& o1, f32& o2) {
                        const f64 inv = 1.0 / det;
                        const f64 x0 = (b0 * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                                        A[0][1] * (b1 * A[2][2] - A[1][2] * b2) +
                                        A[0][2] * (b1 * A[2][1] - A[1][1] * b2)) * inv;
                        const f64 x1 = (A[0][0] * (b1 * A[2][2] - A[1][2] * b2) -
                                        b0 * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                                        A[0][2] * (A[1][0] * b2 - b1 * A[2][0])) * inv;
                        const f64 x2 = (A[0][0] * (A[1][1] * b2 - b1 * A[2][1]) -
                                        A[0][1] * (A[1][0] * b2 - b1 * A[2][0]) +
                                        b0 * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) * inv;
                        o0 = static_cast<f32>(x0);
                        o1 = static_cast<f32>(x1);
                        o2 = static_cast<f32>(x2);
                    };
                    solve(sc.z, scu.z, scv.z, a.base.z, a.du.z, a.dv.z);
                    solve(sc.y, scu.y, scv.y, a.base.y, a.du.y, a.dv.y);
                    solve(sc.x, scu.x, scv.x, a.base.x, a.du.x, a.dv.x);
                } else if (n > 0) {
                    a.base = Vec3f{static_cast<f32>(sc.z / n), static_cast<f32>(sc.y / n),
                                   static_cast<f32>(sc.x / n)};
                }
            } else if (n > 0) {
                a.base = Vec3f{static_cast<f32>(sc.z / n), static_cast<f32>(sc.y / n),
                               static_cast<f32>(sc.x / n)};
            }
            const usize t = static_cast<usize>(ty * ntx + tx);
            s32* q = aff_q.data() + t * 9;
            const f32 src[9] = {a.base.z, a.base.y, a.base.x, a.du.z, a.du.y, a.du.x, a.dv.z, a.dv.y, a.dv.x};
            for (int k = 0; k < 9; ++k) q[k] = t2_qcoord(src[k]);
            // encode against the QUANTIZED affine so decode reconstructs the identical predictor
            aff[t].base = Vec3f{t2_dqcoord(q[0]), t2_dqcoord(q[1]), t2_dqcoord(q[2])};
            aff[t].du = Vec3f{t2_dqcoord(q[3]), t2_dqcoord(q[4]), t2_dqcoord(q[5])};
            aff[t].dv = Vec3f{t2_dqcoord(q[6]), t2_dqcoord(q[7]), t2_dqcoord(q[8])};
        }

    // Residuals in the tile tangent frame -> three scalar fields.
    const usize N = static_cast<usize>(nu * nv);
    std::vector<f32> fh(N, 0.0f), ft1(N, 0.0f), ft2(N, 0.0f);
    for (s64 gv = 0; gv < nv; ++gv)
        for (s64 gu = 0; gu < nu; ++gu) {
            const usize gi = static_cast<usize>(gv * nu + gu);
            if (!valid.empty() && valid[gi] == 0) continue;
            const usize t = static_cast<usize>((gv / kT2N) * ntx + (gu / kT2N));
            const T2Affine& a = aff[t];
            Vec3f tu, tv, n;
            t2_frame(a, tu, tv, n);
            const f32 lu = static_cast<f32>(gu % kT2N), lv = static_cast<f32>(gv % kT2N);
            const Vec3f r = coord[gi] - (a.base + a.du * lu + a.dv * lv);
            fh[gi] = dot(r, n);
            ft1[gi] = dot(r, tu);
            ft2[gi] = dot(r, tv);
        }

    const f32 ct = tau / std::numbers::sqrt3_v<f32>;  // orthonormal frame: 3D err = sqrt(sum of squares)
    std::vector<u8> out;
    detail::t2_put_blob(out, lossless_encode<u32>([&] {
                            std::vector<u32> z(aff_q.size());
                            for (usize i = 0; i < aff_q.size(); ++i) z[i] = detail::t2_zz32(aff_q[i]);
                            return z;
                        }()));
    detail::t2_put_blob(out, encode_field2d(fh, nu, nv, ct, ct, valid));
    detail::t2_put_blob(out, encode_field2d(ft1, nu, nv, ct, ct, valid));
    detail::t2_put_blob(out, encode_field2d(ft2, nu, nv, ct, ct, valid));
    return out;
}

inline Expected<std::vector<Vec3f>> decode_coords2d(std::span<const u8> in, s64 nu, s64 nv) {
    using namespace detail;
    if (nu <= 0 || nv <= 0) return err(Errc::decode_error, "coords2d: bad dims");
    const s64 ntx = (nu + kT2N - 1) / kT2N, nty = (nv + kT2N - 1) / kT2N;
    const usize ntiles = static_cast<usize>(ntx * nty);
    usize cur = 0;
    auto ba = t2_get_blob(in, cur);
    if (!ba) return std::unexpected(ba.error());
    auto bh = t2_get_blob(in, cur);
    if (!bh) return std::unexpected(bh.error());
    auto b1 = t2_get_blob(in, cur);
    if (!b1) return std::unexpected(b1.error());
    auto b2 = t2_get_blob(in, cur);
    if (!b2) return std::unexpected(b2.error());
    auto az = lossless_decode<u32>(*ba, ntiles * 9);
    if (!az) return std::unexpected(az.error());
    auto fh = decode_field2d(*bh, nu, nv);
    if (!fh) return std::unexpected(fh.error());
    auto f1 = decode_field2d(*b1, nu, nv);
    if (!f1) return std::unexpected(f1.error());
    auto f2 = decode_field2d(*b2, nu, nv);
    if (!f2) return std::unexpected(f2.error());

    std::vector<T2Affine> aff(ntiles);
    for (usize t = 0; t < ntiles; ++t) {
        f32 c[9];
        for (int k = 0; k < 9; ++k) c[k] = t2_dqcoord(t2_unzz32((*az)[t * 9 + k]));
        aff[t].base = Vec3f{c[0], c[1], c[2]};
        aff[t].du = Vec3f{c[3], c[4], c[5]};
        aff[t].dv = Vec3f{c[6], c[7], c[8]};
    }
    std::vector<Vec3f> out(static_cast<usize>(nu * nv));
    for (s64 gv = 0; gv < nv; ++gv)
        for (s64 gu = 0; gu < nu; ++gu) {
            const usize gi = static_cast<usize>(gv * nu + gu);
            const usize t = static_cast<usize>((gv / kT2N) * ntx + (gu / kT2N));
            const T2Affine& a = aff[t];
            Vec3f tu, tv, n;
            t2_frame(a, tu, tv, n);
            const f32 lu = static_cast<f32>(gu % kT2N), lv = static_cast<f32>(gv % kT2N);
            out[gi] = a.base + a.du * lu + a.dv * lv + n * (*fh)[gi] + tu * (*f1)[gi] + tv * (*f2)[gi];
        }
    return out;
}

}  // namespace fenix::codec
