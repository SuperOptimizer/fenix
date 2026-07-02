// io/surface.hpp — the .fxsurf container: a hand-rolled binary serialization of core::Surface (the
// (u,v) grid of ZYX world coords + validity + optional normal/conf channels). v2: coordinates are
// QUANTIZED (1/16 voxel — far below trilinear-sampling sensitivity) and plane-predicted
// (left+up-upleft) before rANS (codec/lossless) — neighboring cells on a smooth sheet are ~a grid
// step apart, so residuals are tiny and the coord payload compresses ~10-20x vs raw f32. Validity
// and channels ride the same lossless substrate. Little-endian; magic + version; readers reject
// unknown versions (no migration), per the format invariants. Atomic write-temp-rename.
#pragma once

#include "codec/lossless.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "core/vec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace fenix::io {

namespace detail {

inline constexpr u32 kSurfCoordQ = 16;      // coord quant: 1/16 voxel (≪ trilinear sensitivity)
inline constexpr u32 kSurfNormalQ = 16384;  // unit-normal quant: ~6e-5
inline constexpr u32 kSurfConfQ = 256;      // confidence quant: 1/256

// Plane predictor (left + up - upleft): kills the constant grid-step gradient in both
// directions, leaving only surface curvature + quant noise in the residual.
inline s32 surf_pred(const std::vector<s32>& q, s64 nu, s64 u, s64 v, usize i) {
    const usize w = static_cast<usize>(nu);
    if (u > 0 && v > 0) {
        const s64 p = static_cast<s64>(q[i - 1]) + static_cast<s64>(q[i - w]) - static_cast<s64>(q[i - w - 1]);
        return static_cast<s32>(std::clamp<s64>(p, INT32_MIN, INT32_MAX));
    }
    if (u > 0) return q[i - 1];
    if (v > 0) return q[i - w];
    return 0;
}

inline u32 zigzag32(s32 v) {
    return (static_cast<u32>(v) << 1) ^ static_cast<u32>(v >> 31);
}
inline s32 unzigzag32(u32 v) {
    return static_cast<s32>((v >> 1) ^ (~(v & 1) + 1));
}

// Quantize one component grid, residual-code against left (else up) neighbor, zigzag.
// `fill_invalid`: cells where valid==0 take the predictor exactly (residual 0 — invalid
// runs compress to nothing and the reader reconstructs deterministic filler there).
inline std::vector<u32> surf_encode_plane(s64 nu,
                                          s64 nv,
                                          u32 q,
                                          std::span<const u8> valid,
                                          auto&& comp) {  // comp(idx) -> f32
    const usize n = static_cast<usize>(nu * nv);
    std::vector<s32> qv(n);
    std::vector<u32> out(n);
    for (s64 v = 0; v < nv; ++v)
        for (s64 u = 0; u < nu; ++u) {
            const usize i = static_cast<usize>(v * nu + u);
            const s32 pred = surf_pred(qv, nu, u, v, i);
            s32 cur;
            if (!valid.empty() && valid[i] == 0) {
                cur = pred;
            } else {
                const f64 scaled = static_cast<f64>(comp(i)) * q;
                cur = static_cast<s32>(std::llround(std::clamp(scaled, -2.1e9, 2.1e9)));
            }
            qv[i] = cur;
            out[i] = zigzag32(cur - pred);
        }
    return out;
}

inline std::vector<f32> surf_decode_plane(s64 nu, s64 nv, u32 q, std::span<const u32> zz) {
    const usize n = static_cast<usize>(nu * nv);
    std::vector<s32> qv(n);
    std::vector<f32> out(n);
    for (s64 v = 0; v < nv; ++v)
        for (s64 u = 0; u < nu; ++u) {
            const usize i = static_cast<usize>(v * nu + u);
            const s32 pred = surf_pred(qv, nu, u, v, i);
            qv[i] = pred + unzigzag32(zz[i]);
            out[i] = static_cast<f32>(qv[i]) / static_cast<f32>(q);
        }
    return out;
}

inline void surf_put_blob(std::ofstream& f, const std::vector<u8>& b) {
    const u64 len = b.size();
    f.write(reinterpret_cast<const char*>(&len), sizeof len);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}

inline Expected<std::vector<u8>> surf_get_blob(std::ifstream& f) {
    u64 len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof len);
    if (!f || len > (u64{1} << 33)) return err(Errc::decode_error, "fxsurf: bad blob length");
    std::vector<u8> b(static_cast<usize>(len));
    f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(len));
    if (!f) return err(Errc::decode_error, "fxsurf: short blob");
    return b;
}

}  // namespace detail

inline Expected<void> write_fxsurf(const std::string& path, const Surface& s) {
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return err(Errc::io_error, "cannot open " + tmp);
    const char magic[4] = {'F', 'X', 'S', 'F'};
    const u32 version = 2;
    const u8 hc = s.has_channels() ? 1u : 0u;
    const u32 cq = detail::kSurfCoordQ, nq = detail::kSurfNormalQ, fq = detail::kSurfConfQ;
    f.write(magic, 4);
    f.write(reinterpret_cast<const char*>(&version), sizeof version);
    f.write(reinterpret_cast<const char*>(&hc), 1);
    f.write(reinterpret_cast<const char*>(&s.nu), sizeof s.nu);
    f.write(reinterpret_cast<const char*>(&s.nv), sizeof s.nv);
    f.write(reinterpret_cast<const char*>(&s.scale_u), sizeof s.scale_u);
    f.write(reinterpret_cast<const char*>(&s.scale_v), sizeof s.scale_v);
    f.write(reinterpret_cast<const char*>(&cq), sizeof cq);
    f.write(reinterpret_cast<const char*>(&nq), sizeof nq);
    f.write(reinterpret_cast<const char*>(&fq), sizeof fq);

    detail::surf_put_blob(f, codec::lossless_encode<u8>(s.valid));
    for (int c = 0; c < 3; ++c) {
        const auto zz = detail::surf_encode_plane(s.nu, s.nv, cq, s.valid, [&](usize i) {
            return c == 0 ? s.coord[i].z : c == 1 ? s.coord[i].y : s.coord[i].x;
        });
        detail::surf_put_blob(f, codec::lossless_encode<u32>(zz));
    }
    if (hc) {
        for (int c = 0; c < 3; ++c) {
            const auto zz = detail::surf_encode_plane(s.nu, s.nv, nq, s.valid, [&](usize i) {
                return c == 0 ? s.normal[i].z : c == 1 ? s.normal[i].y : s.normal[i].x;
            });
            detail::surf_put_blob(f, codec::lossless_encode<u32>(zz));
        }
        const auto zz = detail::surf_encode_plane(s.nu, s.nv, fq, s.valid, [&](usize i) { return s.conf[i]; });
        detail::surf_put_blob(f, codec::lossless_encode<u32>(zz));
    }
    f.close();
    if (!f) return err(Errc::io_error, "write failed: " + tmp);
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) return err(Errc::io_error, "rename failed: " + path);
    return {};
}

inline Expected<Surface> read_fxsurf(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return err(Errc::not_found, "cannot open " + path);
    char magic[4] = {};
    f.read(magic, 4);
    if (std::memcmp(magic, "FXSF", 4) != 0) return err(Errc::decode_error, "not a .fxsurf: " + path);
    u32 version = 0;
    f.read(reinterpret_cast<char*>(&version), sizeof version);
    if (version != 2) return err(Errc::unsupported, "fxsurf version " + std::to_string(version));
    u8 hc = 0;
    s64 nu = 0, nv = 0;
    f32 su = 1.0f, sv = 1.0f;
    u32 cq = 0, nq = 0, fq = 0;
    f.read(reinterpret_cast<char*>(&hc), 1);
    f.read(reinterpret_cast<char*>(&nu), sizeof nu);
    f.read(reinterpret_cast<char*>(&nv), sizeof nv);
    f.read(reinterpret_cast<char*>(&su), sizeof su);
    f.read(reinterpret_cast<char*>(&sv), sizeof sv);
    f.read(reinterpret_cast<char*>(&cq), sizeof cq);
    f.read(reinterpret_cast<char*>(&nq), sizeof nq);
    f.read(reinterpret_cast<char*>(&fq), sizeof fq);
    if (!f || nu <= 0 || nv <= 0 || nu * nv > (s64{1} << 30) || cq == 0 || nq == 0 || fq == 0)
        return err(Errc::decode_error, "bad fxsurf header");
    const usize n = static_cast<usize>(nu * nv);

    Surface s(nu, nv);
    s.scale_u = su;
    s.scale_v = sv;

    auto vb = detail::surf_get_blob(f);
    if (!vb) return std::unexpected(vb.error());
    auto valid = codec::lossless_decode<u8>(*vb, n);
    if (!valid) return std::unexpected(valid.error());
    s.valid = std::move(*valid);

    auto plane = [&](u32 q) -> Expected<std::vector<f32>> {
        auto b = detail::surf_get_blob(f);
        if (!b) return std::unexpected(b.error());
        auto zz = codec::lossless_decode<u32>(*b, n);
        if (!zz) return std::unexpected(zz.error());
        return detail::surf_decode_plane(nu, nv, q, *zz);
    };
    for (int c = 0; c < 3; ++c) {
        auto p = plane(cq);
        if (!p) return std::unexpected(p.error());
        for (usize i = 0; i < n; ++i) (c == 0 ? s.coord[i].z : c == 1 ? s.coord[i].y : s.coord[i].x) = (*p)[i];
    }
    if (hc) {
        s.alloc_channels();
        for (int c = 0; c < 3; ++c) {
            auto p = plane(nq);
            if (!p) return std::unexpected(p.error());
            for (usize i = 0; i < n; ++i) (c == 0 ? s.normal[i].z : c == 1 ? s.normal[i].y : s.normal[i].x) = (*p)[i];
        }
        auto p = plane(fq);
        if (!p) return std::unexpected(p.error());
        for (usize i = 0; i < n; ++i) s.conf[i] = (*p)[i];
    }
    return s;
}

}  // namespace fenix::io
