// io/surface.hpp — the .fxsurf container: a hand-rolled binary serialization of core::Surface (the
// (u,v) grid of ZYX world coords + validity + optional normal/conf channels). v3: coords go through
// codec/tile2d's coordinate front-end — per-64²-tile affine fit + tangent-frame projection (3
// correlated channels → 1 height field + 2 near-zero remainders), DCT-64² + dead-zone quantization,
// rANS — with an encode-time-VERIFIED 3D max error ≤ coord_tau per valid cell (raw-tile fallback,
// never assumed; default 1/4 voxel — see kSurfCoordTau). Normals/conf ride the scalar front-end at their own tolerances; validity is rANS'd.
// Little-endian; magic + version; readers reject unknown versions (no migration). Atomic
// write-temp-rename.
#pragma once

#include "codec/lossless.hpp"
#include "codec/tile2d.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "core/vec.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace fenix::io {

namespace detail {

// Default 3D coord error bound. The GT surfaces themselves are only ~±1 voxel accurate
// (segmentation output), so 1/4 voxel is far inside the data's own noise; measured RD on a
// real PHercParis4 segment: 1/16→9.7x, 1/8→12.3x, 1/4→16.1x, 1/2→21.6x, 1→29.5x vs tifxyz.
inline constexpr f32 kSurfCoordTau = 0.25f;
inline constexpr f32 kSurfNormalTau = 1.0f / 4096.0f;  // per-component unit-normal error
inline constexpr f32 kSurfConfTau = 1.0f / 256.0f;     // confidence error

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

inline Expected<void> write_fxsurf(const std::string& path, const Surface& s,
                                   f32 coord_tau = detail::kSurfCoordTau) {
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return err(Errc::io_error, "cannot open " + tmp);
    const char magic[4] = {'F', 'X', 'S', 'F'};
    const u32 version = 3;
    const u8 hc = s.has_channels() ? 1u : 0u;
    const f32 ct = coord_tau, nt = detail::kSurfNormalTau, ft = detail::kSurfConfTau;
    f.write(magic, 4);
    f.write(reinterpret_cast<const char*>(&version), sizeof version);
    f.write(reinterpret_cast<const char*>(&hc), 1);
    f.write(reinterpret_cast<const char*>(&s.nu), sizeof s.nu);
    f.write(reinterpret_cast<const char*>(&s.nv), sizeof s.nv);
    f.write(reinterpret_cast<const char*>(&s.scale_u), sizeof s.scale_u);
    f.write(reinterpret_cast<const char*>(&s.scale_v), sizeof s.scale_v);
    f.write(reinterpret_cast<const char*>(&ct), sizeof ct);
    f.write(reinterpret_cast<const char*>(&nt), sizeof nt);
    f.write(reinterpret_cast<const char*>(&ft), sizeof ft);

    detail::surf_put_blob(f, codec::lossless_encode<u8>(s.valid));
    detail::surf_put_blob(f, codec::encode_coords2d(s.coord, s.valid, s.nu, s.nv, ct));
    if (hc) {
        const usize n = static_cast<usize>(s.nu * s.nv);
        std::vector<f32> plane(n);
        for (int c = 0; c < 3; ++c) {
            for (usize i = 0; i < n; ++i)
                plane[i] = c == 0 ? s.normal[i].z : c == 1 ? s.normal[i].y : s.normal[i].x;
            detail::surf_put_blob(f, codec::encode_field2d(plane, s.nu, s.nv, nt, nt, s.valid));
        }
        detail::surf_put_blob(f, codec::encode_field2d(s.conf, s.nu, s.nv, ft, ft, s.valid));
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
    if (version != 3) return err(Errc::unsupported, "fxsurf version " + std::to_string(version));
    u8 hc = 0;
    s64 nu = 0, nv = 0;
    f32 su = 1.0f, sv = 1.0f, ct = 0, nt = 0, ft = 0;
    f.read(reinterpret_cast<char*>(&hc), 1);
    f.read(reinterpret_cast<char*>(&nu), sizeof nu);
    f.read(reinterpret_cast<char*>(&nv), sizeof nv);
    f.read(reinterpret_cast<char*>(&su), sizeof su);
    f.read(reinterpret_cast<char*>(&sv), sizeof sv);
    f.read(reinterpret_cast<char*>(&ct), sizeof ct);
    f.read(reinterpret_cast<char*>(&nt), sizeof nt);
    f.read(reinterpret_cast<char*>(&ft), sizeof ft);
    if (!f || nu <= 0 || nv <= 0 || nu * nv > (s64{1} << 30) || !(ct > 0) || !(nt > 0) || !(ft > 0))
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

    auto cb = detail::surf_get_blob(f);
    if (!cb) return std::unexpected(cb.error());
    auto coords = codec::decode_coords2d(*cb, nu, nv);
    if (!coords) return std::unexpected(coords.error());
    s.coord = std::move(*coords);

    if (hc) {
        s.alloc_channels();
        for (int c = 0; c < 3; ++c) {
            auto b = detail::surf_get_blob(f);
            if (!b) return std::unexpected(b.error());
            auto p = codec::decode_field2d(*b, nu, nv);
            if (!p) return std::unexpected(p.error());
            for (usize i = 0; i < n; ++i)
                (c == 0 ? s.normal[i].z : c == 1 ? s.normal[i].y : s.normal[i].x) = (*p)[i];
        }
        auto b = detail::surf_get_blob(f);
        if (!b) return std::unexpected(b.error());
        auto p = codec::decode_field2d(*b, nu, nv);
        if (!p) return std::unexpected(p.error());
        s.conf = std::move(*p);
    }
    return s;
}

}  // namespace fenix::io
