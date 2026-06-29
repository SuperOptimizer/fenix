// io/surface.hpp — the .fxsurf container: a hand-rolled binary serialization of core::Surface (the
// (u,v) grid of ZYX world coords + validity + optional normal/conf channels). Dense layout (the surface
// is ~10% occupied, but dense round-trips trivially and is what the streaming tracer dumps per tile so
// the output never has to stay resident). Little-endian; magic + version; readers reject unknown
// versions (no migration), per the format invariants. Atomic write-temp-rename.
#pragma once

#include "core/core.hpp"
#include "core/surface.hpp"
#include "core/vec.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fenix::io {

inline Expected<void> write_fxsurf(const std::string& path, const Surface& s) {
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return err(Errc::io_error, "cannot open " + tmp);
    const char magic[4] = {'F', 'X', 'S', 'F'};
    const u32 version = 1;
    const u8 hc = s.has_channels() ? 1u : 0u;
    f.write(magic, 4);
    f.write(reinterpret_cast<const char*>(&version), sizeof version);
    f.write(reinterpret_cast<const char*>(&hc), 1);
    f.write(reinterpret_cast<const char*>(&s.nu), sizeof s.nu);
    f.write(reinterpret_cast<const char*>(&s.nv), sizeof s.nv);
    f.write(reinterpret_cast<const char*>(&s.scale_u), sizeof s.scale_u);
    f.write(reinterpret_cast<const char*>(&s.scale_v), sizeof s.scale_v);
    const auto n = static_cast<std::streamsize>(static_cast<usize>(s.nu * s.nv));
    f.write(reinterpret_cast<const char*>(s.coord.data()), n * static_cast<std::streamsize>(sizeof(Vec3f)));
    f.write(reinterpret_cast<const char*>(s.valid.data()), n);
    if (hc) {
        f.write(reinterpret_cast<const char*>(s.normal.data()), n * static_cast<std::streamsize>(sizeof(Vec3f)));
        f.write(reinterpret_cast<const char*>(s.conf.data()), n * static_cast<std::streamsize>(sizeof(f32)));
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
    if (version != 1) return err(Errc::unsupported, "fxsurf version " + std::to_string(version));
    u8 hc = 0;
    s64 nu = 0, nv = 0;
    f32 su = 1.0f, sv = 1.0f;
    f.read(reinterpret_cast<char*>(&hc), 1);
    f.read(reinterpret_cast<char*>(&nu), sizeof nu);
    f.read(reinterpret_cast<char*>(&nv), sizeof nv);
    f.read(reinterpret_cast<char*>(&su), sizeof su);
    f.read(reinterpret_cast<char*>(&sv), sizeof sv);
    if (nu < 0 || nv < 0 || nu * nv > (s64{1} << 30)) return err(Errc::decode_error, "bad fxsurf dims");
    Surface s(nu, nv);
    s.scale_u = su;
    s.scale_v = sv;
    const auto n = static_cast<std::streamsize>(static_cast<usize>(nu * nv));
    f.read(reinterpret_cast<char*>(s.coord.data()), n * static_cast<std::streamsize>(sizeof(Vec3f)));
    f.read(reinterpret_cast<char*>(s.valid.data()), n);
    if (hc) {
        s.alloc_channels();
        f.read(reinterpret_cast<char*>(s.normal.data()), n * static_cast<std::streamsize>(sizeof(Vec3f)));
        f.read(reinterpret_cast<char*>(s.conf.data()), n * static_cast<std::streamsize>(sizeof(f32)));
    }
    if (!f) return err(Errc::decode_error, "short read: " + path);
    return s;
}

}  // namespace fenix::io
