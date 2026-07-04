// winding/model_io.hpp — the .fxmodel container: hand-rolled binary serialization of the
// fitted SpiralModel (umbilicus polyline + SVF flow lattice + per-slice affine + gap
// expander + dr/gauge). Raw f32 payloads — the model is tiny (KBs) next to the volumes it
// deforms; no codec needed. Version-gated, no back-compat (root CLAUDE.md §2.6).
#pragma once

#include "core/core.hpp"
#include "winding/spiral_model.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fenix::winding {

namespace detail {

template <typename T>
void put_pod(std::ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof v);
}

template <typename T>
void put_vec(std::ofstream& f, const std::vector<T>& v) {
    const u64 n = v.size();
    put_pod(f, n);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

template <typename T>
bool get_pod(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof v);
    return static_cast<bool>(f);
}

template <typename T>
bool get_vec(std::ifstream& f, std::vector<T>& v) {
    u64 n = 0;
    if (!get_pod(f, n) || n > (u64{1} << 30)) return false;
    v.resize(static_cast<usize>(n));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(f);
}

inline void put_flow_vol(std::ofstream& f, const Volume<f32>& v) {
    put_pod(f, v.dims().z);
    put_pod(f, v.dims().y);
    put_pod(f, v.dims().x);
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.dims().count() * sizeof(f32)));
}

inline bool get_flow_vol(std::ifstream& f, Volume<f32>& v) {
    Extent3 d{};
    if (!get_pod(f, d.z) || !get_pod(f, d.y) || !get_pod(f, d.x)) return false;
    if (d.z < 0 || d.y < 0 || d.x < 0 || d.count() > (s64{1} << 30)) return false;
    v = Volume<f32>::zeros(d);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(d.count() * sizeof(f32)));
    return static_cast<bool>(f);
}

}  // namespace detail

inline Expected<void> write_fxmodel(const std::string& path, const SpiralModel& m) {
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return err(Errc::io_error, "cannot open " + tmp);
    const char magic[4] = {'F', 'X', 'M', 'D'};
    const u32 version = 2;
    f.write(magic, 4);
    detail::put_pod(f, version);
    detail::put_vec(f, m.umbilicus.z);
    detail::put_vec(f, m.umbilicus.y);
    detail::put_vec(f, m.umbilicus.x);
    detail::put_pod(f, m.affine);
    detail::put_pod(f, m.affine_bands.z0);
    detail::put_pod(f, m.affine_bands.dz);
    detail::put_vec(f, m.affine_bands.bands);
    detail::put_pod(f, m.gap.dr);
    detail::put_vec(f, m.gap.logits);
    detail::put_pod(f, m.dr_per_winding);
    detail::put_pod(f, m.winding_offset);
    detail::put_pod(f, m.flow_steps);
    const u8 hf = m.has_flow ? 1u : 0u;
    detail::put_pod(f, hf);
    if (hf) {
        detail::put_pod(f, m.flow.lat_lo);
        detail::put_pod(f, m.flow.lat_scale);
        detail::put_flow_vol(f, m.flow.vz);
        detail::put_flow_vol(f, m.flow.vy);
        detail::put_flow_vol(f, m.flow.vx);
    }
    f.close();
    if (!f) return err(Errc::io_error, "write failed: " + tmp);
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) return err(Errc::io_error, "rename failed: " + path);
    return {};
}

inline Expected<SpiralModel> read_fxmodel(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return err(Errc::not_found, "cannot open " + path);
    char magic[4] = {};
    f.read(magic, 4);
    if (std::memcmp(magic, "FXMD", 4) != 0) return err(Errc::decode_error, "not a .fxmodel: " + path);
    u32 version = 0;
    if (!detail::get_pod(f, version)) return err(Errc::decode_error, "fxmodel: truncated header");
    if (version != 2) return err(Errc::unsupported, "fxmodel version " + std::to_string(version));
    SpiralModel m;
    u8 hf = 0;
    const bool ok = detail::get_vec(f, m.umbilicus.z) && detail::get_vec(f, m.umbilicus.y) &&
                    detail::get_vec(f, m.umbilicus.x) && detail::get_pod(f, m.affine) &&
                    detail::get_pod(f, m.affine_bands.z0) && detail::get_pod(f, m.affine_bands.dz) &&
                    detail::get_vec(f, m.affine_bands.bands) &&
                    detail::get_pod(f, m.gap.dr) && detail::get_vec(f, m.gap.logits) &&
                    detail::get_pod(f, m.dr_per_winding) && detail::get_pod(f, m.winding_offset) &&
                    detail::get_pod(f, m.flow_steps) && detail::get_pod(f, hf);
    if (!ok) return err(Errc::decode_error, "fxmodel: truncated body");
    m.has_flow = hf != 0;
    if (m.has_flow) {
        if (!detail::get_pod(f, m.flow.lat_lo) || !detail::get_pod(f, m.flow.lat_scale) ||
            !detail::get_flow_vol(f, m.flow.vz) || !detail::get_flow_vol(f, m.flow.vy) ||
            !detail::get_flow_vol(f, m.flow.vx))
            return err(Errc::decode_error, "fxmodel: truncated flow");
    }
    return m;
}

}  // namespace fenix::winding
