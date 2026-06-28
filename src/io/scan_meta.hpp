// io/scan_meta.hpp — read a scroll volume's metadata.json (the bm18/nabu export sidecar in the
// open-data bucket) into the scan attributes preprocessing needs: voxel size, beam energy, and the
// reconstruction's Paganin/unsharp parameters (so deconvolution can be MATCHED to what the recon
// did, not guessed). Find-based JSON extraction (the keys we need are uniquely named); reuses the
// same lightweight approach as zarr.hpp's .zarray parsing — no JSON DOM dependency.
#pragma once

#include "core/core.hpp"
#include "io/s3.hpp"
#include "io/zarr.hpp"  // detail::json_string

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

namespace fenix::io {

// Scan + reconstruction attributes pulled from metadata.json. Distances/sizes normalized to
// micrometres; energy in keV. `*_present` is implicit: a field stays at its sentinel if absent.
struct ScanMeta {
    f32 voxel_um = 0.0f;            // sample pixel size -> voxel edge (micrometres)
    f32 energy_keV = 0.0f;          // beam energy
    f32 paganin_delta_beta = 0.0f;  // recon phase-retrieval delta/beta (blur strength)
    f32 unsharp_coeff = 0.0f;       // recon unsharp-mask amount
    f32 unsharp_sigma = 0.0f;       // recon unsharp-mask gaussian sigma (voxels)
    f32 window_min = 0.0f;          // f32 export window (intensity calibration)
    f32 window_max = 0.0f;
    std::string scan_radix;         // human id (e.g. "2.4um_PHerc-Paris4_B_HA")
};

namespace detail {
// First JSON number following "key": (handles ints/floats/scientific/sign). NaN-safe default.
inline std::optional<f64> json_number(const std::string& s, const std::string& key) {
    auto k = s.find("\"" + key + "\"");
    if (k == std::string::npos) return std::nullopt;
    auto colon = s.find(':', k + key.size() + 2);
    if (colon == std::string::npos) return std::nullopt;
    usize i = colon + 1;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    usize j = i;
    while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '-' || s[j] == '+' ||
                            s[j] == '.' || s[j] == 'e' || s[j] == 'E')) ++j;
    if (j == i) return std::nullopt;
    return std::strtod(s.substr(i, j - i).c_str(), nullptr);
}
inline f32 num_or(const std::string& s, const std::string& key, f32 dflt) {
    auto v = json_number(s, key);
    return v ? static_cast<f32>(*v) : dflt;
}
}  // namespace detail

// Parse the metadata.json text. samplePixelSize is in mm in the bm18 sidecar -> *1000 = um.
inline ScanMeta parse_scan_meta(const std::string& js) {
    ScanMeta m;
    m.voxel_um = detail::num_or(js, "samplePixelSize", 0.0f) * 1000.0f;
    m.energy_keV = detail::num_or(js, "energy", 0.0f);
    m.paganin_delta_beta = detail::num_or(js, "delta_beta", 0.0f);
    m.unsharp_coeff = detail::num_or(js, "unsharp_coeff", 0.0f);
    m.unsharp_sigma = detail::num_or(js, "unsharp_sigma", 0.0f);
    m.window_min = detail::num_or(js, "target_window_f32_min", 0.0f);
    m.window_max = detail::num_or(js, "target_window_f32_max", 0.0f);
    m.scan_radix = detail::json_string(js, "scanRadix");
    return m;
}

// Fetch + parse metadata.json from a local path or a remote (s3://, http(s)://) source. Accepts
// either a volume root (we append /metadata.json) or a direct path to the json file itself.
inline Expected<ScanMeta> fetch_scan_meta(const std::string& volume_root) {
    const bool is_file = volume_root.size() >= 5 &&
                         volume_root.compare(volume_root.size() - 5, 5, ".json") == 0;
    const std::string url = is_file ? volume_root : volume_root + "/metadata.json";
    if (is_remote(url)) {
        auto r = http_get(url, HttpConfig{});
        if (!r) return std::unexpected(r.error());
        if (!*r) return err(Errc::not_found, "metadata.json absent: " + url);
        return parse_scan_meta(std::string((*r)->begin(), (*r)->end()));
    }
    FILE* fp = std::fopen(url.c_str(), "rb");
    if (!fp) return err(Errc::not_found, "cannot open " + url);
    std::string js;
    char buf[4096];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, fp)) > 0;) js.append(buf, n);
    std::fclose(fp);
    return parse_scan_meta(js);
}

}  // namespace fenix::io
