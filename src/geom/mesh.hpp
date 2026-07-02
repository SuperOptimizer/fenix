// geom/mesh.hpp — triangle mesh type + OBJ/PLY I/O. Used by marching cubes, flatten,
// and interop export. Vertex component .x/.y/.z map to OBJ X/Y/Z (Vec3f is ZYX-tagged).
#pragma once

#include "core/core.hpp"

#include <array>
#include <charconv>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::geom {

struct Mesh {
    std::vector<Vec3f> vertices;
    std::vector<std::array<s32, 3>> tris;
    std::vector<Vec3f> normals;                  // optional; empty or same size as vertices
    std::vector<std::array<u8, 3>> colors;       // optional per-vertex RGB; empty or == vertices

    [[nodiscard]] s64 vertex_count() const { return static_cast<s64>(vertices.size()); }
    [[nodiscard]] s64 tri_count() const { return static_cast<s64>(tris.size()); }
};

inline Expected<void> write_obj(const std::string& path, const Mesh& m) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) return err(Errc::io_error, "cannot create " + path);
    f << "# fenix mesh\n";
    for (const Vec3f& v : m.vertices) f << "v " << v.x << " " << v.y << " " << v.z << "\n";
    for (const Vec3f& n : m.normals) f << "vn " << n.x << " " << n.y << " " << n.z << "\n";
    for (const auto& t : m.tris) f << "f " << t[0] + 1 << " " << t[1] + 1 << " " << t[2] + 1 << "\n";
    return f ? Expected<void>{} : err(Errc::io_error, "write failed");
}

namespace detail {

// Parse one float token with std::from_chars (no exceptions, no locale, rejects trailing garbage).
[[nodiscard]] inline Expected<f32> parse_f32_tok(std::string_view tok) {
    f32 v{};
    const auto r = std::from_chars(tok.data(), tok.data() + tok.size(), v);
    if (r.ec != std::errc{} || r.ptr != tok.data() + tok.size())
        return err(Errc::decode_error, "obj: bad float token '" + std::string(tok) + "'");
    return v;
}

// Parse one OBJ face-vertex index token ("N", "N/T", "N/T/N", or with negative/relative N). Returns
// the raw 1-based-or-negative OBJ index (resolved against vertex_count by the caller), never -1-offset
// blindly (that mangles OBJ's legal negative relative indices).
[[nodiscard]] inline Expected<s64> parse_face_index_tok(std::string_view tok) {
    const auto slash = tok.find('/');
    const std::string_view num = tok.substr(0, slash);
    if (num.empty()) return err(Errc::decode_error, "obj: empty face index token");
    s64 v{};
    const auto r = std::from_chars(num.data(), num.data() + num.size(), v);
    if (r.ec != std::errc{} || r.ptr != num.data() + num.size())
        return err(Errc::decode_error, "obj: bad face index token '" + std::string(tok) + "'");
    if (v == 0) return err(Errc::decode_error, "obj: face index 0 is invalid (OBJ is 1-based)");
    return v;
}

}  // namespace detail

inline Expected<Mesh> read_obj(const std::string& path) {
    std::ifstream f(path);
    if (!f) return err(Errc::not_found, "cannot open " + path);
    Mesh m;
    for (std::string line; std::getline(f, line);) {
        if (line.size() < 2) continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v" || tag == "vn") {
            std::array<f32, 3> c{};
            for (int k = 0; k < 3; ++k) {
                std::string tok;
                if (!(ss >> tok)) return err(Errc::decode_error, "obj: short '" + tag + "' line: " + line);
                auto pv = detail::parse_f32_tok(tok);
                if (!pv) return std::unexpected(pv.error());
                c[static_cast<usize>(k)] = *pv;
            }
            (tag == "v" ? m.vertices : m.normals).push_back(Vec3f{c[2], c[1], c[0]});  // ZYX <- XYZ
        } else if (tag == "f") {
            std::vector<s64> raw;  // 1-based-or-negative OBJ indices, resolved after the loop
            for (std::string tok; ss >> tok;) {
                auto pv = detail::parse_face_index_tok(tok);
                if (!pv) return std::unexpected(pv.error());
                raw.push_back(*pv);
            }
            if (raw.size() < 3) return err(Errc::decode_error, "obj: face with <3 indices: " + line);
            // Fan-triangulate polygons (v0,vi,vi+1) instead of silently dropping vertices past 3.
            for (usize k = 1; k + 1 < raw.size(); ++k) {
                std::array<s32, 3> t{};
                const std::array<s64, 3> tri_raw{raw[0], raw[k], raw[k + 1]};
                for (int c = 0; c < 3; ++c) {
                    // Resolve OBJ 1-based (positive) or relative (negative, counts back from the
                    // CURRENT vertex count at this point in the file) indices; forward references are
                    // rejected (this reader requires vertices to precede faces).
                    const s64 raw_i = tri_raw[static_cast<usize>(c)];
                    const s64 vcount = static_cast<s64>(m.vertices.size());
                    const s64 idx0 = raw_i > 0 ? raw_i - 1 : vcount + raw_i;  // 0-based
                    if (idx0 < 0 || idx0 >= vcount)
                        return err(Errc::decode_error, "obj: face index " + std::to_string(raw_i) +
                                                            " out of range (" + std::to_string(vcount) +
                                                            " vertices): " + line);
                    t[static_cast<usize>(c)] = static_cast<s32>(idx0);
                }
                m.tris.push_back(t);
            }
        }
    }
    return m;
}

// Minimal binary-little-endian PLY writer (vertices + triangle faces; optional vertex RGB).
inline Expected<void> write_ply(const std::string& path, const Mesh& m) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return err(Errc::io_error, "cannot create " + path);
    const bool rgb = m.colors.size() == m.vertices.size() && !m.colors.empty();
    f << "ply\nformat binary_little_endian 1.0\n";
    f << "element vertex " << m.vertices.size() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    if (rgb) f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    f << "element face " << m.tris.size() << "\n";
    f << "property list uchar int vertex_indices\n";
    f << "end_header\n";
    for (std::size_t i = 0; i < m.vertices.size(); ++i) {
        const Vec3f& v = m.vertices[i];
        f.write(reinterpret_cast<const char*>(&v.x), 4);
        f.write(reinterpret_cast<const char*>(&v.y), 4);
        f.write(reinterpret_cast<const char*>(&v.z), 4);
        if (rgb) f.write(reinterpret_cast<const char*>(m.colors[i].data()), 3);
    }
    for (const auto& t : m.tris) {
        u8 n = 3;
        f.write(reinterpret_cast<const char*>(&n), 1);
        f.write(reinterpret_cast<const char*>(t.data()), 3 * sizeof(s32));
    }
    return f ? Expected<void>{} : err(Errc::io_error, "write failed");
}

}  // namespace fenix::geom
