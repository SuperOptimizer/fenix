// geom/mesh.hpp — triangle mesh type + OBJ/PLY I/O. Used by marching cubes, flatten,
// and interop export. Vertex component .x/.y/.z map to OBJ X/Y/Z (Vec3f is ZYX-tagged).
#pragma once

#include "core/core.hpp"

#include <array>
#include <fstream>
#include <sstream>
#include <string>
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

inline Expected<Mesh> read_obj(const std::string& path) {
    std::ifstream f(path);
    if (!f) return err(Errc::not_found, "cannot open " + path);
    Mesh m;
    for (std::string line; std::getline(f, line);) {
        if (line.size() < 2) continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            f32 x, y, z;
            ss >> x >> y >> z;
            m.vertices.push_back(Vec3f{z, y, x});  // Vec3f(z,y,x); .x=X .y=Y .z=Z
        } else if (tag == "vn") {
            f32 x, y, z;
            ss >> x >> y >> z;
            m.normals.push_back(Vec3f{z, y, x});
        } else if (tag == "f") {
            std::array<s32, 3> t{};
            for (int k = 0; k < 3; ++k) {
                std::string tok;
                ss >> tok;
                t[static_cast<usize>(k)] = std::stoi(tok.substr(0, tok.find('/'))) - 1;  // 1-based -> 0
            }
            m.tris.push_back(t);
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
