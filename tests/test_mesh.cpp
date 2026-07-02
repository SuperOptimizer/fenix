// test_mesh.cpp — Mesh OBJ/PLY I/O.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "geom/mesh.hpp"

#include <cmath>
#include <filesystem>

using namespace fenix;
using namespace fenix::geom;

static Mesh tetra() {
    Mesh m;
    m.vertices = {Vec3f{0, 0, 0}, Vec3f{0, 0, 1}, Vec3f{0, 1, 0}, Vec3f{1, 0, 0}};  // (z,y,x)
    m.tris = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
    return m;
}

TEST(obj_roundtrip) {
    auto m = tetra();
    auto p = (std::filesystem::temp_directory_path() / "fenix_mesh.obj").string();
    std::filesystem::remove(p);
    REQUIRE(write_obj(p, m).has_value());
    auto got = read_obj(p);
    REQUIRE(got.has_value());
    REQUIRE(got->vertex_count() == 4);
    REQUIRE(got->tri_count() == 4);
    // vertex 1 was (z=0,y=0,x=1) -> OBJ "v 1 0 0" -> read back .x=1
    CHECK(std::abs(got->vertices[1].x - 1.0f) < 1e-6f);
    CHECK(got->tris[3][0] == 1);
    CHECK(got->tris[3][2] == 3);
    std::filesystem::remove(p);
}

TEST(ply_writes_binary_header) {
    auto m = tetra();
    auto p = (std::filesystem::temp_directory_path() / "fenix_mesh.ply").string();
    std::filesystem::remove(p);
    REQUIRE(write_ply(p, m).has_value());
    std::ifstream f(p, std::ios::binary);
    std::string l0, l1;
    std::getline(f, l0);
    std::getline(f, l1);
    CHECK(l0 == "ply");
    CHECK(l1 == "format binary_little_endian 1.0");
    std::filesystem::remove(p);
}

namespace {
Expected<Mesh> read_obj_text(const std::string& contents) {
    auto p = (std::filesystem::temp_directory_path() / "fenix_mesh_malformed.obj").string();
    std::ofstream f(p, std::ios::trunc);
    f << contents;
    f.close();
    auto r = read_obj(p);
    std::filesystem::remove(p);
    return r;
}
}  // namespace

// read_obj must return Expected errors (never abort/UB) on malformed files: previously std::stoi on
// an empty/non-numeric face-index token threw under -fno-exceptions -> std::terminate.
TEST(read_obj_rejects_short_face_line) {
    // "f 1 2" has only two indices -> the third token is missing, not just non-numeric.
    auto r = read_obj_text("v 0 0 0\nv 0 0 1\nv 0 1 0\nf 1 2\n");
    CHECK(!r.has_value());
}

TEST(read_obj_rejects_nonnumeric_face_index) {
    auto r = read_obj_text("v 0 0 0\nv 0 0 1\nv 0 1 0\nf a b c\n");
    CHECK(!r.has_value());
}

TEST(read_obj_rejects_huge_face_index_overflow) {
    auto r = read_obj_text("v 0 0 0\nv 0 0 1\nv 0 1 0\nf 99999999999999999999 1 2\n");
    CHECK(!r.has_value());
}

TEST(read_obj_rejects_out_of_range_face_index) {
    // 3 vertices exist; index 1000000 is out of range.
    auto r = read_obj_text("v 0 0 0\nv 0 0 1\nv 0 1 0\nf 1 2 1000000\n");
    CHECK(!r.has_value());
}

TEST(read_obj_rejects_short_vertex_line) {
    auto r = read_obj_text("v 1 2\nv 0 0 1\nv 0 1 0\nf 1 2 3\n");
    CHECK(!r.has_value());
}

TEST(read_obj_accepts_negative_relative_face_index) {
    // "-1" means "the last vertex defined so far" (OBJ relative indexing).
    auto r = read_obj_text("v 0 0 0\nv 0 0 1\nv 0 1 0\nf 1 2 -1\n");
    REQUIRE(r.has_value());
    REQUIRE(r->tri_count() == 1);
    CHECK(r->tris[0][2] == 2);  // -1 resolves to the 3rd (last) vertex, index 2
}

TEST(read_obj_fan_triangulates_quad_faces) {
    Mesh m;
    m.vertices = {Vec3f{0, 0, 0}, Vec3f{0, 0, 1}, Vec3f{0, 1, 1}, Vec3f{0, 1, 0}};
    auto p = (std::filesystem::temp_directory_path() / "fenix_mesh_quad.obj").string();
    std::filesystem::remove(p);
    {
        std::ofstream f(p, std::ios::trunc);
        for (auto& v : m.vertices) f << "v " << v.x << " " << v.y << " " << v.z << "\n";
        f << "f 1 2 3 4\n";  // a quad -> should fan-triangulate to 2 tris, not drop vertex 4
    }
    auto got = read_obj(p);
    std::filesystem::remove(p);
    REQUIRE(got.has_value());
    CHECK(got->tri_count() == 2);
}
