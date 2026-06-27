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
