// test_eig.cpp — symmetric 3x3 eigensolver.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"

#include <cmath>

using namespace fenix;

TEST(diagonal_matrix_eigenvalues_descending) {
    auto e = sym_eig3<f32>(1.0f, 5.0f, 3.0f, 0, 0, 0);  // diag(1,5,3) over z,y,x
    CHECK(std::abs(e.values[0] - 5.0f) < 1e-4f);
    CHECK(std::abs(e.values[1] - 3.0f) < 1e-4f);
    CHECK(std::abs(e.values[2] - 1.0f) < 1e-4f);
}

TEST(eigenvectors_are_orthonormal_and_reconstruct) {
    // A symmetric structure-tensor-like matrix.
    f32 azz = 4, ayy = 1, axx = 1, azy = 1, azx = 0, ayx = 0.5f;
    auto e = sym_eig3<f32>(azz, ayy, axx, azy, azx, ayx);
    // descending
    CHECK(e.values[0] >= e.values[1]);
    CHECK(e.values[1] >= e.values[2]);
    // unit length + orthogonality
    for (std::size_t i = 0; i < 3; ++i) CHECK(std::abs(norm(e.vectors[i]) - 1.0f) < 1e-3f);
    CHECK(std::abs(dot(e.vectors[0], e.vectors[1])) < 1e-3f);
    CHECK(std::abs(dot(e.vectors[0], e.vectors[2])) < 1e-3f);
    // A * v0 ≈ λ0 * v0
    Vec3f v0 = e.vectors[0];
    Vec3f av{azz * v0.z + azy * v0.y + azx * v0.x, azy * v0.z + ayy * v0.y + ayx * v0.x,
             azx * v0.z + ayx * v0.y + axx * v0.x};
    Vec3f lv = v0 * e.values[0];
    CHECK(norm(av - lv) < 1e-3f);
}

TEST(sheetness_from_structure_tensor) {
    // Plate-like tensor: one large eigenvalue (across-sheet), two small (in-plane).
    auto e = sym_eig3<f32>(10.0f, 0.2f, 0.1f, 0, 0, 0);
    f32 sheetness = (e.values[0] - e.values[1]) / (e.values[0] + tol::eps);
    CHECK(sheetness > 0.9f);  // confident sheet
}
