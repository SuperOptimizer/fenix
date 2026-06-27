// core/eig.hpp — THE symmetric 3x3 eigensolver (cyclic Jacobi). One copy for the whole
// codebase: structure tensor, Hessian/Frangi, OOF, PCA sheet repair all use this.
// (taberna had three divergent copies with inconsistent sort orders — the smell we kill.)
// Returns eigenvalues sorted DESCENDING with their orthonormal eigenvectors.
#pragma once

#include "core/types.hpp"
#include "core/vec.hpp"

#include <array>
#include <cmath>

namespace fenix {

template <class T>
struct SymEig3 {
    std::array<T, 3> values{};      // eigenvalues, descending
    std::array<Vec3<T>, 3> vectors{};  // matching orthonormal eigenvectors (ZYX components)
};

// Symmetric matrix given by its 6 unique entries (a = [[azz,azy,azx],[azy,ayy,ayx],
// [azx,ayx,axx]]). Cyclic Jacobi rotations; converges in a handful of sweeps for 3x3.
template <class T>
SymEig3<T> sym_eig3(T azz, T ayy, T axx, T azy, T azx, T ayx) {
    T a[3][3] = {{azz, azy, azx}, {azy, ayy, ayx}, {azx, ayx, axx}};
    T v[3][3] = {{T(1), T(0), T(0)}, {T(0), T(1), T(0)}, {T(0), T(0), T(1)}};

    for (int sweep = 0; sweep < 50; ++sweep) {
        T off = std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]);
        if (off < T(1e-20)) break;
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (std::abs(a[p][q]) < T(1e-20)) continue;
                T theta = (a[q][q] - a[p][p]) / (T(2) * a[p][q]);
                T t = (theta >= T(0) ? T(1) : T(-1)) /
                      (std::abs(theta) + std::sqrt(theta * theta + T(1)));
                T c = T(1) / std::sqrt(t * t + T(1));
                T s = t * c;
                // Apply rotation J^T A J.
                for (int k = 0; k < 3; ++k) {
                    T akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    T apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                    T vkp = v[k][p], vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }

    SymEig3<T> r;
    int idx[3] = {0, 1, 2};
    T ev[3] = {a[0][0], a[1][1], a[2][2]};
    // sort indices by eigenvalue descending (simple 3-element sort)
    for (int i = 0; i < 2; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (ev[idx[j]] > ev[idx[i]]) {
                int tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
    for (usize i = 0; i < 3; ++i) {
        const int k = idx[i];
        r.values[i] = ev[k];
        r.vectors[i] = Vec3<T>{v[0][k], v[1][k], v[2][k]};
    }
    return r;
}

}  // namespace fenix
