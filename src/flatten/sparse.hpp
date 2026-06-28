// flatten/sparse.hpp — first-party sparse linear algebra for the parameterization solves
// (cotangent-Laplacian systems). CSR matrix + Conjugate Gradient (SPD). No Eigen. f64 internally
// (accumulation-sensitive, per conventions). Built from triplets; duplicate entries are summed by
// matvec, so assembly can just push per-edge contributions.
#pragma once

#include "core/types.hpp"

#include <cmath>
#include <span>
#include <vector>

namespace fenix::flatten {

struct Triplets {
    std::vector<s64> r, c;
    std::vector<f64> v;
    void add(s64 i, s64 j, f64 x) { r.push_back(i); c.push_back(j); v.push_back(x); }
    [[nodiscard]] usize size() const { return r.size(); }
};

struct Csr {
    s64 n = 0;
    std::vector<s64> rowptr, col;
    std::vector<f64> val;

    static Csr from_triplets(s64 n, const Triplets& t) {
        Csr a;
        a.n = n;
        a.rowptr.assign(static_cast<usize>(n + 1), 0);
        for (s64 i : t.r) a.rowptr[static_cast<usize>(i) + 1]++;
        for (s64 i = 0; i < n; ++i) a.rowptr[static_cast<usize>(i) + 1] += a.rowptr[static_cast<usize>(i)];
        a.col.resize(t.size());
        a.val.resize(t.size());
        std::vector<s64> cur(a.rowptr.begin(), a.rowptr.end() - 1);
        for (usize k = 0; k < t.size(); ++k) {
            const s64 i = t.r[k];
            const s64 p = cur[static_cast<usize>(i)]++;
            a.col[static_cast<usize>(p)] = t.c[k];
            a.val[static_cast<usize>(p)] = t.v[k];
        }
        return a;
    }

    void matvec(std::span<const f64> x, std::span<f64> y) const {
        for (s64 i = 0; i < n; ++i) {
            f64 s = 0;
            for (s64 k = rowptr[static_cast<usize>(i)]; k < rowptr[static_cast<usize>(i) + 1]; ++k)
                s += val[static_cast<usize>(k)] * x[static_cast<usize>(col[static_cast<usize>(k)])];
            y[static_cast<usize>(i)] = s;
        }
    }
};

// Jacobi-preconditioned CG for SPD A x = b (warm-started). Returns the relative residual reached.
// Diagonal preconditioning is essential — unpreconditioned CG on a 2D-Poisson/cotan system needs
// far more iterations to converge (and a half-converged solve corrupts the local/global loop).
inline f64 cg(const Csr& a, std::span<const f64> b, std::span<f64> x, int maxit, f64 tol) {
    const s64 n = a.n;
    std::vector<f64> r(static_cast<usize>(n)), z(static_cast<usize>(n)), p(static_cast<usize>(n)), ap(static_cast<usize>(n)), invd(static_cast<usize>(n), 1.0);
    for (s64 i = 0; i < n; ++i) {  // diagonal of A (sum duplicate entries)
        f64 d = 0;
        for (s64 k = a.rowptr[static_cast<usize>(i)]; k < a.rowptr[static_cast<usize>(i) + 1]; ++k) if (a.col[static_cast<usize>(k)] == i) d += a.val[static_cast<usize>(k)];
        invd[static_cast<usize>(i)] = d > 1e-30 ? 1.0 / d : 1.0;
    }
    a.matvec(x, ap);
    f64 rz = 0, bn = 0;
    for (s64 i = 0; i < n; ++i) { r[static_cast<usize>(i)] = b[static_cast<usize>(i)] - ap[static_cast<usize>(i)]; z[static_cast<usize>(i)] = invd[static_cast<usize>(i)] * r[static_cast<usize>(i)]; p[static_cast<usize>(i)] = z[static_cast<usize>(i)]; rz += r[static_cast<usize>(i)] * z[static_cast<usize>(i)]; bn += b[static_cast<usize>(i)] * b[static_cast<usize>(i)]; }
    if (bn < 1e-30) bn = 1;
    f64 rs = 0; for (s64 i = 0; i < n; ++i) rs += r[static_cast<usize>(i)] * r[static_cast<usize>(i)];
    for (int it = 0; it < maxit; ++it) {
        if (rs / bn < tol * tol) break;
        a.matvec(p, ap);
        f64 pap = 0;
        for (s64 i = 0; i < n; ++i) pap += p[static_cast<usize>(i)] * ap[static_cast<usize>(i)];
        if (std::abs(pap) < 1e-30) break;
        const f64 alpha = rz / pap;
        rs = 0;
        for (s64 i = 0; i < n; ++i) { x[static_cast<usize>(i)] += alpha * p[static_cast<usize>(i)]; r[static_cast<usize>(i)] -= alpha * ap[static_cast<usize>(i)]; rs += r[static_cast<usize>(i)] * r[static_cast<usize>(i)]; }
        f64 rz2 = 0;
        for (s64 i = 0; i < n; ++i) { z[static_cast<usize>(i)] = invd[static_cast<usize>(i)] * r[static_cast<usize>(i)]; rz2 += r[static_cast<usize>(i)] * z[static_cast<usize>(i)]; }
        const f64 beta = rz2 / rz;
        for (s64 i = 0; i < n; ++i) p[static_cast<usize>(i)] = z[static_cast<usize>(i)] + beta * p[static_cast<usize>(i)];
        rz = rz2;
    }
    return std::sqrt(rs / bn);
}

}  // namespace fenix::flatten
