// core/surface.hpp — the parametric surface type (the in-memory .fxsurf): a (u,v) grid
// where each cell holds a ZYX world coordinate + validity. The output of segmentation/
// flattening and the input to rendering. Named channels (winding, normal, ...) attach
// alongside; this is the coords+validity core.
#pragma once

#include "core/types.hpp"
#include "core/vec.hpp"

#include <vector>

namespace fenix {

struct Surface {
    s64 nu = 0, nv = 0;          // grid: u (around the wrap), v (along z)
    f32 scale_u = 1.0f, scale_v = 1.0f;  // grid-index -> nominal surface coord
    std::vector<Vec3f> coord;    // nu*nv ZYX world coords (row-major: v-major, u-fastest)
    std::vector<u8> valid;       // nu*nv validity (1 = real)
    // Optional named channels (empty = not computed). `normal` is the per-cell across-sheet
    // unit normal; `conf` is the data-term confidence at the snapped cell (>=1 == on a sheet).
    // The tracer fills these at the end of a grow; the patch-graph / winding fit consume them.
    std::vector<Vec3f> normal;   // nu*nv across-sheet unit normals
    std::vector<f32> conf;       // nu*nv data confidence

    Surface() = default;
    Surface(s64 nu_, s64 nv_)
        : nu(nu_), nv(nv_), coord(static_cast<usize>(nu_ * nv_)), valid(static_cast<usize>(nu_ * nv_), 0) {}

    [[nodiscard]] bool has_channels() const { return !normal.empty(); }
    void alloc_channels() {
        normal.assign(static_cast<usize>(nu * nv), Vec3f{0, 0, 0});
        conf.assign(static_cast<usize>(nu * nv), 0.0f);
    }

    [[nodiscard]] usize idx(s64 u, s64 v) const { return static_cast<usize>(v * nu + u); }
    Vec3f& at(s64 u, s64 v) { return coord[idx(u, v)]; }
    [[nodiscard]] const Vec3f& at(s64 u, s64 v) const { return coord[idx(u, v)]; }
    [[nodiscard]] bool is_valid(s64 u, s64 v) const { return valid[idx(u, v)] != 0; }
    void set(s64 u, s64 v, Vec3f c) {
        coord[idx(u, v)] = c;
        valid[idx(u, v)] = 1;
    }

    [[nodiscard]] s64 valid_count() const {
        s64 n = 0;
        for (u8 b : valid) n += b;
        return n;
    }
};

}  // namespace fenix
