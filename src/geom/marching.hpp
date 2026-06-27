// geom/marching.hpp — isosurface extraction via marching TETRAHEDRA. Each cube is split
// into 6 tets (sharing the 0-7 diagonal); each tet has only 16 sign cases with no
// ambiguity, so the surface is watertight by construction (unlike marching cubes' 256-case
// ambiguities). Produces a triangle-soup Mesh (coincident shared vertices). Used for
// surface-from-mask, eval surface metrics, flatten input. See geom/CLAUDE.md.
#pragma once

#include "core/core.hpp"
#include "geom/mesh.hpp"

#include <array>

namespace fenix::geom {

// Extract the `iso` level set of `field` as a triangle mesh. Vertices in ZYX coords.
inline Mesh marching_tetrahedra(VolumeView<const f32> field, f32 iso) {
    const Extent3 d = field.dims();
    Mesh mesh;

    // Cube corner offsets, index = (dz<<2)|(dy<<1)|dx.
    constexpr std::array<std::array<int, 3>, 8> corner = {{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                                           {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}}};
    // 6 tets sharing the 0-7 diagonal.
    constexpr std::array<std::array<int, 4>, 6> tets = {{{0, 7, 1, 3}, {0, 7, 3, 2}, {0, 7, 2, 6},
                                                        {0, 7, 6, 4}, {0, 7, 4, 5}, {0, 7, 5, 1}}};

    auto vtx_pos = [&](int ci, s64 z, s64 y, s64 x) {
        return Vec3f{static_cast<f32>(z + corner[static_cast<usize>(ci)][0]),
                     static_cast<f32>(y + corner[static_cast<usize>(ci)][1]),
                     static_cast<f32>(x + corner[static_cast<usize>(ci)][2])};
    };

    for (s64 z = 0; z + 1 < d.z; ++z)
        for (s64 y = 0; y + 1 < d.y; ++y)
            for (s64 x = 0; x + 1 < d.x; ++x) {
                std::array<f32, 8> val{};
                for (int ci = 0; ci < 8; ++ci)
                    val[static_cast<usize>(ci)] = field(z + corner[static_cast<usize>(ci)][0],
                                                         y + corner[static_cast<usize>(ci)][1],
                                                         x + corner[static_cast<usize>(ci)][2]);
                for (const auto& tet : tets) {
                    // Collect crossing points on the tet's 6 edges (vertex pairs).
                    static constexpr std::array<std::array<int, 2>, 6> tedges = {
                        {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
                    std::array<Vec3f, 6> cross{};
                    int nc = 0;
                    for (const auto& e : tedges) {
                        const int a = tet[static_cast<usize>(e[0])], b = tet[static_cast<usize>(e[1])];
                        const f32 va = val[static_cast<usize>(a)], vb = val[static_cast<usize>(b)];
                        if ((va < iso) != (vb < iso)) {
                            const f32 t = (iso - va) / (vb - va);
                            const Vec3f pa = vtx_pos(a, z, y, x), pb = vtx_pos(b, z, y, x);
                            cross[static_cast<usize>(nc++)] = pa + (pb - pa) * t;
                        }
                    }
                    auto emit = [&](Vec3f p0, Vec3f p1, Vec3f p2) {
                        const s32 base = static_cast<s32>(mesh.vertices.size());
                        mesh.vertices.push_back(p0);
                        mesh.vertices.push_back(p1);
                        mesh.vertices.push_back(p2);
                        mesh.tris.push_back({base, base + 1, base + 2});
                    };
                    if (nc == 3) {
                        emit(cross[0], cross[1], cross[2]);
                    } else if (nc == 4) {
                        emit(cross[0], cross[1], cross[2]);
                        emit(cross[0], cross[2], cross[3]);
                    }
                }
            }
    return mesh;
}

}  // namespace fenix::geom
