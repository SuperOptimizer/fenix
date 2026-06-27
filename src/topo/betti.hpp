// topo/betti.hpp — discrete topology of a binary volume: Euler characteristic (exact
// cubical-complex cell counting) + Betti numbers (b0 components, b1 tunnels, b2 cavities).
// Foundation for eval TopoScore and postproc topo surgery. Cubical PH builds on this.
#pragma once

#include "core/core.hpp"
#include "geom/connected_components.hpp"

namespace fenix::topo {

struct Betti {
    s64 b0 = 0;   // connected components (26-conn fg, matching the closed-cube complex)
    s64 b1 = 0;   // independent tunnels/loops
    s64 b2 = 0;   // enclosed cavities
    s64 chi = 0;  // Euler characteristic = b0 - b1 + b2
};

// Euler characteristic of the foreground as a cubical complex (voxels = closed unit cubes):
// chi = V - E + F - C over the cells incident to foreground voxels.
inline s64 euler_characteristic(VolumeView<const u8> mask) {
    const Extent3 d = mask.dims();
    auto fg = [&](s64 z, s64 y, s64 x) {
        return z >= 0 && z < d.z && y >= 0 && y < d.y && x >= 0 && x < d.x && mask(z, y, x) != 0;
    };
    s64 V = 0, E = 0, F = 0, C = 0;
    // 0-cells: corners (cz,cy,cx), incident to voxels {cz-1,cz}x{cy-1,cy}x{cx-1,cx}.
    for (s64 cz = 0; cz <= d.z; ++cz)
        for (s64 cy = 0; cy <= d.y; ++cy)
            for (s64 cx = 0; cx <= d.x; ++cx) {
                bool p = false;
                for (s64 az = 0; az < 2 && !p; ++az)
                    for (s64 ay = 0; ay < 2 && !p; ++ay)
                        for (s64 ax = 0; ax < 2 && !p; ++ax)
                            p = fg(cz - 1 + az, cy - 1 + ay, cx - 1 + ax);
                V += p;
            }
    // 1-cells along each axis: edge spans one voxel-step on its axis, shared by 4 voxels.
    auto count_edges = [&](int axis) {
        s64 cnt = 0;
        const s64 ez = (axis == 0) ? d.z : d.z + 1;  // along-axis dim is voxel-count
        const s64 ey = (axis == 1) ? d.y : d.y + 1;
        const s64 ex = (axis == 2) ? d.x : d.x + 1;
        for (s64 cz = 0; cz < ez; ++cz)
            for (s64 cy = 0; cy < ey; ++cy)
                for (s64 cx = 0; cx < ex; ++cx) {
                    bool p = false;
                    // incident voxels: fixed along edge axis = c; the other two range {c-1,c}.
                    for (s64 b0 = 0; b0 < 2 && !p; ++b0)
                        for (s64 b1 = 0; b1 < 2 && !p; ++b1) {
                            s64 z = cz, y = cy, x = cx;
                            if (axis == 0) { y = cy - 1 + b0; x = cx - 1 + b1; }
                            else if (axis == 1) { z = cz - 1 + b0; x = cx - 1 + b1; }
                            else { z = cz - 1 + b0; y = cy - 1 + b1; }
                            p = fg(z, y, x);
                        }
                    cnt += p;
                }
        return cnt;
    };
    E = count_edges(0) + count_edges(1) + count_edges(2);
    // 2-cells: face perpendicular to `axis`, shared by 2 voxels stacked along that axis.
    auto count_faces = [&](int axis) {
        s64 cnt = 0;
        const s64 fz = (axis == 0) ? d.z + 1 : d.z;  // perp-axis dim is vertex-count
        const s64 fy = (axis == 1) ? d.y + 1 : d.y;
        const s64 fx = (axis == 2) ? d.x + 1 : d.x;
        for (s64 cz = 0; cz < fz; ++cz)
            for (s64 cy = 0; cy < fy; ++cy)
                for (s64 cx = 0; cx < fx; ++cx) {
                    bool p = false;
                    for (s64 s = 0; s < 2 && !p; ++s) {
                        s64 z = cz, y = cy, x = cx;
                        if (axis == 0) z = cz - 1 + s;
                        else if (axis == 1) y = cy - 1 + s;
                        else x = cx - 1 + s;
                        p = fg(z, y, x);
                    }
                    cnt += p;
                }
        return cnt;
    };
    F = count_faces(0) + count_faces(1) + count_faces(2);
    for (s64 i = 0; i < d.count(); ++i) C += mask.flat()[static_cast<usize>(i)] != 0 ? 1 : 0;
    return V - E + F - C;
}

// Count background components fully enclosed by foreground (not touching the volume border).
inline s64 enclosed_cavities(VolumeView<const u8> mask) {
    const Extent3 d = mask.dims();
    Volume<u8> bg(d);
    for (s64 i = 0; i < d.count(); ++i) bg.flat()[static_cast<usize>(i)] = mask.flat()[static_cast<usize>(i)] ? u8{0} : u8{1};
    auto cc = geom::connected_components(bg.view(), geom::Conn::Six);
    std::vector<u8> touches_border(static_cast<usize>(cc.count + 1), 0);
    VolumeView<s32> lbl = cc.labels.view();
    auto mark = [&](s64 z, s64 y, s64 x) {
        s32 l = lbl(z, y, x);
        if (l) touches_border[static_cast<usize>(l)] = 1;
    };
    for (s64 y = 0; y < d.y; ++y)
        for (s64 x = 0; x < d.x; ++x) { mark(0, y, x); mark(d.z - 1, y, x); }
    for (s64 z = 0; z < d.z; ++z)
        for (s64 x = 0; x < d.x; ++x) { mark(z, 0, x); mark(z, d.y - 1, x); }
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y) { mark(z, y, 0); mark(z, y, d.x - 1); }
    s64 cavities = 0;
    for (s32 l = 1; l <= cc.count; ++l) cavities += touches_border[static_cast<usize>(l)] ? 0 : 1;
    return cavities;
}

// Betti numbers via chi = b0 - b1 + b2, with b0 (26-conn components) and b2 (cavities) direct.
inline Betti betti_numbers(VolumeView<const u8> mask) {
    Betti b;
    b.chi = euler_characteristic(mask);
    b.b0 = geom::connected_components(mask, geom::Conn::TwentySix).count;
    b.b2 = enclosed_cavities(mask);
    b.b1 = b.b0 + b.b2 - b.chi;
    return b;
}

}  // namespace fenix::topo
