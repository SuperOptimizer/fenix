// test_tracer.cpp — NLLS surface tracer converges a patch onto a sheet.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "segment/tracer.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::segment;

TEST(tracer_converges_patch_onto_planar_sheet) {
    // Sheetness field peaking on the plane x = 24 (Gaussian ridge across x).
    const s64 s = 48;
    Volume<f32> sheet = Volume<f32>::zeros({s, s, s});
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x)
                sheet(z, y, x) = std::exp(-0.5f * std::pow((static_cast<f32>(x) - 24.0f) / 2.0f, 2.0f));

    // Seed slightly OFF the sheet; normal points across it (+x).
    Surface patch = trace_patch(sheet.view(), Vec3f{24, 24, 22.0f}, Vec3f{0, 0, 1}, 8, 8,
                                {.unit = 2.0f, .iters = 250, .lr = 0.2f});

    // All corners should converge to x ~ 24 (onto the sheet).
    f32 max_x_dev = 0;
    for (s64 v = 0; v < patch.nv; ++v)
        for (s64 u = 0; u < patch.nu; ++u) max_x_dev = std::max(max_x_dev, std::abs(patch.at(u, v).x - 24.0f));
    CHECK(max_x_dev < 1.5f);  // pulled onto the high-sheetness plane

    // Grid stays regular: neighbour spacing close to the unit (2.0).
    f32 worst = 0;
    for (s64 v = 0; v < patch.nv; ++v)
        for (s64 u = 0; u + 1 < patch.nu; ++u)
            worst = std::max(worst, std::abs(norm(patch.at(u + 1, v) - patch.at(u, v)) - 2.0f));
    CHECK(worst < 1.0f);
}
