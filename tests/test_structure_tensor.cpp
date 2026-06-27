// test_structure_tensor.cpp — sheet detection on a synthetic planar sheet.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "segment/structure_tensor.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::segment;

TEST(structure_tensor_detects_planar_sheet_normal) {
    // A papyrus-like sheet: a bright ridge along the y-z plane, varying only across x.
    // Across-sheet direction = x, so the principal eigenvector (normal) should be ~x and
    // sheetness should be high on the sheet flanks.
    const s64 s = 48;
    Volume<f32> v = Volume<f32>::zeros({s, s, s});
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x)
                v(z, y, x) = 150.0f * std::exp(-0.5f * std::pow((static_cast<f32>(x) - 24.0f) / 3.0f, 2.0f));

    SheetField f = structure_tensor(v.view(), {.sigma_grad = 1.0f, .sigma_tensor = 2.0f});

    // Sample a flank voxel (on the slope of the ridge), centred in y,z.
    const s64 z = 24, y = 24, x = 20;
    const usize idx = static_cast<usize>((z * s + y) * s + x);
    Vec3f nrm = f.normal[idx];
    // Normal should be (almost) axis-aligned with x (zyx -> .x component dominant).
    CHECK(std::abs(nrm.x) > 0.9f);
    CHECK(std::abs(nrm.z) < 0.3f);
    CHECK(std::abs(nrm.y) < 0.3f);
    // Strongly sheet-like there.
    CHECK(f.sheetness(z, y, x) > 0.7f);

    // In a flat interior region far from the ridge, sheetness is low.
    CHECK(f.sheetness(24, 24, 45) < 0.5f);
}
