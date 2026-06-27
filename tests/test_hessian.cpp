// test_hessian.cpp — Hessian/Frangi plate detector on a synthetic sheet.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "segment/hessian.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::segment;

TEST(hessian_detects_bright_sheet_ridge) {
    // A bright papyrus sheet: Gaussian ridge centred at x=24 (peak), varying only across x.
    const s64 s = 48;
    Volume<f32> v = Volume<f32>::zeros({s, s, s});
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x)
                v(z, y, x) = 200.0f * std::exp(-0.5f * std::pow((static_cast<f32>(x) - 24.0f) / 2.0f, 2.0f));

    SheetField f = hessian_sheet(v.view(), {.sigmas = {1.5f, 3.0f}, .alpha = 0.5f});

    // At the ridge peak (centre), strongly plate-like, normal ~ x (across the sheet).
    const s64 z = 24, y = 24, x = 24;
    CHECK(f.sheetness(z, y, x) > 0.5f);
    Vec3f nrm = f.normal[static_cast<usize>((z * s + y) * s + x)];
    CHECK(std::abs(nrm.x) > 0.85f);
    CHECK(std::abs(nrm.z) < 0.4f);

    // Far from the sheet (flat dark region), low plateness.
    CHECK(f.sheetness(24, 24, 45) < 0.4f);
}
