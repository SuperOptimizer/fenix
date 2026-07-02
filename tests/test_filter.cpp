// test_filter.cpp — core/filter.hpp: separable Gaussian blur, incl. the small-line reflect fix
// (filter.hpp:29 used to read out of bounds when the kernel radius >= the line length).
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/filter.hpp"
#include "core/test.hpp"

#include <cmath>

using namespace fenix;

namespace {
// Blur a volume of the given dims filled with a constant value; a correctly-normalized kernel
// must leave a constant field unchanged (mass-preserving), which is exactly what a truncated
// (non-renormalized) kernel would violate on the small-line branch.
void check_constant_preserved(Extent3 dims, f32 sigma) {
    Volume<f32> v(dims);
    for (s64 i = 0; i < dims.count(); ++i) v.flat()[static_cast<usize>(i)] = 7.0f;
    gaussian_blur(v.view(), sigma);
    for (s64 i = 0; i < dims.count(); ++i)
        CHECK(std::abs(v.flat()[static_cast<usize>(i)] - 7.0f) < 1e-3f);
}
}  // namespace

// Dimension of 1 on one axis: the reflect fixed-point (n==1) case, and every axis's line length is
// <= the kernel radius at sigma=2.0 (r=6) for the other axes too.
TEST(gaussian_blur_dim_1_sigma_2_no_oob) {
    check_constant_preserved(Extent3{1, 8, 8}, 2.0f);
    check_constant_preserved(Extent3{8, 1, 8}, 2.0f);
    check_constant_preserved(Extent3{8, 8, 1}, 2.0f);
    check_constant_preserved(Extent3{1, 1, 1}, 2.0f);
}

// Dimension of 2: r=6 >> len-1=1, so single-application reflect would read far out of bounds
// (reflect(-6,2) = 6, reflect(1+6,2) = -5) before the iterated-reflect fix.
TEST(gaussian_blur_dim_2_sigma_2_no_oob) {
    check_constant_preserved(Extent3{2, 10, 10}, 2.0f);
    check_constant_preserved(Extent3{10, 2, 10}, 2.0f);
    check_constant_preserved(Extent3{10, 10, 2}, 2.0f);
}

// Dimension of 4: r=6 > len-1=3, still triggers the small-line (len < ksz) branch.
TEST(gaussian_blur_dim_4_sigma_2_no_oob) {
    check_constant_preserved(Extent3{4, 12, 12}, 2.0f);
    check_constant_preserved(Extent3{12, 4, 12}, 2.0f);
    check_constant_preserved(Extent3{12, 12, 4}, 2.0f);
}

// A non-constant small-line field must still stay bounded (no OOB garbage swamping the result) and
// keep the blur's mass-preservation property (sum before ~= sum after, since reflect boundary is a
// Neumann-like condition that conserves total mass on a finite line).
TEST(gaussian_blur_small_line_preserves_mass) {
    const Extent3 dims{3, 1, 5};  // y=1 triggers the small-line branch at sigma=2 (r=6 > 0)
    Volume<f32> v(dims);
    f32 sum_before = 0.0f;
    for (s64 i = 0; i < dims.count(); ++i) {
        const f32 val = static_cast<f32>(i) * 1.3f;
        v.flat()[static_cast<usize>(i)] = val;
        sum_before += val;
    }
    gaussian_blur(v.view(), 2.0f);
    f32 sum_after = 0.0f;
    for (s64 i = 0; i < dims.count(); ++i) {
        const f32 x = v.flat()[static_cast<usize>(i)];
        CHECK(std::isfinite(x));
        CHECK(std::abs(x) < 1e4f);  // no OOB-garbage blowup
        sum_after += x;
    }
    // Each 1D pass on a reflect boundary conserves the line sum; 3 passes (z,y,x) still conserve
    // the volume sum up to the accumulated float error of the interior branch-free path.
    CHECK(std::abs(sum_after - sum_before) < 1.0f);
}

// Interior (large-line) path is unaffected by the fix: sanity check a normal-sized volume still
// blurs a delta into a symmetric bump.
TEST(gaussian_blur_normal_size_smooths_delta) {
    const Extent3 dims{16, 16, 16};
    Volume<f32> v = Volume<f32>::zeros(dims);
    v.view()(8, 8, 8) = 1.0f;
    gaussian_blur(v.view(), 1.5f);
    CHECK(v.view()(8, 8, 8) < 1.0f);   // spread out
    CHECK(v.view()(8, 8, 9) > 0.0f);   // neighbour picked up mass
    CHECK(std::abs(v.view()(8, 8, 9) - v.view()(8, 8, 7)) < 1e-4f);  // symmetric
}
