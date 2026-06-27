// test_deformation.cpp — Jacobian fold fraction (deformation invertibility).
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "eval/deformation.hpp"

using namespace fenix;

TEST(zero_displacement_has_no_folds) {
    const s64 s = 12;
    Volume<f32> z = Volume<f32>::zeros({s, s, s}), y = Volume<f32>::zeros({s, s, s}),
                x = Volume<f32>::zeros({s, s, s});
    CHECK(eval::jacobian_fold_fraction(z.view(), y.view(), x.view()) == 0.0);  // J = I, det 1
}

TEST(small_smooth_displacement_no_folds) {
    const s64 s = 12;
    Volume<f32> dz = Volume<f32>::zeros({s, s, s}), dy = Volume<f32>::zeros({s, s, s}),
                dx(Extent3{s, s, s});
    // gentle shear dx = 0.1*x -> J_xx = 1.1 > 0 everywhere
    for (s64 zz = 0; zz < s; ++zz)
        for (s64 yy = 0; yy < s; ++yy)
            for (s64 xx = 0; xx < s; ++xx) dx(zz, yy, xx) = 0.1f * static_cast<f32>(xx);
    CHECK(eval::jacobian_fold_fraction(dz.view(), dy.view(), dx.view()) == 0.0);
}

TEST(contracting_displacement_folds) {
    const s64 s = 12;
    Volume<f32> dz = Volume<f32>::zeros({s, s, s}), dy = Volume<f32>::zeros({s, s, s}),
                dx(Extent3{s, s, s});
    // dx = -1.5*x -> J_xx = 1 - 1.5 = -0.5 < 0 -> folded (orientation reversed) everywhere
    for (s64 zz = 0; zz < s; ++zz)
        for (s64 yy = 0; yy < s; ++yy)
            for (s64 xx = 0; xx < s; ++xx) dx(zz, yy, xx) = -1.5f * static_cast<f32>(xx);
    CHECK(eval::jacobian_fold_fraction(dz.view(), dy.view(), dx.view()) > 0.99);
}
