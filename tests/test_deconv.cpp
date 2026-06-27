// test_deconv.cpp — 3D FFT + Wiener deconvolution (blur then deconvolve recovers sharpness).
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "preprocess/deconv.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::preprocess;

static f64 rms_diff(VolumeView<const f32> a, VolumeView<const f32> b) {
    f64 s = 0;
    for (s64 i = 0; i < a.size(); ++i) {
        f64 e = static_cast<f64>(a.flat()[static_cast<usize>(i)]) -
                static_cast<f64>(b.flat()[static_cast<usize>(i)]);
        s += e * e;
    }
    return std::sqrt(s / static_cast<f64>(a.size()));
}

TEST(fft3d_roundtrip) {
    const Extent3 d{8, 8, 16};
    std::vector<cf32> v(static_cast<usize>(d.count()));
    Pcg32 rng{2};
    for (auto& x : v) x = cf32(rng.next_f32(), rng.next_f32());
    std::vector<cf32> orig = v;
    fft3d(v, d, false);
    fft3d(v, d, true);
    f32 me = 0;
    for (usize i = 0; i < v.size(); ++i) me = std::max(me, std::abs(v[i] - orig[i]));
    CHECK(me < 1e-3f);
}

TEST(wiener_deconv_recovers_blurred_detail) {
    const Extent3 d{32, 32, 32};
    // A sharp blob with internal structure.
    Volume<f32> sharp(d);
    Pcg32 rng{8};
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                f32 r2 = (static_cast<f32>(z) - 16) * (static_cast<f32>(z) - 16) +
                         (static_cast<f32>(y) - 16) * (static_cast<f32>(y) - 16) +
                         (static_cast<f32>(x) - 16) * (static_cast<f32>(x) - 16);
                sharp(z, y, x) = (r2 < 100.0f ? 100.0f : 0.0f) + rng.next_f32() * 10.0f;
            }

    const f32 sigma = 1.2f;
    Volume<f32> blurred = apply_psf(sharp.view(), sigma);
    Volume<f32> restored = wiener_deconvolve(blurred.view(), sigma, 0.005f);

    // Deconvolution brings the blurred image closer to the sharp original.
    f64 blur_err = rms_diff(blurred.view(), sharp.view());
    f64 rest_err = rms_diff(restored.view(), sharp.view());
    CHECK(rest_err < blur_err);          // it sharpened
    CHECK(rest_err < blur_err * 0.85);   // by a meaningful margin
}
