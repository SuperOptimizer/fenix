// test_phasecorr.cpp — phase correlation recovers a known translation.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "preprocess/phasecorr.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::preprocess;

// b(x) = a(x - s) with wraparound (matches the circular FFT model).
static Volume<f32> shifted(VolumeView<const f32> a, s64 sz, s64 sy, s64 sx) {
    const Extent3 d = a.dims();
    Volume<f32> b(d);
    auto wrap = [](s64 k, s64 n) { return ((k % n) + n) % n; };
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x)
                b(z, y, x) = a(wrap(z - sz, d.z), wrap(y - sy, d.y), wrap(x - sx, d.x));
    return b;
}

TEST(phase_correlation_recovers_shift) {
    const Extent3 d{16, 32, 32};
    Volume<f32> a(d);
    Pcg32 rng{3};
    for (s64 i = 0; i < a.size(); ++i) a.flat()[static_cast<usize>(i)] = rng.next_f32();
    // Add a strong feature so the peak is sharp.
    a(8, 16, 16) = 50.0f;

    for (Vec3f s : {Vec3f{3, 5, -7}, Vec3f{-2, 10, 4}, Vec3f{0, 0, 1}}) {
        Volume<f32> b = shifted(a.view(), static_cast<s64>(s.z), static_cast<s64>(s.y),
                                static_cast<s64>(s.x));
        Vec3f rec = phase_correlate(a.view(), b.view());
        CHECK(std::abs(rec.z - s.z) < 0.5f);
        CHECK(std::abs(rec.y - s.y) < 0.5f);
        CHECK(std::abs(rec.x - s.x) < 0.5f);
    }
}

TEST(phase_correlation_zero_shift) {
    const Extent3 d{16, 16, 16};
    Volume<f32> a(d);
    Pcg32 rng{4};
    for (s64 i = 0; i < a.size(); ++i) a.flat()[static_cast<usize>(i)] = rng.next_f32();
    Vec3f rec = phase_correlate(a.view(), a.view());
    CHECK(std::abs(rec.z) < 0.5f && std::abs(rec.y) < 0.5f && std::abs(rec.x) < 0.5f);
}
