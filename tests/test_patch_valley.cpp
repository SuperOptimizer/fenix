// test_patch_valley.cpp — the CT-valley separation oracle (segment/ct_valley.hpp). The ML surface
// prediction fuses touching wraps; the raw CT keeps them as distinct density PEAKS. count_air_valleys
// counts the inter-wrap saddles between two on-papyrus points = |Δwrap|, touch-proof (it keys on the
// peaks being distinct via prominence, not on the gap being deep). This locks that behaviour, including
// a deliberately SHALLOW "touching" gap that a depth threshold would miss but prominence catches.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "segment/ct_valley.hpp"

#include <algorithm>
#include <cmath>

using namespace fenix;

namespace {
// CT of planar papyrus shells ⊥ x at x = 8,16,24,32,40 (spacing 8), Gaussian ridges (peak 200). The
// 32→40 gap is deliberately SHALLOW — a "touching" wrap pair: its valley only dips to ~0.75 of the peak
// (vs ~0 for the others). Value depends only on x, so a segment along x crosses gaps and one along y
// stays on a single shell.
Volume<u8> make_shells(s64 n) {
    Volume<u8> v(Extent3{n, n, n});
    auto vv = v.view();
    auto profile = [](f32 x) {
        f32 best = 0.0f;
        for (f32 c : {8.0f, 16.0f, 24.0f, 32.0f, 40.0f}) {
            const f32 d = x - c;
            best = std::max(best, 200.0f * std::exp(-d * d / (2.0f * 1.5f * 1.5f)));
        }
        if (x > 33.0f && x < 39.0f) best = std::max(best, 150.0f);  // shallow "touch" floor in the 32..40 gap
        return best;
    };
    parallel_for_z(v.dims(), [&](s64 z) {
        for (s64 y = 0; y < n; ++y)
            for (s64 x = 0; x < n; ++x) vv(z, y, x) = static_cast<u8>(std::clamp(profile(static_cast<f32>(x)), 0.0f, 255.0f));
    });
    return v;
}
}  // namespace

TEST(count_air_valleys_counts_shell_crossings) {
    const s64 n = 48;
    const Volume<u8> ct = make_shells(n);
    const VolumeView<const u8> v = ct.view();
    const f32 z = 24, y = 24, prom = 0.12f;

    // one deep adjacent gap.
    CHECK(segment::count_air_valleys(v, Vec3f{z, y, 8}, Vec3f{z, y, 16}, prom) == 1);
    // four gaps across all five shells.
    CHECK(segment::count_air_valleys(v, Vec3f{z, y, 8}, Vec3f{z, y, 40}, prom) == 4);
    // the SHALLOW touching gap (32..40, dips only to 0.75*H) still registers via prominence.
    CHECK(segment::count_air_valleys(v, Vec3f{z, y, 32}, Vec3f{z, y, 40}, prom) == 1);
    // same shell (segment along y at the x=8 peak) — no gap.
    CHECK(segment::count_air_valleys(v, Vec3f{z, 10, 8}, Vec3f{z, 38, 8}, prom) == 0);

    // crosses_valley: true across a gap, false along a shell.
    CHECK(segment::crosses_valley(v, Vec3f{z, y, 8}, Vec3f{z, y, 16}, prom));
    CHECK(!segment::crosses_valley(v, Vec3f{z, 10, 8}, Vec3f{z, 38, 8}, prom));
}
