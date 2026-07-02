// test_patch_valley.cpp — the CT-valley separation oracle (segment/ct_valley.hpp). The ML surface
// prediction fuses touching wraps; the raw CT keeps them as distinct density PEAKS. count_air_valleys
// counts the inter-wrap saddles between two on-papyrus points = |Δwrap|, touch-proof (it keys on the
// peaks being distinct via prominence, not on the gap being deep). This locks that behaviour, including
// a deliberately SHALLOW "touching" gap that a depth threshold would miss but prominence catches.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "segment/ct_valley.hpp"
#include "segment/grow.hpp"

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

// DataField::ct_coord must be used (not raw prediction-space coords) whenever crosses_valley samples a
// coarse (ct_ds>1) CT grid, otherwise every probe reads at ct_ds x the intended position (grow.hpp:378,
// ~967 used to bypass the mapping). Build the same shell profile on a HALF-RESOLUTION CT grid (ds=2:
// shells at x=4,8,12,16,20 instead of 8,16,24,32,40) and confirm a prediction-space segment routed
// through fld.ct_coord() sees the same crossings as the full-res prediction-space segment did above.
TEST(ct_coord_maps_prediction_space_into_coarse_ct_grid) {
    const s64 n = 48, ds = 2;
    const Volume<u8> ct_full = make_shells(n);
    // Downsample by `ds` (nearest) to build the coarse CT grid DataField::ct_ds expects.
    Volume<u8> ct_coarse(Extent3{n / ds, n / ds, n / ds});
    for (s64 z = 0; z < n / ds; ++z)
        for (s64 y = 0; y < n / ds; ++y)
            for (s64 x = 0; x < n / ds; ++x) ct_coarse.view()(z, y, x) = ct_full.view()(z * ds, y * ds, x * ds);

    segment::DataField<u8> fld;
    fld.ct = ct_coarse.view();
    fld.ct_thresh = 1.0f;  // has_ct() only needs > 0
    fld.ct_ds = static_cast<f32>(ds);

    const f32 z = 24, y = 24, prom = 0.12f;
    // Same prediction-space endpoints as the full-res "one deep adjacent gap" case above (8->16, which
    // crosses one shell boundary); mapped into the coarse grid they land near coarse x=4->8 (the coarse
    // profile's first gap), so this must also cross exactly one valley.
    const Vec3f pa = fld.ct_coord(Vec3f{z, y, 8});
    const Vec3f pb = fld.ct_coord(Vec3f{z, y, 16});
    CHECK(segment::count_air_valleys(fld.ct, pa, pb, prom) == 1);
    CHECK(segment::crosses_valley(fld.ct, pa, pb, prom));

    // Without the mapping (raw prediction-space coords fed straight to the coarse grid), the same
    // prediction-space points sample far past where the shells are on the coarse grid (coarse dims are
    // n/ds=24, so x=16 is near the far edge) -- demonstrating the bug this test guards against.
    const Vec3f bad_pa{z, y, 8}, bad_pb{z, y, 16};
    // Not asserting a specific wrong count (that would over-fit the synthetic profile) — only that the
    // mapped and unmapped queries diverge, proving ct_coord is load-bearing here.
    const bool same = segment::count_air_valleys(fld.ct, bad_pa, bad_pb, prom) ==
                       segment::count_air_valleys(fld.ct, pa, pb, prom);
    CHECK(!same);
}
