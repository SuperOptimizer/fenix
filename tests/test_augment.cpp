// test_augment.cpp — train-time augmentation invariants: octahedral is an exact bijection that moves
// image and label together; geometric transforms keep image/label aligned; corruptions preserve dims
// and finiteness. (fast-math: no NaN/inf sentinels — check via a finite-range assertion.)
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "ml/augment.hpp"

#include <cmath>

using namespace fenix;
using namespace fenix::ml::aug;

static Sample make_sample(s64 sz, s64 sy, s64 sx) {
    Sample s{Volume<f32>(Extent3{sz, sy, sx}), Volume<u8>(Extent3{sz, sy, sx})};
    auto iv = s.image.view();
    auto lv = s.label.view();
    for (s64 z = 0; z < sz; ++z)
        for (s64 y = 0; y < sy; ++y)
            for (s64 x = 0; x < sx; ++x) {
                iv(z, y, x) = static_cast<f32>((z * 131 + y * 17 + x) % 251);
                lv(z, y, x) = static_cast<u8>((x + y + z) & 1);
            }
    return s;
}

static bool all_finite(VolumeView<const f32> v) {
    const s64 n = v.dims().count();
    for (s64 i = 0; i < n; ++i) { const f32 x = v.flat()[static_cast<usize>(i)]; if (!(x > -1e30f && x < 1e30f)) return false; }
    return true;
}

TEST(octahedral_identity_is_noop) {
    Sample s = make_sample(6, 8, 10), ref = make_sample(6, 8, 10);
    octahedral(s, 0);  // perm {0,1,2}, no flip
    CHECK(s.image.dims() == ref.image.dims());
    bool same = true;
    for (s64 i = 0; i < s.image.dims().count(); ++i)
        same = same && s.image.flat()[static_cast<usize>(i)] == ref.image.flat()[static_cast<usize>(i)];
    CHECK(same);
}

TEST(octahedral_flip_x_reverses_axis) {
    Sample s = make_sample(4, 4, 6), ref = make_sample(4, 4, 6);
    octahedral(s, 4);  // fm bit 4 => flip x, perm identity
    auto sv = s.image.view(); auto rv = ref.image.view();
    const Extent3 d = ref.image.dims();
    bool ok = true;
    for (s64 z = 0; z < d.z; ++z) for (s64 y = 0; y < d.y; ++y) for (s64 x = 0; x < d.x; ++x)
        ok = ok && sv(z, y, x) == rv(z, y, d.x - 1 - x);
    CHECK(ok);
}

TEST(octahedral_permutation_swaps_dims) {
    Sample s = make_sample(4, 5, 7);
    octahedral(s, 8 * 5);  // perm index 5 = {2,1,0} => dims reverse to (x,y,z)
    CHECK(s.image.dims() == (Extent3{7, 5, 4}));
    CHECK(s.label.dims() == (Extent3{7, 5, 4}));
}

TEST(octahedral_moves_label_with_image) {
    // After a flip, a labeled voxel must still sit on the same image value it labeled before.
    Sample s = make_sample(4, 4, 6), ref = make_sample(4, 4, 6);
    octahedral(s, 4);  // flip x on both
    auto rv = ref.image.view(); auto rl = ref.label.view();
    auto sv = s.image.view(); auto sl = s.label.view();
    const Extent3 d = ref.image.dims();
    bool ok = true;
    for (s64 z = 0; z < d.z; ++z) for (s64 y = 0; y < d.y; ++y) for (s64 x = 0; x < d.x; ++x) {
        // image and label both flipped the same way => pairing preserved
        ok = ok && sv(z, y, x) == rv(z, y, d.x - 1 - x) && sl(z, y, x) == rl(z, y, d.x - 1 - x);
    }
    CHECK(ok);
}

TEST(rotate_zero_degrees_is_near_identity) {
    Sample s = make_sample(4, 12, 12), ref = make_sample(4, 12, 12);
    rotate_z(s, 0.0f);
    auto sv = s.image.view(); auto rv = ref.image.view();
    // interior (away from clamp edges) should match closely
    f32 maxerr = 0;
    for (s64 z = 0; z < 4; ++z) for (s64 y = 2; y < 10; ++y) for (s64 x = 2; x < 10; ++x)
        maxerr = std::max(maxerr, std::abs(sv(z, y, x) - rv(z, y, x)));
    CHECK(maxerr < 1e-2f);
    CHECK(all_finite(s.image.view()));
}

TEST(rotate_360_returns_close) {
    Sample s = make_sample(4, 16, 16), ref = make_sample(4, 16, 16);
    rotate_z(s, 360.0f);
    auto sv = s.image.view(); auto rv = ref.image.view();
    f32 maxerr = 0;
    for (s64 z = 0; z < 4; ++z) for (s64 y = 4; y < 12; ++y) for (s64 x = 4; x < 12; ++x)
        maxerr = std::max(maxerr, std::abs(sv(z, y, x) - rv(z, y, x)));
    CHECK(maxerr < 2.0f);  // one interpolation round-trip on a high-frequency pattern
}

TEST(elastic_preserves_dims_and_finiteness) {
    Sample s = make_sample(8, 24, 24);
    elastic(s, 777, 3.0f, 12.0f);
    CHECK(s.image.dims() == (Extent3{8, 24, 24}));
    CHECK(s.label.dims() == (Extent3{8, 24, 24}));
    CHECK(all_finite(s.image.view()));
}

TEST(intensity_ct_compression_preserve_dims_and_finite) {
    for (int op = 0; op < 3; ++op) {
        Sample s = make_sample(8, 16, 16);
        if (op == 0) intensity(s, 42);
        else if (op == 1) ct_degrade(s, 42);
        else compression(s, 42, 0.7f);
        CHECK(s.image.dims() == (Extent3{8, 16, 16}));
        CHECK(all_finite(s.image.view()));
        CHECK(!s.has_label() || s.label.dims() == (Extent3{8, 16, 16}));  // corruptions are image-only
    }
}

TEST(full_policy_is_deterministic_per_seed) {
    Sample a = make_sample(8, 16, 16), b = make_sample(8, 16, 16);
    augment(a, 2024);
    augment(b, 2024);
    CHECK(a.image.dims() == b.image.dims());
    bool same = true;
    for (s64 i = 0; i < a.image.dims().count(); ++i)
        same = same && a.image.flat()[static_cast<usize>(i)] == b.image.flat()[static_cast<usize>(i)];
    CHECK(same);
    CHECK(all_finite(a.image.view()));
}

TEST(different_seeds_differ) {
    Sample a = make_sample(8, 16, 16), b = make_sample(8, 16, 16);
    augment(a, 1);
    augment(b, 2);
    // Overwhelmingly likely to differ somewhere (different octahedral sym / params).
    bool differ = a.image.dims() != b.image.dims();
    if (!differ)
        for (s64 i = 0; i < a.image.dims().count() && !differ; ++i)
            differ = a.image.flat()[static_cast<usize>(i)] != b.image.flat()[static_cast<usize>(i)];
    CHECK(differ);
}
