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
    for (s64 i = 0; i < n; ++i) {
        const f32 x = v.flat()[static_cast<usize>(i)];
        if (!(x > -1e30f && x < 1e30f)) return false;
    }
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
    auto sv = s.image.view();
    auto rv = ref.image.view();
    const Extent3 d = ref.image.dims();
    bool ok = true;
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) ok = ok && sv(z, y, x) == rv(z, y, d.x - 1 - x);
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
    auto rv = ref.image.view();
    auto rl = ref.label.view();
    auto sv = s.image.view();
    auto sl = s.label.view();
    const Extent3 d = ref.image.dims();
    bool ok = true;
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                // image and label both flipped the same way => pairing preserved
                ok = ok && sv(z, y, x) == rv(z, y, d.x - 1 - x) && sl(z, y, x) == rl(z, y, d.x - 1 - x);
            }
    CHECK(ok);
}

TEST(rotate_zero_degrees_is_near_identity) {
    Sample s = make_sample(4, 12, 12), ref = make_sample(4, 12, 12);
    rotate_z(s, 0.0f);
    auto sv = s.image.view();
    auto rv = ref.image.view();
    // interior (away from clamp edges) should match closely
    f32 maxerr = 0;
    for (s64 z = 0; z < 4; ++z)
        for (s64 y = 2; y < 10; ++y)
            for (s64 x = 2; x < 10; ++x) maxerr = std::max(maxerr, std::abs(sv(z, y, x) - rv(z, y, x)));
    CHECK(maxerr < 1e-2f);
    CHECK(all_finite(s.image.view()));
}

TEST(rotate_360_returns_close) {
    Sample s = make_sample(4, 16, 16), ref = make_sample(4, 16, 16);
    rotate_z(s, 360.0f);
    auto sv = s.image.view();
    auto rv = ref.image.view();
    f32 maxerr = 0;
    for (s64 z = 0; z < 4; ++z)
        for (s64 y = 4; y < 12; ++y)
            for (s64 x = 4; x < 12; ++x) maxerr = std::max(maxerr, std::abs(sv(z, y, x) - rv(z, y, x)));
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
        if (op == 0)
            intensity(s, 42);
        else if (op == 1)
            ct_degrade(s, 42);
        else
            compression(s, 42, 0.7f);
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

TEST(teacher_moves_with_image_through_geometry) {
    // teacher initialized as an exact copy of the image: every geometric op transforms both via
    // the same trilinear path, so they must stay equal elementwise through the whole chain.
    Sample s = make_sample(12, 24, 24);
    s.teacher = Volume<f32>(s.image.dims());
    for (s64 i = 0; i < s.image.dims().count(); ++i)
        s.teacher.flat()[static_cast<usize>(i)] = s.image.flat()[static_cast<usize>(i)];
    octahedral(s, 13);
    rotate_z(s, 17.0f);
    scale_jitter(s, 1.1f);
    elastic(s, 99, 2.0f, 8.0f);
    REQUIRE(s.teacher.dims() == s.image.dims());
    bool same = true;
    for (s64 i = 0; i < s.image.dims().count(); ++i)
        same = same && s.teacher.flat()[static_cast<usize>(i)] == s.image.flat()[static_cast<usize>(i)];
    CHECK(same);
    CHECK(all_finite(s.teacher.view()));
}

TEST(scale_jitter_unit_factor_is_identity) {
    Sample a = make_sample(8, 16, 16), b = make_sample(8, 16, 16);
    scale_jitter(a, 1.0f);
    bool same = true;
    for (s64 i = 0; i < a.image.dims().count(); ++i)
        same = same && std::abs(a.image.flat()[static_cast<usize>(i)] - b.image.flat()[static_cast<usize>(i)]) < 1e-3f;
    CHECK(same);
}

TEST(lowres_blurs_but_preserves_dims_and_mean) {
    Sample s = make_sample(16, 16, 16);
    f64 m0 = 0, v0 = 0;
    const s64 n = s.image.dims().count();
    for (s64 i = 0; i < n; ++i) m0 += s.image.flat()[static_cast<usize>(i)];
    m0 /= static_cast<f64>(n);
    for (s64 i = 0; i < n; ++i) {
        const f64 d = s.image.flat()[static_cast<usize>(i)] - m0;
        v0 += d * d;
    }
    lowres(s, 2.0f);
    REQUIRE(s.image.dims().count() == n);
    f64 m1 = 0, v1 = 0;
    for (s64 i = 0; i < n; ++i) m1 += s.image.flat()[static_cast<usize>(i)];
    m1 /= static_cast<f64>(n);
    for (s64 i = 0; i < n; ++i) {
        const f64 d = s.image.flat()[static_cast<usize>(i)] - m1;
        v1 += d * d;
    }
    CHECK(std::abs(m1 - m0) < 10.0);  // mean roughly preserved
    CHECK(v1 < v0);                   // strictly blurrier
    CHECK(all_finite(s.image.view()));
}

TEST(so3_zero_deg_is_identity_and_teacher_follows) {
    Sample a = make_sample(10, 20, 20), b = make_sample(10, 20, 20);
    rotate_so3(a, Vec3f{0.3f, -0.7f, 0.5f}, 0.0f);  // deg=0: exact no-op
    bool same = true;
    for (s64 i = 0; i < a.image.dims().count(); ++i)
        same = same && a.image.flat()[static_cast<usize>(i)] == b.image.flat()[static_cast<usize>(i)];
    CHECK(same);
    // teacher rides through an actual tumble identically to the image
    Sample s = make_sample(10, 20, 20);
    s.teacher = Volume<f32>(s.image.dims());
    for (s64 i = 0; i < s.image.dims().count(); ++i)
        s.teacher.flat()[static_cast<usize>(i)] = s.image.flat()[static_cast<usize>(i)];
    rotate_so3(s, Vec3f{1.0f, 0.4f, -0.2f}, 23.0f);
    bool tsame = true;
    for (s64 i = 0; i < s.image.dims().count(); ++i)
        tsame = tsame && s.teacher.flat()[static_cast<usize>(i)] == s.image.flat()[static_cast<usize>(i)];
    CHECK(tsame);
    CHECK(all_finite(s.image.view()));
}

TEST(so3_about_z_axis_matches_rotate_z) {
    // axis=(1,0,0) in ZYX components IS the z axis: rotate_so3 must agree with rotate_z
    Sample a = make_sample(6, 24, 24), b = make_sample(6, 24, 24);
    rotate_so3(a, Vec3f{1.0f, 0.0f, 0.0f}, 15.0f);
    rotate_z(b, 15.0f);
    f32 mad = 0;
    for (s64 i = 0; i < a.image.dims().count(); ++i)
        mad = std::max(mad, std::abs(a.image.flat()[static_cast<usize>(i)] - b.image.flat()[static_cast<usize>(i)]));
    CHECK(mad < 1e-2f);
}

TEST(label_stays_on_bright_plane_through_geometric_chain) {
    // A bright plane at z=24 with the label ON it: after every geometric op, labeled voxels
    // must still be brighter than unlabeled — the supervision-alignment invariant the feeder
    // depends on (image trilinear vs label NEAREST must not decouple).
    const s64 S = 48;
    Sample s{Volume<f32>(Extent3{S, S, S}), Volume<u8>(Extent3{S, S, S})};
    auto iv = s.image.view();
    auto lv = s.label.view();
    for (s64 z = 0; z < S; ++z)
        for (s64 y = 0; y < S; ++y)
            for (s64 x = 0; x < S; ++x) {
                const bool on = std::abs(z - 24) <= 2;
                iv(z, y, x) = on ? 200.0f : 50.0f;
                lv(z, y, x) = on ? 255 : 0;
            }
    octahedral(s, 21);
    rotate_z(s, 13.0f);
    scale_jitter(s, 1.08f);
    elastic(s, 7, 2.5f, 12.0f);
    rotate_so3(s, Vec3f{0.6f, 0.5f, -0.3f}, 11.0f);
    f64 on_sum = 0, off_sum = 0;
    s64 on_n = 0, off_n = 0;
    auto iv2 = s.image.view();
    auto lv2 = s.label.view();
    const Extent3 d = s.image.dims();
    for (s64 z = 4; z < d.z - 4; ++z)
        for (s64 y = 4; y < d.y - 4; ++y)
            for (s64 x = 4; x < d.x - 4; ++x) {
                if (lv2(z, y, x) == 255) {
                    on_sum += iv2(z, y, x);
                    ++on_n;
                } else {
                    off_sum += iv2(z, y, x);
                    ++off_n;
                }
            }
    REQUIRE(on_n > 1000);
    const f64 on_mean = on_sum / static_cast<f64>(on_n), off_mean = off_sum / static_cast<f64>(off_n);
    // plane=200, elsewhere=50: labeled voxels must stay decisively bright through the chain
    CHECK(on_mean > 150.0);
    CHECK(on_mean - off_mean > 80.0);
}

TEST(cutout_blanks_image_only) {
    Sample s = make_sample(16, 32, 32), ref = make_sample(16, 32, 32);
    cutout(s, 77);
    s64 changed = 0;
    bool label_same = true;
    for (s64 i = 0; i < s.image.dims().count(); ++i) {
        changed += s.image.flat()[static_cast<usize>(i)] != ref.image.flat()[static_cast<usize>(i)];
        label_same = label_same && s.label.flat()[static_cast<usize>(i)] == ref.label.flat()[static_cast<usize>(i)];
    }
    CHECK(changed > 50);  // at least one real box got blanked
    CHECK(label_same);    // supervision untouched
    CHECK(all_finite(s.image.view()));
}
