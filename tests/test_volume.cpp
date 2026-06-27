// test_volume.cpp — Volume<T> / VolumeView / Vec3 substrate tests.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"

using namespace fenix;

TEST(volume_zyx_layout_and_access) {
    Volume<f32> v = Volume<f32>::zeros({2, 3, 4});  // z=2 y=3 x=4
    REQUIRE(v.size() == 24);
    CHECK(v.dims() == Extent3{2, 3, 4});
    // contiguous ZYX strides: z=12, y=4, x=1
    CHECK(v.view().strides() == Index3{12, 4, 1});
    v(1, 2, 3) = 7.0f;
    CHECK(v(1, 2, 3) == 7.0f);
    CHECK(v.flat()[1 * 12 + 2 * 4 + 3] == 7.0f);
}

TEST(volumeview_crop_reuses_strides) {
    Volume<u16> v = Volume<u16>::zeros({4, 4, 4});
    for (s64 z = 0; z < 4; ++z)
        for (s64 y = 0; y < 4; ++y)
            for (s64 x = 0; x < 4; ++x) v(z, y, x) = static_cast<u16>(z * 100 + y * 10 + x);
    auto sub = v.view().crop({1, 1, 1}, {2, 2, 2});
    CHECK(sub.dims() == Extent3{2, 2, 2});
    CHECK(sub(0, 0, 0) == 111);
    CHECK(sub(1, 1, 1) == 222);
    CHECK(!sub.is_contiguous());  // sub-box of a larger volume
}

TEST(volumeview_clamp_at_borders) {
    Volume<u8> v = Volume<u8>::zeros({2, 2, 2});
    v(0, 0, 0) = 5;
    auto view = v.view();
    CHECK(view.at_clamped(-3, -3, -3) == 5);  // clamps to (0,0,0)
    CHECK(view.in_bounds(0, 0, 0));
    CHECK(!view.in_bounds(2, 0, 0));
}

TEST(vec3_is_axis_tagged_zyx) {
    Vec3f a{1.0f, 2.0f, 3.0f};  // z,y,x
    CHECK(a.z == 1.0f);
    CHECK(a.x == 3.0f);
    Vec3f n = normalized(Vec3f{0, 0, 4});
    CHECK(n.x > 0.99f && n.x < 1.01f);
    CHECK(dot(Vec3f{1, 0, 0}, Vec3f{1, 0, 0}) == 1.0f);
}

TEST(units_are_strongly_typed) {
    Micron a{2.4f};
    Micron b = a * 4.0f;
    CHECK(b.value > 9.5f && b.value < 9.7f);
    CHECK(to_u32(lod(3)) == 3);
    CHECK(chunk_of(Index3{70000, 0, 0}).z == 70000 / 64);
}
