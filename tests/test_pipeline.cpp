// test_pipeline.cpp — integration: synthetic scroll through .fxvol roundtrip + the
// segment -> annotate -> winding -> render vertical slice. Proves the modules compose.
#define FENIX_TEST_MAIN
#include "annotate/umbilicus.hpp"
#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/test.hpp"
#include "render/unroll.hpp"
#include "segment/structure_tensor.hpp"
#include "winding/winding_field.hpp"

#include <cmath>
#include <filesystem>

using namespace fenix;

static Volume<f32> synthetic_scroll(Extent3 d, f32 cy, f32 cx, f32 pitch) {
    Volume<f32> v = Volume<f32>::zeros(d);
    constexpr f32 two_pi = 2.0f * std::numbers::pi_v<f32>;
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                f32 r = std::sqrt((static_cast<f32>(y) - cy) * (static_cast<f32>(y) - cy) +
                                  (static_cast<f32>(x) - cx) * (static_cast<f32>(x) - cx));
                v(z, y, x) = 128.0f + 120.0f * std::cos(two_pi * r / pitch);  // concentric wraps
            }
    return v;
}

TEST(fxvol_whole_volume_roundtrip_with_edge_padding) {
    // Dims deliberately NOT multiples of 64 to exercise edge padding.
    Extent3 d{40, 100, 100};
    auto vol = synthetic_scroll(d, 50.0f, 50.0f, 7.0f);

    auto tmp = std::filesystem::temp_directory_path() / "fenix_pipeline.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);

    {
        auto a = codec::VolumeArchive::create(path, d, {.q = 2.0f, .levels = 4});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume(vol.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    auto a = codec::VolumeArchive::open(path);
    REQUIRE(a.has_value());
    auto got = a->read_volume();
    REQUIRE(got.has_value());
    REQUIRE(got->dims() == d);

    f32 max_err = 0;
    for (s64 i = 0; i < d.count(); ++i)
        max_err = std::max(max_err, std::abs(got->flat()[static_cast<usize>(i)] -
                                             vol.flat()[static_cast<usize>(i)]));
    CHECK(max_err < 25.0f);  // q=2 lossy, but recognizable
    std::filesystem::remove(path);
}

TEST(segment_annotate_winding_render_compose) {
    Extent3 d{24, 96, 96};
    const f32 cy = 48.0f, cx = 48.0f, pitch = 8.0f;
    auto vol = synthetic_scroll(d, cy, cx, pitch);

    // segment: structure tensor finds sheet-like structure on the wraps.
    auto sf = segment::structure_tensor(vol.view(), {.sigma_grad = 1.0f, .sigma_tensor = 2.0f});
    f32 max_sheet = 0;
    for (f32 s : sf.sheetness.flat()) max_sheet = std::max(max_sheet, s);
    CHECK(max_sheet > 0.6f);  // wraps are sheet-like

    // annotate: umbilicus estimate recovers the center.
    auto u = annotate::Umbilicus::estimate(vol.view(), 200.0f);
    Vec3f c = u.center(12.0f);
    CHECK(std::abs(c.y - cy) < 5.0f);
    CHECK(std::abs(c.x - cx) < 5.0f);

    // winding + render: build the field and unroll to an image.
    auto w = winding::winding_init(d, u, {.pitch = pitch});
    auto img = render::unroll(vol.view(), w.view(), {.samp = 4.0f});
    CHECK(img.dims().z == 1);
    CHECK(img.dims().y == d.z);
    CHECK(img.dims().x > 1);
    // The unrolled image holds real signal (not all zero).
    f32 img_max = 0;
    for (f32 p : img.flat()) img_max = std::max(img_max, p);
    CHECK(img_max > 1.0f);
}
