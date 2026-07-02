// test_view.cpp — the Qt-free viewer engine: axis panes vs direct decode, LOD pick,
// oblique==axis cross-check, composite modes, prefetch warming, pixel<->volume mapping.
#define FENIX_TEST_MAIN
#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/test.hpp"
#include "view/view.hpp"

#include <cmath>
#include <filesystem>
#include <vector>

using namespace fenix;
using namespace fenix::view;

// Smooth synthetic u8 volume (smooth => codec error stays small at default q).
static Volume<u8> synth_volume(s64 s) {
    Volume<u8> v = Volume<u8>::zeros({s, s, s});
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x) {
                const f32 val = 120.0f + 60.0f * std::sin(0.09f * static_cast<f32>(x)) *
                                             std::cos(0.07f * static_cast<f32>(y)) +
                                40.0f * std::sin(0.05f * static_cast<f32>(z));
                v(z, y, x) = static_cast<u8>(std::clamp(val, 0.0f, 255.0f));
            }
    return v;
}

// Returns "" on failure (REQUIRE can't be used in a value-returning helper).
static std::string make_archive(const char* name, const Volume<u8>& vol) {
    auto tmp = std::filesystem::temp_directory_path() / name;
    const std::string path = tmp.string();
    std::filesystem::remove(path);
    auto a = codec::VolumeArchive::create(path, vol.view().dims(), {});
    if (!a) return "";
    if (!a->write_volume(vol.view())) return "";
    if (!a->close()) return "";
    return path;
}

TEST(axis_slices_match_direct_decode) {
    const s64 s = 96;
    Volume<u8> vol = synth_volume(s);
    const std::string path = make_archive("fenix_view_axis.fxvol", vol);
    REQUIRE(!path.empty());
    auto a = codec::VolumeArchive::open(path);
    REQUIRE(a.has_value());
    auto direct = a->read_volume_as<u8>(0);
    REQUIRE(direct.has_value());

    SliceEngine eng(*a);
    for (SliceAxis ax : {SliceAxis::z, SliceAxis::y, SliceAxis::x}) {
        SliceSpec sp;
        sp.axis = ax;
        sp.slice = 41;
        sp.center_u = static_cast<f32>(s) * 0.5f;
        sp.center_v = static_cast<f32>(s) * 0.5f;
        sp.zoom = 1.0f;
        sp.width = s;
        sp.height = s;
        auto img = eng.render(sp);
        REQUIRE(img.has_value());
        CHECK(img->lod == 0);
        // Compare against the directly-decoded volume at the pane's voxel centers.
        f64 mae = 0;
        for (s64 py = 8; py < s - 8; py += 3)
            for (s64 px = 8; px < s - 8; px += 3) {
                const Vec3f p = img->pixel_to_volume(static_cast<f32>(px), static_cast<f32>(py));
                const u8 want = direct->view()(static_cast<s64>(std::lround(p.z)),
                                               static_cast<s64>(std::lround(p.y)),
                                               static_cast<s64>(std::lround(p.x)));
                mae += std::abs(img->pix[static_cast<usize>(py * s + px)] - static_cast<f32>(want));
            }
        mae /= static_cast<f64>(((s - 16 + 2) / 3) * ((s - 16 + 2) / 3));
        CHECK(mae < 2.0);  // bilinear-at-centers vs decoded voxels on the same lossy data
    }
}

TEST(zoomed_out_render_uses_coarser_lod) {
    const s64 s = 160;  // > one chunk => a real pyramid exists
    Volume<u8> vol = synth_volume(s);
    const std::string path = make_archive("fenix_view_lod.fxvol", vol);
    REQUIRE(!path.empty());
    auto a = codec::VolumeArchive::open(path);
    REQUIRE(a.has_value());
    REQUIRE(a->nlods() >= 2);

    SliceEngine eng(*a);
    CHECK(eng.pick_lod(1.0f) == 0);
    CHECK(eng.pick_lod(0.5f) == 1);
    CHECK(eng.pick_lod(0.26f) == 1);
    SliceSpec sp;
    sp.slice = 80;
    sp.center_u = 80;
    sp.center_v = 80;
    sp.zoom = 0.5f;
    sp.width = 96;
    sp.height = 96;
    auto img = eng.render(sp);
    REQUIRE(img.has_value());
    CHECK(img->lod == 1);
    // Still looks like the volume: compare against the decoded LOD-1 level.
    auto l1 = a->read_volume_as<u8>(1);
    REQUIRE(l1.has_value());
    f64 mae = 0;
    s64 n = 0;
    for (s64 py = 8; py < 88; py += 3)
        for (s64 px = 8; px < 88; px += 3) {
            const Vec3f p = img->pixel_to_volume(static_cast<f32>(px), static_cast<f32>(py));
            const s64 lz = std::clamp<s64>(std::llround(p.z * 0.5f), 0, l1->view().dims().z - 1);
            const s64 ly = std::clamp<s64>(std::llround(p.y * 0.5f), 0, l1->view().dims().y - 1);
            const s64 lx = std::clamp<s64>(std::llround(p.x * 0.5f), 0, l1->view().dims().x - 1);
            mae += std::abs(img->pix[static_cast<usize>(py * 96 + px)] -
                            static_cast<f32>(l1->view()(lz, ly, lx)));
            ++n;
        }
    CHECK(mae / static_cast<f64>(n) < 6.0);  // half-voxel phase + bilinear tolerance
}

TEST(oblique_plane_matches_axis_pane) {
    const s64 s = 96;
    Volume<u8> vol = synth_volume(s);
    const std::string path = make_archive("fenix_view_obl.fxvol", vol);
    REQUIRE(!path.empty());
    auto a = codec::VolumeArchive::open(path);
    REQUIRE(a.has_value());
    SliceEngine eng(*a);

    SliceSpec sp;
    sp.axis = SliceAxis::z;
    sp.slice = 30;
    sp.center_u = 48;
    sp.center_v = 48;
    sp.zoom = 1.0f;
    sp.width = 64;
    sp.height = 64;
    auto ax = eng.render(sp);
    REQUIRE(ax.has_value());

    ObliqueSpec ob;
    ob.origin = Vec3f{30, 48, 48};
    ob.du = Vec3f{0, 0, 1};
    ob.dv = Vec3f{0, 1, 0};
    ob.zoom = 1.0f;
    ob.width = 64;
    ob.height = 64;
    auto obl = eng.render_oblique(ob);
    REQUIRE(obl.has_value());

    f64 mae = 0;
    for (usize i = 0; i < ax->pix.size(); ++i) mae += std::abs(ax->pix[i] - obl->pix[i]);
    mae /= static_cast<f64>(ax->pix.size());
    CHECK(mae < 1.5);  // same plane, bilinear-in-slab vs trilinear (z is integer => identical taps)
}

TEST(pixel_volume_mapping_roundtrip) {
    SliceImage img;
    img.spec = {SliceAxis::y, 12.0f, 100.0f, 40.0f, 2.0f, 128, 96};
    const Vec3f p = img.pixel_to_volume(17, 33);
    CHECK(std::abs(p.y - 12.0f) < 1e-5f);  // the normal axis carries the slice position
    f32 px = 0, py = 0;
    img.volume_to_pixel(p, px, py);
    CHECK(std::abs(px - 17.0f) < 1e-4f);
    CHECK(std::abs(py - 33.0f) < 1e-4f);
}

TEST(composite_modes_reduce_layers) {
    const s64 s = 96;
    Volume<u8> vol = synth_volume(s);
    const std::string path = make_archive("fenix_view_comp.fxvol", vol);
    REQUIRE(!path.empty());
    auto a = codec::VolumeArchive::open(path);
    REQUIRE(a.has_value());

    // A flat z-plane surface through the middle, normals +z.
    Surface surf(48, 48);
    surf.alloc_channels();
    for (s64 v = 0; v < surf.nv; ++v)
        for (s64 u = 0; u < surf.nu; ++u) {
            surf.set(u, v, Vec3f{48.0f, 24.0f + static_cast<f32>(v), 24.0f + static_cast<f32>(u)});
            surf.normal[surf.idx(u, v)] = Vec3f{1, 0, 0};
            surf.conf[surf.idx(u, v)] = 2.0f;
        }

    CompositeSpec cs;
    cs.lo = -4;
    cs.hi = 4;
    cs.step = 1;
    cs.mode = CompositeMode::mean;
    auto mean = render_surface_composite(*a, surf, cs);
    cs.mode = CompositeMode::max;
    auto mx = render_surface_composite(*a, surf, cs);
    cs.mode = CompositeMode::min;
    auto mn = render_surface_composite(*a, surf, cs);
    cs.mode = CompositeMode::alpha;
    auto al = render_surface_composite(*a, surf, cs);
    cs.mode = CompositeMode::beer_lambert;
    auto bl = render_surface_composite(*a, surf, cs);
    REQUIRE(mean.has_value());
    REQUIRE(mx.has_value());
    REQUIRE(mn.has_value());
    REQUIRE(al.has_value());
    REQUIRE(bl.has_value());

    s64 checked = 0;
    for (usize i = 0; i < mean->pix.size(); ++i) {
        if (!mean->valid[i]) continue;
        CHECK(mx->pix[i] >= mean->pix[i] - 1e-3f);
        CHECK(mean->pix[i] >= mn->pix[i] - 1e-3f);
        CHECK(al->pix[i] >= 0.0f);
        CHECK(bl->pix[i] >= 0.0f);
        ++checked;
    }
    CHECK(checked == static_cast<s64>(mean->pix.size()));  // fully valid input => fully rendered

    // Cross-check the mean mode against direct decode along the normal.
    auto direct = a->read_volume_as<u8>(0);
    REQUIRE(direct.has_value());
    f64 mae = 0;
    s64 n = 0;
    for (s64 v = 4; v < surf.nv - 4; v += 5)
        for (s64 u = 4; u < surf.nu - 4; u += 5) {
            const Vec3f base = surf.at(u, v);
            f64 want = 0;
            for (s64 t = -4; t <= 4; ++t)
                want += direct->view()(static_cast<s64>(base.z) + t, static_cast<s64>(base.y),
                                       static_cast<s64>(base.x));
            want /= 9.0;
            mae += std::abs(static_cast<f64>(mean->pix[surf.idx(u, v)]) - want);
            ++n;
        }
    CHECK(mae / static_cast<f64>(n) < 2.0);
}

TEST(prefetch_warms_the_cache) {
    const s64 s = 160;
    Volume<u8> vol = synth_volume(s);
    const std::string path = make_archive("fenix_view_pf.fxvol", vol);
    REQUIRE(!path.empty());
    auto a = codec::VolumeArchive::open(path);
    REQUIRE(a.has_value());
    SliceEngine eng(*a);

    SliceSpec sp;
    sp.slice = 32;
    sp.center_u = 80;
    sp.center_v = 80;
    sp.zoom = 1.0f;
    sp.width = 128;
    sp.height = 128;
    eng.prefetch_around(sp, 1);
    eng.prefetcher().drain();
    CHECK(eng.prefetcher().warmed() > 0);

    // A render after the warmup misses nothing new inside the prefetched ring.
    const u64 miss0 = a->cache_misses();
    auto img = eng.render(sp);
    REQUIRE(img.has_value());
    CHECK(a->cache_misses() == miss0);
}
