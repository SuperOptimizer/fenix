// test_trace_long.cpp — the model-winding gate: growth on a synthetic pair of FUSED
// concentric shells (adjacent wraps welded along one sector — the classic wrap-hop trap)
// must stay on the seed wrap when gated by a spiral model, and demonstrably hops without.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "segment/grow.hpp"
#include "winding/spiral_model.hpp"

#include <cmath>
#include <numbers>

using namespace fenix;

namespace {
// two concentric shells joined by a smooth SPIRAL RAMP across one sector — the realistic
// wrap-hop geometry: the sheet at theta in [0, ramp] climbs continuously from r0 to r1,
// so an ungated grower happily walks around the ring and up onto the next wrap.
Volume<u8> make_ramped_shells(s64 side, f32 cy, f32 cx, f32 r0, f32 r1, f32 ramp) {
    Volume<u8> v = Volume<u8>::zeros({side, side, side});
    for (s64 z = 0; z < side; ++z)
        for (s64 y = 0; y < side; ++y)
            for (s64 x = 0; x < side; ++x) {
                const f32 dy = static_cast<f32>(y) - cy, dx = static_cast<f32>(x) - cx;
                const f32 r = std::sqrt(dy * dy + dx * dx);
                const f32 th = std::atan2(dy, dx);  // (-pi, pi]
                f32 s = 0;
                if (th >= 0 && th <= ramp) {
                    const f32 t = th / ramp;
                    const f32 rr = r0 + (r1 - r0) * (t * t * (3 - 2 * t));  // smoothstep climb
                    s = std::max(s, 1.5f - std::abs(r - rr));
                } else {
                    s = std::max(s, 1.5f - std::abs(r - r0));  // inner shell everywhere else
                    if (th > ramp) s = std::max(s, 1.5f - std::abs(r - r1));  // outer past the ramp
                }
                v(z, y, x) = static_cast<u8>(std::clamp(std::max(0.0f, s) / 1.5f, 0.0f, 1.0f) * 255.0f);
            }
    return v;
}
}  // namespace

TEST(winding_gate_prevents_wrap_hop_at_weld) {
    const s64 side = 96;
    const f32 cy = 48.0f, cx = 48.0f, r0 = 22.0f, r1 = 30.0f;
    auto pred = make_ramped_shells(side, cy, cx, r0, r1, 1.2f);

    winding::SpiralModel model;  // ideal: winding = r / dr, dr = shell spacing
    model.umbilicus.z = {0, static_cast<f32>(side)};
    model.umbilicus.y = {cy, cy};
    model.umbilicus.x = {cx, cx};
    model.dr_per_winding = r1 - r0;
    model.gap.dr = r1 - r0;

    const segment::NormalField nf = segment::compute_normal_field<u8>(pred.view(), 4);
    const Vec3f seed_far{48.0f, cy, cx - r0};  // theta = pi: far from the ramp

    auto span_of = [&](const Surface& S) {
        f32 wl = 1e30f, wh = -1e30f;
        for (usize i = 0; i < S.coord.size(); ++i)
            if (S.valid[i]) {
                const f32 w = model.winding_cont(S.coord[i]);
                wl = std::min(wl, w);
                wh = std::max(wh, w);
            }
        return wh - wl;
    };

    segment::GrowParams gp;
    gp.surf_thresh = 0.4f;
    gp.grid = 400;
    gp.step = 2.0f;
    gp.max_bridge = 2;

    // ungated: the weld is a bright solid bridge — growth colonizes the outer shell too
    Surface free_grow = segment::grow_surface<u8>(pred.view(), VolumeView<const u8>{}, nf, seed_far, gp);
    REQUIRE(free_grow.valid_count() > 300);
    const f32 span_free = span_of(free_grow);

    // gated: same params + the model gate
    gp.winding_fn = [&model](Vec3f p) { return model.winding_at(p); };  // stepped wrap index
    gp.winding_tol = 0.5f;
    Surface gated = segment::grow_surface<u8>(pred.view(), VolumeView<const u8>{}, nf, seed_far, gp);
    REQUIRE(gated.valid_count() > 300);
    const f32 span_gated = span_of(gated);

    std::printf("  [ramp: ungated span %.3f windings, gated %.3f]\n",
                static_cast<double>(span_free), static_cast<double>(span_gated));
    CHECK(span_free > 0.9f);    // the trap is real: ungated growth climbs onto the next wrap
    CHECK(span_gated < 0.3f);  // gate + post-pass filter pin the wrap (measured 0.033)
}
