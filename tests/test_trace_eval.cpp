// test_trace_eval.cpp — synthetic-oracle coverage for the tracer BENCHMARK metric
// (segment/trace_eval.hpp). Every model-ranking verdict in the training program routes
// through this scoring code and it previously had zero coverage (2026-07-13 audit):
// a latent nearest-point or origin-sign bug would masquerade as a "lever is dead" result.
//   1. PointGrid vs brute force on seeded random point sets (the radius-4 probe contract)
//   2. score_points on analytic plane-vs-plane offsets with known exact recall@2/@4
//   3. run_trace_eval end-to-end on a synthetic sheet: correct origin scores, a
//      wrong-sign origin must fail the "<100 GT cells" gate (catches translation bugs)
#define FENIX_TEST_MAIN
#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/test.hpp"
#include "io/surface.hpp"
#include "segment/trace_eval.hpp"

#include <cmath>
#include <filesystem>
#include <random>
#include <vector>

using namespace fenix;

TEST(pointgrid_matches_brute_force_within_probe_radius) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<f32> u(0.0f, 60.0f);
    std::vector<Vec3f> pts(1500);
    for (auto& p : pts) p = Vec3f{u(rng), u(rng), u(rng)};
    segment::detail::PointGrid g;
    g.build(pts, 4.0f);
    for (int i = 0; i < 500; ++i) {
        const Vec3f q{u(rng), u(rng), u(rng)};
        f32 brute = 1e30f;
        for (const Vec3f& p : pts) {
            const Vec3f d = p - q;
            brute = std::min(brute, d.z * d.z + d.y * d.y + d.x * d.x);
        }
        const f32 grid = g.nearest_d2(q);
        if (brute <= 16.0f) {
            // within the probe radius the grid must be exact
            CHECK(std::abs(grid - brute) < 1e-4f);
        } else {
            // beyond it the grid may miss, but must never report closer than truth
            CHECK(grid >= brute - 1e-4f);
        }
    }
}

TEST(score_points_plane_offsets_have_exact_recall) {
    // GT: plane z=20, 32x32 unit grid. "Trace": same plane shifted by dz.
    auto plane = [](f32 z) {
        std::vector<Vec3f> p;
        for (int y = 0; y < 32; ++y)
            for (int x = 0; x < 32; ++x) p.push_back(Vec3f{z, static_cast<f32>(y), static_cast<f32>(x)});
        return p;
    };
    const std::vector<Vec3f> gt = plane(20.0f);
    struct Case { f32 dz; f64 r2, r4; };
    for (const Case c : {Case{0.0f, 1.0, 1.0}, Case{1.5f, 1.0, 1.0}, Case{3.0f, 0.0, 1.0},
                         Case{6.0f, 0.0, 0.0}}) {
        segment::detail::PointGrid tg;
        tg.build(plane(20.0f + c.dz), 4.0f);
        const auto st = segment::detail::score_points(gt, tg);
        CHECK(std::abs(st.r2 - c.r2) < 1e-9);
        CHECK(std::abs(st.r4 - c.r4) < 1e-9);
    }
    // asymmetry sanity: recall(gt->trace) high while precision(trace->gt) low when the
    // trace has many far-away extra points
    std::vector<Vec3f> trace = plane(20.0f);
    for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x) trace.push_back(Vec3f{50.0f, static_cast<f32>(y), static_cast<f32>(x)});
    segment::detail::PointGrid tg, gg;
    tg.build(trace, 4.0f);
    gg.build(gt, 4.0f);
    CHECK(segment::detail::score_points(gt, tg).r2 > 0.999);   // every GT point found
    const auto pre = segment::detail::score_points(trace, gg);
    CHECK(pre.r2 > 0.4 && pre.r2 < 0.6);                        // half the trace is hallucination
}

TEST(trace_eval_end_to_end_origin_translation) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "fenix_test_trace_eval";
    fs::create_directories(dir);

    // prediction: strong flat sheet at z=48 in a 96^3 block (the easiest traceable case)
    const Extent3 D{96, 96, 96};
    Volume<u8> pred = Volume<u8>::zeros(D);
    for (s64 y = 0; y < D.y; ++y)
        for (s64 x = 0; x < D.x; ++x) {
            pred(47, y, x) = 160;
            pred(48, y, x) = 255;
            pred(49, y, x) = 160;
        }
    const std::string pp = (dir / "pred.fxvol").string();
    {
        auto a = codec::VolumeArchive::create(pp, D, codec::DctParams{.q = 0.5f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(pred.view()).has_value());
        REQUIRE(a->close().has_value());
    }

    // GT: the same sheet in ABSOLUTE scroll coords with block origin (1000,2000,3000)
    const Vec3f org{1000.0f, 2000.0f, 3000.0f};
    Surface s(13, 13);
    s.scale_u = 8.0f;
    s.scale_v = 8.0f;
    for (s64 v = 0; v < 13; ++v)
        for (s64 u2 = 0; u2 < 13; ++u2)
            s.set(u2, v, Vec3f{org.z + 48.0f, org.y + static_cast<f32>(v) * 8.0f,
                               org.x + static_cast<f32>(u2) * 8.0f});
    const std::string gp = (dir / "gt.fxsurf").string();
    REQUIRE(io::write_fxsurf(gp, s).has_value());

    Context ctx;
    // correct origin: tracer follows the plane, recall must be near-perfect -> rc 0
    {
        const std::string parg = "pred=" + pp, garg = "gt=" + gp;
        const std::string_view args[] = {parg, garg, "origin=1000,2000,3000", "min_valid=100"};
        auto r = segment::run_trace_eval(args, ctx);
        REQUIRE(r.has_value());
        CHECK(*r == 0);
    }
    // wrong-SIGN origin (the classic translation bug): no GT cell lands in the block,
    // the <100-GT-cells gate must fire as an error, never a silent zero-recall score
    {
        const std::string parg = "pred=" + pp, garg = "gt=" + gp;
        const std::string_view args[] = {parg, garg, "origin=3000,2000,1000", "min_valid=100"};
        auto r = segment::run_trace_eval(args, ctx);
        CHECK(!r.has_value());
    }
    fs::remove_all(dir);
}
