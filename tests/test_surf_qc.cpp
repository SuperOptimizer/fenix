// tests/test_surf_qc.cpp — surf-qc primitives: nearest-prominent-peak (the profile-mode
// offset estimator) + stencil normals. The old estimator (window global max) locked onto
// brighter NEIGHBOR wraps near tight winding — these pin the fixed behavior.
#define FENIX_TEST_MAIN
#include "core/test.hpp"
#include "ml/surf_qc.hpp"

#include <cmath>
#include <vector>

using namespace fenix;

namespace {
// profile over t=-W..W: gaussian bumps at given offsets/heights on a dark floor
std::vector<f32> bumps(s64 W, std::initializer_list<std::pair<f64, f64>> peaks, f32 floor_v = 20.0f) {
    std::vector<f32> p(static_cast<usize>(2 * W + 1), floor_v);
    for (s64 t = -W; t <= W; ++t)
        for (const auto& [c, h] : peaks)
            p[static_cast<usize>(t + W)] += static_cast<f32>(h * std::exp(-0.5 * (t - c) * (t - c) / 4.0));
    return p;
}
}  // namespace

TEST(nearest_peak_prefers_near_over_bright) {
    // the regression: a small on-sheet ridge at +1 must beat a BRIGHTER neighbor wrap at +9
    const s64 W = 12;
    const auto p = bumps(W, {{1.0, 60.0}, {9.0, 120.0}});
    const auto t = ml::detail::nearest_prominent_peak(p, W, /*search=*/8, /*prom=*/15.0f);
    REQUIRE(t.has_value());
    CHECK(std::abs(*t - 1) <= 1);
}

TEST(nearest_peak_respects_search_window_and_prominence) {
    const s64 W = 12;
    // only peak sits at +10, outside search=8 -> no peak
    CHECK(!ml::detail::nearest_prominent_peak(bumps(W, {{10.0, 80.0}}), W, 8, 15.0f).has_value());
    // low-prominence speckle (height 8 < prom 15) -> no peak
    CHECK(!ml::detail::nearest_prominent_peak(bumps(W, {{0.0, 8.0}}), W, 8, 15.0f).has_value());
    // flat profile -> no peak
    CHECK(!ml::detail::nearest_prominent_peak(std::vector<f32>(25, 100.0f), W, 8, 15.0f).has_value());
    // clean centered ridge -> offset ~0
    const auto t = ml::detail::nearest_prominent_peak(bumps(W, {{0.0, 80.0}}), W, 8, 15.0f);
    REQUIRE(t.has_value());
    CHECK(*t == 0);
}

TEST(stencil_normal_matches_plane) {
    // flat z=zc plane: normal must be ±(1,0,0) in ZYX regardless of uv scale
    Surface s(8, 8);
    s.scale_u = 20.0f;
    s.scale_v = 20.0f;
    for (s64 v = 0; v < 8; ++v)
        for (s64 u = 0; u < 8; ++u)
            s.set(u, v, Vec3f{50.0f, static_cast<f32>(v) * 20.0f, static_cast<f32>(u) * 20.0f});
    const auto n = ml::detail::stencil_normal(s, 4, 4);
    REQUIRE(n.has_value());
    CHECK(std::abs(std::abs(n->z) - 1.0f) < 1e-4f);
    CHECK(std::abs(n->y) < 1e-4f);
    CHECK(std::abs(n->x) < 1e-4f);
    // edge cell falls back to a 1-step stencil rather than failing
    CHECK(ml::detail::stencil_normal(s, 1, 1).has_value());
    // corner with no two-sided neighbors -> nullopt
    Surface tiny(2, 2);
    for (s64 v = 0; v < 2; ++v)
        for (s64 u = 0; u < 2; ++u) tiny.set(u, v, Vec3f{0, static_cast<f32>(v), static_cast<f32>(u)});
    CHECK(!ml::detail::stencil_normal(tiny, 0, 0).has_value());
}

#include "io/surface.hpp"
#include "ml/surf_consist.hpp"

#include <cstdio>

namespace {
fenix::Surface plane_at(fenix::f32 zc, fenix::f32 tilt_du = 0.0f) {
    fenix::Surface s(24, 24);
    s.scale_u = 8.0f;
    s.scale_v = 8.0f;
    for (fenix::s64 v = 0; v < 24; ++v)
        for (fenix::s64 u = 0; u < 24; ++u)
            s.set(u,
                  v,
                  fenix::Vec3f{zc + tilt_du * static_cast<fenix::f32>(u - 12),
                               static_cast<fenix::f32>(v) * 8.0f + 32.0f,
                               static_cast<fenix::f32>(u) * 8.0f + 32.0f});
    return s;
}
}  // namespace

TEST(surf_consist_runs_on_agree_offset_cross) {
    // plain string paths: fs::path in test TUs trips a pre-existing libc++ header/dylib
    // mismatch on the macOS dev build (test_feed has the same issue; Linux is fine)
    // A: z=64. B: z=64.6 (duplicate trace -> AGREE). C: z=68 (parallel offset 4 -> OFFSET).
    // D: tilted through z=64 (crosses A -> CROSS).
    const std::string pa = "/tmp/fenix_consist_a.fxsurf", pb = "/tmp/fenix_consist_b.fxsurf",
                      pc = "/tmp/fenix_consist_c.fxsurf", pd = "/tmp/fenix_consist_d.fxsurf";
    REQUIRE(io::write_fxsurf(pa, plane_at(64.0f)).has_value());
    REQUIRE(io::write_fxsurf(pb, plane_at(64.6f)).has_value());
    REQUIRE(io::write_fxsurf(pc, plane_at(68.0f)).has_value());
    REQUIRE(io::write_fxsurf(pd, plane_at(64.0f, 0.35f)).has_value());
    Context ctx;
    const std::string_view args[] = {pa, pb, pc, pd, "near=6", "k=2000"};
    const auto r = ml::run_surf_consist(args, ctx);
    REQUIRE(r.has_value());
    CHECK(*r == 0);
    for (const auto& p : {pa, pb, pc, pd}) std::remove(p.c_str());
}

TEST(mesh_cloud_nearest_and_key) {
    ml::detail::MeshCloud mc;
    mc.pts = {fenix::Vec3f{10, 10, 10}, fenix::Vec3f{10, 10, 20}, fenix::Vec3f{50, 50, 50}};
    mc.nrm.assign(3, fenix::Vec3f{1, 0, 0});
    mc.build_grid(6.0f);
    CHECK(mc.nearest(fenix::Vec3f{10, 10, 11}, 6.0f) == 0);
    CHECK(mc.nearest(fenix::Vec3f{10, 10, 18}, 6.0f) == 1);
    CHECK(mc.nearest(fenix::Vec3f{30, 30, 30}, 6.0f) == -1);  // nothing within reach
}

#include "codec/archive.hpp"
#include "ml/surf_repair.hpp"

TEST(surf_repair_snaps_warped_plane_to_ridge) {
    // CT: bright band at z=67 (value 200 within |z-67|<=1, floor 20), 128^3.
    const Extent3 vd{128, 256, 256};
    Volume<u8> ctv(vd);
    auto cv = ctv.view();
    for (s64 z = 0; z < vd.z; ++z)
        for (s64 y = 0; y < vd.y; ++y)
            for (s64 x = 0; x < vd.x; ++x) cv(z, y, x) = std::abs(z - 67) <= 1 ? 200 : 20;
    const std::string ctp = "/tmp/fenix_repair_ct.fxvol";
    {
        auto a = codec::VolumeArchive::create(ctp, vd, codec::DctParams{.q = 0.5f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(ctv.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    // mesh: WARPED off the ridge — z = 64 + sin-ish drift (+/-1), true offset ~ +3 varying
    Surface s(24, 24);
    s.scale_u = 8.0f;
    s.scale_v = 8.0f;
    for (s64 v = 0; v < 24; ++v)
        for (s64 u = 0; u < 24; ++u)
            s.set(u,
                  v,
                  Vec3f{64.0f + std::sin(static_cast<f32>(u) * 0.4f),
                        static_cast<f32>(v) * 8.0f + 16.0f,
                        static_cast<f32>(u) * 8.0f + 16.0f});
    const std::string in = "/tmp/fenix_repair_in.fxsurf", out = "/tmp/fenix_repair_out.fxsurf";
    REQUIRE(io::write_fxsurf(in, s).has_value());
    Context ctx;
    const std::string_view args[] = {ctp, in, out, "grid=4", "off=12", "smooth=2"};
    const auto r = ml::run_surf_repair(args, ctx);
    REQUIRE(r.has_value());
    auto rs = io::read_fxsurf(out);
    REQUIRE(rs.has_value());
    // interior vertices must now sit within ~1 vox of the ridge plane z=67
    f64 worst = 0, mean = 0;
    s64 n = 0;
    for (s64 v = 4; v < 20; ++v)
        for (s64 u = 4; u < 20; ++u) {
            const f64 e = std::abs(static_cast<f64>(rs->at(u, v).z) - 67.0);
            worst = std::max(worst, e);
            mean += e;
            ++n;
        }
    mean /= static_cast<f64>(n);
    CHECK(mean < 1.0);
    CHECK(worst < 2.0);
    for (const auto& p : {ctp, in, out}) std::remove(p.c_str());
}

#include "ml/label_audit.hpp"

#include <fstream>

TEST(label_audit_flags_off_surface_mesh) {
    // "prediction": bright sheet-prob band at z=64 in a 128x256x256 block at origin 0.
    const Extent3 vd{128, 256, 256};
    Volume<u8> pv(vd);
    auto pvv = pv.view();
    for (s64 z = 0; z < vd.z; ++z)
        for (s64 y = 0; y < vd.y; ++y)
            for (s64 x = 0; x < vd.x; ++x) pvv(z, y, x) = std::abs(z - 64) <= 1 ? 230 : 5;
    const std::string pp = "/tmp/fenix_audit_pred.fxvol";
    {
        auto a = codec::VolumeArchive::create(pp, vd, codec::DctParams{.q = 0.5f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(pv.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    // mesh A on the predicted sheet (agree), mesh B 16 vox off (model contradicts label)
    const std::string ma = "/tmp/fenix_audit_a.fxsurf", mb = "/tmp/fenix_audit_b.fxsurf";
    REQUIRE(io::write_fxsurf(ma, plane_at(64.0f)).has_value());
    REQUIRE(io::write_fxsurf(mb, plane_at(80.0f)).has_value());
    Context ctx;
    const std::string_view args[] = {pp, "0", "0", "0", ma, mb, "tile=8", "out=/tmp/fenix_audit"};
    REQUIRE(ml::run_label_audit(args, ctx).has_value());
    // parse the two queues: A's tiles ~0% miss, B's ~100%
    auto worst_miss = [](const std::string& tsv) -> double {
        std::ifstream f(tsv);
        if (!f) return -1.0;
        std::string ln;
        std::getline(f, ln);                 // header
        if (!std::getline(f, ln)) return -1.0;  // worst row first
        const auto tab = ln.rfind('\t');
        return std::stod(ln.substr(tab + 1));
    };
    CHECK(worst_miss("/tmp/fenix_audit_fenix_audit_a.tsv") < 5.0);
    CHECK(worst_miss("/tmp/fenix_audit_fenix_audit_b.tsv") > 95.0);
    for (const auto& p : {pp, ma, mb}) std::remove(p.c_str());
}

#include "ml/qc_chunk.hpp"

TEST(qc_chunk_exports_ct_and_band) {
    // reuse the repair fixtures: bright band at z=67, mesh near it
    const Extent3 vd{128, 256, 256};
    Volume<u8> ctv(vd);
    auto cv = ctv.view();
    for (s64 z = 0; z < vd.z; ++z)
        for (s64 y = 0; y < vd.y; ++y)
            for (s64 x = 0; x < vd.x; ++x) cv(z, y, x) = std::abs(z - 67) <= 1 ? 200 : 20;
    const std::string ctp = "/tmp/fenix_qcc_ct.fxvol";
    {
        auto a = codec::VolumeArchive::create(ctp, vd, codec::DctParams{.q = 0.5f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(ctv.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    Surface s = plane_at(67.0f);
    const std::string sp = "/tmp/fenix_qcc.fxsurf";
    REQUIRE(io::write_fxsurf(sp, s).has_value());
    Context ctx;
    const std::string_view args[] = {ctp, sp, "/tmp/fenix_qcc", "n=2", "size=64", "thickness=4"};
    REQUIRE(ml::run_qc_chunk(args, ctx).has_value());
    // file = header line + 2*64^3 bytes; band volume must contain sheet voxels
    std::ifstream f("/tmp/fenix_qcc_0.qcchunk", std::ios::binary);
    REQUIRE(static_cast<bool>(f));
    std::string hdr;
    std::getline(f, hdr);
    CHECK(hdr.find("\"size\":64") != std::string::npos);
    std::vector<char> body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    REQUIRE(body.size() == 2u * 64 * 64 * 64);
    s64 sheet = 0, bright = 0;
    for (usize i = 0; i < 64u * 64 * 64; ++i) {
        bright += static_cast<u8>(body[i]) > 100;
        sheet += static_cast<u8>(body[64u * 64 * 64 + i]) == 255;
    }
    CHECK(bright > 1000);  // the CT band is in-frame
    CHECK(sheet > 500);    // the rasterized mesh band is in-frame
    // fixtures intentionally kept on disk for the chunk_viewer.py smoke test
}

namespace {
// profile builder for air_edge: material value where pred(t) true, air value elsewhere
std::vector<f32> slabprof(s64 W, f32 mat, f32 air, auto pred) {
    std::vector<f32> p(static_cast<usize>(2 * W + 1));
    for (s64 t = -W; t <= W; ++t) p[static_cast<usize>(t + W)] = pred(t) ? mat : air;
    return p;
}
}  // namespace

TEST(air_edge_subvoxel_face_localization) {
    const s64 W = 12;
    // sheet up to t=2, air beyond: face between 2 and 3 (alpha crossing ~2.7)
    const auto p1 = slabprof(W, 180.0f, 5.0f, [](s64 t) { return t <= 2; });
    const auto e1 = ml::detail::air_edge(p1, W, 8);
    REQUIRE(e1.has_value());
    CHECK(*e1 > 2.0);
    CHECK(*e1 < 3.0);
    // floating IN the gap: sheet at t in [-9,-5] -> snaps to its near face (~-4.3)
    const auto p2 = slabprof(W, 180.0f, 5.0f, [](s64 t) { return t >= -9 && t <= -5; });
    const auto e2 = ml::detail::air_edge(p2, W, 8);
    REQUIRE(e2.has_value());
    CHECK(*e2 > -5.0);
    CHECK(*e2 < -3.8);
    // papyrus contact (no air anywhere): MUST refuse
    CHECK(!ml::detail::air_edge(slabprof(W, 180.0f, 180.0f, [](s64) { return true; }), W, 8).has_value());
    // single-sample air dip is speckle, not a face
    CHECK(!ml::detail::air_edge(slabprof(W, 180.0f, 5.0f, [](s64 t) { return t != 4; }), W, 8).has_value());
}

TEST(surf_repair_alpha_snaps_to_air_face) {
    // slab z in [55,69] (180) in air (5): mesh at z=66.5 must snap to the top face ~69.5
    const Extent3 vd{128, 256, 256};
    Volume<u8> ctv(vd);
    auto cv = ctv.view();
    for (s64 z = 0; z < vd.z; ++z)
        for (s64 y = 0; y < vd.y; ++y)
            for (s64 x = 0; x < vd.x; ++x) cv(z, y, x) = (z >= 55 && z <= 69) ? 180 : 5;
    const std::string ctp = "/tmp/fenix_alpha_ct.fxvol";
    {
        auto a = codec::VolumeArchive::create(ctp, vd, codec::DctParams{.q = 0.5f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(ctv.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    const std::string in = "/tmp/fenix_alpha_in.fxsurf", out = "/tmp/fenix_alpha_out.fxsurf";
    REQUIRE(io::write_fxsurf(in, plane_at(66.5f)).has_value());
    Context ctx;
    const std::string_view args[] = {ctp, in, out, "mode=alpha", "grid=4", "max_shift=6"};
    REQUIRE(ml::run_surf_repair(args, ctx).has_value());
    auto rs = io::read_fxsurf(out);
    REQUIRE(rs.has_value());
    f64 mean = 0, worst = 0;
    s64 n = 0;
    for (s64 v = 4; v < 20; ++v)
        for (s64 u = 4; u < 20; ++u) {
            const f64 e = std::abs(static_cast<f64>(rs->at(u, v).z) - 69.5);
            mean += e;
            worst = std::max(worst, e);
            ++n;
        }
    mean /= static_cast<f64>(n);
    CHECK(mean < 0.7);
    CHECK(worst < 1.5);
    // papyrus-contact control: uniform volume -> alpha repair must REFUSE to move the mesh
    Volume<u8> uni(vd);
    for (auto& b : uni.flat()) b = 170;
    const std::string up = "/tmp/fenix_alpha_uni.fxvol";
    {
        auto a = codec::VolumeArchive::create(up, vd, codec::DctParams{.q = 0.5f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(uni.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    const std::string_view args2[] = {up, in, "/tmp/fenix_alpha_out2.fxsurf", "mode=alpha", "grid=4"};
    CHECK(!ml::run_surf_repair(args2, ctx).has_value());
    for (const auto& p : {ctp, in, out, up}) std::remove(p.c_str());
}

TEST(surf_repair_alpha_upsample_and_model_prob) {
    // MODEL-alpha snapping: the volume is a sheet-PROBABILITY field with a soft edge
    // (sigmoid around z=67, width ~1.5) — the same mode must localize the confidence
    // edge; upsample=2 must densify the output grid.
    const Extent3 vd{128, 256, 256};
    Volume<u8> pv(vd);
    auto pvv = pv.view();
    for (s64 z = 0; z < vd.z; ++z)
        for (s64 y = 0; y < vd.y; ++y)
            for (s64 x = 0; x < vd.x; ++x) {
                const f64 p = 255.0 / (1.0 + std::exp((static_cast<f64>(z) - 67.0) / 1.5));
                pvv(z, y, x) = static_cast<u8>(p + 0.5);
            }
    const std::string pp = "/tmp/fenix_msnap_prob.fxvol";
    {
        auto a = codec::VolumeArchive::create(pp, vd, codec::DctParams{.q = 0.5f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(pv.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    const std::string in = "/tmp/fenix_msnap_in.fxsurf", out = "/tmp/fenix_msnap_out.fxsurf";
    REQUIRE(io::write_fxsurf(in, plane_at(64.5f)).has_value());
    Context ctx;
    const std::string_view args[] = {pp, in, out, "mode=alpha", "grid=4", "max_shift=6", "upsample=2"};
    REQUIRE(ml::run_surf_repair(args, ctx).has_value());
    auto rs = io::read_fxsurf(out);
    REQUIRE(rs.has_value());
    CHECK(rs->nu == 47);  // (24-1)*2+1: densified grid
    CHECK(rs->nv == 47);
    f64 mean = 0, mn = 1e9, mx = -1e9;
    s64 n = 0;
    for (s64 v = 8; v < 39; ++v)
        for (s64 u = 8; u < 39; ++u) {
            const f64 z = static_cast<f64>(rs->at(u, v).z);
            mean += z;
            mn = std::min(mn, z);
            mx = std::max(mx, z);
            ++n;
        }
    mean /= static_cast<f64>(n);
    CHECK(mean > 66.5);  // moved from 64.5 to the confidence edge (~67.5-68.5)
    CHECK(mean < 69.0);
    CHECK(mx - mn < 1.5);  // and the snapped surface is coherent, not scattered
    for (const auto& p : {pp, in, out}) std::remove(p.c_str());
}

TEST(surf_repair_alpha_side_selection_by_umbilicus) {
    // slab x in [58,69]; mesh at x=61 with +/-x normal. Nearest face is x~57.5, but the
    // RECTO face (air on the umbilicus-OUTWARD side, umbilicus far at x=-1000) is x~69.5.
    const Extent3 vd{128, 224, 128};
    Volume<u8> ctv(vd);
    auto cv = ctv.view();
    for (s64 z = 0; z < vd.z; ++z)
        for (s64 y = 0; y < vd.y; ++y)
            for (s64 x = 0; x < vd.x; ++x) cv(z, y, x) = (x >= 58 && x <= 69) ? 180 : 5;
    const std::string ctp = "/tmp/fenix_side_ct.fxvol";
    {
        auto a = codec::VolumeArchive::create(ctp, vd, codec::DctParams{.q = 0.5f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(ctv.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    Surface s(24, 24);
    s.scale_u = 8.0f;
    s.scale_v = 8.0f;
    for (s64 v = 0; v < 24; ++v)
        for (s64 u = 0; u < 24; ++u)
            s.set(u, v, Vec3f{static_cast<f32>(v) * 8.0f + 16.0f, static_cast<f32>(u) * 8.0f + 16.0f, 61.0f});
    const std::string in = "/tmp/fenix_side_in.fxsurf";
    REQUIRE(io::write_fxsurf(in, s).has_value());
    Context ctx;
    auto mean_x = [&](const char* side_arg) -> f64 {
        const std::string out = "/tmp/fenix_side_out.fxsurf";
        const std::string_view args[] = {ctp, in, out, "mode=alpha", "grid=4",
                                         "search=10", "max_shift=10", side_arg, "umb=112,-1000"};
        auto r = ml::run_surf_repair(args, ctx);
        if (!r) return -1;
        auto rs = io::read_fxsurf(out);
        f64 m = 0;
        s64 n = 0;
        for (s64 v = 6; v < 18; ++v)
            for (s64 u = 6; u < 18; ++u) {
                m += rs->at(u, v).x;
                ++n;
            }
        return m / static_cast<f64>(n);
    };
    const f64 near_x = mean_x("side=near");
    const f64 recto_x = mean_x("side=recto");
    const f64 verso_x = mean_x("side=verso");
    CHECK(std::abs(near_x - 57.5) < 1.0);   // proximity picks the close face
    CHECK(std::abs(recto_x - 69.5) < 1.0);  // recto = air OUTWARD (+x, away from umbilicus)
    CHECK(std::abs(verso_x - 57.5) < 1.0);  // verso = air INWARD
    for (const auto& p : {ctp, in}) std::remove(p.c_str());
}

#include "annotate/annotation.hpp"
#include "ml/stroke_score.hpp"

TEST(stroke_score_matched_offset_missed) {
    // mesh plane at z=64; three human strokes: on it (MATCHED), z=69 (OFFSET), z=100 (MISSED)
    const std::string mp = "/tmp/fenix_ss_mesh.fxsurf";
    REQUIRE(io::write_fxsurf(mp, plane_at(64.0f)).has_value());
    annotate::AnnotationSet a;
    auto mkstroke = [](const char* name, f32 z) {
        annotate::CoWindingStroke s;
        s.name = name;
        s.kind = annotate::StrokeKind::drawing;
        for (int i = 0; i < 12; ++i)
            s.points.push_back(Vec3f{z, 100.0f, 40.0f + static_cast<f32>(i) * 10.0f});
        return s;
    };
    a.strokes.push_back(mkstroke("on_sheet", 64.2f));
    a.strokes.push_back(mkstroke("offset5", 69.0f));
    a.strokes.push_back(mkstroke("nowhere", 100.0f));
    const std::string ap = "/tmp/fenix_ss_anno.toml";
    REQUIRE(annotate::save_annotations(a, ap).has_value());
    Context ctx;
    const std::string_view args[] = {ap, mp, "tol=3", "out=/tmp/fenix_ss.tsv"};
    REQUIRE(ml::run_stroke_score(args, ctx).has_value());
    std::ifstream tf("/tmp/fenix_ss.tsv");
    REQUIRE(static_cast<bool>(tf));
    std::string ln, all;
    while (std::getline(tf, ln)) all += ln + "\n";
    CHECK(all.find("on_sheet") != std::string::npos);
    CHECK(all.find("MATCHED") != std::string::npos);
    CHECK(all.find("OFFSET") != std::string::npos);
    CHECK(all.find("MISSED") != std::string::npos);
    for (const auto& p : {mp, ap, std::string("/tmp/fenix_ss.tsv")}) std::remove(p.c_str());
}

#include "io/import_obj.hpp"

TEST(import_obj_uv_grid_roundtrip) {
    // synthetic VC-style OBJ: a 2-triangle quad in XYZ with uv, plane z=64 (world),
    // spanning 160x160 world units over uv [0,1]^2
    const std::string op = "/tmp/fenix_imp.obj";
    {
        std::ofstream f(op);
        // OBJ is XYZ: x y z per line; our plane z=64: world (z=64, y in 32..192, x in 32..192)
        f << "v 32 32 64\nv 192 32 64\nv 192 192 64\nv 32 192 64\n";
        f << "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n";
        f << "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n";
    }
    Context ctx;
    const std::string out = "/tmp/fenix_imp.fxsurf";
    const std::string_view args[] = {op, out, "grid=8"};
    REQUIRE(io::run_import_obj(args, ctx).has_value());
    auto s = io::read_fxsurf(out);
    REQUIRE(s.has_value());
    CHECK(s->nu >= 15);  // ~160/8 = 20 cells expected (uv-scale estimate has slack)
    CHECK(s->nu <= 30);
    // all covered cells sit on the plane and interpolate x/y across uv
    s64 n = 0;
    for (s64 v = 0; v < s->nv; ++v)
        for (s64 u = 0; u < s->nu; ++u) {
            if (!s->is_valid(u, v)) continue;
            CHECK(std::abs(s->at(u, v).z - 64.0f) < 0.01f);
            ++n;
        }
    CHECK(n > s->nu * s->nv * 9 / 10);  // quad covers (almost) the whole grid
    // affine: scale x3.2958 + shift (the 7.91->2.4 um mapping shape)
    const std::string out2 = "/tmp/fenix_imp2.fxsurf";
    const std::string_view args2[] = {op, out2, "grid=8",
                                      "affine=3.2958,0,0,100,0,3.2958,0,200,0,0,3.2958,300"};
    REQUIRE(io::run_import_obj(args2, ctx).has_value());
    auto s2 = io::read_fxsurf(out2);
    REQUIRE(s2.has_value());
    bool ok = false;
    for (s64 v = 0; v < s2->nv && !ok; ++v)
        for (s64 u = 0; u < s2->nu && !ok; ++u)
            if (s2->is_valid(u, v)) {
                CHECK(std::abs(s2->at(u, v).z - (64.0f * 3.2958f + 100.0f)) < 0.05f);
                ok = true;
            }
    REQUIRE(ok);
    for (const auto& p : {op, out, out2}) std::remove(p.c_str());
}

TEST(import_obj_vc_transform_json) {
    const std::string op = "/tmp/fenix_imp3.obj", tp = "/tmp/fenix_imp3.json";
    {
        std::ofstream f(op);
        f << "v 32 32 64\nv 192 32 64\nv 192 192 64\nv 32 192 64\n";
        f << "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n";
        f << "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n";
    }
    {
        std::ofstream f(tp);  // XYZ rows: swap x<->y, scale z by 2, translate x +10
        f << "{\"transformation_matrix\": [[0, 1, 0, 10], [1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]}\n";
    }
    Context ctx;
    const std::string out = "/tmp/fenix_imp3.fxsurf";
    const std::string targ = "transform=" + tp;
    const std::string_view args[] = {op, out, "grid=8", targ};
    REQUIRE(io::run_import_obj(args, ctx).has_value());
    auto s = io::read_fxsurf(out);
    REQUIRE(s.has_value());
    // original world point (x=32..192, y=32..192, z=64): new_x = y+10, new_y = x, new_z = 128
    bool ok = false;
    for (s64 v = 0; v < s->nv && !ok; ++v)
        for (s64 u = 0; u < s->nu && !ok; ++u)
            if (s->is_valid(u, v)) {
                const Vec3f p = s->at(u, v);
                CHECK(std::abs(p.z - 128.0f) < 0.05f);
                CHECK(p.x >= 41.5f);
                CHECK(p.x <= 202.5f);
                ok = true;
            }
    REQUIRE(ok);
    for (const auto& p : {op, tp, out}) std::remove(p.c_str());
}

#include "render/layers.hpp"
#include "view/sampler.hpp"

TEST(render_layers_stack_matches_offsets) {
    // gradient volume value = z; plane mesh at z=64 with +/-z normal -> layer L INTERIOR
    // pixels read exactly 64 +/- (L-half)*step (sign = normal orientation). Means over
    // interior cells only: a nonzero filter would sweep in codec-ring pixels at the
    // invalid borders (measured: 497 nonzero vs 484 valid cells) and skew everything.
    const Extent3 vd{128, 256, 256};
    Volume<u8> ctv(vd);
    auto cv = ctv.view();
    for (s64 z = 0; z < vd.z; ++z)
        for (s64 y = 0; y < vd.y; ++y)
            for (s64 x = 0; x < vd.x; ++x) cv(z, y, x) = static_cast<u8>(z);
    const std::string ctp = "/tmp/fenix_rl_ct.fxvol";
    {
        auto a = codec::VolumeArchive::create(ctp, vd, codec::DctParams{.q = 0.5f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_volume<u8>(ctv.view()).has_value());
        REQUIRE(a->close().has_value());
    }
    const std::string sp = "/tmp/fenix_rl.fxsurf";
    REQUIRE(io::write_fxsurf(sp, plane_at(64.0f)).has_value());
    Context ctx;
    const std::string out = "/tmp/fenix_rl_stack.fxvol";
    const std::string_view args[] = {ctp, sp, out, "layers=9", "step=2", "q=0.5"};
    REQUIRE(fenix::render::run_render_layers(args, ctx).has_value());
    auto oa = codec::VolumeArchive::open(out);
    REQUIRE(oa.has_value());
    REQUIRE(oa->dims().z == 16);  // 9 layers replicate-padded to the DCT block
    REQUIRE(oa->dims().y == 24);
    std::vector<u8> layer(static_cast<usize>(24 * 24));
    auto interior_mean = [&](s64 L) -> f64 {
        if (!oa->gather_box_u8(0, L, 0, 0, 1, 24, 24, layer.data())) return -1.0;
        f64 m = 0;
        s64 n = 0;
        for (s64 v = 6; v < 18; ++v)
            for (s64 u = 6; u < 18; ++u) {
                m += layer[static_cast<usize>(v * 24 + u)];
                ++n;
            }
        return m / static_cast<f64>(n);
    };
    const f64 m0 = interior_mean(0), m4 = interior_mean(4), m8 = interior_mean(8);
    CHECK(std::abs(m4 - 64.0) < 1.5);                 // offset 0 = on the plane
    CHECK(std::abs(std::abs(m8 - m0) - 16.0) < 2.0);  // full span = 8 layers x step 2
    CHECK(std::abs((m0 + m8) / 2.0 - 64.0) < 1.5);    // symmetric about the plane
    for (const auto& p : {ctp, sp, out}) std::remove(p.c_str());
}
