// test_annotate.cpp — annotation model TOML roundtrip + version gate, the annotation→fit
// bridge lowering, trusted-region extraction, and rel-winding fit recovery.
#define FENIX_TEST_MAIN
#include "annotate/annotation.hpp"
#include "annotate/extract.hpp"
#include "core/core.hpp"
#include "core/test.hpp"
#include "winding/anno_bridge.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

using namespace fenix;
using namespace fenix::annotate;

static std::string tmp_path(const char* name) {
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove(p);
    return p.string();
}

static AnnotationSet sample_set() {
    AnnotationSet a;
    CoWindingStroke s0;
    s0.name = "kollesis-a";
    s0.kind = StrokeKind::kollesis;
    s0.points = {{1.5f, 20.25f, 30.0f}, {2.5f, 21.0f, 31.5f}, {3.5f, 22.0f, 33.0f}};
    s0.weight = 2.0f;
    a.strokes.push_back(s0);
    CoWindingStroke s1;
    s1.name = "fib";
    s1.kind = StrokeKind::fiber;
    s1.points = {{5, 40, 41}, {6, 40, 42}};
    s1.has_winding = true;
    s1.winding = 7.5f;
    a.strokes.push_back(s1);
    RadialLine r;
    r.name = "rl";
    r.points = {{10, 32, 40}, {10, 32, 48}, {10, 32, 57}};
    r.offset = {0, 1, 2};
    r.weight = 1.5f;
    a.radial_lines.push_back(r);
    a.normals.push_back({{4, 5, 6}, {0, 0.6f, 0.8f}, 0.5f});
    a.links.push_back({0, 1, false});
    return a;
}

TEST(annotation_toml_roundtrip) {
    const AnnotationSet a = sample_set();
    const std::string path = tmp_path("fenix_test_anno.toml");
    auto w = save_annotations(a, path);
    REQUIRE(w.has_value());
    auto r = load_annotations(path);
    REQUIRE(r.has_value());
    const AnnotationSet& b = *r;

    REQUIRE(b.strokes.size() == a.strokes.size());
    CHECK(b.strokes[0].name == "kollesis-a");
    CHECK(b.strokes[0].kind == StrokeKind::kollesis);
    CHECK(!b.strokes[0].has_winding);
    CHECK(std::abs(b.strokes[0].weight - 2.0f) < 1e-6f);
    REQUIRE(b.strokes[0].points.size() == 3);
    CHECK(std::abs(b.strokes[0].points[1].y - 21.0f) < 1e-5f);
    CHECK(b.strokes[1].has_winding);
    CHECK(std::abs(b.strokes[1].winding - 7.5f) < 1e-6f);

    REQUIRE(b.radial_lines.size() == 1);
    REQUIRE(b.radial_lines[0].points.size() == 3);
    CHECK(b.radial_lines[0].offset == std::vector<s32>({0, 1, 2}));
    CHECK(!b.radial_lines[0].has_base_winding);

    REQUIRE(b.normals.size() == 1);
    CHECK(std::abs(b.normals[0].dir.x - 0.8f) < 1e-6f);
    REQUIRE(b.links.size() == 1);
    CHECK(b.links[0].a == 0 && b.links[0].b == 1 && !b.links[0].cannot);
    std::filesystem::remove(path);
}

TEST(annotation_rejects_unknown_version) {
    const std::string path = tmp_path("fenix_test_anno_badver.toml");
    std::ofstream(path) << "version = 99\n";
    auto r = load_annotations(path);
    REQUIRE(!r.has_value());
    CHECK(r.error().code == Errc::unsupported);
    std::filesystem::remove(path);
}

TEST(bridge_lowers_annotations) {
    AnnotationSet a;
    // Two unlabeled strokes must-linked -> ONE co-winding group.
    CoWindingStroke u0, u1;
    u0.points = {{1, 2, 3}, {1, 2, 4}};
    u1.points = {{1, 5, 3}, {1, 5, 4}, {1, 5, 5}};
    a.strokes.push_back(u0);
    a.strokes.push_back(u1);
    a.links.push_back({0, 1, false});
    // A labeled stroke -> absolute targets.
    CoWindingStroke lab;
    lab.points = {{2, 2, 3}, {2, 2, 4}};
    lab.has_winding = true;
    lab.winding = 3.0f;
    a.strokes.push_back(lab);
    // Radial line without a base -> consecutive rel pairs; with a base -> targets.
    RadialLine rl;
    rl.points = {{4, 10, 10}, {4, 10, 20}, {4, 10, 31}};
    rl.offset = {0, 1, 3};
    rl.weight = 2.0f;
    a.radial_lines.push_back(rl);
    RadialLine ab = rl;
    ab.has_base_winding = true;
    ab.base_winding = 5.0f;
    a.radial_lines.push_back(ab);

    const winding::AnnoFitInputs fi = winding::to_fit_inputs(a);
    REQUIRE(fi.groups.size() == 1);
    CHECK(fi.groups[0].points.size() == 5);
    // 2 labeled-stroke targets + 3 absolute radial targets.
    REQUIRE(fi.targets.size() == 5);
    CHECK(std::abs(fi.targets[0].target_winding - 3.0f) < 1e-6f);
    CHECK(std::abs(fi.targets[4].target_winding - 8.0f) < 1e-6f);  // base 5 + offset 3
    REQUIRE(fi.rels.size() == 2);
    CHECK(std::abs(fi.rels[0].delta - 1.0f) < 1e-6f);
    CHECK(std::abs(fi.rels[1].delta - 2.0f) < 1e-6f);  // offsets 1 -> 3
    CHECK(std::abs(fi.rels[1].weight - 2.0f) < 1e-6f);
}

TEST(bridge_musklink_propagates_absolute_winding) {
    AnnotationSet a;
    CoWindingStroke u, lab;
    u.points = {{1, 2, 3}, {1, 2, 4}};
    lab.points = {{2, 2, 3}};
    lab.has_winding = true;
    lab.winding = 4.0f;
    a.strokes.push_back(u);
    a.strokes.push_back(lab);
    a.links.push_back({0, 1, false});
    const winding::AnnoFitInputs fi = winding::to_fit_inputs(a);
    CHECK(fi.groups.empty());  // the unlabeled stroke inherited the absolute winding
    REQUIRE(fi.targets.size() == 3);
    for (const auto& t : fi.targets) CHECK(std::abs(t.target_winding - 4.0f) < 1e-6f);
}

TEST(extract_trusted_keeps_good_regions) {
    // Two high-conf blobs separated by a low-conf band; coords encode (u,v) for readback.
    Surface s(64, 32);
    s.alloc_channels();
    for (s64 v = 0; v < s.nv; ++v)
        for (s64 u = 0; u < s.nu; ++u) {
            s.set(u, v, Vec3f{static_cast<f32>(u), static_cast<f32>(v), 0});
            s.conf[s.idx(u, v)] = (u >= 28 && u < 36) ? 0.2f : 1.5f;  // low-conf band splits u
        }
    // An invalid hole inside the left blob (should just erode around it).
    s.valid[s.idx(10, 10)] = 0;

    ExtractParams p;
    p.erode = 1;
    p.min_cells = 32;
    p.stride = 2;
    const auto strokes = extract_trusted(s, p);
    REQUIRE(strokes.size() == 2);
    for (const auto& st : strokes) {
        CHECK(st.kind == StrokeKind::patch_extract);
        CHECK(st.points.size() >= 2);
        for (const Vec3f& pt : st.points) {
            const s64 u = static_cast<s64>(pt.z), v = static_cast<s64>(pt.y);
            CHECK(s.conf[s.idx(u, v)] >= p.conf_min);   // never a low-conf cell
            CHECK(u > 0 && u + 1 < s.nu && v > 0 && v + 1 < s.nv);  // borders eroded
        }
    }
}

TEST(rel_winding_fit_recovers) {
    // Truth spiral; the fit sees a few absolute anchors + radial-line REL pairs from a
    // perturbed start (affine-only for speed). Gates are DIFFERENTIAL: the rel pairs must
    // (a) end up satisfied and (b) buy real held-out accuracy over the anchors alone —
    // absolute loss thresholds are fragile here (dr trades off against affine scale, so
    // sparse-constraint convergence is asymptotic along an ill-conditioned valley).
    const s64 sz = 64;
    winding::SpiralModel truth;
    for (s64 z = 0; z < sz; ++z) {
        truth.umbilicus.z.push_back(static_cast<f32>(z));
        truth.umbilicus.y.push_back(static_cast<f32>(sz) * 0.5f);
        truth.umbilicus.x.push_back(static_cast<f32>(sz) * 0.5f);
    }
    truth.has_flow = false;
    truth.dr_per_winding = 7.0f;
    truth.affine = winding::AffineYX{.ty = 1.0f, .tx = -0.5f};

    const f32 fs = static_cast<f32>(sz);
    std::vector<winding::FitConstraint> anchors;
    std::vector<winding::RelWindingConstraint> rels;
    Pcg32 rng{99};
    for (int i = 0; i < 6; ++i) {
        Vec3f p{rng.next_f32() * fs, 10.0f + rng.next_f32() * (fs - 20), 10.0f + rng.next_f32() * (fs - 20)};
        anchors.push_back({p, truth.winding_at(p)});
    }
    // Radial lines: walk outward from the center along random directions; consecutive pairs
    // carry the TRUTH winding delta (as an annotated +k crossing line would).
    for (int l = 0; l < 24; ++l) {
        const f32 z = rng.next_f32() * fs;
        const f32 ang = rng.next_f32() * 6.2831853f;
        Vec3f prev{};
        f32 prev_w = 0;
        for (int k = 0; k < 4; ++k) {
            const f32 r = 6.0f + 6.5f * static_cast<f32>(k);
            Vec3f p{z, fs * 0.5f + r * std::sin(ang), fs * 0.5f + r * std::cos(ang)};
            const f32 w = truth.winding_at(p);
            if (k > 0) rels.push_back({prev, p, w - prev_w, 1.0f});
            prev = p;
            prev_w = w;
        }
    }

    auto fresh = [&] {
        winding::SpiralModel m;
        m.umbilicus = truth.umbilicus;
        m.has_flow = false;
        m.dr_per_winding = 9.5f;  // wrong pitch, identity affine
        return m;
    };
    auto held_out_err = [&](const winding::SpiralModel& m) {
        f64 e = 0;
        Pcg32 hrng{7};
        for (int i = 0; i < 60; ++i) {
            Vec3f p{hrng.next_f32() * fs, 14.0f + hrng.next_f32() * (fs - 28), 14.0f + hrng.next_f32() * (fs - 28)};
            e += std::abs(static_cast<f64>(m.winding_at(p) - truth.winding_at(p)));
        }
        return e / 60.0;
    };
    winding::DiffeoFitConfig cfg;
    cfg.fit_flow = false;
    cfg.iters_affine = 800;

    winding::SpiralModel with_rels = fresh();
    const winding::FitResult r = winding::fit_spiral_diffeo(with_rels, anchors, {}, rels, cfg);
    CHECK(r.final_loss < r.initial_loss * 0.25f);

    // (a) the rel pairs themselves end up satisfied (RMS residual in windings).
    f64 rel_sq = 0;
    for (const auto& rc : rels) {
        const f64 e = static_cast<f64>(with_rels.winding_at(rc.b)) - with_rels.winding_at(rc.a) - rc.delta;
        rel_sq += e * e;
    }
    CHECK(std::sqrt(rel_sq / static_cast<f64>(rels.size())) < 0.15);

    // (b) the rel annotations buy real held-out accuracy over the 6 anchors alone.
    winding::SpiralModel anchors_only = fresh();
    winding::fit_spiral_diffeo(anchors_only, anchors, {}, {}, cfg);
    const f64 he_with = held_out_err(with_rels), he_without = held_out_err(anchors_only);
    CHECK(he_with < 0.25);
    CHECK(he_with < he_without * 0.5);
}
