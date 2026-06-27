// test_fit.cpp — the unified diffeomorphic fit recovers model parameters from constraints.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "winding/fit.hpp"
#include "winding/spiral_model.hpp"

#include <vector>

using namespace fenix;
using namespace fenix::winding;

static SpiralModel base_model(s64 s) {
    SpiralModel m;
    for (s64 z = 0; z < s; ++z) {
        m.umbilicus.z.push_back(static_cast<f32>(z));
        m.umbilicus.y.push_back(static_cast<f32>(s) * 0.5f);
        m.umbilicus.x.push_back(static_cast<f32>(s) * 0.5f);
    }
    m.has_flow = false;
    m.dr_per_winding = 8.0f;
    return m;
}

TEST(fit_recovers_perturbed_parameters) {
    const s64 s = 64;
    // Ground-truth model with a real affine + a specific pitch.
    SpiralModel truth = base_model(s);
    truth.affine = AffineYX{.a = 0.10f, .b = 0.12f, .c = -0.07f, .d = 0.04f, .ty = 2.0f, .tx = -1.0f};
    truth.dr_per_winding = 7.0f;

    // Sample constraint points across the volume; targets are the truth windings.
    std::vector<FitConstraint> cs;
    Pcg32 rng{21};
    for (int i = 0; i < 300; ++i) {
        Vec3f p{rng.next_f32() * static_cast<f32>(s), 8.0f + rng.next_f32() * (static_cast<f32>(s) - 16),
                8.0f + rng.next_f32() * (static_cast<f32>(s) - 16)};
        cs.push_back({p, truth.winding_at(p)});
    }

    // Start from a perturbed model and fit back.
    SpiralModel model = base_model(s);
    model.affine = AffineYX{};  // identity (wrong)
    model.dr_per_winding = 9.0f;  // wrong pitch

    FitResult r = fit_spiral(model, cs, {.iters = 800, .lr = 0.03f});
    CHECK(r.final_loss < r.initial_loss * 0.1f);  // loss drops sharply
    CHECK(r.final_loss < 0.05f);                  // constraints well satisfied (winding^2 units)

    // Generalization: the fitted model reproduces the truth winding field on HELD-OUT points.
    // (dr alone isn't identifiable — it trades off with the affine scale — but the winding
    // field, densely constrained, is pinned.)
    f64 held_err = 0;
    int held_n = 0;
    Pcg32 hrng{777};
    for (int i = 0; i < 80; ++i) {
        Vec3f p{hrng.next_f32() * static_cast<f32>(s), 12.0f + hrng.next_f32() * (static_cast<f32>(s) - 24),
                12.0f + hrng.next_f32() * (static_cast<f32>(s) - 24)};
        held_err += std::abs(static_cast<f64>(model.winding_at(p) - truth.winding_at(p)));
        ++held_n;
    }
    CHECK(held_err / static_cast<f64>(held_n) < 0.25);  // reproduces the field off the training points
}

TEST(fit_is_noop_when_already_satisfied) {
    const s64 s = 48;
    SpiralModel truth = base_model(s);
    std::vector<FitConstraint> cs;
    for (int i = 0; i < 50; ++i) {
        Vec3f p{24, 10.0f + static_cast<f32>(i) * 0.5f, 30};
        cs.push_back({p, truth.winding_at(p)});
    }
    SpiralModel model = truth;  // already correct
    FitResult r = fit_spiral(model, cs, {.iters = 100});
    CHECK(r.initial_loss < 1e-4f);
    CHECK(r.final_loss < 1e-3f);  // stays satisfied
}
