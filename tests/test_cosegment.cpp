// test_cosegment.cpp — Stage D coupled patch↔field refinement. The middle of three concentric wraps
// has a band of WEAK (low-confidence) cells displaced off its true radius (simulating a bad
// prediction). Because weak cells don't pin the field, the field is set by the clean neighbours, and
// the EM loop pulls the bad cells back onto the correct wrap — coherent with the wraps on either side.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/surface.hpp"
#include "core/test.hpp"
#include "winding/cosegment.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using namespace fenix;

namespace {
constexpr f32 kTwoPi = 2.0f * std::numbers::pi_v<f32>;
constexpr f32 kCy = 200.0f, kCx = 200.0f, kZ0 = 50.0f;

Surface make_wrap(f32 radius, int nu, int nv, f32 step) {
    Surface S(nu, nv);
    S.alloc_channels();
    for (int v = 0; v < nv; ++v)
        for (int u = 0; u < nu; ++u) {
            const f32 a = kTwoPi * static_cast<f32>(u) / static_cast<f32>(nu);
            const f32 sy = std::sin(a), cx = std::cos(a);
            const usize id = S.idx(u, v);
            S.coord[id] = Vec3f{kZ0 + static_cast<f32>(v) * step, kCy + radius * sy, kCx + radius * cx};
            S.valid[id] = 1;
            S.normal[id] = Vec3f{0, sy, cx};
            S.conf[id] = 1.0f;
        }
    return S;
}

f32 radius_of(const Surface& S, int u, int v) {
    const Vec3f p = S.at(u, v);
    return std::sqrt((p.y - kCy) * (p.y - kCy) + (p.x - kCx) * (p.x - kCx));
}

// An angular fragment [u0,u1) of a full nu_full-resolution wrap — mimics a tile-local trace fragment.
Surface make_frag(f32 radius, int u0, int u1, int nu_full, int nv, f32 step) {
    const int nu = u1 - u0;
    Surface S(nu, nv);
    S.alloc_channels();
    for (int v = 0; v < nv; ++v)
        for (int uu = 0; uu < nu; ++uu) {
            const f32 a = kTwoPi * static_cast<f32>(u0 + uu) / static_cast<f32>(nu_full);
            const f32 sy = std::sin(a), cx = std::cos(a);
            const usize id = S.idx(uu, v);
            S.coord[id] = Vec3f{kZ0 + static_cast<f32>(v) * step, kCy + radius * sy, kCx + radius * cx};
            S.valid[id] = 1;
            S.normal[id] = Vec3f{0, sy, cx};
            S.conf[id] = 1.0f;
        }
    return S;
}
}  // namespace

TEST(cosegment_corrects_weak_displaced_band) {
    const f32 R0 = 40.0f, spacing = 8.0f, step = 2.0f, Rmid = R0 + spacing;
    const int nu = 96, nv = 24;
    annotate::Umbilicus umb;
    umb.z = {kZ0 - 4, kZ0 + static_cast<f32>(nv) * step + 4};
    umb.y = {kCy, kCy};
    umb.x = {kCx, kCx};

    std::vector<Surface> sheets;
    sheets.push_back(make_wrap(R0, nu, nv, step));
    sheets.push_back(make_wrap(Rmid, nu, nv, step));
    sheets.push_back(make_wrap(R0 + 2 * spacing, nu, nv, step));

    // Displace a band of the middle wrap inward (to ~R45) and mark it low-confidence (a bad
    // prediction). Keep validity — these are wrong-but-present cells, the Stage D target.
    const int ulo = 20, uhi = 60, vlo = 8, vhi = 16;
    for (int v = vlo; v <= vhi; ++v)
        for (int u = ulo; u <= uhi; ++u) {
            const f32 a = kTwoPi * static_cast<f32>(u) / static_cast<f32>(nu);
            const usize id = sheets[1].idx(u, v);
            sheets[1].coord[id] = Vec3f{kZ0 + static_cast<f32>(v) * step, kCy + (Rmid - 3.0f) * std::sin(a), kCx + (Rmid - 3.0f) * std::cos(a)};
            sheets[1].conf[id] = 0.3f;
        }

    f64 err_before = 0;
    int nb = 0;
    for (int v = vlo; v <= vhi; ++v)
        for (int u = ulo; u <= uhi; ++u) { err_before += static_cast<f64>(std::abs(radius_of(sheets[1], u, v) - Rmid)); ++nb; }
    err_before /= nb;
    CHECK(err_before > 2.0);  // starts clearly off-wrap

    segment::PatchGraphParams gp;
    gp.step = step;
    winding::CosegParams cp;
    cp.full = Extent3{112, 272, 272};
    cp.rounds = 4;
    cp.field.ds = 2;
    cp.field.iters = 150;
    cp.conf_thresh = 0.8f;
    cp.consist_weight = 0.6f;
    cp.fill.step = step;
    const winding::CosegReport rep = winding::cosegment_refine(sheets, umb, gp, cp);

    f64 err_after = 0;
    for (int v = vlo; v <= vhi; ++v)
        for (int u = ulo; u <= uhi; ++u) err_after += static_cast<f64>(std::abs(radius_of(sheets[1], u, v) - Rmid));
    err_after /= nb;

    CHECK(err_after < 1.0);            // pulled back onto the true wrap
    CHECK(err_after < err_before * 0.5);  // a clear improvement
    CHECK(rep.conflicts == 0);         // winding assignment stayed consistent
    CHECK(rep.monotonicity < 0.15f);   // field stays radially ordered (no fold/overlap)
    CHECK(rep.wrap_hi - rep.wrap_lo == 2);  // three wraps spanned
}

// Stage D with the EULERIAN winding path on FRAGMENTED wraps (the tiled-trace scenario): three wraps,
// each cut into 3 disjoint angular fragments (9 patches), with a weak displaced band in one middle-wrap
// fragment. The discrete winding can't coherently stitch fragments; the normal-driven field can — and
// once windings are coherent, the field-fill pulls the weak band back onto its wrap.
TEST(cosegment_eulerian_corrects_fragmented_band) {
    const f32 R0 = 40.0f, spacing = 8.0f, step = 2.0f, Rmid = R0 + spacing;
    const int nu_full = 96, nv = 24;
    annotate::Umbilicus umb;
    umb.z = {kZ0 - 4, kZ0 + static_cast<f32>(nv) * step + 4};
    umb.y = {kCy, kCy};
    umb.x = {kCx, kCx};

    std::vector<Surface> sheets;
    std::vector<int> wrap_of;
    const int cuts[4] = {0, 32, 64, 96};
    for (int w = 0; w < 3; ++w)
        for (int t = 0; t < 3; ++t) {
            sheets.push_back(make_frag(R0 + static_cast<f32>(w) * spacing, cuts[t], cuts[t + 1], nu_full, nv, step));
            wrap_of.push_back(w);
        }

    // displace+weaken a band in the FIRST middle-wrap fragment (index w=1,t=0 -> sheets[3], u-local 8..28)
    Surface& mid = sheets[3];
    const int ulo = 8, uhi = 28, vlo = 8, vhi = 16;
    f64 err_before = 0;
    int nb = 0;
    for (int v = vlo; v <= vhi; ++v)
        for (int uu = ulo; uu <= uhi; ++uu) {
            const f32 a = kTwoPi * static_cast<f32>(cuts[0] + uu) / static_cast<f32>(nu_full);
            const usize id = mid.idx(uu, v);
            mid.coord[id] = Vec3f{kZ0 + static_cast<f32>(v) * step, kCy + (Rmid - 3.0f) * std::sin(a), kCx + (Rmid - 3.0f) * std::cos(a)};
            mid.conf[id] = 0.3f;
            err_before += static_cast<f64>(std::abs(radius_of(mid, uu, v) - Rmid));
            ++nb;
        }
    err_before /= nb;
    CHECK(err_before > 2.0);

    segment::PatchGraphParams gp;
    gp.step = step;
    winding::CosegParams cp;
    cp.full = Extent3{112, 272, 272};
    cp.rounds = 4;
    cp.eulerian = true;       // the robust normal-driven winding for fragments
    cp.efield.ds = 2;
    cp.efield.iters = 600;
    cp.field.ds = 2;
    cp.field.iters = 150;
    cp.conf_thresh = 0.8f;
    cp.consist_weight = 0.6f;
    cp.fill.step = step;
    const winding::CosegReport rep = winding::cosegment_refine(sheets, umb, gp, cp);

    f64 err_after = 0;
    for (int v = vlo; v <= vhi; ++v)
        for (int uu = ulo; uu <= uhi; ++uu) err_after += static_cast<f64>(std::abs(radius_of(mid, uu, v) - Rmid));
    err_after /= nb;

    CHECK(err_after < err_before * 0.6);    // weak band pulled toward the true wrap
    CHECK(rep.wrap_hi - rep.wrap_lo == 2);  // 3 wraps coherently spanned from 9 fragments
}
