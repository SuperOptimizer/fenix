// test_ml_tiler.cpp — sliding-window tiling math (torch-free; runs in the core build).
// Validates coverage/overlap and the Gaussian importance window used to blend tiles.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "ml/tiling.hpp"

using namespace fenix;

TEST(tile_starts_small_volume_single_patch) {
    auto s = ml::tile_starts(64, 256, 0.5);  // dim < patch
    REQUIRE(s.size() == 1);
    CHECK(s[0] == 0);
}

TEST(tile_starts_cover_far_edge) {
    // dim 256, patch 128, overlap 0.5 -> step 64 -> starts 0,64,128 (last == dim-patch).
    auto s = ml::tile_starts(256, 128, 0.5);
    REQUIRE(s.size() == 3);
    CHECK(s.front() == 0);
    CHECK(s.back() == 128);          // 256-128, far edge fully covered
    CHECK(s[1] == 64);
}

TEST(tile_starts_full_coverage_invariant) {
    // Every voxel in [0,dim) must lie in at least one [start, start+patch) window.
    const s64 dim = 300; const int patch = 128; const double ov = 0.5;
    auto s = ml::tile_starts(dim, patch, ov);
    for (s64 v = 0; v < dim; ++v) {
        bool covered = false;
        for (s64 st : s)
            if (v >= st && v < st + patch) { covered = true; break; }
        CHECK(covered);
    }
    CHECK(s.back() == dim - patch);
}

TEST(gaussian_window_peaks_at_center) {
    auto w = ml::gaussian1d(128);
    REQUIRE(static_cast<int>(w.size()) == 128);
    // Center is the max, ends are the 0.1 floor, monotone-ish toward center.
    CHECK(w[64] >= w[0]);
    CHECK(w[64] >= w[127]);
    CHECK(w[0] >= 0.1f - 1e-6f);
    CHECK(w[127] >= 0.1f - 1e-6f);
    float mx = 0.0f;
    for (float v : w) mx = std::max(mx, v);
    CHECK(mx <= 1.0f + 1e-6f);
    CHECK(w[64] > w[16]);  // center heavier than off-center
}
