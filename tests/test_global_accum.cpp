// test_global_accum.cpp — the global (zero-waste) predict grid's torch-free machinery:
// factorized weight profiles, sparse chunk accumulator, z-sweep finalize, resume floor.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"
#include "ml/global_accum.hpp"
#include "ml/tiling.hpp"

#include <cmath>
#include <map>
#include <vector>

using namespace fenix;
using namespace fenix::ml;

namespace {
constexpr s64 T = GlobalAccum::T;

// Scatter a constant-value patch at every grid position; after finalize every covered voxel
// must decode to exactly that constant (the factorized profile cancels the accumulated weight).
struct Sim {
    Index3 org;
    Extent3 ext;
    int P;
    double ov;
    std::vector<s64> zs, ys, xs;
    std::vector<float> g;
    GlobalAccum acc;

    Sim(Index3 o, Extent3 e, int patch, double overlap)
        : org(o), ext(e), P(patch), ov(overlap),
          zs(tile_starts(e.z, patch, overlap)), ys(tile_starts(e.y, patch, overlap)),
          xs(tile_starts(e.x, patch, overlap)), g(gaussian1d(patch)),
          acc(o, e, global_weight_profile(gaussian1d(patch), zs, e.z, patch),
              global_weight_profile(gaussian1d(patch), ys, e.y, patch),
              global_weight_profile(gaussian1d(patch), xs, e.x, patch)) {}

    void scatter_all(float value) {
        std::vector<float> sp(static_cast<usize>(P) * static_cast<usize>(P) * static_cast<usize>(P), value);
        for (s64 z0 : zs)
            for (s64 y0 : ys)
                for (s64 x0 : xs) acc.scatter(org.z + z0, org.y + y0, org.x + x0, P, sp.data(), g, g, g);
    }
};
}  // namespace

TEST(global_profile_matches_scatter) {
    // Profile at coord c == sum of window weights of every start covering c.
    const int P = 64;
    const auto starts = tile_starts(160, P, 0.5);
    const auto g = gaussian1d(P);
    const auto W = global_weight_profile(g, starts, 160, P);
    for (s64 c = 0; c < 160; ++c) {
        float w = 0;
        for (s64 s0 : starts)
            if (c >= s0 && c < s0 + P) w += g[static_cast<usize>(c - s0)];
        CHECK(std::abs(W[static_cast<usize>(c)] - w) < 1e-5f);
    }
}

TEST(global_accum_constant_roundtrip) {
    // Full grid of constant-0.6 patches → every voxel finalizes to round(0.6*255) exactly,
    // regardless of overlap-count variation across the volume.
    Sim s({64, 0, 64}, {192, 128, 192}, 64, 0.5);
    s.scatter_all(0.6f);
    std::map<u64, std::vector<u8>> out;
    s.acc.finalize_rows_below(s.acc.chunk_rows(), [&](ChunkCoord tc, const u8* blk) {
        CHECK(tc.z >= 1 && tc.y >= 0 && tc.x >= 1);  // archive coords offset by org/T
        out[static_cast<u64>((tc.z * 100 + tc.y) * 100 + tc.x)] = std::vector<u8>(blk, blk + T * T * T);
    });
    CHECK(out.size() == 3u * 2u * 3u);
    const u8 want = static_cast<u8>(0.6f * 255.0f + 0.5f);
    usize bad = 0;
    for (const auto& [k, blk] : out)
        for (u8 v : blk)
            if (v != want) ++bad;
    CHECK(bad == 0);
}

TEST(global_accum_sweep_equals_oneshot) {
    // Finalizing row-by-row as the sweep front advances must produce the same bytes as one
    // finalize at the end (chunks are evicted on finalize — each emitted exactly once).
    Sim a({0, 0, 0}, {192, 64, 64}, 64, 0.5), b({0, 0, 0}, {192, 64, 64}, 64, 0.5);
    a.scatter_all(0.25f);
    b.scatter_all(0.25f);
    std::map<u64, std::vector<u8>> oa, ob;
    auto sink = [](std::map<u64, std::vector<u8>>& m) {
        return [&m](ChunkCoord tc, const u8* blk) {
            const u64 k = static_cast<u64>((tc.z * 100 + tc.y) * 100 + tc.x);
            CHECK(!m.contains(k));  // exactly-once emission
            m[k] = std::vector<u8>(blk, blk + T * T * T);
        };
    };
    a.acc.finalize_rows_below(a.acc.chunk_rows(), sink(oa));
    for (s64 r = 1; r <= b.acc.chunk_rows(); ++r) b.acc.finalize_rows_below(r, sink(ob));
    REQUIRE(oa.size() == ob.size());
    for (const auto& [k, blk] : oa) {
        REQUIRE(ob.contains(k));
        CHECK(ob[k] == blk);
    }
}

TEST(global_accum_floor_drops_below) {
    // Resume floor: scatters below the floor row are dropped; finalize starts at the floor.
    Sim s({0, 0, 0}, {256, 64, 64}, 64, 0.5);
    s.acc.set_floor_row(2);  // rows 0,1 (z<128) already finalized in a previous run
    s.scatter_all(1.0f);
    std::vector<s64> rows;
    s.acc.finalize_rows_below(s.acc.chunk_rows(), [&](ChunkCoord tc, const u8*) { rows.push_back(tc.z); });
    REQUIRE(!rows.empty());
    for (s64 r : rows) CHECK(r >= 2);
    CHECK(static_cast<s64>(rows.size()) == 2);  // rows 2,3 × 1×1 chunks
}

TEST(global_accum_unscattered_chunk_is_zero) {
    // A chunk no patch touched (air-skipped) is emitted as explicit zeros (assessed air).
    GlobalAccum acc({0, 0, 0}, {64, 64, 128}, std::vector<float>(64, 1.0f), std::vector<float>(64, 1.0f),
                    std::vector<float>(128, 1.0f));
    std::vector<float> sp(static_cast<usize>(64) * 64 * 64, 0.5f);
    const auto g = gaussian1d(64);
    acc.scatter(0, 0, 0, 64, sp.data(), g, g, g);  // touches chunk x=0 only
    std::map<s64, bool> nonzero;
    acc.finalize_rows_below(1, [&](ChunkCoord tc, const u8* blk) {
        bool nz = false;
        for (s64 i = 0; i < T * T * T; ++i)
            if (blk[static_cast<usize>(i)] != 0) { nz = true; break; }
        nonzero[tc.x] = nz;
    });
    REQUIRE(nonzero.size() == 2);
    CHECK(nonzero[0]);
    CHECK(!nonzero[1]);
}
