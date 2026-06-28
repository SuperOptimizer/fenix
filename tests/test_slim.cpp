// test_slim.cpp — drive the C++ SLIM/ARAP flattening on a grown surface grid (raw x/y/z/valid),
// report symmetric-Dirichlet energy, and dump per-vertex pos+uv for rasterization. Standalone main.
// Usage: test_slim <tracedir> [iters] [outdir]
#include "core/core.hpp"
#include "core/surface.hpp"
#include "flatten/slim.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace fenix;

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "data/fenix_trace";
    const int iters = argc > 2 ? std::atoi(argv[2]) : 20;
    const std::string outdir = argc > 3 ? argv[3] : "data/fenix_flat";

    const s64 G = [&] { FILE* f = std::fopen((dir + "/meta.txt").c_str(), "r"); long g = 0; if (f) { (void)std::fscanf(f, "%ld", &g); std::fclose(f); } return static_cast<s64>(g); }();
    if (G <= 0) { std::printf("bad meta in %s\n", dir.c_str()); return 1; }
    auto rd = [&](const std::string& nm, usize bytes) { std::vector<u8> v(bytes); FILE* f = std::fopen((dir + "/" + nm).c_str(), "rb"); if (f) { (void)std::fread(v.data(), 1, bytes, f); std::fclose(f); } return v; };
    const usize NG = static_cast<usize>(G * G);
    auto xb = rd("x.f32", NG * 4), yb = rd("y.f32", NG * 4), zb = rd("z.f32", NG * 4), mb = rd("valid.u8", NG);
    const f32* X = reinterpret_cast<const f32*>(xb.data());
    const f32* Y = reinterpret_cast<const f32*>(yb.data());
    const f32* Z = reinterpret_cast<const f32*>(zb.data());

    Surface S(G, G);
    for (usize i = 0; i < NG; ++i) if (mb[i]) { S.coord[i] = Vec3f{Z[i], Y[i], X[i]}; S.valid[i] = 1; }
    std::printf("loaded grid %lldx%lld  valid=%lld\n", (long long)G, (long long)G, (long long)S.valid_count());

    auto t0 = std::chrono::steady_clock::now();
    flatten::FlatMesh M = flatten::slim_parameterize(S, iters);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("flatten %.1fs  verts=%zu tris=%zu  sym-Dirichlet %.3f -> %.3f (2.0=isometric)\n",
                std::chrono::duration<double>(t1 - t0).count(), M.pos.size(), M.tri.size(), M.energy_init, M.energy_final);

    std::string mk = "mkdir -p " + outdir; (void)std::system(mk.c_str());
    const usize N = M.pos.size();
    std::vector<f32> pos(N * 3), uv(N * 2);
    for (usize i = 0; i < N; ++i) { pos[i * 3] = M.pos[i].z; pos[i * 3 + 1] = M.pos[i].y; pos[i * 3 + 2] = M.pos[i].x; uv[i * 2] = M.uv[i][0]; uv[i * 2 + 1] = M.uv[i][1]; }
    auto wr = [&](const std::string& nm, const void* d, usize b) { FILE* f = std::fopen((outdir + "/" + nm).c_str(), "wb"); std::fwrite(d, 1, b, f); std::fclose(f); };
    wr("pos.f32", pos.data(), N * 3 * 4);
    wr("uv.f32", uv.data(), N * 2 * 4);
    FILE* mf = std::fopen((outdir + "/meta.txt").c_str(), "w"); std::fprintf(mf, "%zu\n", N); std::fclose(mf);
    std::printf("wrote %s/{pos,uv}.f32 (N=%zu)\n", outdir.c_str(), N);
    return 0;
}
