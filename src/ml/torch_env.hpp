// ml/torch_env.hpp — the ONE place libtorch is included. Everything torch-typed lives behind
// the FENIX_ML firewall (see ml/CLAUDE.md): this header is only pulled in when FENIX_ML is
// defined, and torch types must not leak past the ml/ module's API. Headers are consumed as
// -isystem (deps.cmake) so the project's -Weverything doesn't fire on them.
#pragma once

#include "core/core.hpp"  // fenix::cpu_budget()

#include <torch/torch.h>

#include <cstdlib>

#include <string>

namespace fenix::ml {

// Clamp libtorch's CPU thread pools to the real CPU budget (cgroup quota, not host core count) — same
// container over-subscription fix as core's OpenMP init. Cheap even for GPU runs (torch still spins up an
// intra-op pool at core count on first op). Call once before the first torch op. Idempotent-safe.
inline void init_torch_threads() {
    const int n = ::fenix::cpu_budget();
    torch::set_num_threads(n);
    torch::set_num_interop_threads(n);
    // cuDNN benchmark autotune: off by default (measured net-negative once, but that was on a CPU-clogged
    // box). Enable with FENIX_CUDNN_BENCHMARK=1 to A/B test — autotunes the fastest conv algo for the fixed
    // patch shape (all patches are P³), which can help a big same-shape 3D-conv workload.
    if (const char* e = std::getenv("FENIX_CUDNN_BENCHMARK"); e && std::atoi(e) != 0)
        torch::globalContext().setBenchmarkCuDNN(true);
    // TF32 on the tensor cores for fp32 matmul/conv paths (harmless for fp16; tolerance-only project).
    torch::globalContext().setAllowTF32CuBLAS(true);
    torch::globalContext().setAllowTF32CuDNN(true);
}

// True if a CUDA device is usable (driver + runtime + a visible GPU).
inline bool cuda_available() { return torch::cuda::is_available(); }

// The device inference should run on: FENIX_GPU=<idx> selects a CUDA device (multi-GPU
// inference = one fenix process per GPU over disjoint inputs — embarrassingly parallel,
// no cross-device coupling); default first GPU; CPU when none.
inline torch::Device best_device() {
    if (torch::cuda::is_available()) {
        int idx = 0;
        if (const char* e = std::getenv("FENIX_GPU")) {
            idx = std::atoi(e);
            const int n = static_cast<int>(torch::cuda::device_count());
            if (idx < 0 || idx >= n) idx = 0;
        }
        return torch::Device(torch::kCUDA, static_cast<torch::DeviceIndex>(idx));
    }
    return torch::Device(torch::kCPU);
}

// One-line human summary of the torch/compute environment.
inline std::string device_summary() {
    if (torch::cuda::is_available())
        return "CUDA (" + std::to_string(torch::cuda::device_count()) +
               " device(s)), torch " TORCH_VERSION;
    return std::string("CPU, torch " TORCH_VERSION);
}

}  // namespace fenix::ml
