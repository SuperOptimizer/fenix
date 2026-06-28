// ml/torch_env.hpp — the ONE place libtorch is included. Everything torch-typed lives behind
// the FENIX_ML firewall (see ml/CLAUDE.md): this header is only pulled in when FENIX_ML is
// defined, and torch types must not leak past the ml/ module's API. Headers are consumed as
// -isystem (deps.cmake) so the project's -Weverything doesn't fire on them.
#pragma once

#include <torch/torch.h>

#include <string>

namespace fenix::ml {

// True if a CUDA device is usable (driver + runtime + a visible GPU).
inline bool cuda_available() { return torch::cuda::is_available(); }

// The device inference should run on: first CUDA GPU if available, else CPU.
inline torch::Device best_device() {
    if (torch::cuda::is_available()) return torch::Device(torch::kCUDA, 0);
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
