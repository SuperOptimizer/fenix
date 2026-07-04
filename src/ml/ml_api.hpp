// ml/ml_api.hpp — the TORCH-FREE public surface of the ml module. The stage registry (ml/ml.hpp) and thus
// the driver / every non-ml TU see ONLY this header; the libtorch implementations live in ml/inference.cpp,
// the single TU that includes <torch/torch.h>. This is the build firewall: nothing else parses libtorch, so
// the driver TU drops from minutes to seconds. Firewalled behind FENIX_ML (torch-free stubs when off).
#pragma once

#include "core/core.hpp"

#include <span>
#include <string_view>

namespace fenix::ml {

#ifdef FENIX_ML
// Defined in ml/inference.cpp (the libtorch TU). Signatures are deliberately torch-free.
Expected<int> run_predict_surface(std::span<const std::string_view> args);
Expected<int> run_predict_ink(std::span<const std::string_view> args);
Expected<int> run_predict_ink2d(std::span<const std::string_view> args);
Expected<int> run(std::span<const std::string_view> args, Context& ctx);
#else
// Core build: no torch. The stages exist but report unimplemented.
inline Expected<int> run_predict_surface(std::span<const std::string_view>) { return stage_unimplemented("predict-surface"); }
inline Expected<int> run_predict_ink(std::span<const std::string_view>) { return stage_unimplemented("predict-ink"); }
inline Expected<int> run_predict_ink2d(std::span<const std::string_view>) { return stage_unimplemented("predict-ink2d"); }
inline Expected<int> run(std::span<const std::string_view>, Context&) { return stage_unimplemented("ml"); }
#endif

}  // namespace fenix::ml
