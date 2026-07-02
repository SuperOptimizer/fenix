// ml/ml.hpp — ml stage REGISTRATION. TORCH-FREE (the build firewall): it pulls only the torch-free
// declarations from ml/ml_api.hpp and registers the CLI stages; the libtorch implementations live in
// ml/inference.cpp (the one TU that parses <torch/torch.h>). This is what fenix.hpp includes, so the
// driver / unity TU never touches torch. Firewalled behind FENIX_ML (stubs when off). See ml/CLAUDE.md
// + ADR 0008 (build firewall).
#pragma once

#include "ml/augment_cli.hpp"  // torch-free `augment` stage (always built)
#include "ml/feed.hpp"         // torch-free `train-feed` data plane (always built)
#include "ml/ml_api.hpp"

FENIX_REGISTER_STAGE(ml, "ml stage (libtorch inference)", ::fenix::ml::run)

namespace {
// Hyphenated subcommand names (the macro only makes identifier-named stages).
[[maybe_unused]] const int fenix_stage_predict_surface = ::fenix::register_stage(
    ::fenix::Stage{"predict-surface", "surface-prediction inference (3D ResEnc-UNet)",
                   [](std::span<const std::string_view> a, ::fenix::Context&) {
                       return ::fenix::ml::run_predict_surface(a);
                   }});
[[maybe_unused]] const int fenix_stage_predict_ink = ::fenix::register_stage(
    ::fenix::Stage{"predict-ink", "ink-detection inference (3D ResEnc-UNet, DINO-guided)",
                   [](std::span<const std::string_view> a, ::fenix::Context&) {
                       return ::fenix::ml::run_predict_ink(a);
                   }});
}  // namespace
