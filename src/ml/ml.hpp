// ml/ml.hpp — ml stage REGISTRATION. TORCH-FREE (the build firewall): it pulls only the torch-free
// declarations from ml/ml_api.hpp and registers the CLI stages; the libtorch implementations live in
// ml/inference.cpp (the one TU that parses <torch/torch.h>). This is what fenix.hpp includes, so the
// driver / unity TU never touches torch. Firewalled behind FENIX_ML (stubs when off). See ml/CLAUDE.md
// + ADR 0008 (build firewall).
#pragma once

#include "ml/augment_cli.hpp"  // torch-free `augment` stage (always built)
#include "ml/feed.hpp"         // torch-free `train-feed` data plane (always built)
#include "ml/ingest_band.hpp"  // torch-free `band-blocks` + `ingest-band` (teacher-sweep data plane)
#include "ml/label_audit.hpp"  // torch-free `label-audit` model-vs-label disagreement mining
#include "ml/qc_chunk.hpp"     // torch-free `qc-chunk` 3D-triage chunk export
#include "ml/stroke_score.hpp"  // torch-free `stroke-score` human-drawn-stroke label referee
#include "ml/surf_consist.hpp"  // torch-free `surf-consist` inter-mesh consistency QC
#include "ml/surf_qc.hpp"      // torch-free `surf-qc` mesh<->volume alignment QC
#include "ml/surf_repair.hpp"  // torch-free `surf-repair` snap-to-ridge offset-field repair
#include "ml/surf_sheet.hpp"   // torch-free `surf-sheet` contact-sheet visual QC
#include "ml/ml_api.hpp"
#include "ml/surfaces_cli.hpp"  // torch-free `surfaces` spatial query + GT rasterize (always built)

FENIX_REGISTER_STAGE(ml, "ml stage (libtorch inference)", ::fenix::ml::run)

namespace {
// Hyphenated subcommand names (the macro only makes identifier-named stages).
[[maybe_unused]] const int fenix_stage_predict_surface = ::fenix::register_stage(::fenix::Stage{
    "predict-surface",
    "surface-prediction inference (3D ResEnc-UNet)",
    [](std::span<const std::string_view> a, ::fenix::Context&) { return ::fenix::ml::run_predict_surface(a); }});
[[maybe_unused]] const int fenix_stage_predict_ink = ::fenix::register_stage(::fenix::Stage{
    "predict-ink",
    "ink-detection inference (3D ResEnc-UNet, DINO-guided)",
    [](std::span<const std::string_view> a, ::fenix::Context&) { return ::fenix::ml::run_predict_ink(a); }});
[[maybe_unused]] const int fenix_stage_predict_ink2d = ::fenix::register_stage(::fenix::Stage{
    "predict-ink2d",
    "2D-ink inference (r152 + 3D-FPN, ink_canonical_2um) over a rendered layer stack",
    [](std::span<const std::string_view> a, ::fenix::Context&) { return ::fenix::ml::run_predict_ink2d(a); }});
}  // namespace
