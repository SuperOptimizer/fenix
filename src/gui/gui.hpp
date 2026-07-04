// gui/gui.hpp — module umbrella + the `view` stage: the interactive 4-pane slice/surface
// viewer + constraint annotator (Qt6; header-only, no moc). Only ever included under
// FENIX_GUI (fenix.hpp firewalls it) — Qt never leaks into a non-GUI TU. See gui/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include "gui/chunk3d.hpp"  // native VTK `view-chunk` 3D chunk-triage viewer
#include "gui/surf3d.hpp"   // native VTK `view-surf` real-time PBR segment renderer
#include "gui/vol3d.hpp"    // native VTK `view-vol` interactive volumetric renderer
#include "gui/viewer.hpp"

namespace fenix::gui {

inline Expected<int> run(std::span<const std::string_view> args, Context& ctx) {
    return run_viewer(args, ctx);
}

}  // namespace fenix::gui

FENIX_REGISTER_STAGE(view, "interactive 4-pane slice/surface viewer + constraint annotator (Qt6)",
                     ::fenix::gui::run)
