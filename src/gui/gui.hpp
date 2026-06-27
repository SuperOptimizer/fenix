// gui/gui.hpp — STUB. See gui/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

namespace fenix::gui {

// Stage entry point (stub). Real implementation per gui/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("gui");
}

}  // namespace fenix::gui

FENIX_REGISTER_STAGE(gui, "gui stage (stub)", ::fenix::gui::run)
