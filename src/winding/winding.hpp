// winding/winding.hpp — STUB. See winding/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "winding/fit.hpp"
#include "winding/flow.hpp"
#include "winding/relax.hpp"
#include "winding/transforms.hpp"
#include "winding/spiral_fit.hpp"
#include "winding/spiral_model.hpp"
#include "winding/winding_field.hpp"

namespace fenix::winding {

// Stage entry point (stub). Real implementation per winding/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("winding");
}

}  // namespace fenix::winding

FENIX_REGISTER_STAGE(winding, "winding stage (stub)", ::fenix::winding::run)
