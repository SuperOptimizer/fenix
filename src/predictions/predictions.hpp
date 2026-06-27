// predictions/predictions.hpp — STUB. See predictions/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "predictions/field.hpp"

namespace fenix::predictions {

// Stage entry point (stub). Real implementation per predictions/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("predictions");
}

}  // namespace fenix::predictions

FENIX_REGISTER_STAGE(predictions, "predictions stage (stub)", ::fenix::predictions::run)
