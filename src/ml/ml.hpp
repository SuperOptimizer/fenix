// ml/ml.hpp — STUB. See ml/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

namespace fenix::ml {

// Stage entry point (stub). Real implementation per ml/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("ml");
}

}  // namespace fenix::ml

FENIX_REGISTER_STAGE(ml, "ml stage (stub)", ::fenix::ml::run)
