// annotate/annotate.hpp — STUB. See annotate/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "annotate/umbilicus.hpp"

namespace fenix::annotate {

// Stage entry point (stub). Real implementation per annotate/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("annotate");
}

}  // namespace fenix::annotate

FENIX_REGISTER_STAGE(annotate, "annotate stage (stub)", ::fenix::annotate::run)
