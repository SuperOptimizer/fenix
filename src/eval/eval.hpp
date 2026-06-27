// eval/eval.hpp — STUB. See eval/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "eval/deformation.hpp"
#include "eval/metrics.hpp"
#include "eval/nsd.hpp"

namespace fenix::eval {

// Stage entry point (stub). Real implementation per eval/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("eval");
}

}  // namespace fenix::eval

FENIX_REGISTER_STAGE(eval, "eval stage (stub)", ::fenix::eval::run)
