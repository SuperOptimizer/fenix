// flatten/flatten.hpp — STUB. See flatten/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "flatten/extract_wrap.hpp"

namespace fenix::flatten {

// Stage entry point (stub). Real implementation per flatten/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("flatten");
}

}  // namespace fenix::flatten

FENIX_REGISTER_STAGE(flatten, "flatten stage (stub)", ::fenix::flatten::run)
