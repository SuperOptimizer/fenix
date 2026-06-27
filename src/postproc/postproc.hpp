// postproc/postproc.hpp — STUB. See postproc/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "postproc/cleanup.hpp"

namespace fenix::postproc {

// Stage entry point (stub). Real implementation per postproc/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("postproc");
}

}  // namespace fenix::postproc

FENIX_REGISTER_STAGE(postproc, "postproc stage (stub)", ::fenix::postproc::run)
