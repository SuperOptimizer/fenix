// segment/segment.hpp — STUB. See segment/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "segment/hessian.hpp"
#include "segment/structure_tensor.hpp"
#include "segment/tracer.hpp"
#include "segment/trace_surface.hpp"

namespace fenix::segment {

// Stage entry point (stub). Real implementation per segment/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("segment");
}

}  // namespace fenix::segment

FENIX_REGISTER_STAGE(segment, "segment stage (stub)", ::fenix::segment::run)
