// geom/geom.hpp — STUB. See geom/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "geom/connected_components.hpp"
#include "geom/edt.hpp"
#include "geom/kdtree.hpp"
#include "geom/marching.hpp"
#include "geom/mesh.hpp"
#include "geom/morphology.hpp"

namespace fenix::geom {

// Stage entry point (stub). Real implementation per geom/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("geom");
}

}  // namespace fenix::geom

FENIX_REGISTER_STAGE(geom, "geom stage (stub)", ::fenix::geom::run)
