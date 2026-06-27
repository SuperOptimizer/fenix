// topo/topo.hpp — STUB. See topo/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "topo/betti.hpp"

namespace fenix::topo {

// Stage entry point (stub). Real implementation per topo/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("topo");
}

}  // namespace fenix::topo

FENIX_REGISTER_STAGE(topo, "topo stage (stub)", ::fenix::topo::run)
