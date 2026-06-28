// preprocess/preprocess.hpp — STUB. See preprocess/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "preprocess/aircut.hpp"
#include "preprocess/deconv.hpp"
#include "preprocess/fft.hpp"
#include "preprocess/guided.hpp"
#include "preprocess/phasecorr.hpp"

namespace fenix::preprocess {

// Stage entry point (stub). Real implementation per preprocess/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("preprocess");
}

}  // namespace fenix::preprocess

FENIX_REGISTER_STAGE(preprocess, "preprocess stage (stub)", ::fenix::preprocess::run)
