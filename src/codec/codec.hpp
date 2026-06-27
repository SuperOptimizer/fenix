// codec/codec.hpp — STUB. See codec/CLAUDE.md for scope. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "codec/archive.hpp"
#include "codec/block.hpp"
#include "codec/lossless.hpp"
#include "codec/rans.hpp"
#include "codec/wavelet.hpp"

namespace fenix::codec {

// Stage entry point (stub). Real implementation per codec/CLAUDE.md.
inline Expected<int> run(std::span<const std::string_view> /*args*/, Context& /*ctx*/) {
    return stage_unimplemented("codec");
}

}  // namespace fenix::codec

FENIX_REGISTER_STAGE(codec, "codec stage (stub)", ::fenix::codec::run)
