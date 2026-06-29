// core/core.hpp — fenix substrate (STUB).
// The real substrate (Volume<T>, Vec/units, arenas, parallel_for, sampling kernels,
// reflection serializer, std::simd helpers) lands per src/core/CLAUDE.md. This stub
// provides just enough (Error/Expected, Logger, Context, Stage registry) for the
// single-TU skeleton to build and dispatch. Header-only; #pragma once; self-contained.
#pragma once

#include "core/arena.hpp"
#include "core/config.hpp"
#include "core/eig.hpp"
#include "core/error.hpp"
#include "core/filter.hpp"
#include "core/hash.hpp"
#include "core/log.hpp"
#include "core/optimize.hpp"
#include "core/parallel.hpp"
#include "core/rng.hpp"
#include "core/sampling.hpp"
#include "core/surface.hpp"
#include "core/tolerances.hpp"
#include "core/types.hpp"
#include "core/vec.hpp"
#include "core/volume.hpp"

#include <expected>
#include <functional>
#include <print>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix {

// version + error model (Errc/Error/Expected/err) are in core/error.hpp.

// ---- assertions ------------------------------------------------------------
// Active in debug/sanitizer; compiled out under -Ofast release (NDEBUG).
#ifdef NDEBUG
#define FENIX_ASSERT(cond, ...) ((void)0)
#else
#define FENIX_ASSERT(cond, ...)                                                          \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::println(stderr, "FENIX_ASSERT failed: {} ({}:{})", #cond, __FILE__,     \
                         __LINE__);                                                       \
            __builtin_trap();                                                            \
        }                                                                                \
    } while (0)
#endif

// ---- logging ---------------------------------------------------------------
// LogLevel / log_level() / log() + the FENIX_* macros now live in core/log.hpp (included above).

// ---- context ---------------------------------------------------------------
// Threaded through every stage run(): io handles, resolved config, logger, pool,
// arenas, progress/cancel. STUB: just the knobs the skeleton needs.
struct Context {
    int threads = 0;          // 0 => hardware_concurrency
    bool cancelled = false;
    std::string project_dir;  // .fxproj root, if any
};

// ---- stage registry --------------------------------------------------------
// A stage = a typed pipeline node + a CLI subcommand wrapper. Stages self-register
// via static init so apps/driver.cpp never enumerates them.
struct Stage {
    std::string_view name;
    std::string_view help;
    std::function<Expected<int>(std::span<const std::string_view> args, Context&)> run;
};

inline std::vector<Stage>& registry() {
    static std::vector<Stage> stages;
    return stages;
}

inline int register_stage(Stage s) {
    registry().push_back(std::move(s));
    return 0;  // value lets us use it in a static initializer
}

inline const Stage* find_stage(std::string_view name) {
    for (const auto& s : registry())
        if (s.name == name) return &s;
    return nullptr;
}

// Convenience for module stubs: register a not-yet-implemented stage.
inline Expected<int> stage_unimplemented(std::string_view name) {
    fenix::log(LogLevel::warn, "stage '{}' is not implemented yet", name);
    return fenix::err(Errc::unimplemented, std::string(name) + " not implemented");
}

}  // namespace fenix

// Register a stage from a module header. Use at namespace scope.
#define FENIX_REGISTER_STAGE(NAME, HELP, FN)                                             \
    namespace {                                                                          \
    [[maybe_unused]] const int fenix_stage_##NAME =                                      \
        ::fenix::register_stage(::fenix::Stage{#NAME, HELP, FN});                        \
    }
