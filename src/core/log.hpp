// core/log.hpp — the fenix logging system. Level-gated, THREAD-SAFE (a global mutex serializes whole
// records so parallel_for output never interleaves), and FLUSHED per record to stderr (block-buffered
// stdout was silently clipping logs on timeouts/crashes). Records carry a module tag and, at debug/trace,
// the source location. The FENIX_* macros gate at COMPILE time (trace/debug vanish in release) and at
// RUN time (std::format runs only when the level passes), so logging liberally is cheap. A pluggable sink
// lets tests capture records (and the driver redirect to a file). See src/core/CLAUDE.md.
#pragma once

#include "core/types.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <functional>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

namespace fenix {

// trace < debug < info < warn < error < off. `quiet` is kept as an alias of `off` for back-compat.
enum class LogLevel { trace, debug, info, warn, error, off, quiet = off };

namespace detail {
inline LogLevel parse_log_level(const char* s) {
    if (!s) return LogLevel::info;
    const std::string_view v(s);
    if (v == "trace") return LogLevel::trace;
    if (v == "debug") return LogLevel::debug;
    if (v == "info") return LogLevel::info;
    if (v == "warn") return LogLevel::warn;
    if (v == "error") return LogLevel::error;
    if (v == "off" || v == "quiet") return LogLevel::off;
    return LogLevel::info;
}
}  // namespace detail

// Process-wide minimum level to EMIT (default info, seeded once from $FENIX_LOG_LEVEL). Mutate via the
// reference (the driver maps -v/-vv/-vvv/--quiet onto it).
inline LogLevel& log_level() {
    static LogLevel level = detail::parse_log_level(std::getenv("FENIX_LOG_LEVEL"));
    return level;
}

inline std::mutex& log_mutex() {
    static std::mutex m;
    return m;
}

// A sink receives one fully-formatted, newline-terminated record. Default: write to stderr + flush.
// Swap it (under the lock at startup) to capture records in a test or tee to a file.
using LogSink = std::function<void(std::string_view)>;
inline LogSink& log_sink() {
    static LogSink sink = [](std::string_view rec) {
        std::fwrite(rec.data(), 1, rec.size(), stderr);
        std::fflush(stderr);
    };
    return sink;
}

inline char log_tag(LogLevel l) {
    switch (l) {
        case LogLevel::trace: return 'T';
        case LogLevel::debug: return 'D';
        case LogLevel::info: return 'I';
        case LogLevel::warn: return 'W';
        case LogLevel::error: return 'E';
        default: return '?';
    }
}

// Format one record ("[I][module] msg" + " (file:line)" at debug/trace), then lock + emit through the
// sink. Re-checks the level so the back-compat `log()` and direct callers are also gated.
inline void log_emit(LogLevel lvl, std::string_view mod, const std::source_location& loc, std::string_view msg) {
    if (static_cast<int>(lvl) < static_cast<int>(log_level())) return;
    std::string rec;
    rec.reserve(msg.size() + 48);
    rec.push_back('[');
    rec.push_back(log_tag(lvl));
    rec.push_back(']');
    if (!mod.empty()) {
        rec.push_back('[');
        rec.append(mod);
        rec.push_back(']');
    }
    rec.push_back(' ');
    rec.append(msg);
    if (static_cast<int>(lvl) <= static_cast<int>(LogLevel::debug)) {  // source location at trace/debug
        const char* f = loc.file_name();
        const char* base = std::strrchr(f, '/');
        rec.append("  (");
        rec.append(base ? base + 1 : f);
        rec.push_back(':');
        rec.append(std::to_string(loc.line()));
        rec.push_back(')');
    }
    rec.push_back('\n');
    const std::scoped_lock lk(log_mutex());
    log_sink()(rec);
}

// Back-compat free function (module-less). Keeps existing ml/render/driver call sites working.
template <class... Args>
inline void log(LogLevel lvl, std::format_string<Args...> fmt, Args&&... args) {
    if (static_cast<int>(lvl) < static_cast<int>(log_level())) return;
    log_emit(lvl, {}, std::source_location::current(), std::format(fmt, std::forward<Args>(args)...));
}

// Compile-time floor: levels below this are stripped at compile time (zero cost). Release keeps debug+;
// debug/sanitizer builds keep everything. Override with -DFENIX_LOG_COMPILE_MIN=... if desired.
#ifndef FENIX_LOG_COMPILE_MIN
#ifdef NDEBUG
#define FENIX_LOG_COMPILE_MIN ::fenix::LogLevel::debug
#else
#define FENIX_LOG_COMPILE_MIN ::fenix::LogLevel::trace
#endif
#endif

#define FENIX_LOG(lvl, mod, ...)                                                                       \
    do {                                                                                               \
        if constexpr (static_cast<int>(lvl) >= static_cast<int>(FENIX_LOG_COMPILE_MIN)) {              \
            if (static_cast<int>(lvl) >= static_cast<int>(::fenix::log_level()))                       \
                ::fenix::log_emit((lvl), (mod), std::source_location::current(), std::format(__VA_ARGS__)); \
        }                                                                                              \
    } while (0)

#define FENIX_TRACE(mod, ...) FENIX_LOG(::fenix::LogLevel::trace, mod, __VA_ARGS__)
#define FENIX_DEBUG(mod, ...) FENIX_LOG(::fenix::LogLevel::debug, mod, __VA_ARGS__)
#define FENIX_INFO(mod, ...) FENIX_LOG(::fenix::LogLevel::info, mod, __VA_ARGS__)
#define FENIX_WARN(mod, ...) FENIX_LOG(::fenix::LogLevel::warn, mod, __VA_ARGS__)
#define FENIX_ERROR(mod, ...) FENIX_LOG(::fenix::LogLevel::error, mod, __VA_ARGS__)

// RAII scoped timer: logs "label: X.XXX s" at debug on scope exit (steady_clock).
class ScopeTimer {
public:
    ScopeTimer(std::string_view mod, std::string label, std::source_location loc = std::source_location::current())
        : mod_(mod), label_(std::move(label)), loc_(loc), t0_(std::chrono::steady_clock::now()) {}
    ~ScopeTimer() {
        if (static_cast<int>(LogLevel::debug) < static_cast<int>(log_level())) return;
        const f64 s = std::chrono::duration<f64>(std::chrono::steady_clock::now() - t0_).count();
        log_emit(LogLevel::debug, mod_, loc_, std::format("{}: {:.3f} s", label_, s));
    }
    ScopeTimer(const ScopeTimer&) = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;

private:
    std::string_view mod_;
    std::string label_;
    std::source_location loc_;
    std::chrono::steady_clock::time_point t0_;
};

#define FENIX_CONCAT_(a, b) a##b
#define FENIX_CONCAT(a, b) FENIX_CONCAT_(a, b)
#define FENIX_SCOPE_TIMER(mod, label) const ::fenix::ScopeTimer FENIX_CONCAT(fenix_timer_, __LINE__)((mod), (label))

}  // namespace fenix
