// core/error.hpp — the error model + version. Foundational; included by core.hpp and by
// leaf headers that need Expected without the full core aggregate. Build -fno-exceptions
// -fno-rtti: fallible ops return Expected<T, Error>, never throw.
#pragma once

#include <expected>
#include <source_location>
#include <string>
#include <string_view>

namespace fenix {

// Date + git-hash identification (no SemVer). Injected by the build; fallback below.
#ifndef FENIX_VERSION
#define FENIX_VERSION "0-unknown"
#endif
inline constexpr std::string_view version = FENIX_VERSION;

enum class Errc {
    ok = 0,
    invalid_argument,
    not_found,
    io_error,
    fetch_failed,  // distinct from not_found/absent — never silently treat as air
    decode_error,
    unsupported,
    unimplemented,
    internal,
};

struct Error {
    Errc code = Errc::internal;
    std::string message;
    std::source_location where = std::source_location::current();
};

template <class T>
using Expected = std::expected<T, Error>;

[[nodiscard]] inline std::unexpected<Error> err(
    Errc code, std::string message, std::source_location where = std::source_location::current()) {
    return std::unexpected(Error{code, std::move(message), where});
}

}  // namespace fenix
