// fuzz_read_obj.cpp — libFuzzer harness for geom/mesh.hpp's read_obj, the OBJ mesh import
// path (marching-cubes/flatten/interop). Same per-call-temp-file pattern as
// fuzz_fxvol_open.cpp since read_obj takes a path, not a byte span.
//
// Originally written against a version of read_obj that called std::stoi on raw face-line
// tokens (would abort under -fno-exceptions on malformed OBJ input — see the review at
// docs/review/2026-07-02/apps-tools-tests.md). That has since been fixed in-tree to use
// std::from_chars (no exceptions, rejects trailing garbage); a 60s local run found no
// crash. Keeping this harness for regression coverage of that parser.
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "core/core.hpp"
#include "geom/mesh.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static const std::filesystem::path dir = std::filesystem::temp_directory_path() / "fenix_fuzz_obj";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::filesystem::path path = dir / ("in_" + std::to_string(::getpid()) + ".obj");

    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f)
            return 0;
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }

    {
        (void)fenix::geom::read_obj(path.string());
    }

    std::filesystem::remove(path, ec);
    return 0;
}
