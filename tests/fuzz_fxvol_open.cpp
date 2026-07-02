// fuzz_fxvol_open.cpp — libFuzzer harness for codec/archive.hpp's VolumeArchive::open, the
// .fxvol container parser (superblock + radix page-table). open() takes a path, not a byte
// span, so the harness writes each fuzz input to a per-call temp file and lets open() mmap +
// parse it — this exercises the same untrusted-bytes path a corrupted/truncated/adversarial
// .fxvol on disk or fetched from S3 would hit. Slow (per-call file IO) but this is the only
// entry point VolumeArchive exposes; a libFuzzer -runs=N smoke run is still useful for the
// cheap structural checks (magic/version/crc/bounds) even without native in-memory fuzzing.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "codec/archive.hpp"
#include "core/core.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static const std::filesystem::path dir = std::filesystem::temp_directory_path() / "fenix_fuzz_fxvol";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::filesystem::path path = dir / ("in_" + std::to_string(::getpid()) + ".fxvol");

    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f)
            return 0;
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }

    {
        (void)fenix::codec::VolumeArchive::open(path.string(), /*writable=*/false);
    }

    std::filesystem::remove(path, ec);
    return 0;
}
