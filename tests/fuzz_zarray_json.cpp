// fuzz_zarray_json.cpp — libFuzzer harness for io/zarr.hpp's hand-rolled .zarray JSON
// parsers (detail::json_int_array / detail::json_string), reachable from any zarr root's
// .zarray metadata file (read_zarray, io/zarr.hpp:119) — a scroll-data ingest path that
// reads bytes fenix does not control (local mirrors or S3 objects).
#include <cstdint>
#include <string>

#include "core/core.hpp"
#include "io/zarr.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    const std::string s(reinterpret_cast<const char*>(data), size);
    (void)fenix::io::detail::json_int_array(s, "shape");
    (void)fenix::io::detail::json_int_array(s, "chunks");
    (void)fenix::io::detail::json_string(s, "dtype");
    (void)fenix::io::detail::json_string(s, "compressor");
    (void)fenix::io::detail::json_string(s, "dimension_separator");
    return 0;
}
