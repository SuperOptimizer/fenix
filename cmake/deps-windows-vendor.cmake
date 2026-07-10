# deps-windows-vendor.cmake — vendor mimalloc + blosc2 via FetchContent on native Windows (no
# system packages, no /bin/sh source scripts). Exposes the same canonical fenix::mimalloc and
# fenix::blosc2 targets the Linux deps.cmake would, so CMakeLists.txt consumes them identically
# (and defines FENIX_HAVE_BLOSC2). Built static, with clang + MSVC STL.
include(FetchContent)

# ---- mimalloc (process allocator) ----
set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
set(MI_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(MI_OVERRIDE     OFF CACHE BOOL "" FORCE)  # link the lib; skip the global malloc redirect on Win
FetchContent_Declare(mimalloc
  GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
  GIT_TAG        v2.1.7
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(mimalloc)
if(TARGET mimalloc-static AND NOT TARGET fenix::mimalloc)
  add_library(fenix::mimalloc INTERFACE IMPORTED)
  target_link_libraries(fenix::mimalloc INTERFACE mimalloc-static)
endif()

# ---- c-blosc2 (chunk compression) ----
set(BUILD_SHARED     OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC     ON  CACHE BOOL "" FORCE)
set(BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS    OFF CACHE BOOL "" FORCE)
set(BUILD_PLUGINS    OFF CACHE BOOL "" FORCE)
# blosc2 bundles zlib-ng in internal-complibs; its zlib/zlibstatic targets collide with our
# vendored madler zlib (deps-windows-zlib.cmake). fenix uses blosc2's zstd/blosclz codecs, not
# its zlib codec, so deactivate blosc2's zlib to avoid building/clashing that second zlib.
set(DEACTIVATE_ZLIB  ON  CACHE BOOL "" FORCE)
FetchContent_Declare(blosc2
  GIT_REPOSITORY https://github.com/Blosc/c-blosc2.git
  GIT_TAG        v2.15.1
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(blosc2)
if(TARGET blosc2_static)
  # blosc2 is third-party C: silence its warnings, and demote clang-22's default-error
  # implicit-function-declaration (POSIX names like rmdir/unlink — they link via UCRT oldnames).
  target_compile_options(blosc2_static PRIVATE
    -w -Wno-error=implicit-function-declaration -Wno-implicit-function-declaration -Wno-implicit-int)
  if(NOT TARGET fenix::blosc2)
    add_library(fenix::blosc2 INTERFACE IMPORTED)
    target_link_libraries(fenix::blosc2 INTERFACE blosc2_static)
    target_include_directories(fenix::blosc2 INTERFACE ${blosc2_SOURCE_DIR}/include)
  endif()
endif()
