# fenix — top-level build. Header-only library + ONE translation unit (apps/driver.cpp).
# Clang-only (see cmake/clang-toolchain.cmake + CMakePresets.json). C++26.
cmake_minimum_required(VERSION 3.28)
project(fenix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# We do NOT use C++20 modules (ADR 0001: header-only, classic #include). Disable the
# compiler module-dependency scan: it's pure overhead here and its ninja dyndep path is
# flaky with file(GLOB) test discovery.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- options ---------------------------------------------------------------
option(FENIX_GUI       "Build the Qt6+VTK viewer (firewalled)"      OFF)
option(FENIX_ML        "Build the libtorch ML module (firewalled)"  OFF)
option(FENIX_FS        "Build the fxfs FUSE filesystem (libfuse3, firewalled)" OFF)
option(FENIX_FAST      "Aggressive -Ofast/-ffast-math/-march=native (default ON for Release)" ON)
set(FENIX_SANITIZE     "" CACHE STRING "Sanitizer: '' | address | undefined | address;undefined | address,fuzzer | thread | memory")
option(FENIX_COVERAGE  "Instrument for llvm-cov coverage"           OFF)
option(FENIX_WERROR    "Treat warnings as errors (CI)"              OFF)
option(FENIX_BUILD_TESTS "Build the test suite"                     ON)
option(FENIX_SPLIT     "Dev build: compile each module as its own TU (parallel + incremental) instead of ONE unity TU" OFF)
option(FENIX_SPLIT_PCH "Use a core.hpp precompiled header in the split build (faster per-TU)" ON)
option(FENIX_CCACHE    "Use ccache/sccache as the compiler launcher when available (huge for repeated/docker builds)" ON)
# The 66 per-file test TUs — not the module TUs — dominate a cold parallel build (each re-parses the heavy
# headers and re-optimizes at the release -O3). Tests are run-once, tolerance-based correctness checks, so the
# dev build compiles them at a lighter -O (still optimized, just far cheaper codegen). Empty = inherit the
# release -O3 (canonical for CI, where test numerics/perf should match the shipped binary). The split/ml dev
# presets set -O1. See ADR 0008.
set(FENIX_TEST_OPT     "" CACHE STRING "Optimization level for the test TUs (e.g. -O1); empty inherits release -O3")

# ---- ccache: cache object files across builds ------------------------------
# On docker/runpod a from-scratch rebuild re-pays every TU (the libtorch ml TU alone is minutes). With
# ccache + a PERSISTED cache dir (mount CCACHE_DIR as a volume), unchanged TUs become instant cache hits —
# the single biggest lever for a rebuild loop. Works for unity AND split. Set FENIX_CCACHE=OFF to disable.
# ccache and PCH COEXIST (they don't have to be either/or): with `pch_defines` sloppiness ccache caches
# PCH-using compiles too, so the split build keeps BOTH — the cold/first build gets PCH-fast parsing AND
# warm rebuilds get cache hits (verified: warm rebuild of a PCH build is still an instant ccache hit). We
# inject the sloppiness through the launcher (`cmake -E env …`) so it holds regardless of the caller's env.
# sccache doesn't cache PCH compiles, so with sccache we keep them mutually exclusive (PCH wins, below).
set(FENIX_CCACHE_IS_CCACHE OFF)
if(FENIX_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(FENIX_CCACHE_PROG NAMES ccache sccache)
  if(FENIX_CCACHE_PROG)
    if(FENIX_CCACHE_PROG MATCHES "ccache")
      set(FENIX_CCACHE_IS_CCACHE ON)
      set(CMAKE_CXX_COMPILER_LAUNCHER
          "${CMAKE_COMMAND};-E;env;CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime;${FENIX_CCACHE_PROG}"
          CACHE STRING "compiler launcher" FORCE)
    else()
      set(CMAKE_CXX_COMPILER_LAUNCHER "${FENIX_CCACHE_PROG}" CACHE STRING "compiler launcher" FORCE)
    endif()
    message(STATUS "fenix: ccache launcher ${FENIX_CCACHE_PROG} (pch-coexist=${FENIX_CCACHE_IS_CCACHE})")
  endif()
endif()
# PCH is OK unless a non-ccache launcher (sccache) is active — real ccache caches PCH compiles (sloppiness set
# above), so the two coexist; only sccache forces the old either/or (PCH off, object cache wins).
set(FENIX_PCH_OK OFF)
if(FENIX_SPLIT_PCH AND (NOT FENIX_CCACHE_PROG OR FENIX_CCACHE_IS_CCACHE))
  set(FENIX_PCH_OK ON)
endif()

# ---- version (date + git hash; no SemVer) ----------------------------------
find_program(GIT_EXECUTABLE git)
set(FENIX_VERSION "0-unknown")
if(GIT_EXECUTABLE)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE _git OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  string(TIMESTAMP _date "%Y%m%d")
  if(_git)
    set(FENIX_VERSION "${_date}-g${_git}")
  endif()
endif()

# ---- the header-only "library" (interface target) --------------------------
add_library(fenix_headers INTERFACE)
target_include_directories(fenix_headers INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(fenix_headers INTERFACE cxx_std_26)
target_compile_definitions(fenix_headers INTERFACE FENIX_VERSION="${FENIX_VERSION}")
# No exceptions, no RTTI (errors via std::expected). EXCEPTION: the ml/ module includes
# libtorch headers, which require both — so an ML build (FENIX_ML) keeps exceptions + RTTI on.
if(NOT FENIX_ML)
  target_compile_options(fenix_headers INTERFACE -fno-exceptions -fno-rtti)
endif()

# Warnings: -Weverything minus a denylist (see also .clang-tidy). -Werror in CI.
target_compile_options(fenix_headers INTERFACE
  -Weverything
  -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-pre-c++26-compat
  -Wno-padded -Wno-poison-system-directories -Wno-unsafe-buffer-usage
  -Wno-switch-default -Wno-exit-time-destructors -Wno-global-constructors
  -Wno-ctad-maybe-unsupported
  # Exact float compares are intentional in our domain (air/ZERO/fill detection, sentinels)
  # and fast-math is on globally anyway. Equality is used deliberately, not by accident.
  -Wno-float-equal
  # Our assert/CHECK/log macros expand std::println etc.; the recursive-expansion guard is
  # pure noise here (clang-22 -Weverything).
  -Wno-disabled-macro-expansion)
if(FENIX_WERROR)
  target_compile_options(fenix_headers INTERFACE -Werror)
endif()

# Fast-by-default (Release). Determinism is not a goal anywhere (see ADR 0004).
if(FENIX_FAST)
  # -O3 -ffast-math == the old -Ofast (which clang 21 deprecates). Fast-by-default; no
  # determinism goal anywhere (ADR 0004).
  # Fast-math + native ISA everywhere; the heavy -O3/-funroll codegen is skipped for targets that opt into a
  # lighter opt (the test TUs, via the FENIX_LIGHT_OPT property + FENIX_TEST_OPT) so they keep identical
  # numerics but compile far cheaper. Genex reads the CONSUMING target's property, so `fenix` still gets -O3.
  target_compile_options(fenix_headers INTERFACE
    $<$<CONFIG:Release>:-ffast-math -march=native -fno-semantic-interposition
      $<$<NOT:$<BOOL:$<TARGET_PROPERTY:FENIX_LIGHT_OPT>>>:-O3 -funroll-loops>>)
endif()

# OpenMP for data-parallel loops (the parallelism primitive).
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fenix_headers INTERFACE OpenMP::OpenMP_CXX)
endif()

# Third-party deps: find an installed copy, else build from source (cmake/deps.cmake).
# Each resolved dep exposes a canonical fenix::<name> target + FENIX_HAVE_<NAME>.
# mimalloc's interposer fights the sanitizer runtimes, so force it off under -fsanitize.
if(FENIX_SANITIZE)
  set(FENIX_DEP_MIMALLOC off CACHE STRING "mimalloc disabled under sanitizers" FORCE)
endif()
include(${CMAKE_SOURCE_DIR}/cmake/deps.cmake)

# Allocator (process-wide) + compression are consumed by the header interface so every
# TU and test picks them up. The codec/io guard their blosc2 use behind FENIX_HAVE_BLOSC2.
if(TARGET fenix::mimalloc)
  target_link_libraries(fenix_headers INTERFACE fenix::mimalloc)
endif()
if(TARGET fenix::blosc2)
  target_link_libraries(fenix_headers INTERFACE fenix::blosc2)
  target_compile_definitions(fenix_headers INTERFACE FENIX_HAVE_BLOSC2=1)
endif()

# libcurl — the S3/HTTP client (io/s3.hpp). System package (approved core dep). Optional: the
# core still builds without it (io/s3.hpp degrades to an unsupported-error stub).
find_package(CURL)
if(CURL_FOUND)
  target_link_libraries(fenix_headers INTERFACE CURL::libcurl)
  target_compile_definitions(fenix_headers INTERFACE FENIX_HAVE_CURL=1)
  message(STATUS "fenix dep curl: ${CURL_VERSION_STRING}")
endif()

# Sanitizers / coverage (dedicated build dirs via presets). FENIX_SANITIZE is a CMake list
# (e.g. "address;undefined"); join with commas for the -fsanitize= flag.
if(FENIX_SANITIZE)
  string(REPLACE ";" "," _san "${FENIX_SANITIZE}")
  target_compile_options(fenix_headers INTERFACE -fsanitize=${_san} -fno-omit-frame-pointer)
  target_link_options(fenix_headers INTERFACE -fsanitize=${_san})
endif()
if(FENIX_COVERAGE)
  target_compile_options(fenix_headers INTERFACE -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(fenix_headers INTERFACE -fprofile-instr-generate -fcoverage-mapping)
endif()

# Approved core deps are discovered by io/codec when implemented (find_package system,
# FetchContent fallback). The stub skeleton needs none of them to build.

# ---- the fenix binary: UNITY (default) or SPLIT (dev) ----------------------
# UNITY: apps/driver.cpp includes the umbrella -> the whole program is ONE translation unit (fewest total
# CPU-seconds; best for clean/CI builds). SPLIT (-DFENIX_SPLIT=ON): each module in src/<mod>/<mod>.cpp is its own
# TU compiled in PARALLEL, plus a core.hpp precompiled header, plus driver.cpp built with -DFENIX_SPLIT
# so it includes only what main() touches. The module objects self-register their stages at startup. Same
# binary/behaviour; SPLIT trades more total work for parallelism + fast INCREMENTAL rebuilds. See ADR 0008.
if(FENIX_SPLIT)
  # one TU per module: src/<mod>/<mod>.cpp (each #includes its module umbrella). fxfs is its own binary.
  file(GLOB FENIX_UNIT_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*/*.cpp)
  list(FILTER FENIX_UNIT_SOURCES EXCLUDE REGEX "/fs/")
  add_executable(fenix apps/driver.cpp ${FENIX_UNIT_SOURCES})
  target_compile_definitions(fenix PRIVATE FENIX_SPLIT)
  # core.hpp PCH: enabled whenever requested, EXCEPT under a non-ccache launcher (sccache) which can't cache
  # PCH compiles — there the cross-build object cache wins and PCH is off. Real ccache coexists (see above).
  if(FENIX_PCH_OK)
    target_precompile_headers(fenix PRIVATE ${CMAKE_SOURCE_DIR}/src/core/core.hpp)
    message(STATUS "fenix split: core.hpp PCH ON")
  elseif(FENIX_CCACHE_PROG)
    message(STATUS "fenix split: PCH off (sccache active — cross-build object cache wins)")
  endif()
else()
  add_executable(fenix apps/driver.cpp)
endif()
target_link_libraries(fenix PRIVATE fenix_headers)
if(FENIX_GUI)
  target_compile_definitions(fenix PRIVATE FENIX_GUI)
  # apps/gui.cpp is THE Qt TU (the GUI firewall, mirroring ml/inference.cpp): Qt is parsed only there,
  # the driver TU stays Qt-free in BOTH unity and split builds, and the Qt parse compiles in parallel.
  target_sources(fenix PRIVATE ${CMAKE_SOURCE_DIR}/apps/gui.cpp)
  # Qt6/VTK resolved by cmake/deps.cmake (system or from-source) → canonical aliases.
  if(TARGET fenix::qt6)
    target_link_libraries(fenix PRIVATE fenix::qt6)
  endif()
  if(TARGET fenix::vtk)
    target_link_libraries(fenix PRIVATE fenix::vtk)
  endif()
endif()
if(FENIX_ML)
  target_compile_definitions(fenix PRIVATE FENIX_ML)
  # ml/inference.cpp is THE libtorch TU (the firewall — see ADR 0008): torch lives only here, so the driver
  # and every other TU stay torch-free. The SPLIT build already globs it via src/*/*.cpp; the UNITY build's
  # target is just driver.cpp, so add it explicitly there.
  if(NOT FENIX_SPLIT)
    target_sources(fenix PRIVATE ${CMAKE_SOURCE_DIR}/src/ml/inference.cpp)
  endif()
  if(TARGET fenix::torch)
    target_link_libraries(fenix PRIVATE fenix::torch)
  endif()
endif()

# ---- fxfs: FUSE filesystem (firewalled; links libfuse3) --------------------
if(FENIX_FS)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FUSE3 REQUIRED IMPORTED_TARGET fuse3)
  add_executable(fenix-fxfs src/fs/fxfs.cpp)
  target_link_libraries(fenix-fxfs PRIVATE fenix_headers PkgConfig::FUSE3)
  message(STATUS "fxfs: libfuse3 ${FUSE3_VERSION}")
endif()

# ---- tests (per-test-file binaries) ----------------------------------------
if(FENIX_BUILD_TESTS)
  enable_testing()
  file(GLOB FENIX_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cpp)
  # The ~66 test TUs each re-parse core.hpp (~0.9 s) — ~40 s of duplicated frontend across a cold parallel
  # build. Since tests vastly outnumber cores, ONE shared core.hpp PCH (built on the first test, reused by the
  # rest) amortizes its one build over all of them. Engages under FENIX_PCH_OK (real ccache or no launcher; off
  # only under sccache) — with ccache it coexists: cold builds get the PCH, warm rebuilds still cache-hit. The
  # tests' own core.hpp include becomes a no-op under the PCH; all tests share identical flags so REUSE_FROM valid.
  set(_test_pch ${FENIX_PCH_OK})
  set(_pch_anchor "")
  foreach(_src ${FENIX_TEST_SOURCES})
    get_filename_component(_name ${_src} NAME_WE)
    add_executable(${_name} ${_src})
    target_link_libraries(${_name} PRIVATE fenix_headers)
    # A lighter -O for the test TUs (dev builds). FENIX_LIGHT_OPT suppresses the interface -O3/-funroll (see
    # the FAST block); the test then sets its own -O. Empty FENIX_TEST_OPT leaves the test at the release -O3.
    if(FENIX_TEST_OPT)
      set_property(TARGET ${_name} PROPERTY FENIX_LIGHT_OPT TRUE)
      target_compile_options(${_name} PRIVATE ${FENIX_TEST_OPT})
    endif()
    if(_test_pch)
      if(NOT _pch_anchor)
        target_precompile_headers(${_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/core/core.hpp)
        set(_pch_anchor ${_name})
      else()
        target_precompile_headers(${_name} REUSE_FROM ${_pch_anchor})
      endif()
    endif()
    add_test(NAME ${_name} COMMAND ${_name})
  endforeach()
  if(_test_pch)
    message(STATUS "fenix tests: shared core.hpp PCH ON (anchor ${_pch_anchor})")
  endif()
endif()

message(STATUS "fenix ${FENIX_VERSION}  (GUI=${FENIX_GUI} ML=${FENIX_ML} FS=${FENIX_FS} FAST=${FENIX_FAST} "
               "SANITIZE='${FENIX_SANITIZE}' COVERAGE=${FENIX_COVERAGE})")
