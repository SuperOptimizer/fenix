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
  target_compile_options(fenix_headers INTERFACE
    $<$<CONFIG:Release>:-O3 -ffast-math -funroll-loops -march=native -fno-semantic-interposition>)
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
# CPU-seconds; best for clean/CI builds). SPLIT (-DFENIX_SPLIT=ON): each module in src/units/ is its own
# TU compiled in PARALLEL, plus a core.hpp precompiled header, plus driver.cpp built with -DFENIX_SPLIT
# so it includes only what main() touches. The module objects self-register their stages at startup. Same
# binary/behaviour; SPLIT trades more total work for parallelism + fast INCREMENTAL rebuilds. See ADR 0008.
if(FENIX_SPLIT)
  file(GLOB FENIX_UNIT_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/units/*.cpp)
  add_executable(fenix apps/driver.cpp ${FENIX_UNIT_SOURCES})
  target_compile_definitions(fenix PRIVATE FENIX_SPLIT)
  if(FENIX_SPLIT_PCH)
    target_precompile_headers(fenix PRIVATE ${CMAKE_SOURCE_DIR}/src/core/core.hpp)
  endif()
else()
  add_executable(fenix apps/driver.cpp)
endif()
target_link_libraries(fenix PRIVATE fenix_headers)
if(FENIX_GUI)
  target_compile_definitions(fenix PRIVATE FENIX_GUI)
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
  foreach(_src ${FENIX_TEST_SOURCES})
    get_filename_component(_name ${_src} NAME_WE)
    add_executable(${_name} ${_src})
    target_link_libraries(${_name} PRIVATE fenix_headers)
    add_test(NAME ${_name} COMMAND ${_name})
  endforeach()
endif()

message(STATUS "fenix ${FENIX_VERSION}  (GUI=${FENIX_GUI} ML=${FENIX_ML} FS=${FENIX_FS} FAST=${FENIX_FAST} "
               "SANITIZE='${FENIX_SANITIZE}' COVERAGE=${FENIX_COVERAGE})")
