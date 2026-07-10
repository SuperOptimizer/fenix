# deps-windows-zlib.cmake — vendor zlib via FetchContent on native Windows (no system pkg).
# Exposes the canonical ZLIB::ZLIB imported target the rest of the build links against, exactly
# as find_package(ZLIB) would on Linux. Built static, with clang + MSVC STL.
include(FetchContent)
set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  zlib
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG        v1.3.1
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(zlib)

# madler/zlib builds 'zlibstatic' (and a shared 'zlib'); it does not create ZLIB::ZLIB or
# carry usable include dirs on its targets. Wire both up so consumers `#include <zlib.h>`
# (source dir) + <zconf.h> (generated in the build dir) resolve.
if(TARGET zlibstatic)
  target_include_directories(zlibstatic INTERFACE "${zlib_SOURCE_DIR}" "${zlib_BINARY_DIR}")
  if(NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
  endif()
endif()
