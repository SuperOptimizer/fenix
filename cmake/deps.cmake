# cmake/deps.cmake — fenix third-party dependency resolution.
#
# One coherent rule per dependency: find it installed, else compile it from source.
# Every dep has a cache mode FENIX_DEP_<NAME> ∈ { auto | system | source | off }:
#   auto    (default) : use an installed copy if present; otherwise build from source
#                       into FENIX_DEPS_PREFIX and use that.
#   system            : require an installed copy (find_package REQUIRED); fail if missing.
#   source            : always build from source (ignore any system copy).
#   off               : do not use this dependency at all.
#
# The from-source path runs cmake/scripts/build-<name>.sh AT CONFIGURE TIME — the very same
# scripts the Chimera Docker images bake in — installing into FENIX_DEPS_PREFIX/<name>, then
# re-runs find_package against that prefix. So "look in the system, fall back to building it
# ourselves" is a single path whether you're in the prebuilt image, bare CI, or a fresh box.
#
# Each resolved dep exposes a canonical imported target  fenix::<name>  and a boolean
# FENIX_HAVE_<NAME>. Consumers link fenix::<name> and never care how it was obtained.

set(FENIX_DEPS_PREFIX "${CMAKE_BINARY_DIR}/_fenix_deps" CACHE PATH
    "Install prefix for from-source third-party dependencies")
set(FENIX_DEPS_SRC "${CMAKE_BINARY_DIR}/_fenix_deps_src" CACHE PATH
    "Checkout/build scratch dir for from-source dependencies")
set(_FENIX_DEP_SCRIPTS "${CMAKE_CURRENT_LIST_DIR}/scripts")

# Discover both the image install prefixes (/opt/<dep>, baked by the Dockerfiles) and our
# own from-source prefix. Prepended so they win over anything stale on the system.
list(PREPEND CMAKE_PREFIX_PATH
  "${FENIX_DEPS_PREFIX}/mimalloc" "${FENIX_DEPS_PREFIX}/blosc2"
  "${FENIX_DEPS_PREFIX}/vtk" "${FENIX_DEPS_PREFIX}/qt6" "${FENIX_DEPS_PREFIX}/libtorch"
  /opt/mimalloc /opt/blosc2 /opt/vtk /opt/qt6 /opt/libtorch)

# Build one dependency from source via its script, installing to FENIX_DEPS_PREFIX/<name>,
# and make that prefix discoverable in the caller's scope.
function(_fenix_build_from_source name)
  set(_script "${_FENIX_DEP_SCRIPTS}/build-${name}.sh")
  if(NOT EXISTS "${_script}")
    message(FATAL_ERROR "fenix: no source-build script for '${name}' (expected ${_script})")
  endif()
  set(_prefix "${FENIX_DEPS_PREFIX}/${name}")
  message(STATUS "fenix dep ${name}: not found installed — building from source into ${_prefix} "
                 "(one-time; this can take a while)")
  execute_process(
    COMMAND /bin/sh "${_script}" "${_prefix}"
    RESULT_VARIABLE _rc
    COMMAND_ECHO STDOUT)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "fenix: source build of '${name}' failed (exit ${_rc}); see output above")
  endif()
  list(PREPEND CMAKE_PREFIX_PATH "${_prefix}")
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
endfunction()

# Resolve a find_package-style dependency. Implemented as a macro so find_package() runs in
# the caller's (directory) scope and its imported targets are visible to the rest of the build.
#   fenix_dep(<name> DEFAULT <mode> PACKAGE <PkgName> [VERSION <v>]
#             [COMPONENTS ...] TARGETS <t1> [t2 ...] [ALIAS <fenix::name>])
# TARGETS is the priority-ordered list of upstream imported targets; the first that exists
# becomes the canonical alias.
macro(fenix_dep name)
  cmake_parse_arguments(_FD "" "DEFAULT;PACKAGE;VERSION;ALIAS" "TARGETS;COMPONENTS" ${ARGN})
  string(TOUPPER "${name}" _FD_N)
  set(FENIX_DEP_${_FD_N} "${_FD_DEFAULT}" CACHE STRING
      "Resolve dependency '${name}': auto|system|source|off")
  set_property(CACHE FENIX_DEP_${_FD_N} PROPERTY STRINGS auto system source off)
  set(_FD_MODE "${FENIX_DEP_${_FD_N}}")
  set(FENIX_HAVE_${_FD_N} OFF)

  if(_FD_MODE STREQUAL "off")
    message(STATUS "fenix dep ${name}: off")
  else()
    set(_FD_ARGS "${_FD_PACKAGE}")
    if(_FD_VERSION)
      list(APPEND _FD_ARGS "${_FD_VERSION}")
    endif()
    if(_FD_COMPONENTS)
      list(APPEND _FD_ARGS COMPONENTS ${_FD_COMPONENTS})
    endif()

    set(_FD_FOUND FALSE)
    if(_FD_MODE STREQUAL "auto" OR _FD_MODE STREQUAL "system")
      find_package(${_FD_ARGS} QUIET)
      if(${_FD_PACKAGE}_FOUND)
        set(_FD_FOUND TRUE)
      endif()
    endif()
    if(NOT _FD_FOUND AND _FD_MODE STREQUAL "system")
      message(FATAL_ERROR
        "fenix dep ${name}: FENIX_DEP_${_FD_N}=system but '${_FD_PACKAGE}' was not found.\n"
        "Install it, point CMAKE_PREFIX_PATH at it, or use FENIX_DEP_${_FD_N}=auto/source.")
    endif()
    if(NOT _FD_FOUND)  # source mode, or auto with nothing installed → compile from scratch
      _fenix_build_from_source("${name}")
      find_package(${_FD_ARGS} REQUIRED)
      set(_FD_FOUND TRUE)
    endif()

    if(_FD_ALIAS)
      set(_FD_AL "${_FD_ALIAS}")
    else()
      set(_FD_AL "fenix::${name}")
    endif()
    foreach(_FD_T IN LISTS _FD_TARGETS)
      if(TARGET ${_FD_T} AND NOT TARGET ${_FD_AL})
        add_library(${_FD_AL} INTERFACE IMPORTED)
        target_link_libraries(${_FD_AL} INTERFACE ${_FD_T})
      endif()
    endforeach()
    if(NOT TARGET ${_FD_AL})
      message(FATAL_ERROR "fenix dep ${name}: resolved '${_FD_PACKAGE}' but none of TARGETS "
                          "(${_FD_TARGETS}) exist — fix the TARGETS list in deps.cmake.")
    endif()
    set(FENIX_HAVE_${_FD_N} ON)
    message(STATUS "fenix dep ${name}: ${_FD_MODE} → ${_FD_AL}")
  endif()
endmacro()

# ---- the dependency set ----------------------------------------------------
# Core (always considered): allocator + compression. Both are tiny, musl-clean source builds.
fenix_dep(mimalloc DEFAULT auto PACKAGE mimalloc
          TARGETS mimalloc-static mimalloc)
fenix_dep(blosc2   DEFAULT auto PACKAGE Blosc2
          TARGETS Blosc2::blosc2_static Blosc2::blosc2_shared)

# GUI-only (firewalled behind -DFENIX_GUI). Heavy source builds; the gui image prebakes them.
if(FENIX_GUI)
  fenix_dep(qt6 DEFAULT auto PACKAGE Qt6 COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets
            TARGETS Qt6::Widgets ALIAS fenix::qt6)
  fenix_dep(vtk DEFAULT auto PACKAGE VTK
            COMPONENTS RenderingVolumeOpenGL2 RenderingOpenGL2 InteractionStyle IOImage
            TARGETS VTK::RenderingVolumeOpenGL2 ALIAS fenix::vtk)
endif()

# ML-only (firewalled behind -DFENIX_ML). libtorch is the heaviest; the ml image prebakes it.
if(FENIX_ML)
  fenix_dep(libtorch DEFAULT auto PACKAGE Torch
            TARGETS torch torch_cpu ALIAS fenix::torch)
endif()
