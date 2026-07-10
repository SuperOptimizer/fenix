# clang-windows-toolchain.cmake — NATIVE WINDOWS build: clang (LLVM) + MSVC STL + lld-link.
# Companion to clang-toolchain.cmake (the canonical Linux/libc++ path). This variant targets
# x86_64-pc-windows-msvc, so it uses Microsoft's STL + UCRT (NOT libc++), because no official
# Windows LLVM ships libc++. Everything else stays clang/LLVM: the compiler, lld, llvm-* tools.
# Deps are vendored (FetchContent) — see cmake/deps-windows-zlib.cmake and CMakeLists.txt.

# Locate clang++: an explicit -DFENIX_WIN_CLANG=... wins; else the VS-bundled LLVM; else PATH.
set(FENIX_WIN_CLANG "" CACHE FILEPATH "Path to clang++.exe for the native-Windows build")
if(FENIX_WIN_CLANG)
  set(_cxx "${FENIX_WIN_CLANG}")
else()
  file(GLOB _vs_clang
    "C:/Program Files/Microsoft Visual Studio/*/*/VC/Tools/Llvm/x64/bin/clang++.exe"
    "C:/Program Files (x86)/Microsoft Visual Studio/*/*/VC/Tools/Llvm/x64/bin/clang++.exe"
    "C:/Program Files/LLVM/bin/clang++.exe")
  if(_vs_clang)
    list(GET _vs_clang 0 _cxx)
  else()
    set(_cxx clang++)
  endif()
endif()
get_filename_component(_bindir "${_cxx}" DIRECTORY)

set(CMAKE_CXX_COMPILER "${_cxx}"            CACHE FILEPATH "")
set(CMAKE_C_COMPILER   "${_bindir}/clang.exe" CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_C_COMPILER_TARGET   x86_64-pc-windows-msvc)

# Windows resource compiler: LLVM's llvm-rc (ships next to clang). Without it, enabling the
# implicit RC language on the Windows-Clang platform fails ("No CMAKE_RC_COMPILER could be found").
if(EXISTS "${_bindir}/llvm-rc.exe")
  set(CMAKE_RC_COMPILER "${_bindir}/llvm-rc.exe" CACHE FILEPATH "")
endif()

# lld (lld-link on the MSVC target). CMAKE_LINKER_TYPE needs CMake >= 3.29 (see CMakeLists.txt).
set(CMAKE_LINKER_TYPE LLD)

# LLVM binutils that ship alongside clang (no GNU).
foreach(_pair ar:llvm-ar ranlib:llvm-ranlib nm:llvm-nm objdump:llvm-objdump
              objcopy:llvm-objcopy strip:llvm-strip)
  string(REPLACE ":" ";" _p "${_pair}")
  list(GET _p 0 _n)
  list(GET _p 1 _b)
  string(TOUPPER "${_n}" _N)
  if(EXISTS "${_bindir}/${_b}.exe")
    set(CMAKE_${_N} "${_bindir}/${_b}.exe" CACHE FILEPATH "")
  endif()
endforeach()

# NOTE: deliberately no -stdlib=libc++ / -rtlib=compiler-rt / -unwindlib=libunwind here —
# on the windows-msvc target clang uses MSVC's STL + UCRT, which is the whole point of this
# variant. The Linux toolchain file keeps those flags.

# compiler-rt builtins: __builtin_cpu_supports (blosc2's SIMD dispatch, and -march=native codegen)
# references __cpu_model / __cpu_indicator_init, which live in clang_rt.builtins. clang does not
# auto-link its builtins archive for the windows-msvc target, so link it explicitly.
file(GLOB _fenix_rtbuiltins "${_bindir}/../lib/clang/*/lib/windows/clang_rt.builtins-x86_64.lib")
if(_fenix_rtbuiltins)
  list(GET _fenix_rtbuiltins 0 _fenix_rtb)
  add_link_options("${_fenix_rtb}")
endif()
