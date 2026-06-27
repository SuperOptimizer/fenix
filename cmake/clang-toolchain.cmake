# clang-toolchain.cmake — LLVM/Clang ONLY (no GCC, no GNU binutils). Latest clang + lld +
# llvm-* tools + libc++. Intended for the Chimera Linux dev/CI image (musl + LLVM userland).
# Override the suffix with -DFENIX_LLVM_SUFFIX=-21 etc. if versioned binaries are installed.
set(FENIX_LLVM_SUFFIX "" CACHE STRING "Suffix for versioned LLVM tools, e.g. -21")

find_program(FENIX_CLANG   NAMES clang${FENIX_LLVM_SUFFIX} clang   REQUIRED)
find_program(FENIX_CLANGXX NAMES clang++${FENIX_LLVM_SUFFIX} clang++ REQUIRED)

set(CMAKE_C_COMPILER   "${FENIX_CLANG}"   CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "${FENIX_CLANGXX}" CACHE FILEPATH "")

# lld linker.
set(CMAKE_LINKER_TYPE LLD)

# LLVM binutils (no GNU).
foreach(_t ar:llvm-ar ranlib:llvm-ranlib nm:llvm-nm objdump:llvm-objdump
            objcopy:llvm-objcopy strip:llvm-strip readelf:llvm-readelf
            addr2line:llvm-addr2line)
  string(REPLACE ":" ";" _pair "${_t}")
  list(GET _pair 0 _name)
  list(GET _pair 1 _bin)
  string(TOUPPER "${_name}" _NAME)
  find_program(FENIX_${_NAME} NAMES ${_bin}${FENIX_LLVM_SUFFIX} ${_bin})
  if(FENIX_${_NAME})
    set(CMAKE_${_NAME} "${FENIX_${_NAME}}" CACHE FILEPATH "")
  endif()
endforeach()

# libc++ + compiler-rt; let clang drive the link.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
add_link_options(-stdlib=libc++ -rtlib=compiler-rt -unwindlib=libunwind)
