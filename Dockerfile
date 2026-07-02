# fenix dev/CI image — Chimera Linux (musl libc + LLVM/Clang userland + BSD coreutils,
# ZERO GNU). All development happens inside this image; it doubles as the CI image.
# Multi-arch: the chimeralinux/chimera manifest covers x86_64, aarch64, ppc64le, riscv64 —
# so `docker build` on an arm64 host (fenix's primary target) pulls the aarch64 variant.
# https://chimera-linux.org/   package db: https://pkgs.chimera-linux.org/
#
# Package names verified against the Chimera aarch64 repo (LLVM stack @ 22.1.7). The base
# clang ships compiler-rt and defaults to libc++ + lld, so we only add the libc++/unwind
# *headers* + the tools. mimalloc and blosc2 are NOT in Chimera's repos -> built from source
# in a later layer when wired (both optional). libtorch (ml/) is glibc-based and Qt6/VTK
# (gui/) are heavy -> separate images: Dockerfile.ml / Dockerfile.gui (see docs).
FROM chimeralinux/chimera:latest

# apk (apk-tools 3). chimerautils = the BSD coreutils userland. libomp-devel is the
# OpenMP runtime + omp.h — Chimera packages it separately from clang/llvm, and without it
# find_package(OpenMP) silently fails and every `#pragma omp parallel for` degrades to a
# serial loop (src/core/parallel.hpp's `#else` fallback) with no diagnostic — including in
# the tsan job, which then cannot see OpenMP data races at all. CLAUDE.md §2.4: "OpenMP for
# data-parallel loops" is the stated parallelism primitive, so this is a required package.
RUN apk update && apk add \
      chimerautils \
      clang \
      lld \
      llvm \
      libcxx-devel \
      libunwind-devel \
      clang-tools-extra \
      cmake \
      ninja \
      git \
      pkgconf \
      ccache \
      curl-devel \
      zlib-ng-devel \
      libomp-devel

# git-lfs (golden test artifacts) lives in the `user` repo; best-effort so the core image
# builds even if that repo isn't enabled. Not needed to compile/test the core pipeline.
RUN apk add git-lfs && git lfs install --system || \
    echo "note: git-lfs unavailable (user repo); golden-artifact tests will be skipped"

# Clang is the only toolchain; Ninja the generator; ccache shared across configs.
# CMake seeds CMAKE_CXX_COMPILER_LAUNCHER from this env var, so CMakeLists.txt's
# `NOT CMAKE_CXX_COMPILER_LAUNCHER` guard is false in-container and its own
# CCACHE_SLOPPINESS=pch_defines,... injection (needed for ccache to cache PCH-consuming
# compiles — see the ~20-line comment there) never runs. Set the same sloppiness here so
# ccache+PCH coexist in the canonical container too, not just in a bare `cmake --preset`
# invocation outside Docker.
ENV CC=clang \
    CXX=clang++ \
    CMAKE_GENERATOR=Ninja \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CCACHE_DIR=/ccache \
    CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime \
    CCACHE_COMPILERCHECK="%compiler% -v -march=native -E -x c++ /dev/null"

# Sanity-check the toolchain at build time (fail fast if a package name drifts). The
# -fopenmp compile+link smoke test catches libomp-devel package-name/path drift here,
# at image-build time, instead of silently at `cmake configure` (find_package(OpenMP)
# has no loud failure mode on its own — see CMakeLists.txt).
RUN clang --version && clang++ --version && ld.lld --version && cmake --version && \
    ninja --version && clang-tidy --version && clang-format --version && \
    printf '#include <omp.h>\nint main(){ return omp_get_max_threads() > 0 ? 0 : 1; }\n' > /tmp/omp_check.cpp && \
    clang++ -fopenmp /tmp/omp_check.cpp -o /tmp/omp_check && /tmp/omp_check && rm -f /tmp/omp_check.cpp /tmp/omp_check

# Core third-party deps (allocator + compression) — NOT in Chimera's repos, so built from
# source against musl + libc++ with the same scripts CMake uses for its configure-time
# fallback (cmake/deps.cmake). Prebaking them into /opt means `cmake --preset` finds them
# instantly (no per-build compile). zlib-ng (curl-devel pulls it) covers the zlib dep.
COPY cmake/scripts /opt/fenix/scripts
RUN sh /opt/fenix/scripts/build-mimalloc.sh /opt/mimalloc && \
    sh /opt/fenix/scripts/build-blosc2.sh   /opt/blosc2 && \
    rm -rf /tmp/fenix-deps-src
ENV CMAKE_PREFIX_PATH=/opt/mimalloc:/opt/blosc2

WORKDIR /work
CMD ["/bin/sh"]
