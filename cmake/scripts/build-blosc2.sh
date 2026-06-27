#!/bin/sh
# Build c-blosc2 from source against clang/libc++/musl and install to a prefix.
# Bundled zlib/zstd/lz4 (self-contained). Single source of truth: Dockerfile + CMake
# configure-time source fallback (cmake/deps/blosc2.cmake). Idempotent.
#   usage: build-blosc2.sh <install-prefix>
set -eu
PREFIX="${1:-${FENIX_DEPS_PREFIX:-/opt/blosc2}}"
SRC="${FENIX_DEPS_SRC:-/tmp/fenix-deps-src}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
VER="${BLOSC2_VERSION:-v3.1.5}"
STAMP="$PREFIX/.fenix-blosc2-$VER.stamp"
[ -f "$STAMP" ] && { echo "blosc2 $VER already at $PREFIX"; exit 0; }

mkdir -p "$SRC"
[ -d "$SRC/c-blosc2" ] || git clone --depth=1 -b "$VER" https://github.com/Blosc/c-blosc2 "$SRC/c-blosc2"
cmake -S "$SRC/c-blosc2" -B "$SRC/b-blosc2" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_SHARED=ON -DBUILD_STATIC=ON \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_FUZZERS=OFF
cmake --build "$SRC/b-blosc2" -j"$JOBS"
cmake --install "$SRC/b-blosc2"
touch "$STAMP"
echo "blosc2 $VER -> $PREFIX"
