#!/bin/sh
# Build mimalloc from source against clang/libc++/musl and install to a prefix.
# Single source of truth: invoked by Dockerfile (image bake) AND by CMake's
# configure-time source fallback (cmake/deps/mimalloc.cmake). Idempotent.
#   usage: build-mimalloc.sh <install-prefix>
set -eu
PREFIX="${1:-${FENIX_DEPS_PREFIX:-/opt/mimalloc}}"
SRC="${FENIX_DEPS_SRC:-/tmp/fenix-deps-src}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
VER="${MIMALLOC_VERSION:-v3.3.2}"
STAMP="$PREFIX/.fenix-mimalloc-$VER.stamp"
[ -f "$STAMP" ] && { echo "mimalloc $VER already at $PREFIX"; exit 0; }

mkdir -p "$SRC"
[ -d "$SRC/mimalloc" ] || git clone --depth=1 -b "$VER" https://github.com/microsoft/mimalloc "$SRC/mimalloc"
cmake -S "$SRC/mimalloc" -B "$SRC/b-mimalloc" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DMI_BUILD_TESTS=OFF -DMI_BUILD_OBJECT=ON \
  -DMI_BUILD_SHARED=ON -DMI_BUILD_STATIC=ON
cmake --build "$SRC/b-mimalloc" -j"$JOBS"
cmake --install "$SRC/b-mimalloc"
touch "$STAMP"
echo "mimalloc $VER -> $PREFIX"
