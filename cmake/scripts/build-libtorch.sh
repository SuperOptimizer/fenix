#!/bin/sh
# Build LibTorch (the C++ runtime of PyTorch) from source against clang/libc++/musl and
# install to a prefix. CPU-only here (GPU is a later, separate concern). This is the HARD
# one: prebuilt libtorch is glibc-based, and PyTorch's build assumes glibc in spots, so a
# musl-pure build is best-effort. If it fails, the documented fallbacks (docs/design/docker.md)
# are a gcompat shim over prebuilt glibc libtorch, or a glibc base purely for the firewalled
# ml/ subimage. ml/ is optional (FENIX_ML default OFF) so the core never depends on this.
# Single source of truth: Dockerfile.ml + CMake configure-time source fallback. Idempotent.
#   usage: build-libtorch.sh <install-prefix>
set -eu
PREFIX="${1:-${FENIX_DEPS_PREFIX:-/opt/libtorch}}"
SRC="${FENIX_DEPS_SRC:-/tmp/fenix-deps-src}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
VER="${LIBTORCH_VERSION:-v2.12.1}"
STAMP="$PREFIX/.fenix-libtorch-$VER.stamp"
[ -f "$STAMP" ] && { echo "libtorch $VER already at $PREFIX"; exit 0; }

# Build prerequisites: python + codegen modules (pyyaml/typing_extensions/numpy), BLAS, etc.
# All packaged on Chimera (no pip needed).
if command -v apk >/dev/null 2>&1; then
  apk add python python-devel python-numpy python-pyyaml python-typing_extensions \
          python-setuptools openblas-devel protobuf-devel linux-headers
fi

PATCH_DIR="${FENIX_PATCH_DIR:-$(CDPATH= cd "$(dirname "$0")/../patches" && pwd)}"
apply_patch() {  # apply_patch <patch> <srcdir>  (idempotent; warn-continue if upstream moved)
  if git -C "$2" apply --reverse --check "$1" 2>/dev/null; then
    echo "patch already applied: $(basename "$1")"
  elif git -C "$2" apply "$1" 2>/dev/null; then
    echo "applied: $(basename "$1")"
  else
    echo "WARN: $(basename "$1") did not apply (context drift or fixed upstream); continuing"
  fi
}

mkdir -p "$SRC"
if [ ! -d "$SRC/pytorch" ]; then
  git clone --depth=1 -b "$VER" --recurse-submodules --shallow-submodules \
    https://github.com/pytorch/pytorch "$SRC/pytorch"
fi

# libc++/musl fix: c10/macros/Macros.h __assert_fail glibc-signature mismatch. See patch header.
apply_patch "$PATCH_DIR/pytorch-musl-assert-fail.patch" "$SRC/pytorch"

# BUILD_PYTHON=OFF → just the C++ libtorch. Trim everything we don't use to keep it tractable.
cmake -S "$SRC/pytorch" -B "$SRC/b-torch" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DProtobuf_USE_STATIC_LIBS=ON \
  -DBUILD_PYTHON=OFF -DBUILD_SHARED_LIBS=ON \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_XPU=OFF \
  -DUSE_DISTRIBUTED=OFF -DUSE_MPI=OFF -DUSE_GLOO=OFF \
  -DUSE_MKLDNN=OFF -DUSE_FBGEMM=OFF -DUSE_NNPACK=OFF -DUSE_QNNPACK=OFF -DUSE_XNNPACK=OFF \
  -DUSE_KINETO=OFF -DUSE_OPENMP=ON -DUSE_NUMPY=OFF \
  -DBLAS=OpenBLAS \
  -DCMAKE_CXX_FLAGS="-stdlib=libc++"
cmake --build "$SRC/b-torch" -j"$JOBS"
cmake --install "$SRC/b-torch"
touch "$STAMP"
echo "libtorch $VER -> $PREFIX"
