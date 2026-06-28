#!/bin/sh
# Provision LibTorch (the C++ runtime of PyTorch) into a prefix. TWO paths, one script
# (single source of truth: Dockerfile{.ml,.ml.ubuntu} + CMake configure-time fallback):
#
#   * PREBUILT (glibc, the default on Ubuntu/glibc and whenever LIBTORCH_PREBUILT=1) —
#     download the official prebuilt CUDA libtorch zip and extract it. It links the system
#     CUDA runtime (cudart/cublas/cudnn/cusparseLt/...), which install-ubuntu.sh --ml provides
#     from the NVIDIA apt repo. This is the GPU path used on this project's bare-metal boxes.
#   * SOURCE (musl, the Chimera image) — build CPU-only from source against clang/libc++.
#     PyTorch assumes glibc in spots so it is best-effort; see docs/design/docker.md.
#
# ml/ is optional (FENIX_ML default OFF) so the core never depends on this. Idempotent.
#   usage: build-libtorch.sh <install-prefix>
set -eu
PREFIX="${1:-${FENIX_DEPS_PREFIX:-/opt/libtorch}}"
SRC="${FENIX_DEPS_SRC:-/tmp/fenix-deps-src}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
VER="${LIBTORCH_VERSION:-v2.12.1}"
STAMP="$PREFIX/.fenix-libtorch-$VER.stamp"
[ -f "$STAMP" ] && { echo "libtorch $VER already at $PREFIX"; exit 0; }

# ---- prebuilt path (glibc + CUDA) ------------------------------------------------------
# Default ON for glibc (musl libc has no /lib/ld-musl + the prebuilt is glibc-linked). Force
# with LIBTORCH_PREBUILT=1 / disable with LIBTORCH_PREBUILT=0.
_is_glibc() { ldd --version 2>&1 | head -1 | grep -qiE 'glibc|gnu libc'; }
if [ "${LIBTORCH_PREBUILT:-$(_is_glibc && echo 1 || echo 0)}" = "1" ]; then
  CU="${LIBTORCH_CUDA:-cu130}"           # CUDA tag; cu130 matches driver 595/CUDA 13.x
  PVER="${VER#v}"                         # strip leading v -> 2.12.1
  ZIP="libtorch-shared-with-deps-${PVER}%2B${CU}.zip"
  URL="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/${CU}/${ZIP}}"
  echo "libtorch: fetching prebuilt ${PVER}+${CU}  <-  $URL"
  mkdir -p "$SRC" "$PREFIX"
  [ -f "$SRC/libtorch.zip" ] || curl -fL -o "$SRC/libtorch.zip" "$URL"
  rm -rf "$SRC/libtorch"
  ( cd "$SRC" && unzip -q -o libtorch.zip )   # extracts a top-level libtorch/ dir
  cp -a "$SRC/libtorch/." "$PREFIX/"          # PREFIX becomes the libtorch root
  rm -rf "$SRC/libtorch" "$SRC/libtorch.zip"
  [ -f "$PREFIX/share/cmake/Torch/TorchConfig.cmake" ] || \
    { echo "libtorch: extract failed (no TorchConfig.cmake under $PREFIX)" >&2; exit 1; }
  touch "$STAMP"
  echo "libtorch ${PVER}+${CU} -> $PREFIX  (links SYSTEM CUDA; see install-ubuntu.sh --ml)"
  exit 0
fi

# ---- source path (musl / Chimera, CPU-only) --------------------------------------------

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
  -DUSE_PRIORITIZED_TEXT_FOR_LD=OFF \
  -DBUILD_PYTHON=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_TEST=OFF \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_XPU=OFF \
  -DUSE_DISTRIBUTED=OFF -DUSE_MPI=OFF -DUSE_GLOO=OFF \
  -DUSE_MKLDNN=OFF -DUSE_FBGEMM=OFF -DUSE_NNPACK=OFF -DUSE_QNNPACK=OFF -DUSE_XNNPACK=OFF \
  -DUSE_KINETO=OFF -DUSE_OPENMP=ON -DUSE_NUMPY=OFF \
  -DBLAS=OpenBLAS \
  -DCMAKE_C_FLAGS="-D_GNU_SOURCE -include sys/types.h" \
  -DCMAKE_CXX_FLAGS="-stdlib=libc++ -D_GNU_SOURCE -include sys/types.h"
cmake --build "$SRC/b-torch" -j"$JOBS"
cmake --install "$SRC/b-torch"
touch "$STAMP"
echo "libtorch $VER -> $PREFIX"
