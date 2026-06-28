#!/bin/sh
# install-ubuntu.sh — provision + build fenix on a bare-metal Ubuntu 26.04 host (glibc +
# LLVM/Clang). The no-Docker counterpart to Dockerfile.ubuntu; same package set, same
# clang-only/no-GNU-compiler stance (glibc is the only GNU concession — it's what enables the
# prebuilt CUDA libtorch used by the ml/ module).
#
#   ./install-ubuntu.sh              # apt deps + configure + build + test (core, no ML)
#   ./install-ubuntu.sh --ml         # also fetch prebuilt CUDA libtorch + build with -DFENIX_ML=ON
#   ./install-ubuntu.sh --no-build   # install system deps only
#   FENIX_PRESET=relwithdebinfo ./install-ubuntu.sh   # pick a configure preset (default: release)
#
# Idempotent: apt install is a no-op if present; the dep source-builds short-circuit on a
# version stamp; ccache makes rebuilds cheap. Needs sudo for apt only.
set -eu

ROOT=$(CDPATH= cd "$(dirname "$0")" && pwd)
PRESET="${FENIX_PRESET:-release}"
WITH_ML=0
DO_BUILD=1
for arg in "$@"; do
  case "$arg" in
    --ml)       WITH_ML=1 ;;
    --no-build) DO_BUILD=0 ;;
    *) echo "unknown flag: $arg (use --ml / --no-build)" >&2; exit 2 ;;
  esac
done

if [ -r /etc/os-release ]; then
  . /etc/os-release
  [ "${VERSION_ID:-}" = "26.04" ] || \
    echo "note: tuned for Ubuntu 26.04; detected '${PRETTY_NAME:-unknown}' — continuing anyway."
fi

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "==> installing system packages (apt)"
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
  clang lld llvm clang-tidy clang-format clang-tools libclang-rt-dev \
  libc++-dev libc++abi-dev libunwind-dev libomp-dev \
  cmake ninja-build ccache git git-lfs pkg-config \
  libcurl4-openssl-dev zlib1g-dev ca-certificates curl
git lfs install --skip-repo || true

export CC=clang CXX=clang++
export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Core deps (mimalloc, c-blosc2) are resolved by cmake/deps.cmake in `auto` mode: if not
# installed system-wide it compiles them from source at configure time (the same scripts the
# Docker images bake). Nothing to do here — the first configure handles it.

# CUDA stack for the firewalled ml/ module. The NVIDIA driver is assumed already installed
# (this box ships 595 / CUDA 13.x); we add the CUDA *toolkit* + the runtime libs the prebuilt
# libtorch links: cudnn 9 and cusparseLt (NOT part of the base toolkit). Sourced from NVIDIA's
# CUDA apt repo — the ubuntu2404 repo is glibc-compatible with 26.04 and carries CUDA 13.0,
# which exactly matches libtorch's cu130 build.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
install_cuda() {
  if [ -x "$CUDA_HOME/bin/nvcc" ] && ldconfig -p | grep -q 'libcudnn.so.9'; then
    echo "==> CUDA toolkit + cuDNN already present ($("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release //p'))"
    return 0
  fi
  echo "==> installing CUDA toolkit + cuDNN + cuSPARSELt (NVIDIA apt repo)"
  command -v nvidia-smi >/dev/null 2>&1 || \
    echo "warn: no nvidia-smi — install the NVIDIA driver too (e.g. 'apt install cuda-drivers') for GPU."
  _kr=/tmp/cuda-keyring.deb
  curl -fsSL -o "$_kr" \
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
  $SUDO dpkg -i "$_kr"
  $SUDO apt-get update
  $SUDO apt-get install -y --no-install-recommends \
    cuda-toolkit-13-0 libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcusparselt0 libcusparselt-dev
  # cudart/cublas/etc. land under /usr/local/cuda-13.0/targets/.../lib, which is not on the
  # default loader path → register it so libtorch_cuda.so resolves them at runtime.
  echo "/usr/local/cuda-13.0/targets/x86_64-linux/lib" | \
    $SUDO tee /etc/ld.so.conf.d/fenix-cuda.conf >/dev/null
  $SUDO ldconfig
}

CONFIG_ARGS=""
if [ "$WITH_ML" -eq 1 ]; then
  install_cuda
  echo "==> fetching prebuilt CUDA libtorch (firewalled ml/ module)"
  LIBTORCH_PREFIX="${FENIX_LIBTORCH_PREFIX:-$ROOT/.fenix-libtorch}"
  LIBTORCH_PREBUILT=1 sh "$ROOT/cmake/scripts/build-libtorch.sh" "$LIBTORCH_PREFIX"
  export PATH="$CUDA_HOME/bin:$PATH"
  CONFIG_ARGS="-DFENIX_ML=ON -DCMAKE_PREFIX_PATH=$LIBTORCH_PREFIX;$CUDA_HOME -DCUDAToolkit_ROOT=$CUDA_HOME"
fi

if [ "$DO_BUILD" -eq 0 ]; then
  echo "==> system deps installed; skipping build (--no-build)"
  exit 0
fi

echo "==> configuring (preset: $PRESET${CONFIG_ARGS:+, $CONFIG_ARGS})"
cmake --preset "$PRESET" $CONFIG_ARGS

echo "==> building"
cmake --build --preset "$PRESET"

echo "==> testing"
ctest --preset "$PRESET" || echo "note: some tests failed (see output above)"

echo "==> done.  binary: $ROOT/build-$PRESET/fenix"
