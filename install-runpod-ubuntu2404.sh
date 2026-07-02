#!/bin/sh
# install-runpod-ubuntu2404.sh — provision fenix (with ML) on a RunPod-style **Ubuntu 24.04** GPU box
# that ships **torch 2.8+cuXXX preinstalled and GPU-working** (RTX 5090 / driver-in-host). This is the
# counterpart to install-ubuntu.sh, but tuned for THIS image profile instead of a bare 26.04 host:
#
#   * The base image has torch + cuDNN 9 + the CUDA runtime already (via pip/nvidia wheels) — we do NOT
#     install a CUDA toolkit or download a separate libtorch. We point FENIX_ML at the preinstalled
#     torch as libtorch (a complete lib+include+cxx11-ABI tree). Saves a ~2.5 GB download + apt CUDA.
#   * Ubuntu's default clang-18 does NOT support C++26 — we install LLVM 22 from apt.llvm.org.
#   * FENIX_ML switches the toolchain to libstdc++ (libtorch ABI) whose <print> needs libstdc++-14.
#   * apt's cmake 3.28 doesn't know clang-22 speaks C++26 → install cmake>=4 via pip.
#   * libcusparseLt (torch_cuda needs it) ships as an nvidia pip wheel, not on the loader path → we
#     register torch's bundled CUDA libs + the cusparseLt wheel dir with ldconfig.
#
#   ./install-runpod-ubuntu2404.sh            # toolchain + deps + configure + build (ML) + smoke test
#   ./install-runpod-ubuntu2404.sh --no-build # install/setup only
#   ./install-runpod-ubuntu2404.sh --core     # skip ML (core build only)
#
# Idempotent: apt/pip are no-ops if present; re-running just reconfigures/rebuilds. Run as root.
set -eu

ROOT=$(CDPATH= cd "$(dirname "$0")" && pwd)
DO_BUILD=1
WITH_ML=1
for arg in "$@"; do
  case "$arg" in
    --no-build) DO_BUILD=0 ;;
    --core)     WITH_ML=0 ;;
    *) echo "unknown flag: $arg (use --no-build / --core)" >&2; exit 2 ;;
  esac
done

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
export DEBIAN_FRONTEND=noninteractive

echo "==> [1/6] apt: base tooling"
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
  curl ca-certificates gnupg lsb-release software-properties-common \
  ninja-build ccache git git-lfs pkg-config \
  libcurl4-openssl-dev zlib1g-dev libstdc++-14-dev >/dev/null
git lfs install --skip-repo >/dev/null 2>&1 || true

echo "==> [2/6] LLVM 22 (apt.llvm.org — clang-18 lacks C++26)"
if ! command -v clang-22 >/dev/null 2>&1; then
  curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh
  chmod +x /tmp/llvm.sh
  $SUDO /tmp/llvm.sh 22 all
fi
# clang tooling + libc++ (the toolchain file wants these present). Not `|| true`-masked
# any more: a failure here left the symlink loop below silently killing the script under
# `set -eu` (both /usr/bin/$p and /usr/bin/$p-22 missing -> the `[ -e ... ] || { ... }`
# brace group's own exit status is the failed test -> set -e aborts mid-loop, no message).
# Surface the real error instead.
if ! $SUDO apt-get install -y --no-install-recommends \
  clang-tools-22 libclang-rt-22-dev libc++-22-dev libc++abi-22-dev \
  lld-22 clang-tidy-22 clang-format-22 libomp-22-dev >/dev/null 2>&1; then
  echo "ERROR: apt-get install of clang-22 tooling failed — see above for the apt error." >&2
  exit 1
fi
# apt.llvm.org installs ONLY versioned binaries (clang-22, ld.lld-22). The cmake toolchain's
# `CMAKE_LINKER_TYPE LLD` and CC/CXX=clang want UNVERSIONED names on PATH — create the symlinks
# (idempotent).
for p in clang clang++ ld.lld lld llvm-ar llvm-ranlib llvm-nm llvm-objdump llvm-objcopy llvm-strip; do
  if [ ! -e "/usr/bin/$p" ] && [ -e "/usr/bin/$p-22" ]; then
    $SUDO ln -sf "/usr/bin/$p-22" "/usr/bin/$p"
  fi
done
clang-22 --version | head -1

echo "==> [3/6] cmake >= 4 (apt 3.28 rejects clang-22 C++26)"
CMAKE_MAJ=$(cmake --version 2>/dev/null | sed -n 's/cmake version \([0-9]*\).*/\1/p' || echo 0)
if [ "${CMAKE_MAJ:-0}" -lt 4 ]; then
  pip3 install --break-system-packages -q -U cmake
fi
hash -r 2>/dev/null || true
echo "cmake: $(cmake --version | head -1)"

# ---- ML: wire the preinstalled torch as libtorch + fix the CUDA runtime loader path ----------------
CONFIG_ARGS="-DFENIX_LLVM_SUFFIX=-22"
if [ "$WITH_ML" -eq 1 ]; then
  echo "==> [4/6] ML: locate preinstalled torch + register CUDA runtime libs"
  TORCH_DIR=$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null || true)
  if [ -z "${TORCH_DIR:-}" ] || [ ! -f "$TORCH_DIR/share/cmake/Torch/TorchConfig.cmake" ]; then
    echo "ERROR: preinstalled torch not found — this script expects the torch-preloaded image." >&2
    echo "       (use ./install-ubuntu.sh --ml to fetch a standalone libtorch instead.)" >&2
    exit 1
  fi
  echo "torch: $(python3 -c 'import torch;print(torch.__version__)')  at $TORCH_DIR"
  # torch_cuda needs libcusparseLt (ships as an nvidia pip wheel) + torch's bundled cuDNN/cudart on the
  # loader path. The nvidia/*/lib wheel dirs sit alongside torch under its dist-packages parent — glob
  # from there (NOT sys.prefix, which is /usr, not the dist-packages root). Register all so the fenix
  # binary resolves them at runtime without LD_LIBRARY_PATH.
  SITE_DIR=$(dirname "$TORCH_DIR")
  {
    echo "$TORCH_DIR/lib"
    ls -d "$SITE_DIR"/nvidia/*/lib 2>/dev/null || true
  } | $SUDO tee /etc/ld.so.conf.d/fenix-torch.conf >/dev/null
  $SUDO ldconfig
  ldconfig -p | grep -q 'libcusparseLt.so' || echo "warn: libcusparseLt still not resolved (torch_cuda may fail to load)"
  CONFIG_ARGS="$CONFIG_ARGS -DFENIX_ML=ON -DFENIX_LIBTORCH_ROOT=$TORCH_DIR"
else
  echo "==> [4/6] ML: skipped (--core)"
fi

if [ "$DO_BUILD" -eq 0 ]; then
  echo "==> setup complete (--no-build). configure with: cmake --preset release $CONFIG_ARGS"
  exit 0
fi

# ---- build location note: on the FUSE-mount variant of this image, /workspace is slow and builds hang
# on 'Re-checking globbed directories'. If cmake stalls, clone/build under /dev/shm (RAM) and copy the
# binary to /root to run (/dev/shm is noexec). On this box /workspace is XFS, so building in-place is OK.
export CC=clang-22 CXX=clang++-22
export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
JOBS=$(nproc)
# Respect a cgroup CPU quota if present (don't spawn nproc build jobs on a quota-limited container).
if [ -r /sys/fs/cgroup/cpu.max ]; then
  Q=$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max); P=$(cut -d' ' -f2 /sys/fs/cgroup/cpu.max)
  case "$Q" in max) : ;; *) JOBS=$(( (Q + P - 1) / P )); [ "$JOBS" -lt 1 ] && JOBS=1 ;; esac
fi

echo "==> [5/6] configure + build (preset release, -j$JOBS)"
# A half-configured build-release/ from an earlier failed run poisons re-configure (stale compiler
# cache). If the cache exists but has no generated build system, wipe it for a clean configure.
if [ -f "$ROOT/build-release/CMakeCache.txt" ] && [ ! -f "$ROOT/build-release/build.ninja" ]; then
  echo "    (clearing stale build-release cache from a prior failed configure)"
  rm -rf "$ROOT/build-release"
fi
cmake --preset release $CONFIG_ARGS
cmake --build --preset release -j "$JOBS"

echo "==> [6/6] smoke test"
BIN="$ROOT/build-release/fenix"
if "$BIN" --help >/dev/null 2>&1; then
  echo "core: ok ($BIN)"
else
  echo "ERROR: smoke test failed — '$BIN --help' exited nonzero." >&2
  exit 1
fi
if [ "$WITH_ML" -eq 1 ]; then
  "$BIN" ml 2>&1 | grep -qi 'cuda\|self-test' && echo "ml: cuda ok" || echo "ml: built (check 'fenix ml' for GPU)"
fi
echo "==> done.  binary: $BIN"
