#!/bin/sh
# bootstrap.sh — build the Chimera Linux dev image and drop into a dev shell (or run a
# command). All fenix development happens inside this container (musl + LLVM, zero GNU).
# A persistent ccache volume + a build-dir volume keep rebuilds fast across runs. Usage:
#   ./bootstrap.sh              # build image, open a shell in the repo
#   ./bootstrap.sh build        # configure + build the release preset in the container
#   ./bootstrap.sh test         # build + run tests (asan preset) in the container
#   ./bootstrap.sh ci           # the full local CI gate (release + asan + tests)
#   ./bootstrap.sh <cmd...>     # run an arbitrary command in the container
#
# Uses docker if present, else podman (same CLI surface).
set -eu

IMAGE=fenix-dev
ROOT=$(CDPATH= cd "$(dirname "$0")" && pwd)
ENGINE=$(command -v docker || command -v podman || { echo "need docker or podman" >&2; exit 1; })

"$ENGINE" build -t "$IMAGE" "$ROOT"

# Persist ccache so the unity-build TU isn't recompiled cold every run.
run() {
  "$ENGINE" run --rm -it \
    -v "$ROOT":/work -w /work \
    -v fenix-ccache:/ccache \
    "$IMAGE" "$@"
}

case "${1:-shell}" in
  shell) run /bin/sh ;;
  build) run sh -c 'cmake --preset release && cmake --build --preset release' ;;
  test)  run sh -c 'cmake --preset asan && cmake --build --preset asan && ctest --test-dir build-asan --output-on-failure' ;;
  ci)    run sh -c 'cmake --preset release && cmake --build --preset release && ctest --test-dir build-release --output-on-failure &&
                    cmake --preset asan && cmake --build --preset asan && ctest --test-dir build-asan --output-on-failure' ;;
  *)     run "$@" ;;
esac
