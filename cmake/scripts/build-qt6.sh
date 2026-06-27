#!/bin/sh
# Build Qt6 qtbase from source against clang/libc++/musl and install to a prefix. Only
# qtbase (gui + widgets + opengl) — the 4-pane viewer needs no more. ICU/DBus/SQL off to
# keep it lean and musl-clean. Single source of truth: Dockerfile.gui + CMake configure-time
# source fallback (cmake/deps/qt6.cmake). Idempotent.
#   usage: build-qt6.sh <install-prefix>
set -eu
PREFIX="${1:-${FENIX_DEPS_PREFIX:-/opt/qt6}}"
SRC="${FENIX_DEPS_SRC:-/tmp/fenix-deps-src}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
VER="${QT6_VERSION:-v6.11.1}"
STAMP="$PREFIX/.fenix-qt6-$VER.stamp"
[ -f "$STAMP" ] && { echo "qt6 $VER already at $PREFIX"; exit 0; }

if command -v apk >/dev/null 2>&1; then
  apk add openssl3-devel libxkbcommon-devel fontconfig-devel freetype-devel \
          libxcb-devel wayland-devel mesa-devel libpng-devel zlib-ng-devel \
          double-conversion-devel pcre2-devel glib-devel >/dev/null
fi

mkdir -p "$SRC"
[ -d "$SRC/qtbase" ] || git clone --depth=1 -b "$VER" https://github.com/qt/qtbase "$SRC/qtbase"
cmake -S "$SRC/qtbase" -B "$SRC/b-qt" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DQT_BUILD_EXAMPLES=OFF -DQT_BUILD_TESTS=OFF -DQT_BUILD_BENCHMARKS=OFF \
  -DFEATURE_gui=ON -DFEATURE_widgets=ON -DFEATURE_opengl=ON \
  -DFEATURE_sql=OFF -DFEATURE_dbus=OFF -DFEATURE_icu=OFF
cmake --build "$SRC/b-qt" -j"$JOBS"
cmake --install "$SRC/b-qt"
touch "$STAMP"
echo "qt6 $VER -> $PREFIX"
