#!/bin/sh
# Build a MINIMAL VTK from source (only the modules the gui/ 4-pane viewer needs:
# volume ray-cast + OpenGL2 + interaction + image IO). EGL/offscreen by default so it
# works headless in CI; X is off (musl, no GNU). Optionally enables the Qt support module
# when Qt6 is present (pass its prefix via QT6_PREFIX). Single source of truth: Dockerfile.gui
# + CMake configure-time source fallback (cmake/deps/vtk.cmake). Idempotent.
#   usage: build-vtk.sh <install-prefix>
set -eu
PREFIX="${1:-${FENIX_DEPS_PREFIX:-/opt/vtk}}"
SRC="${FENIX_DEPS_SRC:-/tmp/fenix-deps-src}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
VER="${VTK_VERSION:-v9.6.2}"
STAMP="$PREFIX/.fenix-vtk-$VER.stamp"
[ -f "$STAMP" ] && { echo "vtk $VER already at $PREFIX"; exit 0; }

# Runtime/headers for OpenGL + EGL + GBM offscreen (Chimera package names).
if command -v apk >/dev/null 2>&1; then
  apk add mesa-devel mesa-gl-libs mesa-egl-libs mesa-gbm-libs mesa-gbm-devel \
          mesa-dri glu libxml2-devel >/dev/null
fi

# Chimera/musl mesa ships the legacy combined libGL.so but NOT the GLVND split libs
# (libOpenGL/libGLX) that CMake's FindOpenGL looks for. Provide compat symlinks so VTK's
# OpenGL discovery resolves; mesa's libGL exports both GL and GLX symbols, and we render
# headless via EGL anyway.
for _d in /usr/lib /usr/lib64 /lib; do
  [ -e "$_d/libGL.so" ] || continue
  for _gl in libOpenGL libGLX; do
    [ -e "$_d/$_gl.so" ] || ln -sf libGL.so "$_d/$_gl.so"
  done
  break
done

QT_ARGS=""
if [ -n "${QT6_PREFIX:-}" ]; then
  QT_ARGS="-DVTK_GROUP_ENABLE_Qt=YES -DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES \
           -DCMAKE_PREFIX_PATH=$QT6_PREFIX"
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
[ -d "$SRC/VTK" ] || git clone --depth=1 -b "$VER" https://gitlab.kitware.com/vtk/vtk.git "$SRC/VTK"

# libc++/musl fix for VTK's vendored diy2 fmt (std::char_traits<char8_t>). See the patch header.
apply_patch "$PATCH_DIR/vtk-diy2-fmt-char8t.patch" "$SRC/VTK"
# shellcheck disable=SC2086
cmake -S "$SRC/VTK" -B "$SRC/b-vtk" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_SHARED_LIBS=ON \
  -DVTK_BUILD_TESTING=OFF -DVTK_BUILD_EXAMPLES=OFF -DVTK_BUILD_DOCUMENTATION=OFF \
  -DVTK_GROUP_ENABLE_Rendering=DONT_WANT \
  -DVTK_GROUP_ENABLE_StandAlone=DONT_WANT \
  -DVTK_GROUP_ENABLE_Views=DONT_WANT \
  -DVTK_GROUP_ENABLE_Web=DONT_WANT \
  -DVTK_GROUP_ENABLE_Imaging=DONT_WANT \
  -DVTK_GROUP_ENABLE_MPI=DONT_WANT \
  -DVTK_MODULE_ENABLE_VTK_RenderingVolumeOpenGL2=YES \
  -DVTK_MODULE_ENABLE_VTK_RenderingOpenGL2=YES \
  -DVTK_MODULE_ENABLE_VTK_RenderingVolume=YES \
  -DVTK_MODULE_ENABLE_VTK_RenderingCore=YES \
  -DVTK_MODULE_ENABLE_VTK_InteractionStyle=YES \
  -DVTK_MODULE_ENABLE_VTK_IOImage=YES \
  -DVTK_MODULE_ENABLE_VTK_ImagingCore=YES \
  -DVTK_OPENGL_HAS_EGL=ON \
  -DOpenGL_GL_PREFERENCE=LEGACY \
  -DVTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=ON \
  -DVTK_USE_X=OFF -DVTK_USE_COCOA=OFF \
  -DVTK_WRAP_PYTHON=OFF \
  $QT_ARGS
cmake --build "$SRC/b-vtk" -j"$JOBS"
cmake --install "$SRC/b-vtk"
touch "$STAMP"
echo "vtk $VER -> $PREFIX"
