# Docker / Chimera image strategy + dependency resolution

All fenix development and CI happen inside Docker images based on **Chimera Linux** (musl
libc + LLVM/Clang userland + BSD coreutils — zero GNU), matching the project's clang-only,
no-GNU stance (ADR 0001). Base image `chimeralinux/chimera:latest` is a multi-arch manifest
(x86_64, **aarch64**, ppc64le, riscv64), so it pulls the right variant on the arm64 dev box.

## Dependency resolution: find system, else build from source

One coherent rule, implemented in [`cmake/deps.cmake`](../../cmake/deps.cmake): every
third-party dependency is **found if installed, else compiled from source**. Each dep has a
cache mode `FENIX_DEP_<NAME>` ∈ `{ auto | system | source | off }` (default `auto`):

| mode     | behavior                                                              |
|----------|----------------------------------------------------------------------|
| `auto`   | use an installed copy if `find_package` succeeds; else build from source |
| `system` | require an installed copy; hard-fail if missing                      |
| `source` | always build from source (ignore any system copy)                   |
| `off`    | don't use the dependency                                            |

The from-source path runs **`cmake/scripts/build-<name>.sh` at configure time** — the *same*
scripts the Docker images bake in — installing into `FENIX_DEPS_PREFIX/<name>` and re-running
`find_package` against it. So image-baked and from-scratch builds never drift, and a bare box
with no prebuilt deps still configures (it just compiles them once). Each resolved dep exposes
a canonical imported target `fenix::<name>` (+ `FENIX_HAVE_<NAME>`); consumers link the alias
and never care how it was obtained.

The build scripts are idempotent (a per-version stamp short-circuits re-builds), pin a version,
and build with `clang`/`clang++`/libc++. They are the single source of truth for *how* each dep
is built on musl.

| dep       | script                 | upstream target(s)                          | mode default | image          |
|-----------|------------------------|---------------------------------------------|--------------|----------------|
| mimalloc  | `build-mimalloc.sh`    | `mimalloc-static`, `mimalloc`               | auto         | core           |
| c-blosc2  | `build-blosc2.sh`      | `Blosc2::blosc2_static/_shared`             | auto         | core           |
| Qt6       | `build-qt6.sh`         | `Qt6::Widgets` (+Core/Gui/OpenGL)           | auto (GUI)   | gui            |
| VTK       | `build-vtk.sh`         | `VTK::RenderingVolumeOpenGL2`               | auto (GUI)   | gui            |
| libtorch  | `build-libtorch.sh`    | `torch` / `torch_cpu`                       | auto (ML)    | ml             |

## Three layered images

### 1. Core — `Dockerfile` (the dev/CI image)
The header-only pipeline + the single-TU driver + tests. Verified Chimera packages (LLVM @
22.1.7, cmake 4.3, ninja 1.13, ccache 4.12): `clang lld llvm libcxx-devel libunwind-devel
clang-tools-extra cmake ninja git pkgconf ccache curl-devel zlib-ng-devel` + `chimerautils`;
`git-lfs` (best-effort, `user` repo). Chimera's `clang` bundles compiler-rt and defaults to
libc++ + lld. **mimalloc and c-blosc2 are not packaged → prebaked from source into
`/opt/{mimalloc,blosc2}`** by the same scripts CMake uses. A toolchain sanity-check runs at
build time so a drifted package name fails fast. Builds + tests everything except `ml/`/`gui/`.

### 2. GUI — `Dockerfile.gui` (Qt6 + VTK, firewalled to `gui/`)
Builds **Qt6 qtbase** then a **minimal VTK** from source against musl + libc++ (volume
ray-cast + OpenGL2 + Qt support modules only). VTK renders headless via **EGL/offscreen**;
Chimera's mesa ships the legacy combined `libGL.so` but not the GLVND split libs, so the VTK
script adds `libOpenGL.so`/`libGLX.so` compat symlinks and configures with
`OpenGL_GL_PREFERENCE=LEGACY`. VTK's bundled `fmt` needs `-DFMT_USE_CHAR8_T=0` under libc++
(no `char_traits<char8_t>`). Opt-in `-DFENIX_GUI=ON`.

### 3. ML — `Dockerfile.ml` (libtorch, firewalled to `ml/`)
**Watch-item (from the research):** prebuilt **libtorch is glibc-based**, so it does not drop
onto musl. Options, in preference order:
1. Build LibTorch CPU-only from source against musl + libc++ (`build-libtorch.sh`; the clean
   musl-pure path — heavy, and PyTorch assumes glibc in spots so it is best-effort).
2. A `gcompat`/glibc-compat shim over the prebuilt glibc libtorch (faster, less pure).
3. A separate glibc base purely for this firewalled subimage (diverges from no-GNU, but `ml/`
   is optional, so acceptable as a fallback).
The core pipeline never depends on this image. Opt-in `-DFENIX_ML=ON`.

## Build status (validated 2026-06-27, aarch64, in-image clang/libc++ @ 22.1.7)
- **mimalloc 2.1.7** — built, installed, statically linked into `fenix`; overrides `malloc`. ✅
- **c-blosc2 2.15.1** — built, installed, linked (`FENIX_HAVE_BLOSC2`); awaiting io/codec use. ✅
- **Qt6 6.8.1 (qtbase)** — built, installed (`Qt6Config.cmake` + libs). ✅
- **VTK 9.3.1 (minimal)** — built from source (EGL/offscreen, legacy GL, fmt char8_t fix). ✅
- **libtorch** — source build attempt in progress; see `cmake/scripts/build-libtorch.sh`. ⏳
- **CMake resolver** — `auto` finds prebaked `/opt/*`; `source` compiles from scratch at
  configure time into `FENIX_DEPS_PREFIX`. Both paths verified for mimalloc + blosc2. ✅

## Usage
`./bootstrap.sh` builds the core image and opens a dev shell (docker or podman); `build` /
`test` / `ci` run the presets inside it, with a persistent `fenix-ccache` volume. The GUI/ML
images build on top: `docker build -f Dockerfile.gui -t fenix-gui .` /
`docker build -f Dockerfile.ml -t fenix-ml .`. CI uses the same `Dockerfile` as the job
container (see `.github/workflows/ci.yml`).
