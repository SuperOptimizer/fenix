# fenix ML image — adds LibTorch (the C++ runtime of PyTorch), built CPU-only from source
# against musl + libc++, on top of the core Chimera image. Firewalled to the ml/ module
# (opt-in -DFENIX_ML=ON). Prebuilt libtorch is glibc-based and not on Chimera, so a musl-pure
# source build is the clean path; the recipe is the SAME script CMake uses for its
# configure-time source fallback (cmake/deps.cmake). If the musl source build proves
# intractable, the documented fallbacks (docs/design/docker.md) are a gcompat shim over the
# prebuilt glibc libtorch, or a glibc base for this firewalled subimage only. The core image
# never depends on this one. Build with: docker build -f Dockerfile.ml -t fenix-ml .
FROM fenix-dev AS base

COPY cmake/scripts /opt/fenix/scripts
COPY cmake/patches /opt/fenix/patches

# LibTorch (CPU). The script apk-adds python/openblas/protobuf codegen deps. Heavy: pin
# LIBTORCH_VERSION to match the model export toolchain. GPU is a later, separate concern.
RUN sh /opt/fenix/scripts/build-libtorch.sh /opt/libtorch && rm -rf /tmp/fenix-deps-src /src

ENV CMAKE_PREFIX_PATH=/opt/mimalloc:/opt/blosc2:/opt/libtorch \
    Torch_DIR=/opt/libtorch/share/cmake/Torch

# Then build fenix with ML:  cmake --preset release -DFENIX_ML=ON
WORKDIR /work
CMD ["/bin/sh"]
