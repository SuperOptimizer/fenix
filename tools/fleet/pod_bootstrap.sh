#!/bin/sh
# One-paste fleet-pod bootstrap: fenix + TRT + the b1 teacher engine. Idempotent.
# Assumes a RunPod Ubuntu 24.04 torch-preloaded image (see install-runpod-ubuntu2404.sh).
#   curl -sf https://raw.githubusercontent.com/SuperOptimizer/fenix/main/tools/fleet/pod_bootstrap.sh | sh
set -eu

[ -d /workspace/fenix ] || git clone -q https://github.com/SuperOptimizer/fenix /workspace/fenix
cd /workspace/fenix && git pull -q
[ -x build-release/fenix ] || ./install-runpod-ubuntu2404.sh

pip install -q --break-system-packages tensorrt onnxscript huggingface_hub dynamic-network-architectures

# TRT public headers (pip wheels ship libs only) from the OSS repo at the wheel's release
if [ ! -e /opt/trt-include/NvInfer.h ]; then
  mkdir -p /opt/trt-include && cd /opt/trt-include
  for h in NvInfer.h NvInferImpl.h NvInferLegacyDims.h NvInferPluginBase.h NvInferRuntime.h \
           NvInferRuntimeBase.h NvInferRuntimeCommon.h NvInferRuntimePlugin.h NvInferVersion.h; do
    curl -sfO "https://raw.githubusercontent.com/NVIDIA/TensorRT/release/11.1/include/$h"
  done
  sed -i "s/TRT_MAJOR_ENTERPRISE/11/; s/TRT_MINOR_ENTERPRISE/1/; s/TRT_PATCH_ENTERPRISE/0/; \
          s/TRT_BUILD_ENTERPRISE/106/; /^#define [0-9]/d" NvInferVersion.h
fi

TRTLIB=$(ls /usr/local/lib/python3.12/dist-packages/tensorrt_libs/libnvinfer.so.* 2>/dev/null | head -1)
cd /workspace/fenix
if ! grep -q FENIX_TRT build-release/CMakeCache.txt 2>/dev/null; then
  cmake -B build-release --preset release -DFENIX_TRT_LIB="$TRTLIB" -DFENIX_TRT_INCLUDE=/opt/trt-include
  cmake --build build-release -j "$(nproc)"
fi

CKPT=$(python3 -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('scrollprize/surface_recto_3dunet','checkpoint_inference_ready.pth'))")
[ -e /root/surface_256b1.plan ] || \
  (cd tools/ml-export && python3 build_engine.py "$CKPT" /root/surface_256b1.plan 256 1)

echo "READY: $(nvidia-smi --query-gpu=name --format=csv,noheader) — run:"
echo "  sh tools/fleet/teacher_sweep.sh <sweep.txt> <shard> <nshards>"
