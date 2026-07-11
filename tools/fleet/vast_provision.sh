#!/bin/sh
# Provision a bare Ubuntu vast box for fenix training: toolchain via install-ubuntu.sh
# (core build, no --ml — train-feed is torch-free), python torch via pip wheel (bundles
# its own CUDA libs; no toolkit needed), then launch train_run.sh.
# Expects /workspace/payload_{src,meshes,eval}.tar.gz already scp'd.
set -eux
export DEBIAN_FRONTEND=noninteractive
mkdir -p /opt/fenix /workspace/train /workspace/caches /workspace/gtqc
cd /opt/fenix
tar xzf /workspace/payload_src.tar.gz
tar -C / -xzf /workspace/payload_meshes.tar.gz
tar -C /workspace/gtqc -xzf /workspace/payload_eval.tar.gz

sh ./install-ubuntu.sh                      # apt toolchain + configure + build + test (core)

apt-get install -y --no-install-recommends python3-pip python3-venv
python3 -m venv /opt/venv
/opt/venv/bin/pip install --no-cache-dir torch numpy

export TOOLS_DIR=/opt/fenix/tools/train
export FENIX_BIN=/opt/fenix/build-release/fenix
export PATH=/opt/venv/bin:$PATH
nohup sh /opt/fenix/tools/train/train_run.sh \
  /tmp/gtqc/pairs_multi_box.txt /tmp/gtqc/pairs_val_box.txt /workspace/train/studentV \
  BASE=32 STEPS=60000 BATCH=8 ACCUM=1 GPU=0 THREADS=24 ECHO=2 \
  CROPS=/workspace/gtqc/eval > /workspace/train/run.log 2>&1 &
echo "LAUNCHED pid $!"
