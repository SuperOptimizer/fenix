#!/bin/sh
# Provision a bare Ubuntu vast box for fenix training: toolchain via install-ubuntu.sh
# (core build, no --ml — train-feed is torch-free), python torch via pip wheel (bundles
# its own CUDA libs; no toolkit needed), then launch train_run.sh.
# Expects /workspace/payload_{src,meshes,eval}.tar.gz already scp'd.
#
# RENT with an onstart hook or the box is a one-boot wonder: vast provisions sshd into
# a bare image at instance CREATE only — any container restart (reboot / stop+start /
# host hiccup) comes back from the pristine image with NO sshd and NO processes, and
# no API path reaches a running instance without ssh (measured 2026-07-11: lost a
# training box that way; reboot and stop/start both failed to revive ssh).
#   vastai create instance <offer> --image ubuntu:26.04 --disk 200 --ssh --onstart-cmd \
#     'which sshd >/dev/null 2>&1 || (apt-get update && apt-get install -y openssh-server); \
#      mkdir -p /run/sshd; /usr/sbin/sshd || true; \
#      [ -x /workspace/onboot.sh ] && sh /workspace/onboot.sh || true'
# This script writes /workspace/onboot.sh so restarts also RESUME TRAINING unattended
# (meshes re-extract from the persisted payload — /tmp does not survive a restart;
# train_run.sh's trainer supervisor resumes from the newest studentV_step*.pt).
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

cat > /workspace/onboot.sh <<'EOB'
#!/bin/sh
# Runs on EVERY container start (vast --onstart-cmd). /opt and /tmp are wiped by a
# restart; /workspace persists. Rebuild is cached-cheap (ccache in /root is NOT
# persisted, so it is a full ~4min rebuild — still fully unattended).
set -ux
export DEBIAN_FRONTEND=noninteractive
[ -x /opt/fenix/build-release/fenix ] || {
  mkdir -p /opt/fenix && cd /opt/fenix
  tar xzf /workspace/payload_src.tar.gz
  sh ./install-ubuntu.sh
  apt-get install -y --no-install-recommends python3-pip python3-venv
  python3 -m venv /opt/venv
  /opt/venv/bin/pip install --no-cache-dir torch numpy
}
[ -d /tmp/gtqc ] || tar -C / -xzf /workspace/payload_meshes.tar.gz
[ -d /workspace/gtqc/eval ] || tar -C /workspace/gtqc -xzf /workspace/payload_eval.tar.gz
pgrep -f train_run.sh >/dev/null 2>&1 || {
  export TOOLS_DIR=/opt/fenix/tools/train FENIX_BIN=/opt/fenix/build-release/fenix
  export PATH=/opt/venv/bin:$PATH
  nohup sh /opt/fenix/tools/train/train_run.sh \
    /tmp/gtqc/pairs_multi_box.txt /tmp/gtqc/pairs_val_box.txt /workspace/train/studentV \
    BASE=32 STEPS=60000 BATCH=8 ACCUM=1 GPU=0 THREADS=24 ECHO=2 \
    CROPS=/workspace/gtqc/eval >> /workspace/train/run.log 2>&1 &
}
EOB
chmod +x /workspace/onboot.sh

export TOOLS_DIR=/opt/fenix/tools/train
export FENIX_BIN=/opt/fenix/build-release/fenix
export PATH=/opt/venv/bin:$PATH
nohup sh /opt/fenix/tools/train/train_run.sh \
  /tmp/gtqc/pairs_multi_box.txt /tmp/gtqc/pairs_val_box.txt /workspace/train/studentV \
  BASE=32 STEPS=60000 BATCH=8 ACCUM=1 GPU=0 THREADS=24 ECHO=2 \
  CROPS=/workspace/gtqc/eval > /workspace/train/run.log 2>&1 &
echo "LAUNCHED pid $!"
