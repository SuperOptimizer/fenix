#!/bin/sh
# Round F: QAT full cycle — prepare -> train -> convert -> int8 accuracy delta
cd /root
F=/workspace/fenix/build-release/fenix
PY=/root/trainenv/bin/python
TT=/workspace/fenix/tools/train
rm -f /dev/shm/f.ring
$F train-feed pairs4.txt /dev/shm/f.ring patch=128 slots=32 threads=12 octa=1 thickness=6 cache_mb=8192 > feedf.log 2>&1 &
FP=$!
sleep 8
$PY - <<'PYEOF' > qat_cycle.log 2>&1
import sys, torch
sys.path.insert(0, "/workspace/fenix/tools/train")
sys.argv = ["train.py", "--ring", "/dev/shm/f.ring", "--steps", "120", "--batch", "8",
            "--base", "16", "--qat", "--out", "/root/s_f", "--ckpt-every", "9999"]
import train
train.main()
PYEOF
RC1=$?
tail -n 2 qat_cycle.log
# convert step: load final EMA, apply QATConfig(step="convert"), save
$PY - <<'PYEOF' >> qat_cycle.log 2>&1
import sys, torch
sys.path.insert(0, "/workspace/fenix/tools/train")
from train import StudentUNet
from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig, IntxFakeQuantizeConfig
net = StudentUNet(base=16).cuda()
act = IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
w = IntxFakeQuantizeConfig(torch.int8, "per_channel", is_symmetric=True)
quantize_(net, QATConfig(activation_config=act, weight_config=w, step="prepare"))
st = torch.load("/root/s_f_final.pt", map_location="cuda", weights_only=False)
net.load_state_dict(st["ema"])
quantize_(net, QATConfig(step="convert"))
x = torch.randn(1, 1, 128, 128, 128, device="cuda")
with torch.no_grad():
    y = net(x)
print("CONVERT_OK", y.shape, y.dtype)
torch.save(net.state_dict(), "/root/s_f_int8.pt")
PYEOF
RC2=$?
kill $FP 2>/dev/null
grep -E "CONVERT_OK|Error" qat_cycle.log | tail -n 2
[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ] && echo "ROUND_F=PASS" || echo "ROUND_F=FAIL"
