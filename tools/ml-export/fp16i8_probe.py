"""3090-class lanes probe: fp16 and int8 through the v2 conv kernel (dtype-driven).

Ampere (sm86, the 3090 fleet) has NO fp8 tensor cores — its fast lanes are fp16 and
int8 (2x fp16 TC rate). The v2 implicit-GEMM kernel is dtype-driven (fp8/fp16 loads,
IS_INT for int8 -> int32 acc), so both lanes reuse the whole fused-inference structure.
This probe: per-layer corr vs f32 conv + depth-chain corr (int8's real risk) + a
micro-bench vs cuDNN channels-last fp16, per channel width.

Run on the idle GPU: CUDA_VISIBLE_DEVICES=1 python fp16i8_probe.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from fp8_conv3d_op import (Fp8Tensor, fp8_conv3d_v2, pack_weight_f16, pack_weight_fp8,
                           pack_weight_i8, quantize_fp8, quantize_i8)
from fp8_train import _amax_scale


def corr(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def bench(fn, it=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def lane_conv(x, w, lane):
    N, Ci, P = x.shape[0], x.shape[1], x.shape[2]
    Co = w.shape[0]
    xm = x.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, Ci)
    if lane == "fp16":
        x_l, sx = xm.half(), torch.ones(1, device=x.device)
        w8, sw = pack_weight_f16(w)
    elif lane == "int8":
        sx = _amax_scale(xm) * (448.0 / 127.0)      # amax/127
        x_l = quantize_i8(xm, sx)
        w8, sw = pack_weight_i8(w)
    else:
        sx = _amax_scale(xm)
        x_l = quantize_fp8(xm, sx)
        w8, sw = pack_weight_fp8(w)
    xf = Fp8Tensor(x_l, sx, (N, P, P, P, Ci))
    return fp8_conv3d_v2(xf, w8, sw, Co, 3, 1, out_dtype=torch.float16, xs_f=sx)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}")
    print("\n== per-layer corr vs f32 conv + kernel time ==")
    for C, P in ((64, 64), (128, 48), (320, 16)):
        x = torch.randn(1, C, P, P, P, device=dev) * 0.5
        w = torch.randn(C, C, 3, 3, 3, device=dev) * (1.0 / (C * 27) ** 0.5)
        yr = F.conv3d(x, w, padding=1)
        xh = x.half().to(memory_format=torch.channels_last_3d)
        wh = w.half()
        t_cudnn = bench(lambda: F.conv3d(xh, wh, padding=1))
        row = f"  C={C:>3} P={P:>3}: cuDNN-CL {t_cudnn:5.2f}ms"
        for lane in ("fp16", "int8", "fp8"):
            y = lane_conv(x, w, lane)
            t = bench(lambda ln=lane: lane_conv(x, w, ln))
            row += f" | {lane} corr={corr(y, yr):.4f} {t:5.2f}ms"
        print(row)

    print("\n== depth chain (8 conv+lrelu blocks, C=64, 32^3): compounding ==")
    C, P = 64, 32
    x0 = torch.randn(1, C, P, P, P, device=dev) * 0.5
    ws = [torch.randn(C, C, 3, 3, 3, device=dev) * (1.0 / (C * 27) ** 0.5)
          for _ in range(8)]
    ref = x0
    for w in ws:
        ref = F.leaky_relu(F.conv3d(ref, w, padding=1), 0.01)
        ref = ref / ref.std().clamp(min=1e-3)
    for lane in ("fp16", "int8", "fp8"):
        h = x0
        for w in ws:
            h = F.leaky_relu(lane_conv(h.float(), w, lane).float(), 0.01)
            h = h / h.std().clamp(min=1e-3)
        print(f"  {lane}: depth-8 corr={corr(h, ref):.5f}")


if __name__ == "__main__":
    main()
