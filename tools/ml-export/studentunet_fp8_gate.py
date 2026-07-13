#!/usr/bin/env python3
"""StudentUNet deploy-precision gate: fp8-resident lane on our from-scratch student.

The 2026-07-11 trace-eval arbiter obsoleted the surface_recto_3dunet distillation
target — the graded-corpus StudentUNet doubles its end-task recall at the same size.
This reruns the canonical resident bench (fp8_resident_bench.py) on StudentUNet:
fp16-autocast-CL baseline vs fp8-resident (+CUDA graph), corr + SurfaceDice@2 gate
(bar 0.998) on a real CT patch from the holdout crops.

StudentUNet compat notes: all convs swap-eligible (3^3 s1/s2 bias-free, 1^1 skips,
k2s2 transpconv, InstanceNorm3d affine); activations are FUNCTIONAL F.leaky_relu, so
the fused norm+act (normfuse) lane cannot bind — this measures the per-conv path.

Usage: studentunet_fp8_gate.py [--ckpt /tmp/gtqc/real/studentM_best.pt] [--base 32]
       [--ct /tmp/gtqc/m8/eval/crop00/ct.npy]
"""
import sys, os, argparse, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import numpy as np
import torch

import fp8_forward as ff
from train import StudentUNet

def bench(fn, it=10, wu=3):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def corr(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/gtqc/real/studentM_best.pt")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--ct", default="/tmp/gtqc/m8/eval/crop00/ct.npy")
    args = ap.parse_args()

    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = st.get("ema", st.get("net")) if isinstance(st, dict) else st
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    net = StudentUNet(base=args.base)
    net.load_state_dict(sd)
    net = net.to("cuda").eval().to(memory_format=torch.channels_last_3d)

    ct = np.load(args.ct)[:128, :128, :128].astype(np.float32) / 255.0  # StudentUNet's own norm
    x = torch.from_numpy(ct)[None, None].to("cuda").to(memory_format=torch.channels_last_3d)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            y16 = net(x).float().softmax(1)[:, 1]
            t16 = bench(lambda: net(x))
        # NOTE: fp8_resident_patched() binds the REFERENCE net's block classes
        # (ConvDropoutNormReLU/BasicBlockD) and silently no-ops on other archs — measured
        # here as corr exactly 1.0 and 0.99x "speedup". The arch-agnostic lane is
        # fp8_conv3d_patched (patches nn.Conv3d.forward): per-conv fp8, unfused.
        with ff.fp8_conv3d_patched():
            net(x)  # calibration pass
            ff.freeze_calibration()
            with torch.autocast("cuda", dtype=torch.float16):
                t8 = bench(lambda: net(x))
                sx = x.clone()
                for _ in range(3):
                    net(sx)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    sy = net(sx)
                def run(inp):
                    sx.copy_(inp); g.replay(); return sy
                tg = bench(lambda: run(x))
                yg = run(x).float().softmax(1)[:, 1]

    print(f"StudentUNet base={args.base}: autocast-CL {t16:.1f}ms | fp8-perconv {t8:.1f}ms | "
          f"fp8-perconv+GRAPH {tg:.1f}ms | speedup {t16/tg:.2f}x")
    sd2 = ff.surface_dice(yg, y16, tol_vox=2)
    print(f"graph corr {corr(yg, y16):.5f} SurfaceDice@2 {sd2:.4f} [{'PASS' if sd2 >= 0.998 else 'FAIL'}]")


if __name__ == "__main__":
    main()
