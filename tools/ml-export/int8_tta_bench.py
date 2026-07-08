"""Flip-TTA batched driver + batched-inference bench for the int8 resident lane.

Answers two board items in one harness:
  1. batched inference: ms/patch at N=1/2/4 through the fused int8 path
     (kernels are N-general; autotune keys include N)
  2. flip-TTA quality: all 8 axis-flip variants batched (N=8 or 2x4), un-flipped
     and averaged — SD@2 / corr vs the fp16 flip-TTA reference, plus ms/variant.

Flip equivariance holds for the seg-logit head (channel semantics unchanged by
spatial flips). Static frozen int8 calibration is reused across variants —
flipped inputs have identical value distributions, so (scale, zp) transfer
exactly.

Usage: python int8_tta_bench.py   (expects /tmp/ct/patch.npy)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools
import statistics
import time

import torch

import fp8_forward as ff
from reference import build_and_load

dev = "cuda"
FLIPS = [f for r in range(4) for f in itertools.combinations((2, 3, 4), r)]  # 8 subsets


def bench(fn, it=10, wu=3):
    for _ in range(wu):
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


def corr(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def tta_batch(x1):
    """[1,1,D,H,W] -> [8,1,D,H,W] of all axis-flip variants (order = FLIPS)."""
    return torch.cat([x1.flip(f) if f else x1 for f in FLIPS], 0)


def tta_merge(y8):
    """Un-flip each variant's prob map and average. y8: [8,D,H,W] (post-softmax)."""
    outs = [y8[i:i + 1].flip(tuple(a - 1 for a in f)) if f else y8[i:i + 1]
            for i, f in enumerate(FLIPS)]
    return torch.stack(outs, 0).mean(0)[0]


def main():
    net = build_and_load(
        "/home/forrest/fenix/models/surface_recto_3dunet/checkpoint_inference_ready.pth"
    ).to(dev).eval().to(memory_format=torch.channels_last_3d)
    x1 = ff.load_ct_patch("/tmp/ct/patch.npy", 128, dev).to(memory_format=torch.channels_last_3d)

    with torch.no_grad():
        # fp16 references: single + flip-TTA (the quality bar)
        with torch.autocast("cuda", dtype=torch.float16):
            y16 = net(x1).float().softmax(1)[:, 1][0]
            xb = tta_batch(x1).to(memory_format=torch.channels_last_3d)
            y16_tta = tta_merge(net(xb).float().softmax(1)[:, 1])
            t16 = bench(lambda: net(x1))

        with ff.fp8_conv3d_patched():
            net(x1)
        ff.freeze_calibration()

        with ff.fp8_resident_patched(net), torch.autocast("cuda", dtype=torch.float16):
            rows = []
            for n in (1, 2, 4, 8):
                xn = x1.expand(n, -1, -1, -1, -1).contiguous(
                    memory_format=torch.channels_last_3d)
                tn = bench(lambda: net(xn))
                rows.append((n, tn, tn / n))
            for n, tn, per in rows:
                print(f"int8 N={n}: {tn:6.1f} ms total  {per:6.1f} ms/patch"
                      f"  (fp16 N=1 {t16:.1f})")

            # flip-TTA through int8, batched N=8
            xb8 = xb.contiguous(memory_format=torch.channels_last_3d)
            t_tta = bench(lambda: net(xb8))
            y8_tta = tta_merge(net(xb8).float().softmax(1)[:, 1])
            y8_one = net(x1).float().softmax(1)[:, 1][0]

    sd_one = ff.surface_dice(y8_one, y16, tol_vox=2)
    sd_tta = ff.surface_dice(y8_tta, y16_tta, tol_vox=2)
    sd_gain = ff.surface_dice(y8_tta, y16, tol_vox=2)
    print(f"int8 TTA x8 batched: {t_tta:.1f} ms total, {t_tta / 8:.1f} ms/variant")
    print(f"int8 N=1  vs fp16 N=1 : corr {corr(y8_one, y16):.5f} SD@2 {sd_one:.4f}"
          f" [{'PASS' if sd_one >= 0.998 else 'FAIL'}]")
    print(f"int8 TTA8 vs fp16 TTA8: corr {corr(y8_tta, y16_tta):.5f} SD@2 {sd_tta:.4f}"
          f" [{'PASS' if sd_tta >= 0.998 else 'FAIL'}]")
    print(f"int8 TTA8 vs fp16 N=1 : SD@2 {sd_gain:.4f} (TTA-vs-single delta on int8 lane)")


if __name__ == "__main__":
    main()
