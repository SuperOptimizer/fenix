"""Probe: does this Triton/driver miscompile TD>=4 cube tiles in _conv3d_f8_v2?

On Triton 3.5.1/sm120 every TD in {4,8} config produced wrong results (corr 0.85-0.99)
while TD in {1,2} was exact — the reason _cfgs_v2() caps TD at 2. Re-run this after any
Triton or driver upgrade: if all TDs pass, larger cube tiles can go back into the
autotune space (more L2 tap reuse -> possible inference speedup).

Usage: python fp8_td_probe.py    # prints PASS/FAIL per TD config + a verdict line
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import triton

import fp8_conv3d_op as op


def probe(td, th, tw, bn=64, bk=64, warps=4, stages=3):
    cfg = triton.Config({"TD": td, "TH": th, "TW": tw, "BLOCK_N": bn, "BLOCK_K": bk},
                        num_warps=warps, num_stages=stages)
    orig = op._conv3d_f8_v2
    op._conv3d_f8_v2 = triton.autotune(
        configs=[cfg], key=["Cin", "Cout", "D", "H", "W", "KD", "STRIDE"],
        reset_to_zero=["ps_ptr", "pq_ptr"])(orig.fn)
    try:
        torch.manual_seed(0)
        N, C, P, k = 1, 64, 32, 3
        x = torch.randn(N, C, P, P, P, device="cuda") * 0.5
        w = torch.randn(C, C, k, k, k, device="cuda") * (1.0 / (C * 27) ** 0.5)
        xm = x.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, C)
        sx = (xm.abs().amax().float() / op.E4M3_MAX).reshape(1)
        x8 = (xm / sx).clamp(-op.E4M3_MAX, op.E4M3_MAX).to(torch.float8_e4m3fn)
        w8, sw = op.pack_weight_fp8(w)
        xf = op.Fp8Tensor(x8, sx, (N, P, P, P, C))
        y = op.fp8_conv3d_v2(xf, w8, float(sw), C, k, 1, out_dtype=torch.float16,
                             xs_f=float(sx))
        # reference on the DEQUANTIZED operands: isolates the kernel from quant noise
        xd = (x8.float() * sx).reshape(N, P, P, P, C).permute(0, 4, 1, 2, 3)
        # unpack [Cout, tap, Cin] -> [Cout, Cin, k,k,k]
        wd = w8.float().reshape(C, k ** 3, C).permute(0, 2, 1).reshape(C, C, k, k, k) * sw
        yr = F.conv3d(xd, wd, padding=1)
        c = torch.corrcoef(torch.stack([y.flatten().float(), yr.flatten().float()]))[0, 1].item()
        return c
    finally:
        op._conv3d_f8_v2 = orig


def main():
    results = []
    for td, th, tw in ((1, 8, 16), (2, 4, 16), (4, 4, 8), (4, 8, 8), (8, 4, 4), (8, 8, 8)):
        c = probe(td, th, tw)
        ok = c > 0.9999
        results.append((td, ok))
        print(f"  TD={td} TH={th} TW={tw}: corr={c:.6f}  {'PASS' if ok else 'FAIL'}")
    big_ok = all(ok for td, ok in results if td >= 4)
    small_ok = all(ok for td, ok in results if td <= 2)
    if not small_ok:
        print("VERDICT: BROKEN BASELINE — even TD<=2 fails; do not trust this stack")
    elif big_ok:
        print("VERDICT: FIXED — TD>=4 correct on this Triton; widen _cfgs_v2() and re-tune")
    else:
        print("VERDICT: STILL MISCOMPILED — keep TD <= 2 in _cfgs_v2()")


if __name__ == "__main__":
    main()
