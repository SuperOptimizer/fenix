"""Probe: wgrad as chunked im2col + cuBLASLt fp8 GEMM (torch._scaled_mm) vs the fused
Triton kernels (v1/v2). The Triton wgrad is L2-bandwidth-bound at ~4x worse efficiency
than the fwd conv; a materialized-chunk GEMM pays 2 extra activation passes (im2col
write+read) but runs the MMA at full tensor-core rate via cuBLASLt.

Caveat probed here: the GEMM is SKINNY (M=Cout as small as 32, K up to 2M) — cuBLASLt
may handle the split-K badly. Measures the three representative net shapes.

Usage: python fp8_wgrad_gemm_probe.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
import triton.language as tl

import fp8_conv3d_op as op


@triton.jit
def _im2colT(x_ptr, out_ptr, N, D, H, W, Cin, Do, Ho, Wo, m0, MC, MCP,
             KD: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    # write im2col^T [NTAP*Cin, MCP] row-major for output rows m0..m0+MC; pad cols
    # BEYOND MC are written as 0 (0 x uninit-NaN = NaN in the GEMM otherwise).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    NTAP: tl.constexpr = KD * KD * KD
    total = NTAP * Cin * MCP
    mask = offs < total
    row = offs // MCP
    mm = offs % MCP
    m = m0 + mm
    tap = row // Cin
    ci = row % Cin
    kz = tap // (KD * KD)
    ky = (tap % (KD * KD)) // KD
    kx = tap % KD
    n_ = m // (Do * Ho * Wo)
    r = m % (Do * Ho * Wo)
    od = r // (Ho * Wo)
    oh = (r // Wo) % Ho
    ow = r % Wo
    pad = (KD - 1) // 2
    zz = od * STRIDE + kz - pad
    yy = oh * STRIDE + ky - pad
    xx = ow * STRIDE + kx - pad
    ib = (mask & (mm < MC) & (m < N * Do * Ho * Wo)
          & (zz >= 0) & (zz < D) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W))
    v = tl.load(x_ptr + ((((n_ * D + zz) * H + yy) * W + xx) * Cin + ci),
                mask=ib, other=0.0)
    tl.store(out_ptr + offs, v, mask=mask)


def wgrad_gemm(dy_mc, x_mc, shapes, Cin, Cout, k, stride, sdy, sx, chunk=1 << 19):
    N, D, H, W, Do, Ho, Wo = shapes
    M = N * Do * Ho * Wo
    K27 = k ** 3 * Cin
    dw = torch.zeros(Cout, K27, dtype=torch.float32, device=dy_mc.device)
    for m0 in range(0, M, chunk):
        MC = min(chunk, M - m0)
        MCP = (MC + 15) // 16 * 16
        col = torch.empty(K27, MCP, dtype=torch.float8_e4m3fn, device=x_mc.device)
        total = K27 * MCP
        _im2colT[(triton.cdiv(total, 4096),)](
            x_mc, col, N, D, H, W, Cin, Do, Ho, Wo, m0, MC, MCP,
            KD=k, STRIDE=stride, BLOCK=4096)
        dyT = torch.zeros(Cout, MCP, dtype=torch.float8_e4m3fn, device=dy_mc.device)
        dyT[:, :MC] = dy_mc[m0:m0 + MC].t()
        dw += torch._scaled_mm(dyT, col.t(), scale_a=sdy, scale_b=sx,
                               out_dtype=torch.float32)
    return dw.reshape(Cout, k ** 3, Cin).permute(0, 2, 1).reshape(Cout, Cin, k, k, k)


def corr(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def bench(fn, it=15):
    for _ in range(4):
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


def main():
    torch.manual_seed(0)
    dev = "cuda"
    # (Cin, Cout, P, stride): stage0 shallow, stage2 mid, stage5 deep
    for Ci, Co, P, s in ((32, 32, 128, 1), (128, 128, 32, 1), (320, 320, 8, 1)):
        Po = P // s
        x = torch.randn(1, Ci, P, P, P, device=dev) * 0.5
        gy = torch.randn(1, Co, Po, Po, Po, device=dev) * 0.1
        xm = x.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, Ci)
        sx = (xm.abs().amax().float() / 448).reshape(1)
        x8 = (xm / sx).clamp(-448, 448).to(torch.float8_e4m3fn)
        dym = gy.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, Co)
        sd = (dym.abs().amax().float() / 448).reshape(1)
        d8 = (dym / sd).clamp(-448, 448).to(torch.float8_e4m3fn)
        shapes = (1, P, P, P, Po, Po, Po)

        xr = (x8.float() * sx).reshape(1, P, P, P, Ci).permute(0, 4, 1, 2, 3)
        dyr = (d8.float() * sd).reshape(1, Po, Po, Po, Co).permute(0, 4, 1, 2, 3)
        wref = torch.nn.grad.conv3d_weight(xr, (Co, Ci, 3, 3, 3), dyr, padding=1)

        dwg = wgrad_gemm(d8, x8, shapes, Ci, Co, 3, s, sd, sx)
        tg = bench(lambda: wgrad_gemm(d8, x8, shapes, Ci, Co, 3, s, sd, sx))
        op.WGRAD_V2 = True
        dw2 = op.fp8_conv3d_wgrad(d8, x8, shapes, Ci, Co, 3, s, sd, sx)
        t2 = bench(lambda: op.fp8_conv3d_wgrad(d8, x8, shapes, Ci, Co, 3, s, sd, sx))
        print(f"Ci={Ci:>3} Co={Co:>3} P={P:>3}: GEMM corr={corr(dwg, wref):.5f} "
              f"{tg:6.2f}ms | tritonV2 corr={corr(dw2, wref):.5f} {t2:6.2f}ms | "
              f"GEMM speedup {t2/tg:.2f}x")


if __name__ == "__main__":
    main()
