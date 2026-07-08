"""Fused fp8 e4m3 conv3d (implicit GEMM, Triton) — sm120 consumer Blackwell.

The spike (fp8_conv3d_spike.py) proved: the fp8 e4m3 GEMM is ~3.5x faster than cuDNN
fp16 conv3d on this box, but an EXPLICIT im2col buries the win (5-7x slower end-to-end
from materializing the 27x-inflated column tensor). This kernel is the payoff test: a
FUSED implicit GEMM that gathers the 27 taps inside the reduction loop — no im2col ever
materialized — to see whether the GEMM ceiling survives fusion.

Layout: NHWC-style (channels-last), the natural layout for tensor-core conv. Input is
[N, D, H, W, Cin] fp8; weight is [Cout, 3,3,3, Cin] -> flattened per tap. Output tile is
[BLOCK_M spatial rows] x [BLOCK_N output channels]; the K reduction runs over
27*Cin, with the tap index decoded per K-block to pick the (dz,dy,dx) shift and the Cin
slice. Boundary voxels (outside the volume) contribute zero (implicit zero-pad).

Per-tensor scaling (sa,sb) applied after the f32 accumulation, matching the spike recipe.

Usage: python fp8_conv3d_triton.py [--patch 64] [--iters 30]
"""
import argparse
import statistics
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

E4M3_MAX = 448.0


def _autotune_configs():
    cfgs = []
    for bm in (64, 128):
        for bn in (64, 128):
            for bk in (32, 64):
                for w in (4, 8):
                    for s in (2, 3, 4):
                        cfgs.append(triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                            num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_autotune_configs(), key=["Cin", "Cout", "D", "H", "W"])
@triton.jit
def _conv3d_fp8_kernel(
    x_ptr, w_ptr, y_ptr,
    sa, sb,
    N, D, H, W, Cin, Cout,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program computes y[m0:m0+BLOCK_M spatial, n0:n0+BLOCK_N cout]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    offs_m = m0 + tl.arange(0, BLOCK_M)          # flat spatial index over N*D*H*W
    offs_n = n0 + tl.arange(0, BLOCK_N)          # output channel

    # decode flat spatial index -> (n, d, h, w)
    w_idx = offs_m % W
    t = offs_m // W
    h_idx = t % H
    t = t // H
    d_idx = t % D
    n_idx = t // D
    m_valid = offs_m < (N * D * H * W)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = 27 * Cin, iterate tap-outer, cin-inner. BLOCK_K must divide Cin (asserted host-side).
    for tap in range(27):
        dz = tap // 9 - 1
        dy = (tap % 9) // 3 - 1
        dx = tap % 3 - 1
        zz = d_idx + dz
        yy = h_idx + dy
        xx = w_idx + dx
        in_bounds = m_valid & (zz >= 0) & (zz < D) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        base = ((n_idx * D + zz) * H + yy) * W + xx          # row base in x (channels-last)
        for k0 in range(0, Cin, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_valid = offs_k < Cin
            # gather x[base, offs_k]: [BLOCK_M, BLOCK_K]
            x_off = base[:, None] * Cin + offs_k[None, :]
            xmask = in_bounds[:, None] & k_valid[None, :]
            a = tl.load(x_ptr + x_off, mask=xmask, other=0.0)         # fp8
            # weight w[cout, tap, cin] flattened as [Cout, 27*Cin] row-major (tap outer)
            w_off = offs_n[:, None] * (27 * Cin) + (tap * Cin + offs_k)[None, :]
            wmask = (offs_n < Cout)[:, None] & k_valid[None, :]
            b = tl.load(w_ptr + w_off, mask=wmask, other=0.0)         # fp8  [BLOCK_N, BLOCK_K]
            acc += tl.dot(a, b.T)                                     # fp8 inputs -> f32 acc

    acc = acc * (sa * sb)
    y_off = offs_m[:, None] * Cout + offs_n[None, :]
    ymask = m_valid[:, None] & (offs_n < Cout)[None, :]
    tl.store(y_ptr + y_off, acc, mask=ymask)


def conv3d_fp8_triton(x_f, w_f):
    """x_f: [N,Cin,D,H,W] float ; w_f: [Cout,Cin,3,3,3] float -> y: [N,Cout,D,H,W] float."""
    N, Cin, D, H, W = x_f.shape
    Cout = w_f.shape[0]
    dev = x_f.device
    sa = (x_f.abs().amax().clamp(min=1e-8) / E4M3_MAX).to(torch.float32)
    sb = (w_f.abs().amax().clamp(min=1e-8) / E4M3_MAX).to(torch.float32)
    # channels-last fp8 input [N,D,H,W,Cin]
    xc = (x_f / sa).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    xc = xc.permute(0, 2, 3, 4, 1).contiguous()
    # weight [Cout, 27*Cin], tap-outer cin-inner
    wc = (w_f / sb).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    wc = wc.reshape(Cout, Cin, 27).permute(0, 2, 1).reshape(Cout, 27 * Cin).contiguous()

    y = torch.empty(N * D * H * W, Cout, dtype=torch.float32, device=dev)
    M = N * D * H * W
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(Cout, meta["BLOCK_N"]))
    _conv3d_fp8_kernel[grid](
        xc, wc, y, sa.item(), sb.item(),
        N, D, H, W, Cin, Cout,
    )
    return y.reshape(N, D, H, W, Cout).permute(0, 4, 1, 2, 3).contiguous()


def bench(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    P = args.patch
    print(f"device={torch.cuda.get_device_name()} cap={torch.cuda.get_device_capability()} "
          f"patch={P}^3\n")
    print(f"{'layer':>14} | {'corr':>8} {'max|err|':>9} | "
          f"{'fp8T us':>8} {'fp16 us':>8} {'speedup':>7}")
    print("-" * 68)
    for C in (64, 128, 256, 320):
        x = torch.randn(1, C, P, P, P, device=dev) * 0.5
        w = torch.randn(C, C, 3, 3, 3, device=dev) * (1.0 / (C * 27) ** 0.5)

        y_ref = F.conv3d(x.float(), w.float(), padding=1)
        y_fp8 = conv3d_fp8_triton(x, w)
        corr = torch.corrcoef(torch.stack([y_fp8.flatten(), y_ref.flatten()]))[0, 1].item()
        maxe = (y_fp8 - y_ref).abs().amax().item() / (y_ref.abs().amax().item() + 1e-8)
        del y_ref, y_fp8
        torch.cuda.empty_cache()

        t_fp8 = bench(lambda: conv3d_fp8_triton(x, w), args.iters)
        xh, wh = x.half(), w.half()
        t_fp16 = bench(lambda: F.conv3d(xh, wh, padding=1), args.iters)
        print(f"{C:>6}->{C:<6} | {corr:>8.4f} {maxe:>9.4f} | "
              f"{t_fp8:>8.1f} {t_fp16:>8.1f} {t_fp16/t_fp8:>6.2f}x")
        del x, w, xh, wh
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
