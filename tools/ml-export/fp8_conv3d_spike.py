"""fp8 e4m3 conv3d spike (unfold + cuBLASLt GEMM) — sm120 consumer Blackwell.

Answers the go/no-go question ADR 0010 parked: can a custom fp8 conv3d beat the
adopted TRT fp16 path (1.53x over eager)? This is the FAST de-risking probe: it uses
the proven cuBLASLt fp8 e4m3 GEMM (torch._scaled_mm, measured corr 0.9993 vs fp32 on
this box) as the tensor-core engine, with an explicit im2col unfold — NOT a fused
CUTLASS kernel. If the number wins here, the next step is the hand-assembled CUTLASS
dense sm120 fp8 GEMM (higher ceiling, no im2col materialization). If it loses even the
GEMM-optimal cuBLASLt path, custom fp8 conv3d is dead and TRT fp16 stays.

Method (per representative ResEnc-UNet layer C->C, 3x3x3, stride 1, pad 1):
  x[N,C,D,H,W] -> unfold to cols[N*D*H*W, 27*C] -> fp8
  weight[C, 27*C]                                -> fp8
  cols @ weight^T via _scaled_mm (fp8->f32)      -> y[N,C,D,H,W]
Per-tensor scales are amax/448 (e4m3 max), the standard fp8 dynamic-scaling recipe.

Baselines: cuDNN fp16 conv3d (torch), fp32 conv3d (reference for numerics).
Reports: correlation + max|err| vs fp16; median us/call for fp8 vs fp16.

Usage: python fp8_conv3d_spike.py [--patch 128] [--iters 50]
"""
import argparse
import statistics
import time

import torch
import torch.nn.functional as F

E4M3_MAX = 448.0


def to_fp8_pertensor(t: torch.Tensor):
    """Per-tensor symmetric fp8 e4m3 cast. Returns (fp8_tensor, scale) with
    t ~= fp8.float() * scale."""
    amax = t.abs().amax().clamp(min=1e-8)
    scale = (amax / E4M3_MAX).to(torch.float32)
    q = (t / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def conv3d_fp8_unfold(x: torch.Tensor, w: torch.Tensor):
    """3x3x3 stride-1 pad-1 conv via im2col + cuBLASLt fp8 GEMM.
    x: [N,C,D,H,W] f32/f16 ; w: [Co,Ci,3,3,3] -> y: [N,Co,D,H,W] f32."""
    N, Ci, D, H, W = x.shape
    Co = w.shape[0]
    K = Ci * 27
    M = N * D * H * W
    # im2col via padded gather (F.unfold is 2D-only): 27 shifted views of x.
    # Build A[M,K] DIRECTLY in fp8 tap-by-tap so the full-float im2col (27x input,
    # the OOM culprit) is never materialized. A single input-wide scale suffices: the
    # 27 taps are shifts of the same x, so amax over x bounds amax over every tap.
    xp = F.pad(x, (1, 1, 1, 1, 1, 1))
    sa = (x.abs().amax().clamp(min=1e-8) / E4M3_MAX).to(torch.float32)
    a_fp8 = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    a_view = a_fp8.view(N, D, H, W, 27, Ci)   # K = (tap outer, Ci inner)
    tap = 0
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                v = xp[:, :, dz:dz + D, dy:dy + H, dx:dx + W]          # [N,Ci,D,H,W]
                q = (v / sa).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
                a_view[:, :, :, :, tap, :] = q.permute(0, 2, 3, 4, 1)  # -> [N,D,H,W,Ci]
                tap += 1
    wmat = w.reshape(Co, Ci, 27).permute(0, 2, 1).reshape(Co, K).contiguous()  # [Co, K]
    # cuBLASLt wants B column-major [K,Co]: quantize contiguous [Co,K] then transpose-view.
    b_fp8_km, sb = to_fp8_pertensor(wmat)                                     # row-major [Co,K]
    b_fp8 = b_fp8_km.t()                                                      # col-major [K,Co] view
    y = torch._scaled_mm(a_fp8, b_fp8, scale_a=sa, scale_b=sb, out_dtype=torch.float32)
    return y.reshape(N, D, H, W, Co).permute(0, 4, 1, 2, 3).contiguous()


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
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    P = args.patch
    print(f"device={torch.cuda.get_device_name()} cap={torch.cuda.get_device_capability()} "
          f"patch={P}^3\n")
    print(f"{'layer':>14} | {'corr':>8} {'max|err|':>9} | "
          f"{'fp8 us':>8} {'fp16 us':>8} {'speedup':>7}")
    print("-" * 68)
    # representative ResEnc-UNet C->C 3x3x3 layers; N=1 patch
    for C in (64, 128, 256, 320):
        x = torch.randn(1, C, P, P, P, device=dev) * 0.5
        w = torch.randn(C, C, 3, 3, 3, device=dev) * (1.0 / (C * 27) ** 0.5)

        y_ref = F.conv3d(x.float(), w.float(), padding=1)
        y_fp8 = conv3d_fp8_unfold(x, w)

        corr = torch.corrcoef(torch.stack([y_fp8.flatten(), y_ref.flatten()]))[0, 1].item()
        maxe = (y_fp8 - y_ref).abs().amax().item() / (y_ref.abs().amax().item() + 1e-8)
        del y_ref, y_fp8
        torch.cuda.empty_cache()

        t_fp8 = bench(lambda: conv3d_fp8_unfold(x, w), args.iters)
        xh, wh = x.half(), w.half()
        t_fp16 = bench(lambda: F.conv3d(xh, wh, padding=1), args.iters)
        print(f"{C:>6}->{C:<6} | {corr:>8.4f} {maxe:>9.4f} | "
              f"{t_fp8:>8.1f} {t_fp16:>8.1f} {t_fp16/t_fp8:>6.2f}x")
        del x, w, xh, wh
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
