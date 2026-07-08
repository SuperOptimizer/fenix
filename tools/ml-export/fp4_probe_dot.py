"""Probe tl.dot_scaled (MX fp4/fp8) on sm120: does it produce correct results, and does it
hit native block-scaled MMA (fast) or the bf16-upcast emulation (slow)?

Run FIRST after any Triton/driver change, before trusting fp4_conv3d_op. Prints a GEMM
correctness check (e4m3 x e2m1 vs f32 reference) and a speed comparison vs plain fp8
tl.dot at the same shape — native fp4 should be >= fp8; emulation will be ~5-10x slower.

Usage: python fp4_probe_dot.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time

import torch
import triton
import triton.language as tl

from fp4_conv3d_op import pack_weight_fp4, unpack_weight_fp4


@triton.jit
def _gemm_f8f4(a_ptr, b_ptr, bs_ptr, y_ptr, M, N, K,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # b_ptr: K-major packed [K//2, N]; bs_ptr: [K//32, N] — no in-kernel transpose
    # (tl.trans on packed-fp4 operands fails to lower on Triton 3.5.1).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    a_scale = tl.full((BM, BK // 32), 127, dtype=tl.uint8)   # 1.0
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :],
                    mask=(om < M)[:, None] & (ok < K)[None, :], other=0.0)
        okp = (k0 // 2) + tl.arange(0, BK // 2)
        b = tl.load(b_ptr + okp[:, None] * N + on[None, :],
                    mask=(okp < K // 2)[:, None] & (on < N)[None, :], other=0)
        oks = (k0 // 32) + tl.arange(0, BK // 32)
        bs = tl.load(bs_ptr + on[:, None] * (K // 32) + oks[None, :],
                     mask=(on < N)[:, None] & (oks < K // 32)[None, :], other=127)
        acc = tl.dot_scaled(a, a_scale, "e4m3", b, bs, "e2m1", acc=acc)
    tl.store(y_ptr + om[:, None] * N + on[None, :], acc,
             mask=(om < M)[:, None] & (on < N)[None, :])


@triton.jit
def _gemm_f8f8(a_ptr, b_ptr, y_ptr, M, N, K,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + om[:, None] * K + ok[None, :],
                    mask=(om < M)[:, None] & (ok < K)[None, :], other=0.0)
        b = tl.load(b_ptr + on[:, None] * K + ok[None, :],
                    mask=(on < N)[:, None] & (ok < K)[None, :], other=0.0)
        acc += tl.dot(a, b.T)
    tl.store(y_ptr + om[:, None] * N + on[None, :], acc,
             mask=(om < M)[:, None] & (on < N)[None, :])


def bench(fn, it=50, wu=10):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e6


def main():
    dev = "cuda"
    torch.manual_seed(0)
    M, N, K = 4096, 256, 1728        # ~ a mid-net conv GEMM shape
    a = torch.randn(M, K, device=dev) * 0.5
    wf = torch.randn(N, K // 27, 3, 3, 3, device=dev) * 0.05   # [N, Ci=K/27, 3,3,3]
    a8 = a.clamp(-448, 448).to(torch.float8_e4m3fn)
    bp, bs = pack_weight_fp4(wf)
    bref = unpack_weight_fp4(bp, bs, N, K).reshape(N, K)
    ref = a8.float() @ bref.t()
    bp_kn = bp.t().contiguous()      # K-major packed [K//2, N] (no in-kernel trans)

    y4 = torch.empty(M, N, dtype=torch.float32, device=dev)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 64))
    _gemm_f8f4[grid](a8, bp_kn, bs, y4, M, N, K, BM=128, BN=64, BK=64)
    corr = torch.corrcoef(torch.stack([y4.flatten(), ref.flatten()]))[0, 1].item()
    err = (y4 - ref).abs().max().item() / ref.abs().max().item()
    print(f"fp8xfp4 dot_scaled: corr vs packed-ref {corr:.6f}  maxrel {err:.2e}"
          f"  [{'OK' if corr > 0.9999 else 'BAD — investigate'}]")

    b8 = bref.clamp(-448, 448).to(torch.float8_e4m3fn)
    y8 = torch.empty(M, N, dtype=torch.float32, device=dev)
    _gemm_f8f8[grid](a8, b8, y8, M, N, K, BM=128, BN=64, BK=64)
    t4 = bench(lambda: _gemm_f8f4[grid](a8, bp_kn, bs, y4, M, N, K, BM=128, BN=64, BK=64))
    t8 = bench(lambda: _gemm_f8f8[grid](a8, b8, y8, M, N, K, BM=128, BN=64, BK=64))
    verdict = "NATIVE (good)" if t4 < t8 * 2 else "EMULATED (slow — fp4 not worth it via Triton)"
    print(f"speed: fp8xfp4 {t4:.0f}us  fp8xfp8 {t8:.0f}us  ratio {t4/t8:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
