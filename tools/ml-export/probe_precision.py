#!/usr/bin/env python3
"""Probe what the installed torch+cuDNN can ACTUALLY do on this GPU: bf16/fp16/fp8
conv3d + matmul paths, with timings. Run after any torch upgrade; paste results into
docs/design/training-pipeline.md's precision table."""
import time, torch

def bench(fn, warm=3, iters=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1e3

print(f"torch {torch.__version__}  cuda {torch.version.cuda}  cudnn {torch.backends.cudnn.version()}")
print(f"gpu: {torch.cuda.get_device_name(0)}  cc {torch.cuda.get_device_capability(0)}")
dev = "cuda"
x32 = torch.randn(4, 32, 64, 64, 64, device=dev)
conv = torch.nn.Conv3d(32, 32, 3, padding=1).to(dev)
print(f"conv3d fp32: {bench(lambda: conv(x32)):7.2f} ms")
for dt, name in [(torch.bfloat16, "bf16"), (torch.float16, "fp16")]:
    c, x = conv.to(dt), x32.to(dt)
    print(f"conv3d {name}: {bench(lambda: c(x)):7.2f} ms")
# fp8: try the dtypes + a conv; report exactly what works
for dt in ("float8_e4m3fn", "float8_e5m2"):
    if not hasattr(torch, dt):
        print(f"{dt}: dtype ABSENT"); continue
    try:
        c8 = conv.to(getattr(torch, dt)); x8 = x32.to(getattr(torch, dt))
        ms = bench(lambda: c8(x8))
        print(f"conv3d {dt}: {ms:7.2f} ms  <-- WORKS")
    except Exception as e:
        print(f"conv3d {dt}: no ({type(e).__name__}: {str(e)[:80]})")
try:
    a = torch.randn(4096, 4096, device=dev).to(torch.float8_e4m3fn)
    b = torch.randn(4096, 4096, device=dev).to(torch.float8_e4m3fn).t()
    sa = torch.tensor(1.0, device=dev); sb = torch.tensor(1.0, device=dev)
    ms = bench(lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
    print(f"_scaled_mm fp8 4096^2: {ms:7.2f} ms  <-- WORKS")
except Exception as e:
    print(f"_scaled_mm fp8: no ({type(e).__name__}: {str(e)[:80]})")
