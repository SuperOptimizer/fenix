"""Canonical fp8-resident whole-net benchmark + accuracy gate (max-perf sprint harness).

Measures the surface ResEnc-UNet 128^3 forward three ways on real CT:
  1. fp16 autocast + channels_last_3d  (the FAIR baseline -- NCHW eager is ~1.8x slower)
  2. fp8-resident fused execution      (fp8_forward.fp8_resident_patched)
  3. same, CUDA-graph captured         (all scale syncs eliminated -> capturable)
and checks corr + SurfaceDice@2 (bar 0.998) of the graphed output vs the fp16 reference.

Sprint result 2026-07-06: 109ms -> 41.0ms (2.66x), SD 1.0000. See
docs/design/fp8-conv3d-sm120.md for the step-by-step lever table and the correctness traps
(TD>=4 Triton miscompile, autotune atomic pollution, transpconv bias non-absorption).

Usage: python fp8_resident_bench.py   (expects /tmp/ct/patch.npy; see fp8_forward.py)
"""
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, time, statistics
import fp8_forward as ff
from reference import build_and_load
dev='cuda'
net=build_and_load('/home/forrest/fenix/models/surface_recto_3dunet/checkpoint_inference_ready.pth').to(dev).eval().to(memory_format=torch.channels_last_3d)
x=ff.load_ct_patch('/tmp/ct/patch.npy',128,dev).to(memory_format=torch.channels_last_3d)
def bench(fn,it=10,wu=3):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        torch.cuda.synchronize(); t0=time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    return statistics.median(ts)
def corr(a,b): return torch.corrcoef(torch.stack([a.flatten().float(),b.flatten().float()]))[0,1].item()
with torch.no_grad():
    with torch.autocast("cuda",dtype=torch.float16):
        y16=net(x).float().softmax(1)[:,1]; t16=bench(lambda: net(x))
    with ff.fp8_conv3d_patched(): net(x)
    ff.freeze_calibration()
    with ff.fp8_resident_patched(net), torch.autocast("cuda",dtype=torch.float16):
        t8=bench(lambda: net(x))
        # CUDA graph capture
        sx=x.clone()
        for _ in range(3): net(sx)                     # warm autotune/allocs
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            sy=net(sx)
        def run(inp):
            sx.copy_(inp); g.replay(); return sy
        tg=bench(lambda: run(x))
        yg=run(x).float().softmax(1)[:,1]
print(f"autocast-CL {t16:.1f}ms | fp8-res {t8:.1f}ms | fp8-res+GRAPH {tg:.1f}ms | speedup {t16/tg:.2f}x")
sd=ff.surface_dice(yg,y16,tol_vox=2)
print(f"graph corr {corr(yg,y16):.5f} SurfaceDice@2 {sd:.4f} [{'PASS' if sd>=0.998 else 'FAIL'}]")
from fp8_conv3d_op import dump_tuned
n=dump_tuned("/tmp/fp8_tuned_configs.json")
print(f"pinned {n} tuned conv configs -> /tmp/fp8_tuned_configs.json (load_tuned() skips the sweep)")
