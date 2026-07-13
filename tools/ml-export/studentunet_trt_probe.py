#!/usr/bin/env python3
"""StudentUNet TRT probe: batch sweep + quantized (int8/fp8 QDQ) engines, accuracy-gated.

Follows the studentunet_fp8_gate verdict (TRT fp16 b1 = 29.5ms/patch, the deploy lane).
Two questions:
  1. Does batching fill the GPU better? (11M params at b1 underfills.) fp16 engine at
     b1/2/4/8, ms/patch = wall/batch.
  2. Do TRT-11 QDQ engines (modelopt PTQ) beat fp16 now, on Blackwell? The old
     "quantized conv3d dead in the export toolchain" verdict predates this stack.
     Unlike trt_probe.py (random calib, speed-only), calibration here uses REAL CT
     patches (memory: synthetic noise is OOD and misleads fp8 accuracy) and every
     engine is SD@2-gated vs the eager fp32 reference on a held-out crop.

Usage: studentunet_trt_probe.py [--ckpt ...] [--base 32] [--crops /tmp/gtqc/m8/eval]
"""
import sys, os, glob, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import numpy as np
import torch

import trt_probe as tp
import fp8_forward as ff
from train import StudentUNet


def load(ckpt, base):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st.get("ema", st.get("net")) if isinstance(st, dict) else st
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    n = StudentUNet(base=base)
    n.load_state_dict(sd)
    return n.eval()


def ct_patch(crops, crop, dev="cuda"):
    ct = np.load(f"{crops}/{crop}/ct.npy")[:128, :128, :128].astype(np.float32) / 255.0
    return torch.from_numpy(ct)[None, None].to(dev)


def engine_prob(eng, x_half):
    ctx = eng.create_execution_context()
    b = x_half.shape[0]
    ctx.set_input_shape("x", tuple(x_half.shape))
    y = torch.empty(tuple(ctx.get_tensor_shape("y")), device="cuda", dtype=x_half.dtype)
    ctx.set_tensor_address("x", x_half.data_ptr())
    ctx.set_tensor_address("y", y.data_ptr())
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        ctx.execute_async_v3(s.cuda_stream)
    s.synchronize()
    return y.float().softmax(1)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/gtqc/real/studentM_best.pt")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--crops", default="/tmp/gtqc/m8/eval")
    ap.add_argument("--outdir", default="/tmp/gtqc/real")
    args = ap.parse_args()

    x = ct_patch(args.crops, "crop00")
    ref = load(args.ckpt, args.base).cuda().float()
    with torch.no_grad():
        pr = ref(x).softmax(1)[:, 1]

    print("=== 1. fp16 engine batch sweep ===")
    onnx16 = f"{args.outdir}/studentM_fp16.onnx"
    if not os.path.exists(onnx16):
        tp.export_onnx(load(args.ckpt, args.base).cuda(), onnx16, 128, True)
    for b in (1, 2, 4, 8):
        eng = tp.build_engine(onnx16, 128, b, f"fp16-b{b}")
        if eng is None:
            continue
        ms = tp.bench_engine(eng, b, 128, True)
        pt = engine_prob(eng, x.half().expand(b, -1, -1, -1, -1).contiguous())[0]
        sd2 = ff.surface_dice(pt, pr[0], tol_vox=2)
        print(f"fp16 b{b}: {ms/b:.1f} ms/patch (wall {ms:.1f})  SD@2 {sd2:.4f}")

    print("=== 2. QDQ engines (real-CT calibration) ===")
    calib = [ct_patch(args.crops, os.path.basename(c))
             for c in sorted(glob.glob(f"{args.crops}/crop*"))[:6]]

    import modelopt.torch.quantization as mtq
    for mode, cfgname in (("int8", "INT8_DEFAULT_CFG"), ("fp8", "FP8_DEFAULT_CFG")):
        cfg = getattr(mtq, cfgname, None)
        if cfg is None:
            print(f"{mode}: {cfgname} missing — skip")
            continue
        net = load(args.ckpt, args.base).cuda().float()

        def loop(m):
            with torch.no_grad():
                for c in calib:
                    m(c)

        onnx_q = f"{args.outdir}/studentM_{mode}.onnx"
        try:
            q = mtq.quantize(net, cfg, loop)
            torch.onnx.export(q, x, onnx_q, input_names=["x"], output_names=["y"],
                              opset_version=17, dynamo=False)
        except Exception as e:  # noqa: BLE001 — probe: the failure IS the verdict
            print(f"{mode}: quantize/export failed — {type(e).__name__}: {e}")
            continue
        for b in (1, 4):
            eng = tp.build_engine(onnx_q, 128, b, f"{mode}-b{b}")
            if eng is None:
                break
            ms = tp.bench_engine(eng, b, 128, False)  # QDQ ONNX is fp32-typed I/O
            pt = engine_prob(eng, x.float().expand(b, -1, -1, -1, -1).contiguous())[0]
            sd2 = ff.surface_dice(pt, pr[0], tol_vox=2)
            gate = "PASS" if sd2 >= 0.998 else "FAIL"
            print(f"{mode} b{b}: {ms/b:.1f} ms/patch (wall {ms:.1f})  SD@2 {sd2:.4f} [{gate}]")


if __name__ == "__main__":
    main()
