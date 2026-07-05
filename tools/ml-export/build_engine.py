# build_engine.py — checkpoint/ONNX -> static-shape fp16 TensorRT .plan for `fenix predict-surface`.
# Usage: build_engine.py <ckpt.pth|model.onnx> <out.plan> [patch=256] [batch=3] [ink]
# Engines are TRT-version + GPU-arch locked: build on the box that runs them (rebuild ~1min).
# Static shape: batch*32*patch^3 must stay under TRT's 2^31-element tensor cap -> 256^3 caps
# at batch 3 (measured optimum anyway: 86ms/patch flat across b1-b3 on the RTX PRO 6000).
import os
import sys
import time

import torch


def export_onnx(ckpt, path, patch, ink):
    from reference import build_and_load

    # Export a WEAK-TYPED fp32 ONNX and let the TRT builder's FP16 flag pick per-layer precision.
    # A strongly-typed fp16 ONNX (the old `.half()` export) makes TRT 11.x demand an fp16 conv3d
    # kernel that DOES NOT EXIST — the build fails with "No low-precision conv kernel available for
    # this strongly-typed Conv" / Error 10 at the stem. fp32 ONNX + BuilderFlag.FP16 keeps convs in
    # whatever precision has a kernel (fp16 where available, fp32 fallback where not) and builds.
    # Trace on the GPU (exporter executes the model; fp32 conv3d on a slow CPU is the 37-min path).
    # Trace at a SMALL spatial size with the spatial axes marked dynamic: an fp32 trace of the full
    # 256^3 patch OOMs a 24 GB card (fp32 activations ~2x the old .half() trace). The ONNX graph is
    # spatially generic (all convs are shape-agnostic), so the runtime 256^3 shape is pinned by the
    # TRT optimization profile below, not by the trace size. no_grad keeps activations out of memory.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_and_load(ckpt, ink=ink).eval().to(dev)
    trace_p = min(patch, 128)  # 128^3 bottoms at 2^3 (InstanceNorm3d needs >1 spatial elem); fp32 fits 24 GB
    x = torch.randn(1, 1, trace_p, trace_p, trace_p, dtype=torch.float32, device=dev)
    dyn = {0: "batch", 2: "d", 3: "h", 4: "w"}
    with torch.no_grad():
        torch.onnx.export(net, x, path, input_names=["x"], output_names=["y"],
                          dynamic_axes={"x": dyn, "y": dyn}, opset_version=17)
    print(f"onnx: {path} (traced at {trace_p}^3, spatial axes dynamic)")


def main():
    src, out = sys.argv[1], sys.argv[2]
    patch = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    ink = "ink" in sys.argv[5:]
    if batch * 32 * patch**3 >= 2**31:
        sys.exit(f"batch {batch} @ {patch}^3 exceeds TRT's 2^31-element cap (stem is 32ch)")

    onnx_path = src
    if not src.endswith(".onnx"):
        onnx_path = out + ".onnx"
        export_onnx(src, onnx_path, patch, ink)

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("onnx-parse:", parser.get_error(i))
            sys.exit(1)
    cfg = builder.create_builder_config()
    # Workspace pool: cap to a fraction of FREE VRAM, not a fixed 24 GB — on a 24 GB card (RTX 4090)
    # a 24 GB workspace leaves no room for the weights + tactic scratch and every tactic OOMs, failing
    # the build. Default to min(free*0.75, 12 GB); override with FENIX_TRT_WORKSPACE_GB.
    ws_gb = float(os.environ.get("FENIX_TRT_WORKSPACE_GB", "0"))
    if ws_gb <= 0:
        free_b, _ = torch.cuda.mem_get_info()
        ws_gb = min(free_b * 0.75 / (1 << 30), 12.0)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(ws_gb * (1 << 30)))
    # FP16 flag on a weak-typed fp32 network: TRT picks fp16 kernels where they exist (the speed win)
    # and falls back to fp32 where no low-precision kernel is available (e.g. some conv3d shapes) —
    # instead of failing the build like a strongly-typed fp16 ONNX does.
    cfg.set_flag(trt.BuilderFlag.FP16)
    print(f"trt: workspace pool {ws_gb:.1f} GB, FP16 flag set (weak-typed net)")
    prof = builder.create_optimization_profile()
    s = (batch, 1, patch, patch, patch)
    prof.set_shape("x", s, s, s)
    cfg.add_optimization_profile(prof)
    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, cfg)
    if blob is None:
        sys.exit("engine build FAILED")
    with open(out, "wb") as f:
        f.write(memoryview(blob))
    print(f"plan: {out} ({blob.nbytes / 1e6:.0f} MB, built in {time.perf_counter() - t0:.0f}s, "
          f"static [{batch},1,{patch},{patch},{patch}], FP16 flag)")


if __name__ == "__main__":
    main()
