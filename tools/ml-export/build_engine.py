# build_engine.py — checkpoint/ONNX -> static-shape fp16 TensorRT .plan for `fenix predict-surface`.
# Usage: build_engine.py <ckpt.pth|model.onnx> <out.plan> [patch=256] [batch=3] [ink]
# Engines are TRT-version + GPU-arch locked: build on the box that runs them (rebuild ~1min).
# Static shape: batch*32*patch^3 must stay under TRT's 2^31-element tensor cap -> 256^3 caps
# at batch 3 (measured optimum anyway: 86ms/patch flat across b1-b3 on the RTX PRO 6000).
import sys
import time

import torch


def export_fp16_onnx(ckpt, path, patch, ink):
    from reference import build_and_load

    # trace on the GPU: the exporter EXECUTES the model, and fp16 conv3d on CPU is
    # emulated (a 37-min export on a slow-clock host vs seconds on the card)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_and_load(ckpt, ink=ink).eval().half().to(dev)
    x = torch.randn(1, 1, patch, patch, patch, dtype=torch.float16, device=dev)
    torch.onnx.export(net, x, path, input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, opset_version=17)
    print(f"onnx: {path}")


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
        export_fp16_onnx(src, onnx_path, patch, ink)

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
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 24 << 30)
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
          f"static [{batch},1,{patch},{patch},{patch}] fp16)")


if __name__ == "__main__":
    main()
