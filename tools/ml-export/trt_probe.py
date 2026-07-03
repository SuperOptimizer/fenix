# trt_probe.py — the TensorRT decision gate: is int8/fp16 TRT >=1.5x over eager bf16?
# Usage: trt_probe.py <checkpoint.pth> [patch=256] [batch=6]
# Exports the surface ResEnc-UNet to ONNX, builds fp16 and int8 engines, benches them
# against eager fp16/bf16 forward at the same shapes. Speed probe only — int8 uses a
# random-data calibrator, so outputs are NOT accuracy-valid (QAT covers that if we adopt).
import sys
import time

import numpy as np
import torch

from reference import build_and_load


def bench_eager(net, dev, shape, dtype, n=12):
    x = torch.randn(shape, device=dev, dtype=torch.float32)
    net = net.to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for _ in range(3):
            net(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            net(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def export_onnx(net, path, patch):
    x = torch.randn(1, 1, patch, patch, patch)
    torch.onnx.export(
        net,
        x,
        path,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        opset_version=17,
    )
    print(f"onnx: {path}")


class RandCalib:  # random-data int8 calibrator: speed probe only
    def __init__(self, trt, shape, n=8):
        import tensorrt

        class C(tensorrt.IInt8EntropyCalibrator2):
            def __init__(c):
                super().__init__()
                c.i = 0
                c.buf = torch.randn(shape, device="cuda")

            def get_batch_size(c):
                return shape[0]

            def get_batch(c, names):
                if c.i >= n:
                    return None
                c.i += 1
                return [int(c.buf.data_ptr())]

            def read_calibration_cache(c):
                return None

            def write_calibration_cache(c, cache):
                pass

        self.c = C()


def build_engine(onnx_path, patch, max_batch, int8):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("onnx-parse:", parser.get_error(i))
            sys.exit(1)
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 24 << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    if int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = RandCalib(trt, (1, 1, patch, patch, patch)).c
    prof = builder.create_optimization_profile()
    s = (patch, patch, patch)
    prof.set_shape("x", (1, 1) + s, (max_batch, 1) + s, (max_batch, 1) + s)
    cfg.add_optimization_profile(prof)
    if int8:
        cfg.set_calibration_profile(prof)
    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, cfg)
    if blob is None:
        print("build FAILED", "int8" if int8 else "fp16")
        return None
    print(f"engine ({'int8' if int8 else 'fp16'}): built in {time.perf_counter() - t0:.0f}s")
    return trt.Runtime(logger).deserialize_cuda_engine(blob)


def bench_engine(engine, batch, patch, n=12):
    import tensorrt as trt  # noqa: F401

    ctx = engine.create_execution_context()
    ctx.set_input_shape("x", (batch, 1, patch, patch, patch))
    x = torch.randn(batch, 1, patch, patch, patch, device="cuda")
    y = torch.empty(tuple(ctx.get_tensor_shape("y")), device="cuda")
    ctx.set_tensor_address("x", x.data_ptr())
    ctx.set_tensor_address("y", y.data_ptr())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(n):
            ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ckpt = sys.argv[1]
    patch = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    net = build_and_load(ckpt).eval()

    print(f"=== eager baselines (patch={patch}) ===")
    res = {}
    for b in (1, batch):
        for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            ms = bench_eager(net, "cuda", (b, 1, patch, patch, patch), dt)
            res[f"eager-{name}-b{b}"] = ms
            print(f"eager {name} b{b}: {ms:.1f}ms  ({ms / b:.1f}ms/patch)")
    net = net.cpu()
    torch.cuda.empty_cache()

    onnx_path = "/root/surface.onnx"
    export_onnx(net, onnx_path, patch)
    del net
    for mode in ("fp16", "int8"):
        eng = build_engine(onnx_path, patch, batch, int8=mode == "int8")
        if eng is None:
            continue
        for b in (1, batch):
            ms = bench_engine(eng, b, patch)
            res[f"trt-{mode}-b{b}"] = ms
            print(f"trt {mode} b{b}: {ms:.1f}ms  ({ms / b:.1f}ms/patch)")
        del eng
        torch.cuda.empty_cache()

    base = res.get(f"eager-bf16-b{batch}")
    print("=== VERDICT (gate: >=1.5x over eager bf16 to adopt) ===")
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        print(f"  {k}: {v:.1f}ms  {base / v:.2f}x")


if __name__ == "__main__":
    main()
