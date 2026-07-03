# trt_probe.py — the TensorRT decision gate: is a quantized TRT engine >=1.5x over eager?
# Usage: trt_probe.py <checkpoint.pth> [patch=256] [batch=6]
# TRT >=11 is strongly typed (no FP16/INT8 builder flags, no implicit calibration):
# fp16 comes from a half-precision ONNX export; int8/fp8 need QDQ nodes, inserted by
# nvidia-modelopt with a random-data forward loop. Speed probe only — quantized outputs
# are NOT accuracy-valid (QAT covers that if we adopt).
import sys
import time

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


def export_onnx(net, path, patch, half):
    net = net.half() if half else net.float()
    x = torch.randn(1, 1, patch, patch, patch, dtype=torch.float16 if half else torch.float32)
    torch.onnx.export(
        net.cpu(),
        x,
        path,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        opset_version=17,
    )
    print(f"onnx: {path} ({'fp16' if half else 'fp32'})")


def quantize_qdq(net, mode, patch, path):
    """modelopt QDQ insertion (int8/fp8/fp4) + ONNX export. Returns False if unsupported."""
    try:
        import modelopt.torch.quantization as mtq
    except ImportError:
        print(f"{mode}: nvidia-modelopt not installed — skipping")
        return False
    cfgs = {"int8": "INT8_DEFAULT_CFG", "fp8": "FP8_DEFAULT_CFG", "fp4": "NVFP4_DEFAULT_CFG"}
    cfg = getattr(mtq, cfgs[mode], None)
    if cfg is None:
        print(f"{mode}: {cfgs[mode]} not in this modelopt — skipping")
        return False
    net = net.float().cuda().eval()

    def loop(m):
        with torch.no_grad():
            for _ in range(4):
                m(torch.randn(1, 1, patch, patch, patch, device="cuda"))

    try:
        q = mtq.quantize(net, cfg, loop)
        x = torch.randn(1, 1, patch, patch, patch, device="cuda")
        # dynamo exporter can't trace modelopt's QDQ ops — use the legacy exporter
        torch.onnx.export(
            q,
            x,
            path,
            input_names=["x"],
            output_names=["y"],
            opset_version=17,
            dynamo=False,
        )
        print(f"onnx: {path} ({mode} QDQ)")
        return True
    except Exception as e:  # noqa: BLE001 — probe: any failure is itself the verdict
        print(f"{mode}: quantize/export failed — {type(e).__name__}: {e}")
        return False


def build_engine(onnx_path, patch, max_batch, tag):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # strongly typed, explicit batch
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("onnx-parse:", parser.get_error(i))
            return None
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 24 << 30)
    # STATIC batch: dynamic-to-max profiles push the 32ch stem activation past TRT's
    # 2^31-element tensor limit (32*256^3*6 = 3.2G) -> zero conv3d tactics, build fails.
    prof = builder.create_optimization_profile()
    s = (max_batch, 1, patch, patch, patch)
    prof.set_shape("x", s, s, s)
    cfg.add_optimization_profile(prof)
    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, cfg)
    if blob is None:
        print(f"engine ({tag}): build FAILED")
        return None
    print(f"engine ({tag}): built in {time.perf_counter() - t0:.0f}s")
    return trt.Runtime(logger).deserialize_cuda_engine(blob)


def bench_engine(engine, batch, patch, in_half, n=12):
    ctx = engine.create_execution_context()
    ctx.set_input_shape("x", (batch, 1, patch, patch, patch))
    dt = torch.float16 if in_half else torch.float32
    x = torch.randn(batch, 1, patch, patch, patch, device="cuda", dtype=dt)
    y = torch.empty(tuple(ctx.get_tensor_shape("y")), device="cuda", dtype=dt)
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
    res = {}

    print(f"=== eager baselines (patch={patch}) ===")
    net = build_and_load(ckpt).eval()
    for b in (1, batch):
        for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            ms = bench_eager(net, "cuda", (b, 1, patch, patch, patch), dt)
            res[f"eager-{name}-b{b}"] = ms
            print(f"eager {name} b{b}: {ms:.1f}ms  ({ms / b:.1f}ms/patch)")
    net = net.cpu()
    torch.cuda.empty_cache()

    # (path, tag, half-io, exporter)
    jobs = [("/root/surface_fp16.onnx", "fp16", True, lambda p: (export_onnx(net, p, patch, True), True)[1])]
    for mode in ("int8", "fp8", "fp4"):
        jobs.append(
            (
                f"/root/surface_{mode}.onnx",
                mode,
                False,
                lambda p, m=mode: quantize_qdq(build_and_load(ckpt).eval(), m, patch, p),
            )
        )
    for path, tag, half_io, make in jobs:
        if not make(path):
            continue
        torch.cuda.empty_cache()
        eng = build_engine(path, patch, batch, tag)
        if eng is None:
            continue
        ms = bench_engine(eng, batch, patch, half_io)
        res[f"trt-{tag}-b{batch}"] = ms
        print(f"trt {tag} b{batch}: {ms:.1f}ms  ({ms / batch:.1f}ms/patch)")
        del eng
        torch.cuda.empty_cache()

    base = res.get(f"eager-fp16-b{batch}")
    print("=== VERDICT (gate: >=1.5x over eager fp16 to adopt) ===")
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        print(f"  {k}: {v:.1f}ms  {base / v:.2f}x  ({v / (batch if k.endswith(f'b{batch}') else 1):.1f}ms/patch)")


if __name__ == "__main__":
    main()
