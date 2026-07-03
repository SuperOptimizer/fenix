# 0010 — TensorRT engine path for bulk ML inference

- Status: accepted
- Date: 2026-07-03
- Deciders: forrest, Claude

## Context

The tta=48 teacher sweep over the training corpus is the largest GPU bill ahead. The TRT
probe (`tools/ml-export/trt_probe.py`, RTX PRO 6000) measured a TensorRT fp16 engine at
86 ms/patch @256³ vs 131.7 eager-best — 1.53× — while every quantized conv3d lane (torchao
int8, TRT int8 QDQ, fp8, fp4) is dead in the NVIDIA toolchain (no kernels / unexportable).
forrest exempted the ML island from the minimal-deps rule (root CLAUDE.md §2.5): NVIDIA
ML-stack libs are pre-approved; the rule guards the core C++ stack.

## Decision

`fenix predict-surface` accepts a serialized TensorRT engine (`.plan`/`.engine`) as the
weights argument, dispatched through a `TrtNet` adapter (`src/ml/trt_engine.hpp`) into the
same net-generic `run_predict_core` machinery as `.fxweights`/`.ts`. TRT headers are parsed
ONLY in `inference.cpp` (the existing ML-firewall TU). CMake: `-DFENIX_TRT_LIB=<libnvinfer.so>`
`-DFENIX_TRT_INCLUDE=<headers>` (optional; absent ⇒ the branch compiles out). Engines are
built by `tools/ml-export/build_engine.py` (checkpoint → fp16 ONNX → static-shape plan).

## Consequences

- Engines are STATIC-shape (batch·32·patch³ < 2³¹ — TRT's tensor cap: 256³ ⇒ batch ≤3) and
  TRT-version + GPU-arch locked. This matches the project's no-compat philosophy: rebuild
  per box (~1 min), deserialize fails loudly on mismatch. The engine's batch/patch override
  the CLI's.
- Measured end-to-end on a 768³ volume: 24.6 s → 14.9 s (1.65× incl. IO); output agreement
  vs eager fp16: SurfaceDice@2 = 0.9987, Dice = 0.984 (fp16 reorder noise — tolerance-only).
- Headers ship no runtime: pip wheels provide `libnvinfer.so`, public headers come from the
  NVIDIA/TensorRT OSS repo at the matching release (NvInferVersion.h placeholders patched).
- Rejected alternatives: quantized conv3d engines (unbuildable, see model-registry.md);
  custom CUTLASS fp8/fp4 conv3d kernels (pioneering unsupported territory — parked);
  standalone Python runner (would fork the tiling/TTA/resume machinery fenix already has).
