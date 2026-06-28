# ml-export — ScrollPrize checkpoints → fenix C++ inference

Offline Python tooling (a venv, **not** part of the build) that converts the published
ScrollPrize PyTorch checkpoints into the artifacts fenix's `src/ml/` C++ runtime consumes.
fenix **reimplements** the network graphs in `torch::nn` (we own the architecture; libtorch is
just the tensor/CUDA library) and loads the trained tensors from a hand-rolled `.fxweights`
file — so there is no Python at inference time.

## Setup

```sh
python3 -m venv .venv-ml && . .venv-ml/bin/activate
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130   # match system CUDA + libtorch
pip install -r tools/ml-export/requirements.txt
pip install dynamic-network-architectures        # only for reference.py (validation)
```

## Scripts

| script | purpose |
|--------|---------|
| `introspect.py <ckpt.pth>` | dump every weight tensor's name/shape/dtype (the arch spec) |
| `convert_weights.py <ckpt.pth> <out.fxweights>` | write the flat `.fxweights` + a `.toml` registry entry |
| `reference.py {gen,run,cmp}` | authoritative PyTorch reference (real upstream blocks) to validate the C++ reimpl |

## Surface model — end-to-end

```sh
# 1. download + convert
python -c "from huggingface_hub import hf_hub_download as g; [g('scrollprize/surface_recto_3dunet', f, local_dir='models/surface_recto_3dunet') for f in ('config.json','checkpoint_inference_ready.pth')]"
python tools/ml-export/convert_weights.py \
    models/surface_recto_3dunet/checkpoint_inference_ready.pth \
    models/surface_recto_3dunet/surface.fxweights \
    --key model --config models/surface_recto_3dunet/config.json --norm zscore

# 2. validate the C++ reimplementation is numerically exact (bit-identical on GPU)
python tools/ml-export/reference.py gen 128 128 128 /tmp/in.raw
python tools/ml-export/reference.py run models/surface_recto_3dunet/checkpoint_inference_ready.pth /tmp/in.raw 128 128 128 /tmp/ref.raw
./build-ml/fenix ml run-raw models/surface_recto_3dunet/surface.fxweights /tmp/in.raw /tmp/cpp.raw 128 128 128
python tools/ml-export/reference.py cmp /tmp/ref.raw /tmp/cpp.raw     # -> VERDICT: MATCH

# 3. real inference
./build-ml/fenix predict-surface in.nrrd models/surface_recto_3dunet/surface.fxweights out.nrrd 256 0.5
```

**VRAM note:** a 256³ patch through this net needs >8 GB. On an 8 GB card (e.g. RTX 4060) pass
a smaller patch, e.g. `... out.nrrd 128 0.5` (fully-convolutional, so any multiple of 64 works).

## The `.fxweights` format
`"FXWT" | u32 version | u32 count | u32 reserved`, then per tensor
`u16 name_len | name | u8 dtype | u8 ndim | u64 nbytes | u64 data_off(64-aligned) | i64 shape[ndim]`,
then the raw little-endian blobs. Read by `src/ml/weights.hpp`.
