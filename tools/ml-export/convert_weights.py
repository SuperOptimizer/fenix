#!/usr/bin/env python3
"""convert_weights.py — convert a ScrollPrize checkpoint into fenix's `.fxweights` file.

fenix's C++ ml/ module reimplements the network with torch::nn and loads the trained tensors
by name. We don't want a runtime dependency on Python pickle / torch serialization, so we
write a hand-rolled, mmap-friendly flat file (the fenix way):

  magic "FXWT" | u32 version | u32 count | u32 reserved
  count × manifest entry:  u16 name_len | name | u8 dtype | u8 ndim | u64 nbytes |
                           u64 data_offset (abs, 64-aligned) | i64 shape[ndim]
  data blobs at their offsets, raw little-endian, each 64-byte aligned.

dtype codes: 0=f32 1=f16 2=bf16 3=f64 4=i64 5=i32 6=u8 7=i16 8=bool

Also emits a TOML model-registry entry (architecture + I/O + normalization) next to it.

  usage: convert_weights.py <checkpoint.pth> <out.fxweights> [--key model|ema_model]
                            [--config config.json] [--toml out.toml]
"""
import argparse, json, os, struct, sys
import numpy as np
import torch

DTYPE = {  # torch dtype -> (code, numpy dtype)
    torch.float32: (0, "<f4"), torch.float16: (1, "<f2"), torch.bfloat16: (2, None),
    torch.float64: (3, "<f8"), torch.int64: (4, "<i8"), torch.int32: (5, "<i4"),
    torch.uint8: (6, "|u1"), torch.int16: (7, "<i2"), torch.bool: (8, "|u1"),
}
ALIGN = 64


def to_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        # store bf16 as raw uint16 little-endian (no numpy bf16)
        return t.view(torch.int16).numpy().astype("<i2").tobytes()
    np_dt = DTYPE[t.dtype][1]
    return t.numpy().astype(np_dt, copy=False).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--key", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--toml", default=None)
    ap.add_argument("--arch", default="resenc_unet")
    ap.add_argument("--norm", default="zscore", help="zscore | percentile_minmax")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = None
    if args.key:
        sd = ck[args.key]
    else:
        for k in ("ema_model", "model", "state_dict", "network_weights"):
            if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
                sd, args.key = ck[k], k
                break
    if sd is None:
        sys.exit("could not locate state_dict; pass --key")

    items = [(n, t) for n, t in sd.items() if torch.is_tensor(t)]
    for _, t in items:
        if t.dtype not in DTYPE:
            sys.exit(f"unsupported dtype {t.dtype}")

    # Pass 1: manifest sizes -> data offsets. Header + manifest first, then aligned blobs.
    head = 4 + 4 + 4 + 4
    man = head
    for n, t in items:
        man += 2 + len(n.encode()) + 1 + 1 + 8 + 8 + 8 * t.dim()
    off = (man + ALIGN - 1) // ALIGN * ALIGN
    blobs, offsets = [], []
    for n, t in items:
        b = to_bytes(t)
        offsets.append(off)
        blobs.append(b)
        off = (off + len(b) + ALIGN - 1) // ALIGN * ALIGN

    nbytes = [len(b) for b in blobs]
    with open(args.out, "wb") as f:
        f.write(b"FXWT")
        f.write(struct.pack("<III", 1, len(items), 0))
        for (n, t), o, nb_ in zip(items, offsets, nbytes):
            name = n.encode()
            f.write(struct.pack("<H", len(name))); f.write(name)
            f.write(struct.pack("<BB", DTYPE[t.dtype][0], t.dim()))
            f.write(struct.pack("<QQ", nb_, o))
            for s in t.shape:
                f.write(struct.pack("<q", int(s)))
        for o, b in zip(offsets, blobs):
            f.seek(o)
            f.write(b)
    print(f"wrote {args.out}: {len(items)} tensors, {sum(nbytes)/1e6:.1f} MB data")

    # TOML registry entry.
    cfg = {}
    if args.config and os.path.exists(args.config):
        cfg = json.load(open(args.config))
    toml_path = args.toml or os.path.splitext(args.out)[0] + ".toml"
    patch = cfg.get("patch_size", [256, 256, 256])
    in_ch = cfg.get("in_channels", 1)
    out_ch = next(iter(cfg.get("targets", {}).values()), {}).get("out_channels", 2) if cfg else 2
    with open(toml_path, "w") as f:
        f.write(f'name = "{os.path.splitext(os.path.basename(args.out))[0]}"\n')
        f.write(f'arch = "{args.arch}"\n')
        f.write(f'weights = "{os.path.basename(args.out)}"\n')
        f.write(f'in_channels = {in_ch}\n')
        f.write(f'out_channels = {out_ch}\n')
        f.write(f'patch_size = [{patch[0]}, {patch[1]}, {patch[2]}]\n')
        f.write(f'normalization = "{args.norm}"\n')
        if cfg:
            f.write(f'features_per_stage = {cfg.get("features_per_stage", [])}\n')
            f.write(f'n_blocks_per_stage = {cfg.get("n_blocks_per_stage", [])}\n')
            f.write(f'strides = {cfg.get("strides", [])}\n')
    print(f"wrote {toml_path}")


if __name__ == "__main__":
    main()
