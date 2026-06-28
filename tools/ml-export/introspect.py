#!/usr/bin/env python3
"""introspect.py — dump the exact parameter layout of a ScrollPrize checkpoint.

The fenix C++ ml/ module reimplements these networks with torch::nn (we own the graph; libtorch
is just the tensor/conv/CUDA library). To do that 1:1 we need the authoritative list of every
tensor name + shape + dtype in the checkpoint. This script prints that and writes it to JSON so
the C++ reimpl and the weight converter agree on names.

  usage: introspect.py <checkpoint.pth> [--key model|ema_model] [--json out.json]

The checkpoint's network weights live under a top-level key (`model` for surface, `ema_model`
for ink). We load weights_only=False because the file also pickles a config dict; the weights
themselves are a plain tensor OrderedDict, so no model class is needed just to read shapes.
"""
import argparse, json, sys
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--key", default=None, help="state-dict key (auto-detected if omitted)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Find the state dict: an explicit key, else the first dict-of-tensors value.
    sd = None
    if args.key:
        sd = ck[args.key]
    elif isinstance(ck, dict):
        for k in ("ema_model", "model", "state_dict"):
            if k in ck and isinstance(ck[k], dict):
                sd, args.key = ck[k], k
                break
        if sd is None and all(torch.is_tensor(v) for v in ck.values()):
            sd, args.key = ck, "(root)"
    if sd is None:
        sys.exit("could not locate a state_dict; pass --key")

    print(f"# state_dict key: {args.key}   tensors: {len(sd)}")
    entries, total = [], 0
    for name, t in sd.items():
        if not torch.is_tensor(t):
            continue
        shape = list(t.shape)
        n = t.numel()
        total += n
        entries.append({"name": name, "shape": shape, "dtype": str(t.dtype).replace("torch.", "")})
        print(f"{name:60s} {str(shape):26s} {str(t.dtype).replace('torch.',''):8s} {n}")
    print(f"# total parameters: {total:,}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"key": args.key, "count": len(entries), "total_params": total,
                       "tensors": entries}, f, indent=2)
        print(f"# wrote {args.json}")


if __name__ == "__main__":
    main()
