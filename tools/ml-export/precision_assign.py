"""Per-layer precision auto-assignment: find the cheapest lane each conv tolerates.

Greedy sensitivity scan over the QAT layers: start all-int8 (static-calibrated affine),
measure whole-net SD@2 vs the fp16 teacher on real CT; while below the gate, flip the
most-suspect layers back to fp8 (deepest-error-first via per-layer output corr) until
the gate passes. Output: a per-layer lane map (JSON) — the mixed-precision deploy
config where only the layers that NEED fp8 pay for it (on Ampere they'd get fp16).

Run on the idle GPU: CUDA_VISIBLE_DEVICES=1 python precision_assign.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy

import torch

import fp8_forward as ff
from fp8_conv3d_op import load_tuned
from fp8_train import Fp8Conv3dLayer, int8_calibrate, set_int8_qat, swap_convs_fp8
from reference import build_and_load

GATE = 0.998


def sd_of(net8, net, x):
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            yr = net(x).float().softmax(1)[:, 1]
        y8 = net8(x).float().softmax(1)[:, 1]
    return ff.surface_dice(y8, yr, tol_vox=2)


def main():
    load_tuned(os.path.expanduser("~/.cache/fenix-fp8-tuned-train.json"))
    torch.manual_seed(0)
    ckpt = ("/home/forrest/fenix/models/surface_recto_3dunet/"
            "checkpoint_inference_ready.pth")
    net = build_and_load(ckpt).cuda().eval()
    net8 = copy.deepcopy(net)
    swap_convs_fp8(net8)
    set_int8_qat(net8)
    net8.eval()
    x = ff.load_ct_patch("/tmp/ct/patch.npy", 128, "cuda")
    int8_calibrate(net8, [x])

    layers = {n: m for n, m in net8.named_modules() if isinstance(m, Fp8Conv3dLayer)}
    base = sd_of(net8, net, x)
    print(f"all-int8 (static affine): SD@2={base:.4f} over {len(layers)} layers")

    # per-layer sensitivity: flip ONE layer to fp8, measure SD gain
    gains = []
    for i, (n, m) in enumerate(layers.items()):
        m.int8_fwd = False
        gains.append((sd_of(net8, net, x) - base, n))
        m.int8_fwd = True
        if i % 30 == 0:
            print(f"  scanned {i}/{len(layers)}")
    gains.sort(reverse=True)
    print("top-8 most int8-sensitive layers:")
    for g, n in gains[:8]:
        print(f"  {g:+.5f} {n[:70]}")

    # greedy: promote highest-gain layers to fp8 until the gate passes
    promoted = []
    cur = base
    for g, n in gains:
        if cur >= GATE:
            break
        layers[n].int8_fwd = False
        promoted.append(n)
        cur = sd_of(net8, net, x)
        print(f"  promote {n[:60]} -> fp8 : SD {cur:.4f}")
    lane_map = {n: ("fp8" if n in promoted else "int8") for n in layers}
    out = os.path.expanduser("~/.cache/fenix-precision-map.json")
    json.dump({"sd_all_int8": base, "sd_final": cur, "promoted": promoted,
               "lanes": lane_map}, open(out, "w"), indent=1)
    print(f"final: SD@2={cur:.4f} with {len(promoted)}/{len(layers)} layers on fp8 "
          f"-> {out}")


if __name__ == "__main__":
    main()
