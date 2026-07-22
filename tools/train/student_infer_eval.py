#!/usr/bin/env python3
"""student_infer_eval.py — sliding-window inference + blob-recall referee for the distilled
ink student (finetune_ink_compression.py --student-base). Windows match the training patch
(128³, per-window pct_minmax — the norm the student was trained under), stride 96, uniform
overlap averaging. Compares q32- and raw-input predictions against the teacher npy with the
same blob-recall protocol as the full-net referee (2026-07-20 findings doc).

  python3 student_infer_eval.py --ckpt ink_student_b64.pth --data ~/inkft --region r12288a \
      [--base 64] [--stem 2] [--patch 128] [--stride 96] [--out-prefix stu]
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from train import StudentUNet  # noqa: E402
from finetune_ink_compression import pct_minmax, zscore  # noqa: E402


def infer_volume(net, vol, patch, stride, dev, batch=16):
    D, H, W = vol.shape
    acc = np.zeros(vol.shape, np.float32)
    cnt = np.zeros(vol.shape, np.float32)
    orgs = [(z, y, x)
            for z in list(range(0, D - patch, stride)) + [D - patch]
            for y in list(range(0, H - patch, stride)) + [H - patch]
            for x in list(range(0, W - patch, stride)) + [W - patch]]
    for i in range(0, len(orgs), batch):
        grp = orgs[i:i + batch]
        xs = torch.stack([torch.from_numpy(
            np.ascontiguousarray(vol[z:z + patch, y:y + patch, x:x + patch])).float()
            for z, y, x in grp]).unsqueeze(1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(net(NORM(xs.to(dev)))).float().cpu().numpy()[:, 0]
        for k, (z, y, x) in enumerate(grp):
            acc[z:z + patch, y:y + patch, x:x + patch] += p[k]
            cnt[z:z + patch, y:y + patch, x:x + patch] += 1
    return acc / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--region", default="r12288a")
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--stem", type=int, default=2)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--out-prefix", default="stu")
    ap.add_argument("--task", choices=["ink", "surface"], default="ink")
    args = ap.parse_args()

    dev = "cuda"
    net = StudentUNet(base=args.base, classes=1, stem_stride=args.stem).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net.load_state_dict(ck["ema_model"])
    print(f"loaded {args.ckpt} step {ck.get('step')}", flush=True)

    global NORM
    NORM = zscore if args.task == "surface" else pct_minmax
    tsuf = ".steacher.npy" if args.task == "surface" else ".teacher.npy"
    D = args.data
    t = np.load(os.path.join(D, f"{args.region}{tsuf}"))
    import time
    from scipy import ndimage
    lab, nb = ndimage.label(t > 128)
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, range(1, nb + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= 200]
    print(f"teacher blobs (>=200 vox): {len(keep)}", flush=True)

    for tag in ["q32", "raw"]:
        vol = np.load(os.path.join(D, f"{args.region}.{tag}.npy"))
        t0 = time.time()
        prob = infer_volume(net, vol, args.patch, args.stride, dev)
        el = time.time() - t0
        u8 = (np.clip(prob, 0, 1) * 255 + 0.5).astype(np.uint8)
        np.save(os.path.join(D, f"{args.out_prefix}_{tag}.npy"), u8)
        d = np.abs(u8.astype(np.float32) - t.astype(np.float32))
        mse = (d ** 2).mean()
        row = []
        if args.task == "surface":
            tm_, vm_ = t > 128, u8 > 128
            dice = 2 * (tm_ & vm_).sum() / max(tm_.sum() + vm_.sum(), 1)
            row.append(f"sheet-dice@0.5: {dice:.4f}")
        else:
            for thr in [128, 64, 32]:
                vm = u8 > thr
                hit = sum(1 for i in keep if vm[lab == i].any())
                row.append(f"thr{thr}: {hit}/{len(keep)}")
        print(f"student_{tag}: PSNR {10*np.log10(255**2/max(mse,1e-9)):.2f} dB  MAE {d.mean():.3f}  "
              + "  ".join(row) + f"  infer {el:.0f}s", flush=True)
    print("STUDENT_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
