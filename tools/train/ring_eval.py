#!/usr/bin/env python3
"""Ring-based held-out patch eval: score models on patches drained from a live
`fenix train-feed` ring instead of pre-cut crops.

Why: eval crops cut at a scroll's NATIVE grid mis-scale models trained on the
canonical 2.4um grid; the feeder's um=/canon= resampling already solves this
(CT trilinear, GT rasterized post-scale — never interpolated). Draining the ring
gives correctly-resampled (CT, band GT) pairs with CLEAN tri-state labels
(255/128/0 — no codec round-trip smear), at the cost of losing per-patch mesh
provenance (so no trace-eval tier here; patch metrics only).

Run the feeder with aug=0 and a fixed seed for a deterministic, augmentation-free
eval set:
  fenix train-feed pairs_heldout.txt /dev/shm/eval.ring patch=128 slots=16 \
      threads=8 seed=123 aug=0 &
  ring_eval.py --ring /dev/shm/eval.ring --n 64 --out results.json \
      M=/tmp/gtqc/real/studentM_best.pt:32 G=/tmp/gtqc/m8/studentG_final.pt:16
"""
import sys, os, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feed_reader import FeedRing
from eval_students import load_student, auprc_maxdice, ece, surface_dice

DEV = "cuda:0"


@torch.no_grad()
def predict_patch(net, ct_u8):
    x = torch.from_numpy(ct_u8.astype(np.float32) / 255.0)[None, None].to(DEV)
    with torch.autocast("cuda", torch.bfloat16):
        lo = net(x)
    return torch.softmax(lo.float(), 1)[0, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True)
    ap.add_argument("--n", type=int, default=64, help="patches to drain and score")
    ap.add_argument("--out", default="ring_eval.json")
    ap.add_argument("--per-mesh", action="store_true",
                    help="group metrics by slot mesh id (label-audit mode: score how well "
                         "the model corroborates each mesh's labels — the M11 rescue signal)")
    ap.add_argument("models", nargs="+", help="NAME=ckpt.pt[:base]")
    args = ap.parse_args()

    nets = {}
    for spec in args.models:
        name, rest = spec.split("=", 1)
        ckpt, _, base = rest.partition(":")
        nets[name] = load_student(ckpt, int(base) if base else 16)

    ring = FeedRing(args.ring)
    per = {n: [] for n in nets}
    got = 0
    while got < args.n:
        batch = ring.next_batch(1, timeout_s=600.0)
        ct = batch["ct"][0]
        gt = batch["gt"][0]
        sheet = gt == 255
        lab = gt > 0
        if sheet.sum() < 500 or lab.sum() < 5000:
            continue  # patch with almost no supervision: skip, keep the count honest
        got += 1
        mesh_id = int(batch["meta"][0][0])
        for name, net in nets.items():
            prob = predict_patch(net, ct)
            pred = (prob > 0.5) & lab
            inter = (pred & sheet).sum()
            dice = float(2 * inter / max(pred.sum() + sheet.sum(), 1))
            ap_, md, mt = auprc_maxdice(prob, sheet, lab)
            per[name].append({
                "mesh": mesh_id,
                "dice": dice, "auprc": ap_, "maxdice": md, "maxdice_t": mt,
                "sd2": surface_dice((prob > mt) & lab, sheet, 2.0),
                "ece": ece(prob, sheet, lab),
                "n_sheet": int(sheet.sum()),
            })
        if got % 8 == 0:
            print(f"{got}/{args.n} patches", flush=True)

    if args.per_mesh:
        meshes = open(args.ring + ".meshes").read().split()  # line order IS the id order
        for name, rows in per.items():
            by = {}
            for r in rows:
                by.setdefault(r["mesh"], []).append(r)
            print(f"== {name} per-mesh (n, auprc mean/p10, sd2 mean/p10):")
            for mid in sorted(by):
                v = by[mid]
                a = np.array([r["auprc"] for r in v]); sd = np.array([r["sd2"] for r in v])
                nm = os.path.basename(meshes[mid]) if mid < len(meshes) else str(mid)
                print(f"  {nm[:60]:60s} n={len(v):3d} auprc {a.mean():.3f}/{np.percentile(a,10):.3f} "
                      f"sd2 {sd.mean():.3f}/{np.percentile(sd,10):.3f}")
    summary = {}
    for name, rows in per.items():
        summary[name] = {}
        for m in ("dice", "auprc", "maxdice", "sd2", "ece"):
            v = np.array([r[m] for r in rows])
            summary[name][m] = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
                                "min": float(v.min())}
            print(f"{name} {m}: mean {v.mean():.4f}  p10 {np.percentile(v,10):.4f}  "
                  f"min {v.min():.4f}")
    json.dump({"patches": per, "summary": summary}, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
