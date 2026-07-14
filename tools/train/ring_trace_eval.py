#!/usr/bin/env python3
"""Trace-eval tier for ring-based held-out eval: end-task recall on feeder-resampled
patches, using the ring's per-slot provenance (origin in the feeder's canonical frame).

Frame note (measured on PHerc0500P2, 2026-07-12): the ring origin lives in the frame
mesh_coords * (canon_um/source_um) — for a um=4.317 pair that is coords * 0.555941.
Pre-scale GT meshes into that frame ONCE (export-tifxyz -> import-tifxyz coordscale=)
and pass the directory here; verified 99.3% of a rasterized band lands inside ring GT.

For each drained sheet-bearing patch: predict -> import-npy (pred prob + raw CT) ->
`fenix trace-eval` against every scaled mesh whose bbox intersects the patch box, and
aggregate cell-weighted RECALL per model.

Usage:
  fenix train-feed pairs_heldout.txt /dev/shm/tr.ring patch=128 slots=16 threads=8 \
      seed=123 aug=0 &
  ring_trace_eval.py --ring /dev/shm/tr.ring --meshes /tmp/gtqc/canon0500 --n 16 \
      --out results.json M=studentM_best.pt:32 G=studentG_final.pt:16
"""
import sys, os, re, json, glob, argparse, subprocess, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feed_reader import FeedRing
from eval_students import load_student

FENIX = os.environ.get("FENIX_BIN", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "build-release", "fenix"))
DEV = "cuda:0"


def mesh_bboxes(meshdir):
    out = []
    for m in sorted(glob.glob(f"{meshdir}/*.fxsurf")):
        r = subprocess.run([FENIX, "fxinfo", m], capture_output=True, text=True)
        mm = re.search(r"bbox \(ZYX\) \[([\d,]+)\] \.\. \[([\d,]+)\]", r.stdout)
        if not mm:
            continue
        lo = [int(v) for v in mm.group(1).split(",")]
        hi = [int(v) for v in mm.group(2).split(",")]
        out.append((m, lo, hi))
    return out


@torch.no_grad()
def predict_patch(net, ct_u8):
    x = torch.from_numpy(ct_u8.astype(np.float32) / 255.0)[None, None].to(DEV)
    with torch.autocast("cuda", torch.bfloat16):
        lo = net(x)
    return torch.softmax(lo.float(), 1)[0, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True)
    ap.add_argument("--meshes", required=True, help="dir of ring-frame-scaled .fxsurf GT")
    ap.add_argument("--n", type=int, default=16, help="sheet-bearing patches to trace")
    ap.add_argument("--min-sheet", type=float, default=0.02)
    ap.add_argument("--out", default="ring_trace_eval.json")
    ap.add_argument("models", nargs="+", help="NAME=ckpt.pt[:base]")
    args = ap.parse_args()

    nets = {}
    for spec in args.models:
        name, rest = spec.split("=", 1)
        ckpt, _, base = rest.partition(":")
        nets[name] = load_student(ckpt, int(base) if base else 16)
    boxes = mesh_bboxes(args.meshes)
    print(f"{len(boxes)} GT meshes indexed")

    ring = FeedRing(args.ring)
    wd = tempfile.mkdtemp(prefix="ringtrace_")
    per = {n: [] for n in nets}
    got = 0
    while got < args.n:
        b = ring.next_batch(1, timeout_s=900.0)
        ct, gt = b["ct"][0], b["gt"][0]
        if (gt == 255).mean() < args.min_sheet:
            continue
        mesh, draw, oz, oy, ox = b["meta"][0]
        P = ct.shape[0]
        hits = [m for m, lo, hi in boxes
                if lo[0] < oz + P and hi[0] > oz and lo[1] < oy + P and hi[1] > oy
                and lo[2] < ox + P and hi[2] > ox]
        if not hits:
            continue
        got += 1
        np.save(f"{wd}/ct.npy", ct)
        subprocess.run([FENIX, "import-npy", f"{wd}/ct.npy", f"{wd}/ct.fxvol", "q=2"],
                       capture_output=True)
        for name, net in nets.items():
            prob = predict_patch(net, ct).astype(np.float32)
            np.save(f"{wd}/pred.npy", prob)
            subprocess.run([FENIX, "import-npy", f"{wd}/pred.npy", f"{wd}/pred.fxvol",
                            "q=2", "scale255"], capture_output=True)
            tot_r2 = tot_r4 = tot_c = 0
            for gtm in hits:
                r = subprocess.run([FENIX, "trace-eval", f"pred={wd}/pred.fxvol",
                                    f"gt={gtm}", f"origin={oz},{oy},{ox}",
                                    f"ct={wd}/ct.fxvol"],
                                   capture_output=True, text=True, timeout=1800)
                txt = r.stdout + r.stderr
                m2 = re.search(r"RECALL @2 ([0-9.]+) @4 ([0-9.]+)", txt)
                mc = re.search(r"(\d+) sheets / (\d+) cells", txt)
                if m2 and mc:
                    c = int(mc.group(2))
                    tot_r2 += float(m2.group(1)) * c
                    tot_r4 += float(m2.group(2)) * c
                    tot_c += c
            if tot_c:
                per[name].append({"draw": int(draw), "org": [int(oz), int(oy), int(ox)],
                                  "recall2": tot_r2 / tot_c, "recall4": tot_r4 / tot_c,
                                  "cells": tot_c, "n_meshes": len(hits)})
        r2s = {n: per[n][-1]["recall2"] if per[n] else None for n in nets}
        print(f"patch {got}/{args.n} (meshes {len(hits)}): " +
              " ".join(f"{n} r2={v:.3f}" for n, v in r2s.items() if v is not None), flush=True)

    summary = {}
    for name, rows in per.items():
        for k in ("recall2", "recall4"):
            v = np.array([r[k] for r in rows])
            summary.setdefault(name, {})[k] = {
                "mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "min": float(v.min())}
            print(f"{name} {k}: mean {v.mean():.3f}  p10 {np.percentile(v,10):.3f}  min {v.min():.3f}")
    json.dump({"patches": per, "summary": summary}, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
