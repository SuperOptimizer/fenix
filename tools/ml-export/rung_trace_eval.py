#!/usr/bin/env python3
"""Trace-eval for reference-family surface nets (teacher + distilled rungs).

The rung-A gate (docs/design/student-distill-plan.md) left SD@2-vs-teacher at 0.94 with
the verdict "trace-eval on GT meshes is the real arbiter" — this runs that arbiter.
Same crop protocol as tools/train/trace_eval_run.py but with the teacher's recipe
(per-patch z-score, 2-class softmax; NOT StudentUNet's /255), so numbers are comparable
across both families: predict -> import-npy scale255 -> `fenix trace-eval` -> scrape
RECALL/PRECISION.

Usage:
  rung_trace_eval.py --out results.json NAME=ckpt.pth[:rung] ...
  e.g. rung_trace_eval.py teacher=models/surface_recto_3dunet/checkpoint_inference_ready.pth \
       rungA=models/students/rungA_50k.pth:A
No rung suffix = full-size teacher Net. student8-style checkpoints (post norm-swap
gamma/beta naming) are renamed back on load; absorbed conv biases zero-fill.
"""
import sys, os, glob, json, argparse, subprocess, re
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R
from student import build_student

FENIX = os.environ.get("FENIX_BIN", "/home/forrest/fenix/build-release/fenix")
GTQC = os.environ.get("GTQC_DIR", "/tmp/gtqc")
DEV = "cuda:0"
PATCH, STRIDE = 128, 64


def load_net(ckpt, rung):
    net = build_student(rung) if rung else R.Net()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and not any(hasattr(v, "shape") for v in sd.values()):
        # e2e --save dict: {student16, student8, args}; prefer the fp16-trained lane
        # (the gate showed fp16-train-then-quantize beats QAT by ~0.075 SD).
        sd = sd.get("student16", sd.get("model", sd.get("net", sd.get("state_dict", sd))))
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    if any(k.endswith(".gamma") for k in sd):  # post norm-swap save (student8 lane)
        sd = {k.replace(".gamma", ".weight").replace(".beta", ".bias"): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith(".bias")]  # absorbed conv biases: zero is exact
    if missing or unexpected:
        raise SystemExit(f"{ckpt}: missing {missing[:4]} unexpected {unexpected[:4]}")
    return net.to(DEV).eval()


def gauss3(n, sigma_frac=0.25):
    ax = np.arange(n) - (n - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * (n * sigma_frac) ** 2)).astype(np.float32)
    return torch.from_numpy(g[:, None, None] * g[None, :, None] * g[None, None, :])


@torch.no_grad()
def predict(net, ct):
    D, H, W = ct.shape
    x = torch.from_numpy(ct.astype(np.float32)).to(DEV)
    prob = torch.zeros((D, H, W), device=DEV)
    wsum = torch.zeros((D, H, W), device=DEV)
    w = gauss3(PATCH).to(DEV)
    zs = sorted(set(list(range(0, max(D - PATCH, 0) + 1, STRIDE)) + [max(D - PATCH, 0)]))
    for z0 in zs:
        for y0 in zs:
            for x0 in zs:
                p = x[z0:z0+PATCH, y0:y0+PATCH, x0:x0+PATCH][None, None]
                m, s = p.mean(), p.std().clamp(min=1e-6)
                with torch.autocast("cuda", torch.float16):
                    lo = net((p - m) / s)
                if isinstance(lo, (list, tuple)):
                    lo = lo[0]
                pr = torch.softmax(lo.float(), 1)[0, 1]
                prob[z0:z0+PATCH, y0:y0+PATCH, x0:x0+PATCH] += pr * w
                wsum[z0:z0+PATCH, y0:y0+PATCH, x0:x0+PATCH] += w
    return (prob / wsum.clamp_min(1e-6)).cpu().numpy()


def find_gt_fxsurf(seg):
    hits = glob.glob(f"{GTQC}/fxsurf/*__{seg}.fxsurf") + glob.glob(f"{GTQC}/repaired/*__{seg}.repaired.fxsurf")
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=f"{GTQC}/m8/eval")
    ap.add_argument("--out", default=f"{GTQC}/rung_trace_eval.json")
    ap.add_argument("models", nargs="+", help="NAME=ckpt.pth[:rung]")
    args = ap.parse_args()

    nets = {}
    for spec in args.models:
        name, rest = spec.split("=", 1)
        ckpt, _, rung = rest.partition(":")
        nets[name] = load_net(ckpt, rung or None)

    results = {}
    for name, net in nets.items():
        rows = []
        for cd in sorted(glob.glob(f"{args.crops}/crop*")):
            box = json.load(open(f"{cd}/box.json"))
            gt = find_gt_fxsurf(box["seg"])
            if not gt:
                continue
            ct = np.load(f"{cd}/ct.npy")
            prob = predict(net, ct).astype(np.float32)
            np.save(f"{cd}/pred_{name}.npy", prob)
            r = subprocess.run([FENIX, "import-npy", f"{cd}/pred_{name}.npy", f"{cd}/pred_{name}.fxvol",
                                "q=2", "scale255"], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{cd}: import-npy failed: {r.stderr[-200:]}")
                continue
            org = box["org"]
            r = subprocess.run([FENIX, "trace-eval", f"pred={cd}/pred_{name}.fxvol", f"gt={gt}",
                                f"origin={org[0]},{org[1]},{org[2]}", f"ct={cd}/ct.fxvol"],
                               capture_output=True, text=True, timeout=1800)
            txt = r.stdout + r.stderr
            rec = {"crop": os.path.basename(cd), "rc": r.returncode}
            m = re.search(r"RECALL @2 ([0-9.]+) @4 ([0-9.]+) \(mean d ([0-9.]+)", txt)
            if m:
                rec["recall2"], rec["recall4"], rec["mean_d"] = map(float, m.groups())
            rows.append(rec)
            print(f"{name} {rec}", flush=True)
        results[name] = rows
        r2 = [x["recall2"] for x in rows if "recall2" in x]
        r4 = [x["recall4"] for x in rows if "recall4" in x]
        if r2:
            print(f"{name}: recall2 mean {np.mean(r2):.3f} min {min(r2):.3f} | "
                  f"recall4 mean {np.mean(r4):.3f} min {min(r4):.3f}", flush=True)

    json.dump(results, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
