#!/usr/bin/env python3
"""audit_persist.py — cross-run label-noise persistence.

A mesh in the worst per-mesh-CE tier of ONE run may be unlucky sampling; a mesh that
stays there across independent runs has labels the model cannot fit — drop or repair it.
Usage: audit_persist.py <meshloss.json...> [--thr 0.8] [--min-n 30]
Prints meshes suspect (ce_ema > thr, n >= min_n) in >= 2 of the given runs.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="meshloss.json files (one per training run)")
    ap.add_argument("--thr", type=float, default=0.8, help="suspect CE-EMA threshold")
    ap.add_argument("--min-n", type=int, default=30, help="min samples for a verdict")
    args = ap.parse_args()

    hits = {}   # path -> {run_name: ce}
    seen = {}   # path -> #runs the mesh appeared in at all
    for rp in args.runs:
        d = json.load(open(rp))
        name = os.path.basename(rp).replace("_meshloss.json", "")
        for r in d["meshes"]:
            seen[r["path"]] = seen.get(r["path"], 0) + 1
            if r["ce_ema"] > args.thr and r["n"] >= args.min_n:
                hits.setdefault(r["path"], {})[name] = r["ce_ema"]

    persistent = {p: rs for p, rs in hits.items() if len(rs) >= 2}
    if not persistent:
        print(f"no mesh is suspect (ce_ema>{args.thr}) in >=2 of {len(args.runs)} runs")
        return
    print(f"# persistent label-noise suspects (ce_ema>{args.thr} in >=2/{len(args.runs)} runs)")
    for p, rs in sorted(persistent.items(), key=lambda kv: -max(kv[1].values())):
        per = "  ".join(f"{k}:{v:.3f}" for k, v in sorted(rs.items()))
        print(f"{os.path.basename(p)}  ({len(rs)}/{seen[p]} runs)  {per}")


if __name__ == "__main__":
    main()
