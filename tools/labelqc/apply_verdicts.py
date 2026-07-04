#!/usr/bin/env python3
"""apply_verdicts.py — turn triage-gallery verdicts into pairs-file edits (stdlib only).

Reads verdicts.json (exported by the gallery) and a train-feed pairs file; writes a new
pairs file where:
  drop   -> the mesh's line is removed
  trust  -> `trust=<trust_dir>/<base>.trust` appended (if the grid file exists)
  keep   -> unchanged
  unsure -> unchanged, listed for the scrub-clip review round
Usage: apply_verdicts.py verdicts.json pairs.txt out_pairs.txt [--trust-dir /workspace/trust]
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdicts")
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--trust-dir", default="/workspace/trust")
    args = ap.parse_args()

    V = json.load(open(args.verdicts))
    stats = {"drop": 0, "trust": 0, "keep": 0, "unsure": 0, "untouched": 0}
    unsure = []
    with open(args.pairs) as f, open(args.out, "w") as o:
        for ln in f:
            if not ln.strip() or ln.lstrip().startswith("#"):
                o.write(ln)
                continue
            base = os.path.basename(ln.split()[0]).replace(".fxsurf", "")
            v = V.get(base)
            if v is None:
                stats["untouched"] += 1
                o.write(ln)
                continue
            verdict = v.get("verdict", "keep")
            stats[verdict] = stats.get(verdict, 0) + 1
            if verdict == "drop":
                continue
            if verdict == "trust" and "trust=" not in ln:
                tp = os.path.join(args.trust_dir, base + ".trust")
                if os.path.exists(tp):
                    ln = ln.rstrip("\n") + f" trust={tp}\n"
                else:
                    print(f"warn: trust verdict but no grid at {tp} — kept unchanged")
            if verdict == "unsure":
                unsure.append(base)
            o.write(ln)
    print(f"apply_verdicts: {stats} -> {args.out}")
    if unsure:
        print("unsure (scrub-clip round):")
        for b in unsure:
            print(" ", b)


if __name__ == "__main__":
    main()
