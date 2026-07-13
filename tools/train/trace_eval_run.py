#!/usr/bin/env python3
"""End-task metric for a trained student: can the TRACER follow wraps in its prediction?

Dice/sd2 are patch-level proxies scored against imperfect GT; the deliverable is tracing.
For each holdout crop: predict prob (TorchScript-free, same sliding window as
eval_students), import-npy scale255 -> pred.fxvol, then `fenix trace-eval` against the
crop's source GT mesh (winding metrics: sheets traced, jumps, coverage).

Usage: trace_eval_run.py --ckpt /path/student_final.pt:32 [--crops DIR] [--out json]
"""
import sys, os, glob, json, argparse, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_students import load_student, predict

FENIX = os.environ.get("FENIX_BIN", "/home/forrest/fenix/build-release/fenix")
GTQC = os.environ.get("GTQC_DIR", "/tmp/gtqc")


def find_gt_fxsurf(seg):
    hits = glob.glob(f"{GTQC}/fxsurf/*__{seg}.fxsurf") + glob.glob(f"{GTQC}/repaired/*__{seg}.repaired.fxsurf")
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint.pt[:base]")
    ap.add_argument("--crops", default=f"{GTQC}/m8/eval")
    ap.add_argument("--out", default=f"{GTQC}/trace_eval_results.json")
    ap.add_argument("--thresh", default=None, help="trace-eval thresh= override")
    args = ap.parse_args()
    ckpt, _, base = args.ckpt.partition(":")
    net = load_student(ckpt, int(base) if base else 16)

    rows = []
    for cd in sorted(glob.glob(f"{args.crops}/crop*")):
        box = json.load(open(f"{cd}/box.json"))
        gt = find_gt_fxsurf(box["seg"])
        if not gt:
            print(f"{cd}: no GT fxsurf for {box['seg']}, skipping")
            continue
        ct = np.load(f"{cd}/ct.npy")
        prob = predict(net, ct).astype(np.float32)
        np.save(f"{cd}/pred.npy", prob)
        r = subprocess.run([FENIX, "import-npy", f"{cd}/pred.npy", f"{cd}/pred.fxvol", "q=2", "scale255"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{cd}: import-npy failed: {r.stderr[-200:]}")
            continue
        org = box["org"]
        cmd = [FENIX, "trace-eval", f"pred={cd}/pred.fxvol", f"gt={gt}",
               f"origin={org[0]},{org[1]},{org[2]}", f"ct={cd}/ct.fxvol"]
        if args.thresh:
            cmd.append(f"thresh={args.thresh}")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        txt = r.stdout + r.stderr
        rec = {"crop": os.path.basename(cd), "seg": box["seg"], "rc": r.returncode}
        # scrape trace-eval's summary line:
        # "RECALL @2 0.280 @4 0.520 (mean d 2.96, med 3.97) | PRECISION @2 0.001 @4 0.009 ... | 89 sheets / 111460 cells"
        # RECALL (GT cells recovered by the trace) is the load-bearing number; PRECISION is
        # only meaningful when the GT meshes cover every wrap in the block (usually they don't).
        import re
        m = re.search(r"RECALL @2 ([0-9.]+) @4 ([0-9.]+) \(mean d ([0-9.]+)", txt)
        if m:
            rec["recall2"], rec["recall4"], rec["mean_d"] = float(m.group(1)), float(m.group(2)), float(m.group(3))
        m = re.search(r"PRECISION @2 ([0-9.]+) @4 ([0-9.]+)", txt)
        if m:
            rec["precision2"], rec["precision4"] = float(m.group(1)), float(m.group(2))
        m = re.search(r"(\d+) sheets / (\d+) cells", txt)
        if m:
            rec["sheets"], rec["cells"] = int(m.group(1)), int(m.group(2))
        rec["tail"] = txt.strip().splitlines()[-3:]
        rows.append(rec)
        print(f"{rec['crop']}: rc={r.returncode} " +
              " ".join(f"{k}={rec[k]}" for k in rec if k not in ("crop", "seg", "tail", "rc")), flush=True)

    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
