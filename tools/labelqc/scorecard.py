"""Unified multi-axis GT scorecard. Fuses the independent quality axes into one record per
segment with a composite grade that requires passing EVERY axis (a segment training-grade
only if aligned AND intrinsically healthy AND consistent). Axes:

  alignment  : crop_qc (ridge_med, iqr_med, frac_crops_good)          -- CT, crop-local
  health     : mesh-qual (folds, self_intersect, tears, holes, edge_cv) -- CT-free
  consistency: surf-consist AGREE/OFFSET/CROSS                        -- geometry, no CT
  model      : label-audit miss%                                      -- flywheel (later)

Grade: E if any axis is disqualifying (garbage), else the worst repairable tier.
  A = aligned + healthy + consistent
  B/C = alignment repairable (offset/warp) but health/consistency ok
  D = partial (some crops good) -> trust-mask
  E = fails health (folds/self-int/tears) OR alignment-unfixable OR CROSS OR fragment

Plan: docs/design/gt-autograde-improve.md (step 4). Reads whatever axis inputs exist;
missing axes are simply not gated (so this runs before consist/model are wired).

Usage: scorecard.py --scroll PHercParis4 [--cropdir DIR] [--meshqual JSONL] [--out DIR]
"""
import sys, os, json, argparse, glob
import numpy as np


def load_meshqual(path):
    by = {}
    if path and os.path.exists(path):
        for l in open(path):
            if not l.strip():
                continue
            r = json.loads(l)
            seg = os.path.basename(r["mesh"]).replace(".fxsurf", "")
            by[seg] = r
    return by


def health_verdict(mq):
    """CT-free intrinsic health -> ok / suspect / bad."""
    if not mq:
        return "unknown", {}
    fold = mq.get("fold_detj", 0); si = mq.get("self_intersect", 0)
    tear = mq.get("frac_long", 0); holes = mq.get("holes", 0); cv = mq.get("edge_cv", 0)
    flags = {"fold": fold, "self_intersect": si, "tear": tear, "holes": holes, "edge_cv": cv}
    if fold > 0.05 or si > 0.10 or tear > 0.08 or holes > 100 or cv > 0.5:
        return "bad", flags
    if fold > 0.02 or si > 0.05 or tear > 0.03 or holes > 20 or cv > 0.2:
        return "suspect", flags
    return "ok", flags


def align_verdict(crop):
    """crop_qc alignment -> tier + verdict."""
    if not crop:
        return "unknown", {}
    rm = crop.get("ridge_med", 0); iqr = crop.get("iqr_med", 99)
    good = crop.get("frac_crops_good", 0)
    v = {"ridge_med": rm, "iqr_med": iqr, "frac_good": good}
    if rm >= 72 and iqr <= 5:
        return "A", v
    if rm >= 62 and iqr <= 6:
        return "B", v          # small offset -> shift/repair
    if rm >= 50 and iqr <= 10:
        return "C", v          # warp -> offset-field repair
    if good >= 0.4:
        return "D", v          # mixed -> trust-mask the bad crops
    return "E", v              # mostly off the ridge


def compose(align, health, consist=None):
    """Composite grade: E if any axis disqualifies, else the alignment tier (capped by health)."""
    a_tier, _ = align
    h_verdict, _ = health
    if h_verdict == "bad":
        return "E"             # geometrically corrupt regardless of alignment
    if a_tier == "E":
        return "E"
    if consist and consist.get("cross"):
        return "E"             # physically impossible
    if h_verdict == "suspect" and a_tier in ("A", "B"):
        return "C"             # demote: health concern makes it not clean-A
    return a_tier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--cropdir", default="/tmp/gtqc/cropcards")
    ap.add_argument("--meshqual", default="/tmp/gtqc/meshqual.jsonl")
    ap.add_argument("--out", default="/tmp/gtqc/scorecards")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mq = load_meshqual(args.meshqual)

    # index crop cards by segment
    crops = {}
    for p in glob.glob(f"{args.cropdir}/{args.scroll}__*.json"):
        seg = os.path.basename(p).replace(".json", "")
        crops[seg] = json.load(open(p))

    segs = sorted(set(list(crops) + [s for s in mq if s.startswith(args.scroll)]))
    from collections import Counter
    grades = Counter()
    for seg in segs:
        crop = crops.get(seg)
        mqr = mq.get(seg)
        align = align_verdict(crop)
        health = health_verdict(mqr)
        grade = compose(align, health)
        card = {"scroll": args.scroll, "segment": seg,
                "align": {"tier": align[0], **align[1]},
                "health": {"verdict": health[0], **health[1]},
                "grade": grade,
                "decision": {"A": "use", "B": "shift-repair", "C": "warp-repair",
                             "D": "trust-mask", "E": "quarantine"}.get(grade, "review")}
        json.dump(card, open(f"{args.out}/{seg}.card.json", "w"), indent=2)
        grades[grade] += 1

    print(f"{args.scroll}: {len(segs)} segments scored")
    print("grades:", dict(sorted(grades.items())))
    # health-flag summary
    bad_h = [s for s in segs if health_verdict(mq.get(s))[0] == "bad"]
    print(f"health=bad: {len(bad_h)}  (auto-quarantine)")
    for s in bad_h[:10]:
        print("   ", s)


if __name__ == "__main__":
    main()
