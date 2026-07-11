"""Accept-set export: graded+repaired corpus -> feeder pairs.txt (closes GT loop 1).

Reads the final scorecards and emits the training manifest:
  - grade A/B/C (incl. borderline base tiers): the repaired fxsurf when repair improved
    (decision=use-repaired), else the original; D: original + trust= sidecar (bad tiles
    blanked); E: excluded (quarantine).
  - DEDUP (measured corpus semantics): DISAGREE partners at overlap>=15% are redundant
    VERSIONS of the same region disagreeing 2-3 vox. Keeping both double-counts those
    wraps with conflicting labels -> keep the better-graded member per cluster, drop the
    rest (greedy by grade rank then pap_med).

Usage: export_acceptset.py --scroll PHercParis4 --ct <path-or-cache-url> [--out pairs.txt]
"""
import sys, os, json, argparse, glob

RANK = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def rank(g):
    return RANK.get(g.rstrip("?"), 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--ct", required=True, help="CT path/url token for the pairs line")
    ap.add_argument("--um", type=float, default=None,
                    help="voxel um of this scroll's grid when != the 2.4 canon (emitted as um= "
                         "on every pair line; the feeder resamples to canon at feed time)")
    ap.add_argument("--scorecards", default="/tmp/gtqc/scorecards")
    ap.add_argument("--out", default="/tmp/gtqc/pairs.txt")
    args = ap.parse_args()

    cards = {}
    for p in glob.glob(f"{args.scorecards}/{args.scroll}__*.card.json"):
        c = json.load(open(p))
        cards[c["segment"]] = c

    # dedup clusters from DISAGREE partners (overlap>=15%)
    dropped = set()
    for seg, c in sorted(cards.items()):
        if seg in dropped or rank(c["grade"]) >= 4:
            continue
        for pr in c.get("consist", {}).get("partners", []):
            other = pr["seg"]
            if pr.get("verdict") != "DISAGREE" or other not in cards:
                continue
            oc = cards[other]
            if rank(oc["grade"]) >= 4:
                continue  # already quarantined
            # keep the better-graded (tiebreak: higher pap_med); drop the other
            a_key = (rank(c["grade"]), -(c["align"].get("pap_med") or 0))
            b_key = (rank(oc["grade"]), -(oc["align"].get("pap_med") or 0))
            dropped.add(other if a_key <= b_key else seg)

    lines, stats = [], {"use": 0, "use-repaired": 0, "trust-mask": 0, "dedup-dropped": 0, "quarantined": 0}
    for seg, c in sorted(cards.items()):
        g = c["grade"].rstrip("?")
        if g == "E":
            stats["quarantined"] += 1
            continue
        if seg in dropped:
            stats["dedup-dropped"] += 1
            continue
        fx = c["fxsurf"]
        rep = c.get("repair", {})
        if c.get("decision") == "use-repaired" and rep.get("repaired_fxsurf") and os.path.exists(rep["repaired_fxsurf"]):
            fx = rep["repaired_fxsurf"]
            stats["use-repaired"] += 1
        elif g == "D":
            stats["trust-mask"] += 1
        else:
            stats["use"] += 1
        tok = f"{fx} {args.ct}"
        if args.um:
            tok += f" um={args.um}"
        trust = f"/tmp/gtqc/_trust_{seg}.txt"
        if g == "D" and os.path.exists(trust):
            tok += f" trust={trust}"
        lines.append(tok)

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{args.scroll}: {len(lines)} pairs -> {args.out}")
    print("stats:", stats)
    if dropped:
        print("dedup-dropped:", sorted(dropped))


if __name__ == "__main__":
    main()
