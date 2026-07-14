#!/usr/bin/env python3
"""Mesh lineage manifest — ONE joined view of the corpus (2026-07-13 audit P0).

Joins, per mesh: import provenance (corpus_import.txt), health (meshqual.jsonl),
grade (scorecards/*.card.json), pairs-file membership (every pairs_*.txt), and which
training runs consumed it (inferred from the pairs file each run's harness was given —
pass --runs "name=pairs_file" to record that mapping explicitly).

This replaces grep archaeology across a dozen dated ad-hoc files and makes silent
exclusion-flip-flops (the PHerc0172 excluded->re-included history) visible in one query.
Regenerate on every corpus refresh; the output is the authoritative record of what is
in the corpus and where each mesh stands.

Usage:
  manifest.py [--gtqc /tmp/gtqc] [--out /tmp/gtqc/manifest.tsv]
              [--runs studentH=pairs_m12.txt studentK=pairs_m15.txt ...]
"""
import os, glob, json, argparse, collections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtqc", default="/tmp/gtqc")
    ap.add_argument("--out", default="/tmp/gtqc/manifest.tsv")
    ap.add_argument("--runs", nargs="*", default=[],
                    help="name=pairs_file.txt bindings recording which run trained on which file")
    args = ap.parse_args()
    G = args.gtqc

    rows = collections.defaultdict(dict)  # seg name -> fields

    def seg_of(path):
        b = os.path.basename(path)
        for suf in (".repaired.fxsurf", ".canon.fxsurf", ".fxsurf"):
            if b.endswith(suf):
                return b[: -len(suf)]
        return b

    # import provenance
    imp = os.path.join(G, "corpus_import.txt")
    if os.path.exists(imp):
        for l in open(imp):
            p = l.rstrip("\n").split("\t")
            if len(p) >= 5:
                rows[f"{p[0]}__{p[1]}"].update(scroll=p[0], tifxyz=p[2], scan=p[3], scan_um=p[4])

    # mesh health
    mq = os.path.join(G, "meshqual.jsonl")
    if os.path.exists(mq):
        for l in open(mq):
            try:
                j = json.loads(l)
            except json.JSONDecodeError:
                continue
            s = seg_of(j.get("mesh", ""))
            rows[s]["self_intersect"] = j.get("self_intersect")
            rows[s]["edge_cv"] = j.get("edge_cv")
            rows[s]["holes"] = j.get("holes")

    # grades
    for cp in glob.glob(f"{G}/scorecards/*.card.json"):
        try:
            c = json.load(open(cp))
        except json.JSONDecodeError:
            continue
        rows[c.get("segment", seg_of(cp))]["grade"] = c.get("grade")

    # pairs membership
    pair_files = sorted(glob.glob(f"{G}/pairs_*.txt"))
    for pf in pair_files:
        col = os.path.basename(pf)
        for l in open(pf):
            if l.strip() and not l.startswith("#"):
                s = seg_of(l.split()[0])
                rows[s].setdefault("pairs", set()).add(col)

    # run bindings
    for spec in args.runs:
        name, _, pf = spec.partition("=")
        fp = pf if os.path.isabs(pf) else os.path.join(G, pf)
        if not os.path.exists(fp):
            print(f"warn: {fp} missing, skipping run {name}")
            continue
        for l in open(fp):
            if l.strip() and not l.startswith("#"):
                rows[seg_of(l.split()[0])].setdefault("runs", set()).add(name)

    cols = ["seg", "scroll", "scan", "scan_um", "grade", "self_intersect", "edge_cv", "holes",
            "pairs", "runs", "tifxyz"]
    with open(args.out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for s in sorted(rows):
            r = rows[s]
            f.write("\t".join([
                s, str(r.get("scroll", "")), str(r.get("scan", "")), str(r.get("scan_um", "")),
                str(r.get("grade", "")), str(r.get("self_intersect", "")), str(r.get("edge_cv", "")),
                str(r.get("holes", "")), ",".join(sorted(r.get("pairs", []))),
                ",".join(sorted(r.get("runs", []))), str(r.get("tifxyz", "")),
            ]) + "\n")
    n_trained = sum(1 for r in rows.values() if r.get("runs"))
    n_ungraded_trained = sum(1 for r in rows.values() if r.get("runs") and not r.get("grade"))
    print(f"{len(rows)} meshes -> {args.out}")
    print(f"trained-on: {n_trained}; trained-on but NEVER graded: {n_ungraded_trained}"
          + (" <- flag for post-hoc QC" if n_ungraded_trained else ""))


if __name__ == "__main__":
    main()
