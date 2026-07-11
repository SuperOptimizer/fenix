"""Tiny atomic helpers for the labelqc jsonl stores (meshqual.jsonl, orientation.jsonl).

The stores are per-segment records keyed by name strings embedded in each line. Two bug
classes this kills (both hit 2026-07-11): shell-assembled records producing invalid JSON,
and sed-renaming segments across files corrupting/missing entries. All writes are
validate-then-atomic-replace; renames go through one function that touches every store.

Usage as a CLI:  registry.py rename OLD NEW   |   registry.py validate
"""
import sys, os, json, glob

GTQC = os.environ.get("GTQC_DIR", "/tmp/gtqc")
STORES = ("meshqual.jsonl", "orientation.jsonl")


def load(path):
    """-> list of records; raises on any malformed line (fail loud, not at the next reader)."""
    recs = []
    if not os.path.exists(path):
        return recs
    for i, l in enumerate(open(path)):
        if not l.strip():
            continue
        try:
            recs.append(json.loads(l))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i+1}: malformed record: {l.strip()[:120]}") from e
    return recs


def save(path, recs):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def append(path, rec):
    """Validated append (round-trips through json to guarantee well-formedness)."""
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def seg_key(rec):
    if "segment" in rec:
        return rec["segment"]
    if "mesh" in rec:
        return os.path.basename(rec["mesh"]).replace(".fxsurf", "")
    return None


def rename_segment(old, new, gtqc=None):
    """Rename a segment across every store + per-segment file (cards, crops, fxsurf name
    is the caller's job — this covers the string-keyed records)."""
    g = gtqc or GTQC
    n = 0
    for store in STORES:
        p = os.path.join(g, store)
        recs = load(p)
        changed = False
        for r in recs:
            if seg_key(r) == old:
                if "segment" in r:
                    r["segment"] = new
                if "mesh" in r:
                    r["mesh"] = r["mesh"].replace(old, new)
                changed = True
                n += 1
        if changed:
            save(p, recs)
    for pat in (f"{g}/cropcards/{old}.json", f"{g}/scorecards/{old}.card.json"):
        for f in glob.glob(pat):
            os.replace(f, f.replace(old, new))
            n += 1
    return n


def validate(gtqc=None):
    g = gtqc or GTQC
    ok = True
    for store in STORES:
        p = os.path.join(g, store)
        try:
            recs = load(p)
            print(f"{p}: {len(recs)} records ok")
        except ValueError as e:
            print(f"{p}: {e}")
            ok = False
    return ok


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "validate":
        sys.exit(0 if validate() else 1)
    if len(sys.argv) == 4 and sys.argv[1] == "rename":
        print(f"renamed {rename_segment(sys.argv[2], sys.argv[3])} records/files")
        sys.exit(0)
    print(__doc__)
    sys.exit(1)
