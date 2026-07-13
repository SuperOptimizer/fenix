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
    rm = crop.get("ridge_med", 0); iqr = crop.get("iqr_med")  # None = unmeasured (fail closed)
    good = crop.get("frac_crops_good", 0)
    pap = crop.get("pap_med")  # raw on-papyrus % — the PRIMARY gate when present
    coh = crop.get("coh_med"); med = crop.get("med_off_med")
    aiqr = crop.get("alpha_iqr_med")
    v = {"ridge_med": rm, "iqr_med": iqr, "frac_good": good, "pap_med": pap,
         "coh_med": coh, "med_off_med": med}

    def tier_at(p_):
        """pap-branch tier at a given pap value (dispersion/med gates unchanged)."""
        def disp(iq_max, coh_min):
            if iqr is not None and iqr <= iq_max:
                return True
            if aiqr is not None and aiqr <= iq_max + 1:  # M2: alpha-edge dispersion substitute
                return True
            return coh is not None and coh >= coh_min
        if p_ >= 90 and disp(7, 70) and (med is None or abs(med) <= 3):
            return "A"
        if p_ >= 84 and disp(8, 60) and (med is None or abs(med) <= 4):
            return "B"
        if p_ >= 70 and disp(12, 45):
            return "C"
        if good >= 0.4 or p_ >= 50:
            return "D"
        return "E"

    ci = crop.get("pap_ci")
    if pap is not None and ci:
        # M1: a tier is only assigned when it is CI-STABLE — the tier at both CI edges
        # matches the tier at the median. Otherwise "borderline": the grade sits inside
        # sampling noise and the segment needs more crops (escalate 8->16->32), not a coin flip.
        t_med, t_lo, t_hi = tier_at(pap), tier_at(ci[0]), tier_at(ci[1])
        v["pap_ci"] = ci
        if not (t_med == t_lo == t_hi) and crop.get("n_crops", 8) < 32:
            v["borderline"] = True
            return t_med + "?", v          # tier?"borderline" — escalation driver re-runs
        return t_med, v
    if pap is not None:
        # pap is a one-sided bright-material gate (adversarial-review finding): a mesh
        # coherently on the WRONG wrap scores pap~100. Every tier therefore ALSO requires
        # dispersion evidence (iqr or coherence) + a bounded median offset; iqr=None
        # (unmeasured — too few peak-firing points) does not count as tight.
        def disp(iq_max, coh_min):
            if iqr is not None and iqr <= iq_max:
                return True
            if aiqr is not None and aiqr <= iq_max + 1:  # M2
                return True
            return coh is not None and coh >= coh_min
        med_ok = med is None or abs(med) <= 3
        if pap >= 90 and disp(7, 70) and med_ok:
            return "A", v
        if pap >= 84 and disp(8, 60) and (med is None or abs(med) <= 4):
            return "B", v
        if pap >= 70 and disp(12, 45):
            return "C", v
        if good >= 0.4 or pap >= 50:
            return "D", v
        return "E", v
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
    """Composite grade: E if any axis disqualifies, else the alignment tier (capped by health
    and consistency). DISAGREE (redundant coverage conflicting by 2-3 vox — measured corpus
    semantics, see surf_consist.hpp) caps at B and marks dedup candidates; it never
    quarantines (the alignment axis separates the good member of the pair from the bad)."""
    a_tier, _ = align
    h_verdict, _ = health
    if h_verdict == "bad":
        return "E"             # geometrically corrupt regardless of alignment
    if a_tier == "E":
        return "E"
    if h_verdict == "suspect" and a_tier in ("A", "B"):
        return "C"             # demote: health concern makes it not clean-A
    if consist and consist.get("n_disagree", 0) > 0 and a_tier == "A":
        return "B"             # a peer trace disagrees -> can't be clean-A
    return a_tier


def load_consist(path):
    """Parse surf-consist pair lines -> per-segment {n_agree, n_offset, n_disagree, partners}.
    partners = overlap>=15%% peers (dedup candidates: redundant versions of the same region)."""
    import re
    by = {}
    if not path or not os.path.exists(path):
        return by
    pat = re.compile(r"surf-consist pair (\S+) <-> (\S+)\s+overlap (\d+)% .* (AGREE|OFFSET|DISAGREE|CROSS)\s*$")
    for l in open(path):
        m = pat.search(l)
        if not m:
            continue
        a, b, ov, v = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        a = os.path.basename(a).replace(".fxsurf", ""); b = os.path.basename(b).replace(".fxsurf", "")
        v = "DISAGREE" if v == "CROSS" else v  # legacy naming
        for s1, s2 in ((a, b), (b, a)):
            e = by.setdefault(s1, {"n_agree": 0, "n_offset": 0, "n_disagree": 0, "partners": []})
            e["n_" + v.lower()] += 1
            if ov >= 15:
                e["partners"].append({"seg": s2, "overlap": ov, "verdict": v})
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--cropdir", default="/tmp/gtqc/cropcards")
    ap.add_argument("--meshqual", default="/tmp/gtqc/meshqual.jsonl")
    ap.add_argument("--consist", default="/tmp/gtqc/consist_paris4.txt")
    ap.add_argument("--out", default="/tmp/gtqc/scorecards")
    ap.add_argument("--um", type=float, default=2.4,
                    help="voxel um of this scroll's mesh grid (area-normalizes the fragment gate)")
    ap.add_argument("--no-align", action="store_true",
                    help="alignment axis unmeasurable at this scan LOD (wrap pitch ~ capture "
                         "range, e.g. 7.91um: fault-injection shows every corruption grades A). "
                         "Grade on health+consist+orientation only; cap at B; NO repair.")
    ap.add_argument("--profile", default=None,
                    help="per-domain calibration profile from fault_inject --emit-profile; "
                         "its measured align_valid verdict sets no-align (the gate's judgment, "
                         "not a manual flag)")
    args = ap.parse_args()
    if args.profile:
        prof = json.load(open(args.profile))
        if not prof.get("align_valid", False):
            args.no_align = True
            print(f"profile {prof.get('domain')}: alignment axis BLIND (fault gate) -> no-align grading")
    os.makedirs(args.out, exist_ok=True)
    mq = load_meshqual(args.meshqual)
    cons = load_consist(args.consist)
    orient = {}
    if os.path.exists("/tmp/gtqc/orientation.jsonl"):
        for l in open("/tmp/gtqc/orientation.jsonl"):
            if l.strip():
                r = json.loads(l)
                orient[r["segment"]] = r

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
        # --no-align: local intensity QC is BLIND when wrap pitch ~= peak capture range
        # (measured 7.91um fault injection: wrapshift/offset/warp/scramble ALL grade A —
        # every position is within ~2-3 vox of SOME crest). Treat alignment as B evidence
        # ceiling, never A, and never repair (snap range spans the inter-sheet gap).
        align = ("B", {"mode": "no-align-lod-cap"}) if args.no_align else align_verdict(crop)
        health = health_verdict(mqr)
        cn = cons.get(seg)
        grade = compose(align, health, cn)
        orr = orient.get(seg)
        # M5: mixed orientation (normal field flips sign vs the curved axis) caps at C —
        # measured baseline: a good mesh reads ~0.85 consistency; below 0.7 = damaged field.
        if orr and orr.get("consistency") is not None and orr["consistency"] < 0.70                 and grade in ("A", "B"):
            grade = "C"
        # fragment gate, AREA-normalized: the legacy 100k-cell cutoff assumed Paris4's
        # 20-vox cells at 2.4um (= 0.23 cm^2). Cell counts don't transfer across grids
        # (an 8-vox/cell Khartes wrap at 4.317um has ~7x less area per cell).
        if mqr:
            if mqr.get("area_vox2") is not None:
                if mqr["area_vox2"] * (args.um ** 2) / 1e8 < 0.23:
                    grade = "E"
            elif mqr.get("valid", 10**9) < 100_000:
                grade = "E"  # legacy cell-count gate (pre-area meshqual records)
        card = {"scroll": args.scroll, "segment": seg,
                "fxsurf": f"/tmp/gtqc/fxsurf/{seg}.fxsurf",
                "align": {"tier": align[0], **align[1]},
                "health": {"verdict": health[0], **health[1]},
                "consist": cn or {},
                "orientation": orient.get(seg) or {},
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
