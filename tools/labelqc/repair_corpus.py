"""Triage + repair the graded GT corpus (loop 1, no model). Reads scorecards from
grade_corpus.py, applies the repair operator matched to each grade, RE-GRADES the repaired
mesh, and records the before/after. Provenance-preserving: never overwrites the original
.fxsurf — writes a <seg>.repaired.fxsurf + updates the scorecard.

  B (uniform offset)  -> surf-repair (small max_shift, tight smoothing) then re-grade
  C (warped)          -> surf-repair offset-field (larger max_shift) then re-grade
  D (partial)         -> keep the trust grid; rasterizer blanks FAIL tiles (no geom change)
  A                   -> accept as-is
  E                   -> quarantine (no repair attempt)

Plan: docs/design/gt-autograde-improve.md.

Usage: repair_corpus.py --scroll PHercParis4 --ct <zarr-url> --cache <ct.fxvol> \
       [--scorecards DIR] [--outdir DIR]
"""
import sys, os, json, argparse, subprocess, glob, re

FENIX = "/home/forrest/fenix/build-release/fenix"


def run(cmd, timeout=600):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def profile_of(ct, fx, k=100, off=12):
    txt = run([FENIX, "surf-qc", ct, fx, f"k={k}", f"off={off}", "profile=1"])
    m = {}
    for key, pat in [("pap", r"on-papyrus (\d+)%"), ("ridge", r"ridge (\d+)%"), ("med_off", r"median-offset (-?\d+)"),
                     ("iqr", r"offset-IQR (-?\d+)"), ("coherent", r"coherent (\d+)%")]:
        g = re.search(pat, txt)
        if g:
            m[key] = int(g.group(1))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--scorecards", default="/tmp/gtqc/scorecards")
    ap.add_argument("--outdir", default="/tmp/gtqc/repaired")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ct = f"{args.cache}@{args.ct}"

    # ONLY the canonical scorecard files (fixes the {seg}.json / {seg}.card.json collision
    # that KeyError'd on c["fxsurf"] — adversarial-review P0.1)
    cards = sorted(glob.glob(f"{args.scorecards}/{args.scroll}__*.card.json"))
    print(f"{args.scroll}: {len(cards)} scorecards to triage")
    moved = {"A->A": 0, "repaired_up": 0, "repaired_flat": 0, "quarantine": 0, "trustmask": 0}

    for cp in cards:
        c = json.load(open(cp))
        g, fx = c["grade"].rstrip("?"), c["fxsurf"]  # borderline (X?) repairs as its base tier
        seg = c["segment"]
        if g == "A":
            # capture-range doctrine (gt-metrics-hardening.md): QC cannot see sub-5-vox
            # systematic offsets, so even A segments get one gentle unconditional snap
            # (measured-conservative on good meshes; fixes what QC can't detect).
            out = f"{args.outdir}/{seg}.repaired.fxsurf"
            run([FENIX, "surf-repair", ct, fx, out, "grid=8", "off=12",
                 "max_shift=3", "smooth=2"])
            c["decision"] = "use-repaired" if os.path.exists(out) else "use"
            if os.path.exists(out):
                c["repair"] = {"applied": True, "kind": "gentle-unconditional",
                               "max_shift": 3, "repaired_fxsurf": out}
            moved["A->A"] += 1
        elif g == "E":
            c["decision"] = "quarantine"; moved["quarantine"] += 1
        elif g == "D":
            c["decision"] = "trust-mask"; moved["trustmask"] += 1  # trust grid already emitted
        elif g in ("B", "C"):
            out = f"{args.outdir}/{seg}.repaired.fxsurf"
            max_shift = 4 if g == "B" else 8
            before = c.get("profile", {})
            # ITERATE to convergence: pass 1 fixes the normals (repair searches along the
            # local normal — scrambled normals miss the adjacent sheet), pass 2+ then snap
            # what pass 1 couldn't. Measured on the ridge-6% crop: 18%->33%->43%, fixpoint
            # by pass 3 (mean|shift| ~0.6). Stop when mean|shift| < 0.7 or 3 passes.
            src = fx
            for it in range(3):
                r = run([FENIX, "surf-repair", ct, src, out, "grid=8", "off=12",
                         f"max_shift={max_shift}", "smooth=2"])
                g2 = re.search(r"mean\|shift\| ([0-9.]+)", r)
                src = out
                if g2 and float(g2.group(1)) < 0.7:
                    break
            after = profile_of(ct, out) if os.path.exists(out) else {}
            # gated WIDE-SEARCH escalation: if much of the mesh still finds no peak, the
            # offset may be large-but-coherent (recoverable) — try one wide pass; accept
            # only if ridge improves AND coherence doesn't drop (a wrong-wrap snap scrambles
            # coherence; measured: incoherent far peaks get refused by outlier-rejection).
            if after.get("ridge", 0) < 60 and 100 - after.get("ridge", 0) > 30:
                wide = f"{args.outdir}/{seg}.wide.fxsurf"
                run([FENIX, "surf-repair", ct, out, wide, "grid=8", "off=32",
                     "search=32", "max_shift=28", "smooth=2"])
                w = profile_of(ct, wide) if os.path.exists(wide) else {}
                if (w.get("ridge", 0) > after.get("ridge", 0) + 5
                        and w.get("coherent", 0) >= after.get("coherent", 0) - 3):
                    os.replace(wide, out)
                    after = w
                else:
                    # M3 (gt-metrics-hardening.md): the wide pass FOUND far peaks but the
                    # accept-gate refused them — the only computed hint of coherent
                    # wrong-wrap capture. Surface it instead of discarding silently.
                    c.setdefault("cautions", []).append("wide-search-far-peaks-refused")
                    if os.path.exists(wide):
                        os.remove(wide)
            def _iqr(d):
                v = d.get("iqr", -1)
                return v if v >= 0 else None  # -1 = unmeasured, never "tight"
            b_iqr, a_iqr = _iqr(before), _iqr(after)
            b_ridge, a_ridge = before.get("ridge", 0), after.get("ridge", 0)
            b_pap, a_pap = before.get("pap", 0), after.get("pap", 0)
            improved = (a_pap > b_pap + 3)                 or (a_iqr is not None and b_iqr is not None and a_iqr < b_iqr - 1)                 or (a_ridge > b_ridge + 5)
            c["repair"] = {"applied": True, "kind": "offset-field", "max_shift": max_shift,
                           "before": before, "after": after, "improved": improved,
                           "repaired_fxsurf": out}
            if improved:
                from scorecard import align_verdict  # SINGLE grading source (P0.1)
                ga, _ = align_verdict({"pap_med": a_pap, "iqr_med": a_iqr,
                                       "coh_med": after.get("coherent"),
                                       "med_off_med": after.get("med_off")})
                c["decision"] = "use-repaired"; c["grade_after"] = ga
                moved["repaired_up"] += 1
            else:
                c["decision"] = "trust-mask"; moved["repaired_flat"] += 1
        json.dump(c, open(cp, "w"), indent=2)
        print(f"{seg[:44]:44s} {g} -> {c.get('decision')}  "
              + (f"iqr {c['repair']['before'].get('iqr')}→{c['repair']['after'].get('iqr')}"
                 if "repair" in c else ""))

    print(f"\n{args.scroll} triage:", moved)


if __name__ == "__main__":
    main()
