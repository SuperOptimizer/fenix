"""Grade the GT segment corpus for training-fitness -> per-segment scorecard JSON.

Chains the existing torch-free QC oracles (surf-qc delta+profile, surf-consist) over the
imported .fxsurf corpus and emits one scorecard per segment. Deterministic auto-tiling:
the trust-tile size is derived from the offset-field autocorrelation length (½ the length
at which du decorrelates), floored by the min-cells-per-tile significance requirement — not
a magic constant.

Plan: docs/design/gt-autograde-improve.md. This is the GRADE stage (loop 1, no model).

Usage:
  grade_corpus.py --scroll PHercParis4 --ct <zarr-url> [--fxdir /tmp/gtqc/fxsurf] \
      [--out /tmp/gtqc/scorecards] [--jobs 8]
"""
import sys, os, json, argparse, subprocess, re, math, glob
import numpy as np

FENIX = "/home/forrest/fenix/build-release/fenix"


def run(cmd, timeout=600):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def parse_profile(txt):
    """surf-qc-profile line -> dict of the fields we grade on."""
    m = {}
    for k, pat in [("pap", r"on-papyrus (\d+)%"), ("n_offs", r"n_offs=(\d+)"),
                   ("ridge", r"ridge (\d+)%"), ("air", r"AIR (\d+)%"),
                   ("nopeak", r"no-peak (\d+)%"), ("certified", r"certified (\d+)%"),
                   ("med_off", r"median-offset (-?\d+)"), ("iqr", r"offset-IQR (-?\d+)"),
                   ("coherent", r"coherent (-?\d+)%"), ("n", r"n=(\d+)")]:
        g = re.search(pat, txt)
        if g:
            m[k] = int(g.group(1))
    return m


def offset_autocorr_len(tsv_path, nu, nv):
    """Decorrelation length (in uv cells) of the per-cell offset field du.
    Sparse samples -> bin du onto the uv grid, estimate the lag at which the spatial
    autocorrelation of du drops below 1/e. Returns a length in uv-CELLS, or None."""
    try:
        d = np.loadtxt(tsv_path)
    except Exception:
        return None
    if d.ndim != 2 or len(d) < 20:
        return None
    u, v, du = d[:, 0], d[:, 1], d[:, 2]
    du = du - du.mean()
    if du.std() < 1e-6:
        return 0.0  # perfectly flat offset -> uniform, tile can be large
    # pairwise: correlation of du vs uv-distance, binned
    n = len(du)
    idx = np.random.default_rng(0).choice(n, min(n, 400), replace=False)
    uu, vv, dd = u[idx], v[idx], du[idx]
    dist = np.sqrt((uu[:, None] - uu[None, :])**2 + (vv[:, None] - vv[None, :])**2)
    prod = dd[:, None] * dd[None, :]
    var = dd.var()
    maxd = np.percentile(dist, 90)
    bins = np.linspace(0, maxd, 12)
    ac = []
    for i in range(len(bins)-1):
        mask = (dist >= bins[i]) & (dist < bins[i+1])
        ac.append(prod[mask].mean() / var if mask.sum() > 5 else np.nan)
    ac = np.array(ac)
    centers = (bins[:-1] + bins[1:]) / 2
    below = np.where(ac < 1/math.e)[0]
    return float(centers[below[0]]) if len(below) else float(centers[-1])


def auto_tile(nu, nv, acorr_len, rk=8):
    """Deterministic trust-tile size (uv cells). Half the offset-field decorrelation length,
    floored so each tile still holds >= ~rk sampleable cells for a significant delta."""
    min_tile = max(16, int(math.ceil(math.sqrt(rk))) * 4)  # significance floor
    if acorr_len is None or acorr_len <= 0:
        return min(256, max(min_tile, min(nu, nv) // 4))
    t = int(round(acorr_len / 2))
    return int(np.clip(t, min_tile, min(nu, nv)))


def grade(card):
    """Delegates to the CANONICAL grading in scorecard.py (adversarial-review P0.1 —
    three divergent graders collapsed to one). Keeps the fragment/cross/coherence gates."""
    from scorecard import align_verdict
    p = card.get("profile", {})
    if card.get("n_valid", 0) < 100_000:
        return "E"
    if 0 <= p.get("coherent", 100) < 25:
        return "E"
    iqr = p.get("iqr", -1)
    tier, _ = align_verdict({"pap_med": p.get("pap"), "iqr_med": iqr if iqr >= 0 else None,
                             "coh_med": p.get("coherent"), "med_off_med": p.get("med_off"),
                             "ridge_med": p.get("ridge", 0)})
    return tier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--ct", required=True, help="zarr level-0 URL")
    ap.add_argument("--cache", default=None, help="persistent local CT cache .fxvol (huge speedup)")
    ap.add_argument("--fxdir", default="/tmp/gtqc/fxsurf")
    ap.add_argument("--out", default="/tmp/gtqc/scorecards")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--off", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # persistent local cache: cold-fill once, then every re-QC of an overlapping region is warm
    cache = args.cache or f"/tmp/gtqc/ctcache_{args.scroll}.fxvol"
    ct = f"{cache}@{args.ct}"
    metas = sorted(glob.glob(f"{args.fxdir}/{args.scroll}__*.fxsurf"))
    print(f"{args.scroll}: {len(metas)} segments to grade")

    for i, fx in enumerate(metas):
        seg = os.path.basename(fx).replace(".fxsurf", "")
        card_path = f"{args.out}/{seg}.json"
        if os.path.exists(card_path):
            print(f"[{i+1}/{len(metas)}] {seg}  (cached)"); continue
        off_tsv = f"/tmp/gtqc/_off_{seg}.tsv"
        info = run([FENIX, "fxinfo", fx, "--json"])
        try:
            meta = json.loads(info.strip().splitlines()[-1])
            nu, nv, nvalid = meta["nu"], meta["nv"], meta["valid"]
        except Exception:
            nu = nv = nvalid = 0
        prof_txt = run([FENIX, "surf-qc", ct, fx, f"k={args.k}", f"off={args.off}",
                        "profile=1", f"offsets={off_tsv}"])
        prof = parse_profile(prof_txt)
        acorr = offset_autocorr_len(off_tsv, nu, nv)
        tile = auto_tile(nu, nv, acorr)
        # region trust grid at the auto-derived tile size
        trust_path = f"/tmp/gtqc/_trust_{seg}.txt"
        run([FENIX, "surf-qc", ct, fx, f"off={args.off}", f"tile={tile}",
             f"regions={trust_path}", f"min_delta=5"])
        tp = tf = tu = 0
        try:
            for ln in open(trust_path).read().splitlines()[1:]:
                tp += ln.count("P"); tf += ln.count("F"); tu += ln.count("?")
        except Exception:
            pass
        card = {"scroll": args.scroll, "segment": seg, "fxsurf": fx,
                "n_valid": nvalid, "nu": nu, "nv": nv,
                "profile": prof, "acorr_len": acorr, "tile": tile,
                "trust": {"pass": tp, "fail": tf, "unk": tu},
                "consist": []}  # filled by the consist pass (needs neighbors)
        card["grade"] = grade(card)
        json.dump(card, open(card_path, "w"), indent=2)
        print(f"[{i+1}/{len(metas)}] {seg}  grade={card['grade']}  "
              f"ridge={prof.get('ridge','?')}% iqr={prof.get('iqr','?')} "
              f"coh={prof.get('coherent','?')}% tile={tile} "
              f"trust P/F={tp}/{tf}")

    # summary
    cards = [json.load(open(p)) for p in glob.glob(f"{args.out}/{args.scroll}__*.json")]
    from collections import Counter
    print(f"\n{args.scroll} grades:", dict(sorted(Counter(c['grade'] for c in cards).items())))


if __name__ == "__main__":
    main()
