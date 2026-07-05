#!/usr/bin/env python3
"""box_prep_inst_corpus.py — reconstruct the PHercParis4 instance-training corpus on a
fresh GPU box, from segments-manifest.json + a fitted spiral model (corpus_v7.fxmodel).

Chain: manifest -> download each tifxyz (x/y/z.tif) from S3 -> `fenix import-tifxyz`
-> `fenix wrap-label` (produce .wrapcolor sidecars) -> emit pairs_inst_full/val.txt.

This is the checked-in successor to the ad-hoc pod steps. Idempotent: skips a mesh
whose .fxsurf + .wrapcolor already exist. Run on the box after the fenix build.

  python3 box_prep_inst_corpus.py \
      --manifest /root/segments-manifest.json \
      --model    /root/corpus_v7.fxmodel \
      --fenix    /workspace/fenix/build-release/fenix \
      --scroll   PHercParis4 --volume 20260411134726 \
      --workdir  /workspace/corpus --val 5 --wrapk 8
"""
import argparse, json, os, re, subprocess, sys, urllib.request, urllib.parse

B = "https://vesuvius-challenge-open-data.s3.amazonaws.com"


def fetch(url, dst):
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with open(dst, "wb") as f:
        f.write(data)
    return len(data)


def list_vols(scroll):
    url = f"{B}/?list-type=2&delimiter=/&prefix={scroll}/volumes/"
    x = urllib.request.urlopen(url, timeout=30).read().decode()
    return re.findall(rf"<Prefix>{re.escape(scroll)}/volumes/([^<]+)/</Prefix>", x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fenix", required=True)
    ap.add_argument("--scroll", default="PHercParis4")
    ap.add_argument("--volume", default="20260411134726")
    ap.add_argument("--workdir", default="/workspace/corpus")
    ap.add_argument("--caches", default="/workspace/caches")
    ap.add_argument("--val", type=int, default=5)
    ap.add_argument("--wrapk", type=int, default=8)
    ap.add_argument("--conf-tol", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=0, help="cap #meshes (0=all)")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    segs = man["scrolls"][args.scroll]
    fxdir = os.path.join(args.workdir, args.scroll)
    tifdir = os.path.join(args.workdir, "_tifxyz")
    os.makedirs(fxdir, exist_ok=True)
    os.makedirs(tifdir, exist_ok=True)
    os.makedirs(args.caches, exist_ok=True)

    # resolve the volume's zarr dir name (e.g. 20260411134726 -> full prefix)
    vmap = {v.split("-")[0]: v for v in list_vols(args.scroll)}
    zarr = vmap.get(args.volume)
    if not zarr:
        sys.exit(f"volume {args.volume} not found under {args.scroll}/volumes/")
    cache = os.path.join(args.caches, f"{args.scroll}_{args.volume}.fxvol")
    zurl = f"{cache}@{B}/{args.scroll}/volumes/{zarr}/0"

    # collect meshes on the target volume
    todo = []
    for seg, meshes in segs.items():
        for me in meshes:
            if me.get("volume") != args.volume:
                continue
            um = me.get("um")
            if not um or um > 10:
                continue
            pfx = me["tifxyz"].rstrip("/")
            base = pfx.split("/")[-1]
            base = base[:-7] if base.endswith(".tifxyz") else base
            todo.append((seg, pfx, base, um))
    todo.sort(key=lambda t: t[2])
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} meshes on {args.scroll}/{args.volume}", flush=True)

    fxsurfs, um_of = [], {}
    for i, (seg, pfx, base, um) in enumerate(todo):
        fx = os.path.join(fxdir, base + ".fxsurf")
        wc = fx + ".wrapcolor"
        um_of[fx] = um
        if os.path.exists(fx) and os.path.exists(wc):
            fxsurfs.append(fx)
            continue
        if not os.path.exists(fx):
            # download x/y/z.tif
            d = os.path.join(tifdir, base + ".tifxyz")
            os.makedirs(d, exist_ok=True)
            try:
                for t in ("x.tif", "y.tif", "z.tif", "meta.json"):
                    dst = os.path.join(d, t)
                    if not os.path.exists(dst):
                        fetch(f"{B}/{pfx}/{t}", dst)
            except Exception as e:
                print(f"  [{i}] FETCH FAIL {base}: {e}", flush=True)
                continue
            r = subprocess.run([args.fenix, "import-tifxyz", d, fx],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(fx):
                print(f"  [{i}] IMPORT FAIL {base}: {r.stderr.strip()[:160]}", flush=True)
                continue
        fxsurfs.append(fx)
        if i % 10 == 0:
            print(f"  imported {i}/{len(todo)}", flush=True)

    print(f"{len(fxsurfs)} fxsurf ready; wrap-labeling...", flush=True)
    # wrap-label in one batch (writes <fx>.wrapcolor next to each)
    need = [fx for fx in fxsurfs if not os.path.exists(fx + ".wrapcolor")]
    if need:
        cmd = [args.fenix, "wrap-label", f"model={args.model}",
               f"k={args.wrapk}", f"conf_tol={args.conf_tol}"] + \
              [f"surf={fx}" for fx in need]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout.strip()[-400:], flush=True)
        if r.returncode != 0:
            print("WRAP-LABEL stderr:", r.stderr.strip()[-400:], flush=True)

    labeled = [fx for fx in fxsurfs if os.path.exists(fx + ".wrapcolor")]
    print(f"{len(labeled)} wrap-labeled", flush=True)

    # deterministic val split (last N by name), rest train
    labeled.sort()
    val = labeled[-args.val:] if args.val else []
    train = labeled[:-args.val] if args.val else labeled

    def line(fx):
        return f"{fx} {zurl} - um={um_of[fx]} wrap={fx}.wrapcolor"

    with open("/root/pairs_inst_full.txt", "w") as f:
        f.write("\n".join(line(fx) for fx in train) + "\n")
    with open("/root/pairs_inst_val.txt", "w") as f:
        f.write("\n".join(line(fx) for fx in val) + "\n")
    print(f"WROTE /root/pairs_inst_full.txt ({len(train)}) + "
          f"/root/pairs_inst_val.txt ({len(val)}); cache={cache}", flush=True)


if __name__ == "__main__":
    main()
