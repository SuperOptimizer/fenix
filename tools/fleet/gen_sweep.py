#!/usr/bin/env python3
"""Generate the teacher-sweep manifest: PER-VOLUME union band blocks (deduped across meshes).

Runs WHERE THE FXSURF CORPUS LIVES. Naive per-mesh bbox sweeps measured 1.75 Pvox (245
overlapping meshes; spiral bboxes cover the scroll) — the unit of teacher work is a VOLUME
sub-block intersecting the union surface band, each voxel predicted once.

  gen_sweep.py <fenix-binary> <segments-manifest.json> <fxsurf-root> <out.sweep> [band_r=384]

Line format (consumed by teacher_sweep.sh; fxsurf paths are corpus-relative):
  <name> <zarr-url> <um> <z0> <y0> <x0> <D> <H> <W> <fxsurf,fxsurf,...>
"""
import json
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict

B = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
BLOCK, HALO = 1792, 128  # 2048^3 crops (halo overlaps seams), dense-decodable on a 62GB pod


def list_vols(scroll):
    url = f"{B}/?list-type=2&delimiter=/&prefix={scroll}/volumes/"
    x = urllib.request.urlopen(url, timeout=30).read().decode()
    return re.findall(r"<Prefix>[^<]*/volumes/([^<]+)/</Prefix>", x)


def main():
    fenix, man_path, fxroot, out_path = sys.argv[1:5]
    band_r = int(sys.argv[5]) if len(sys.argv) > 5 else 384
    man = json.load(open(man_path))
    volmap = {}
    for scroll in man["scrolls"]:
        for v in list_vols(scroll):
            volmap[(scroll, v.split("-")[0])] = v

    groups = defaultdict(list)  # (scroll, vol, um) -> [corpus-relative fxsurf]
    for scroll, segs in man["scrolls"].items():
        for seg, meshes in segs.items():
            for m in meshes:
                name = m["tifxyz"].rstrip("/").split("/")[-1]
                base = name[:-7] if name.endswith(".tifxyz") else f"{seg}-{name}"
                rel = f"{scroll}/{base}.fxsurf"
                vol, um = m.get("volume"), m.get("um")
                if not vol or not um or um > 10 or (scroll, vol) not in volmap:
                    continue
                groups[(scroll, vol, um)].append(rel)

    out = []
    for (scroll, vol, um), rels in sorted(groups.items()):
        url = f"{B}/{scroll}/volumes/{volmap[(scroll, vol)]}"  # bare root: ingest-band appends /<level>
        surfs = ",".join(f"{fxroot}/{r}" for r in rels)
        blocks = subprocess.run(
            [fenix, "band-blocks", surfs, f"block={BLOCK}", f"halo={HALO}", f"band_r={band_r}"],
            capture_output=True, text=True, check=True,
        ).stdout.split("\n")
        rel_list = ",".join(rels)
        for bl in blocks:
            if not bl.strip():
                continue
            z, y, x, d, h, w = bl.split()
            out.append(f"{scroll}_{vol}_{z}_{y}_{x} {url} {um} {z} {y} {x} {d} {h} {w} {rel_list}")
        print(f"{scroll}/{vol}: {len(rels)} meshes -> {sum(1 for b in blocks if b.strip())} blocks")

    with open(out_path, "w") as f:
        f.write("\n".join(out) + "\n")
    gvox = sum(int(l.split()[6]) * int(l.split()[7]) * int(l.split()[8]) for l in out) / 1e9
    print(f"{len(out)} block items, {gvox:.0f} Gvox gross (band-filtered further at ingest/predict)")


if __name__ == "__main__":
    main()
