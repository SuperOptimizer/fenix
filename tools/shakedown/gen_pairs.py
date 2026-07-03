#!/usr/bin/env python3
"""Generate the FULL-CORPUS pairs file: every imported .fxsurf paired with its source
volume's zarr URL (per-VOLUME on-demand cache) + um= from the manifest."""
import json, os, re, urllib.request

MAN = json.load(open("/root/segments-manifest.json"))
FX = "/workspace/fenix/data/training/fxsurf"
CACHES = "/workspace/caches"
os.makedirs(CACHES, exist_ok=True)
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com"

def list_vols(scroll):
    url = f"{B}/?list-type=2&delimiter=/&prefix={scroll}/volumes/"
    x = urllib.request.urlopen(url, timeout=30).read().decode()
    return re.findall(r"<Prefix>[^<]*/volumes/([^<]+)/</Prefix>", x)

volmap = {}
for scroll in MAN["scrolls"]:
    for v in list_vols(scroll):
        volmap[(scroll, v.split("-")[0])] = v

out, missing = [], 0
for scroll, segs in MAN["scrolls"].items():
    for seg, meshes in segs.items():
        for m in meshes:
            name = m["tifxyz"].rstrip("/").split("/")[-1]
            base = name[:-7] if name.endswith(".tifxyz") else f"{seg}-{name}"
            fx = f"{FX}/{scroll}/{base}.fxsurf"
            vol, um = m.get("volume"), m.get("um")
            if not vol or not um or um > 10:  # skip unnamed-variant + 45um
                continue
            if not os.path.exists(fx):
                missing += 1
                continue
            z = volmap.get((scroll, vol))
            if not z:
                continue
            cache = f"{CACHES}/{scroll}_{vol}.fxvol"
            out.append(f"{fx} {cache}@{B}/{scroll}/volumes/{z}/0 - um={um}")
with open("/root/pairs_full.txt", "w") as f:
    f.write("\n".join(out) + "\n")
vols = len(set(l.split()[1] for l in out))
print(f"{len(out)} pairs written ({missing} fxsurf missing), {vols} distinct volume caches")
