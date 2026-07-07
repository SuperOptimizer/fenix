#!/usr/bin/env python3
"""Crawl the Vesuvius Challenge open-data bucket into a TOML manifest.

Records, per scroll: the CT volumes (OME-zarr name + shape/chunks/dtype/pyramid levels)
and the segments (id + available tifxyz coordinate-grid variants). Photos, surface
renders, ink/surface predictions etc. are deliberately ignored. Only delimiter listings
and per-volume metadata objects are fetched — no key floods over zarr chunk trees.

Usage: python3 tools/opendata_manifest.py docs/data/open-data-manifest.toml

URL conventions (derivable from the manifest, not repeated per entry):
  volume zarr root : {base_url}/{scroll}/volumes/{volume}
  segment tifxyz   : {base_url}/{scroll}/segments/{segment}/mesh/{tifxyz}
"""
import datetime
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

BASE = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 3:
                raise
        except Exception:
            if attempt == 3:
                raise
    return None


def list_prefixes(prefix):
    out, token = [], None
    while True:
        url = f"{BASE}/?list-type=2&delimiter=/&prefix={urllib.parse.quote(prefix)}"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        root = ET.fromstring(get(url))
        out += [p.find(NS + "Prefix").text for p in root.iter(NS + "CommonPrefixes")]
        token = root.findtext(NS + "NextContinuationToken")
        if not token:
            break
    return out


def volume_meta(scroll, vol):
    m = {"name": vol.rstrip("/")}
    za = get(f"{BASE}/{scroll}volumes/{vol}0/.zarray")  # zarr v2 level 0
    if za is None:
        za = get(f"{BASE}/{scroll}volumes/{vol}zarr.json")  # zarr v3
    if za:
        j = json.loads(za)
        m["shape"] = j.get("shape")
        m["chunks"] = j.get("chunks") or j.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape")
        m["dtype"] = j.get("dtype") or j.get("data_type")
    attrs = get(f"{BASE}/{scroll}volumes/{vol}.zattrs")
    if attrs:
        m["levels"] = len(re.findall(rb'"path"', attrs))
    return m


def segment_meta(seg):
    sid = seg.rstrip("/").split("/")[-1]
    tifxyz = [
        sub.rstrip("/").split("/")[-1]
        for sub in list_prefixes(f"{seg}mesh/")
        if sub.rstrip("/").endswith(".tifxyz")
    ]
    return {"id": sid, "tifxyz": sorted(tifxyz)}


def toml_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main(out_path):
    lines = [
        "# Vesuvius Challenge open-data manifest — scrolls, CT volumes, segments.",
        f"# Generated {datetime.date.today().isoformat()} by tools/opendata_manifest.py (regenerate with:",
        "#   python3 tools/opendata_manifest.py docs/data/open-data-manifest.toml)",
        "# volume zarr root: {base_url}/{scroll}/volumes/{volume}",
        "# segment tifxyz:   {base_url}/{scroll}/segments/{segment}/mesh/{tifxyz}",
        f'base_url = "{BASE}"',
    ]
    for sc in list_prefixes(""):
        name = sc.rstrip("/")
        if name.startswith("_"):
            continue
        vols = list_prefixes(f"{sc}volumes/")
        segs = list_prefixes(f"{sc}segments/")
        with ThreadPoolExecutor(16) as ex:
            vmeta = list(ex.map(lambda v: volume_meta(sc, v[len(sc) + len("volumes/"):]), vols))
            smeta = list(ex.map(segment_meta, segs))
        if not vmeta and not smeta:
            continue  # photos-only fragments etc.
        print(f"{name}: {len(vmeta)} volumes, {len(smeta)} segments", file=sys.stderr)
        lines += ["", "[[scroll]]", f"id = {toml_str(name)}"]
        for v in vmeta:
            lines += ["", "  [[scroll.volume]]", f"  name = {toml_str(v['name'])}"]
            if v.get("shape"):
                lines.append(f"  shape = {v['shape']}  # z y x")
            if v.get("chunks"):
                lines.append(f"  chunks = {v['chunks']}")
            if v.get("dtype"):
                lines.append(f"  dtype = {toml_str(v['dtype'])}")
            if v.get("levels"):
                lines.append(f"  levels = {v['levels']}")
        for s in smeta:
            lines += ["", "  [[scroll.segment]]", f"  id = {toml_str(s['id'])}"]
            if s["tifxyz"]:
                items = ",\n    ".join(toml_str(t) for t in s["tifxyz"])
                lines.append(f"  tifxyz = [\n    {items},\n  ]")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
