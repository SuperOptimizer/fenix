#!/usr/bin/env python3
"""ink_gallery.py — review gallery for hunt.sh output (stdlib only).

One card per segment: the max-projected INK probability map beside the mean-projected
papyrus texture, plus the hunt log verdict (OK / FRAME_FAIL / *_FAIL). Keyboard triage
(i = ink!, n = nothing, u = unsure) with verdicts.json export — same pattern as the
label-QC gallery.

Usage: ink_gallery.py --work /workspace/hunt/paris3 --log /root/hunt_p3.log --out index.html
"""
import argparse
import html
import json
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    verdicts = {}
    if args.log and os.path.exists(args.log):
        for ln in open(args.log, errors="replace"):
            m = re.match(r"(OK|FRAME_FAIL|IMPORT_FAIL|RENDER_FAIL|INK_FAIL) (\S+)", ln.strip())
            if m:
                verdicts[m.group(2)] = m.group(1)

    outdir = os.path.dirname(os.path.abspath(args.out))
    cards = []
    for f in sorted(os.listdir(args.work)):
        if not f.endswith(".ink.jpg"):
            continue
        seg = f[:-8]
        ink = os.path.relpath(os.path.join(args.work, f), outdir)
        tex = os.path.relpath(os.path.join(args.work, seg + ".tex.jpg"), outdir)
        size = os.path.getsize(os.path.join(args.work, f))
        cards.append({"seg": seg, "ink": ink, "tex": tex,
                      "status": verdicts.get(seg, "OK"), "jpg_bytes": size})
    # JPEG size of the ink map is a crude busy-ness prior: blank maps compress tiny
    cards.sort(key=lambda c: -c["jpg_bytes"])
    fails = [(s, v) for s, v in verdicts.items() if v != "OK"]

    page = """<!doctype html><html><head><meta charset="utf-8"><title>fenix ink hunt</title>
<style>
body{font:14px system-ui;margin:0;background:#0d0d0d;color:#ddd}
#bar{position:sticky;top:0;background:#191919;padding:8px 12px;z-index:9;border-bottom:1px solid #333}
button{background:#333;color:#ddd;border:1px solid #555;padding:4px 10px;cursor:pointer;margin-right:6px}
.card{margin:14px;background:#181818;border:2px solid #333;padding:8px}
.card.ink{border-color:#2a6}.card.nothing{border-color:#555}.card.unsure{border-color:#59f}
.pair{display:flex;gap:8px}
.pair img{max-width:46vw;max-height:70vh;object-fit:contain;background:#000}
.meta{font:12px monospace;color:#9ab;padding:4px 2px}
.failbox{font:12px monospace;color:#e77;padding:6px 12px;white-space:pre}
</style></head><body>
<div id="bar"><button onclick="exportV()">Export verdicts.json</button>
<span id="count"></span>
<span style="color:#888;margin-left:12px">click a card: <b>i</b>nk! / <b>n</b>othing / <b>u</b>nsure — ink map LEFT, papyrus texture RIGHT</span></div>
__FAILS__
<div id="grid"></div>
<script>
const cards = __CARDS__;
let V = JSON.parse(localStorage.getItem("fenixink")||"{}");
let sel = null;
function render(){
 const g = document.getElementById("grid"); g.innerHTML = "";
 for(const c of cards){
  const v = V[c.seg];
  const div = document.createElement("div");
  div.className = "card" + (v ? " "+v : "");
  div.onclick = ()=>{sel=c.seg; document.querySelectorAll(".card").forEach(e=>e.style.outline=""); div.style.outline="2px solid #fff";};
  div.innerHTML = `<div class="meta"><b>${c.seg}</b>  ${c.status}  ${(c.jpg_bytes/1024).toFixed(0)}kB ink-map  ${v?("— "+v.toUpperCase()):""}</div>
   <div class="pair"><img loading="lazy" src="${c.ink}"><img loading="lazy" src="${c.tex}"></div>`;
  g.appendChild(div);
 }
 document.getElementById("count").textContent = ` ${cards.length} segments | ${Object.keys(V).length} reviewed`;
}
document.addEventListener("keydown", e=>{
 if(!sel) return;
 const m = {i:"ink", n:"nothing", u:"unsure"}[e.key];
 if(!m) return;
 V[sel]=m; localStorage.setItem("fenixink", JSON.stringify(V)); render();
});
function exportV(){
 const blob = new Blob([JSON.stringify(V,null,1)],{type:"application/json"});
 const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="ink_verdicts.json"; a.click();
}
render();
</script></body></html>"""
    failbox = ""
    if fails:
        failbox = '<div class="failbox">pipeline failures:\n' + "\n".join(
            f"  {v}  {html.escape(s)}" for s, v in sorted(fails)) + "</div>"
    page = page.replace("__CARDS__", json.dumps(cards)).replace("__FAILS__", failbox)
    with open(args.out, "w") as f:
        f.write(page)
    print(f"ink-gallery: {len(cards)} segments, {len(fails)} failures -> {args.out}")


if __name__ == "__main__":
    main()
