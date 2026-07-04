#!/usr/bin/env python3
"""gallery.py — label-QC triage gallery generator (stdlib only).

Collects every label-quality signal we have per mesh (intensity delta, per-run audit CE,
consist involvement, trust-grid fail tiles) and emits ONE static index.html next to the
contact-sheet JPEGs: cards sorted worst-first, filterable, keyboard triage
(k=keep d=drop t=trust u=unsure, n=note), decisions in localStorage, Export button
downloads verdicts.json for tools/labelqc/apply_verdicts.py.

Usage:
  gallery.py --sheets /workspace/sheets --out /workspace/sheets/index.html \
      --qc /root/qc_p4.txt --qc /root/qc_cross.txt \
      --meshloss /workspace/run/p4v3_meshloss.json --meshloss /workspace/run/p4v4_meshloss.json \
      --consist /root/qc3.txt --trust /workspace/trust
"""
import argparse
import html
import json
import os
import re


def parse_qc(paths):
    delta = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for ln in open(p, errors="replace"):
            m = re.match(r"surf-qc (\S+?/([^/]+)\.fxsurf)\s+n=\d+.*delta ([+-][\d.]+)\s+(PASS|FAIL)", ln)
            if m:
                delta[m.group(2)] = (float(m.group(3)), m.group(4))  # last measurement wins
    return delta


def parse_meshloss(paths):
    ce = {}  # base -> {run: ce}
    for p in paths:
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        run = os.path.basename(p).replace("_meshloss.json", "")
        for r in d["meshes"]:
            base = os.path.basename(r["path"]).replace(".fxsurf", "")
            ce.setdefault(base, {})[run] = r["ce_ema"]
    return ce


def parse_consist(paths):
    inv = {}  # base -> {"CROSS": n, "OFFSET": n, "AGREE": n}
    for p in paths:
        if not os.path.exists(p):
            continue
        for ln in open(p, errors="replace"):
            m = re.match(r"surf-consist pair (\S+) <-> (\S+)\s+.*\s(AGREE|OFFSET|CROSS)\s*$", ln)
            if m:
                for mp in (m.group(1), m.group(2)):
                    base = os.path.basename(mp).replace(".fxsurf", "")
                    inv.setdefault(base, {}).setdefault(m.group(3), 0)
                    inv[base][m.group(3)] += 1
    return inv


def parse_trust(trust_dir):
    fails = {}
    if not trust_dir or not os.path.isdir(trust_dir):
        return fails
    for f in os.listdir(trust_dir):
        if not f.endswith(".trust"):
            continue
        body = open(os.path.join(trust_dir, f), errors="replace").read()
        grid = body.split("\n", 1)[1] if "\n" in body else ""
        fails[f[:-6]] = (grid.count("F"), sum(c in "PF?" for c in grid))
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", required=True, help="directory of <mesh>.jpg contact sheets")
    ap.add_argument("--out", required=True, help="output index.html (put it in the sheets dir)")
    ap.add_argument("--qc", action="append", default=[], help="surf-qc output file(s)")
    ap.add_argument("--meshloss", action="append", default=[], help="meshloss.json file(s)")
    ap.add_argument("--consist", action="append", default=[], help="surf-consist output file(s)")
    ap.add_argument("--trust", default="", help="trust-grid directory")
    args = ap.parse_args()

    delta = parse_qc(args.qc)
    ce = parse_meshloss(args.meshloss)
    consist = parse_consist(args.consist)
    trust = parse_trust(args.trust)

    cards = []
    outdir = os.path.dirname(os.path.abspath(args.out))
    for f in sorted(os.listdir(args.sheets)):
        if not f.endswith(".jpg"):
            continue
        base = f[:-4]
        d, verdict = delta.get(base, (None, ""))
        ces = ce.get(base, {})
        cv = consist.get(base, {})
        tf = trust.get(base, (None, None))
        scroll = "Paris4"
        m = re.search(r"-on-(\d+)-", base)
        if m:
            vol = m.group(1)
            scroll = {"20260102150214": "0139", "20251217075048": "1667",
                      "20241024131838": "0172", "20260411134726": "Paris4"}.get(vol, vol)
        # badness score for the default sort: low/negative delta, high CE, CROSS involvement
        bad = 0.0
        if d is not None:
            bad += max(0.0, 5.0 - d)
        bad += 3.0 * max(ces.values(), default=0.0)
        bad += 0.2 * cv.get("CROSS", 0)
        rel = os.path.relpath(os.path.join(os.path.abspath(args.sheets), f), outdir)
        cards.append({
            "id": base, "img": rel, "scroll": scroll, "delta": d, "qc": verdict,
            "ce": ces, "consist": cv, "trust_fail": tf[0], "trust_tot": tf[1], "bad": round(bad, 2),
        })
    cards.sort(key=lambda c: -c["bad"])

    payload = json.dumps(cards)
    page = """<!doctype html><html><head><meta charset="utf-8"><title>fenix label-QC triage</title>
<style>
body{font:14px system-ui;margin:0;background:#111;color:#ddd}
#bar{position:sticky;top:0;background:#1b1b1b;padding:8px 12px;z-index:9;border-bottom:1px solid #333}
#bar button{margin-right:6px;background:#333;color:#ddd;border:1px solid #555;padding:4px 10px;cursor:pointer}
#bar button.on{background:#2a6}
.card{display:inline-block;vertical-align:top;margin:10px;background:#1b1b1b;border:2px solid #333;padding:6px;max-width:840px}
.card.keep{border-color:#2a6}.card.drop{border-color:#c33}.card.trust{border-color:#fa0}.card.unsure{border-color:#59f}
.card img{max-width:820px;display:block}
.meta{font:12px monospace;color:#9ab;padding:4px 2px;white-space:pre-wrap}
.vlabel{font-weight:700;padding:1px 6px;border-radius:3px}
kbd{background:#333;padding:0 4px;border-radius:3px}
#help{color:#888;font-size:12px;margin-left:14px}
textarea{width:100%;background:#222;color:#ddd;border:1px solid #444;font:12px monospace}
</style></head><body>
<div id="bar">
 <button onclick="exportV()">Export verdicts.json</button>
 <button id="funsorted" onclick="toggleF('un')">unreviewed</button>
 <button id="ffail" onclick="toggleF('fail')">QC-FAIL only</button>
 <span id="scrolls"></span>
 <span id="count"></span>
 <span id="help">click a card, then <kbd>k</kbd>eep <kbd>d</kbd>rop <kbd>t</kbd>rust-mask <kbd>u</kbd>nsure <kbd>n</kbd>ote — auto-saved locally</span>
</div>
<div id="grid"></div>
<script>
const cards = __PAYLOAD__;
let V = JSON.parse(localStorage.getItem("fenixqc")||"{}");
let filt = {un:false, fail:false, scroll:null};
let sel = null;
function fmt(c){
 let s = `delta ${c.delta===null?"?":c.delta>=0?"+"+c.delta:c.delta} ${c.qc}  scroll ${c.scroll}`;
 const ces = Object.entries(c.ce).map(([k,v])=>`${k}:${v.toFixed(2)}`).join(" ");
 if(ces) s += `\\naudit CE ${ces}`;
 const cv = Object.entries(c.consist).map(([k,v])=>`${k}:${v}`).join(" ");
 if(cv) s += `\\nconsist ${cv}`;
 if(c.trust_fail!==null) s += `\\ntrust ${c.trust_fail}F/${c.trust_tot} tiles`;
 return s;
}
function render(){
 const g = document.getElementById("grid"); g.innerHTML = "";
 let shown = 0;
 for(const c of cards){
  const v = V[c.id];
  if(filt.un && v) continue;
  if(filt.fail && c.qc !== "FAIL") continue;
  if(filt.scroll && c.scroll !== filt.scroll) continue;
  const div = document.createElement("div");
  div.className = "card" + (v ? " "+v.verdict : "");
  div.id = "c_"+c.id;
  div.onclick = ()=>{sel=c.id; document.querySelectorAll(".card").forEach(e=>e.style.outline=""); div.style.outline="2px solid #fff";};
  div.innerHTML = `<div class="meta"><b>${c.id}</b>  <span class="vlabel">${v?v.verdict.toUpperCase():""}</span>\\n${fmt(c)}${v&&v.note?"\\nnote: "+v.note:""}</div><img loading="lazy" src="${c.img}">`;
  g.appendChild(div); shown++;
 }
 const done = Object.keys(V).length;
 document.getElementById("count").textContent = ` ${shown} shown | ${done}/${cards.length} reviewed`;
}
function setV(verdict){
 if(!sel) return;
 V[sel] = V[sel]||{}; V[sel].verdict = verdict;
 localStorage.setItem("fenixqc", JSON.stringify(V)); render();
}
document.addEventListener("keydown", e=>{
 if(e.target.tagName==="TEXTAREA") return;
 if(e.key==="k") setV("keep"); if(e.key==="d") setV("drop");
 if(e.key==="t") setV("trust"); if(e.key==="u") setV("unsure");
 if(e.key==="n" && sel){ const t=prompt("note for "+sel, (V[sel]&&V[sel].note)||""); if(t!==null){V[sel]=V[sel]||{verdict:"unsure"}; V[sel].note=t; localStorage.setItem("fenixqc", JSON.stringify(V)); render();} }
});
function toggleF(k){ filt[k]=!filt[k]; document.getElementById("f"+(k==="un"?"unsorted":k)).classList.toggle("on"); render(); }
const scrolls = [...new Set(cards.map(c=>c.scroll))];
const sp = document.getElementById("scrolls");
for(const s of scrolls){
 const b = document.createElement("button"); b.textContent = s;
 b.onclick = ()=>{filt.scroll = filt.scroll===s?null:s; [...sp.children].forEach(x=>x.classList.remove("on")); if(filt.scroll) b.classList.add("on"); render();};
 sp.appendChild(b);
}
function exportV(){
 const blob = new Blob([JSON.stringify(V,null,1)], {type:"application/json"});
 const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "verdicts.json"; a.click();
}
render();
</script></body></html>"""
    page = page.replace("__PAYLOAD__", payload)
    with open(args.out, "w") as f:
        f.write(page)
    print(f"gallery: {len(cards)} cards -> {args.out}")


if __name__ == "__main__":
    main()
