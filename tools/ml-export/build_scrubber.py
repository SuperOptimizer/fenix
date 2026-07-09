"""Build a self-contained interactive HTML scrubber from a slab's ct.npy + prob.npy.

Bakes every z-slice as CT-grayscale PNG + prob PNG (data URIs) into one HTML file:
z-slider, CT/pred/overlay toggles, live threshold + opacity, click-to-annotate with
notes, JSON export of annotations. Opens standalone in any browser.

Usage: build_scrubber.py <slab_dir> [--out panel.html] [--maxdim 768]
"""
import sys, os, io, base64, argparse
import numpy as np
from PIL import Image


def png_datauri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def jpg_datauri(arr, q=80):
    buf = io.BytesIO()
    Image.fromarray(arr).convert("L").save(buf, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slab_dir")
    ap.add_argument("--out")
    ap.add_argument("--maxdim", type=int, default=512, help="downscale slices to this max side")
    ap.add_argument("--jpq", type=int, default=80, help="CT jpeg quality")
    args = ap.parse_args()
    d = args.slab_dir
    ct = np.load(f"{d}/ct.npy")      # [D,H,W] u8
    prob = np.load(f"{d}/prob.npy")  # [D,H,W] u8 (prob*255)
    D, H, W = ct.shape
    meta = open(f"{d}/meta.txt").read().strip() if os.path.exists(f"{d}/meta.txt") else ""
    org = [0, 0, 0]
    for tok in meta.split():
        if tok.startswith("org="):
            org = [int(v) for v in tok[4:].split(",")]

    sc = min(1.0, args.maxdim / max(H, W))
    dw, dh = int(W*sc), int(H*sc)
    ct_uris, prob_uris = [], []
    for zi in range(D):
        ci = Image.fromarray(ct[zi]).resize((dw, dh), Image.BILINEAR)
        pi = Image.fromarray(prob[zi]).resize((dw, dh), Image.BILINEAR)
        ct_uris.append(jpg_datauri(np.array(ci), args.jpq))
        prob_uris.append(png_datauri(np.array(pi)))
    print(f"encoded {D} slices at {dw}x{dh}")

    import json
    payload = json.dumps({"ct": ct_uris, "prob": prob_uris, "D": D, "W": dw, "H": dh,
                          "org": org, "sx": W, "meta": meta})
    out = args.out or f"{d}/scrubber.html"
    open(out, "w").write(HTML.replace("__PAYLOAD__", payload))
    mb = os.path.getsize(out) / 1e6
    print(f"wrote {out} ({mb:.1f} MB)")


HTML = r"""<!doctype html><html><head><meta charset=utf8><title>surface scrubber</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:13px system-ui}
 #wrap{display:flex;height:100vh}
 #main{flex:1;display:flex;align-items:center;justify-content:center;position:relative}
 #cv{image-rendering:pixelated;max-width:100%;max-height:100vh;cursor:crosshair}
 #side{width:250px;padding:12px;background:#191919;overflow:auto}
 .row{margin:10px 0}
 label{display:block;margin-bottom:3px;color:#9cf}
 input[type=range]{width:100%}
 button{background:#2a2a3a;color:#ddd;border:1px solid #445;padding:5px 9px;border-radius:4px;cursor:pointer;margin:2px}
 button:hover{background:#3a3a4a}
 .k{color:#7c7;font-weight:bold}
 #notes{width:100%;height:60px;background:#0d0d0d;color:#ddd;border:1px solid #334}
 #anns{font-size:11px;max-height:200px;overflow:auto;margin-top:6px}
 .a{padding:2px 4px;border-bottom:1px solid #222;cursor:pointer}
 .a:hover{background:#222}
</style></head><body>
<div id=wrap>
 <div id=main><canvas id=cv></canvas></div>
 <div id=side>
  <div class=row><label>z-slice: <span id=zlab class=k></span></label>
   <input type=range id=z min=0 value=0></div>
  <div class=row>
   <button onclick="mode='ct'">CT</button>
   <button onclick="mode='pred'">Pred</button>
   <button onclick="mode='over'">Overlay</button>
  </div>
  <div class=row><label>threshold: <span id=tlab class=k>0.40</span></label>
   <input type=range id=thr min=0 max=100 value=40></div>
  <div class=row><label>overlay opacity: <span id=olab class=k>0.6</span></label>
   <input type=range id=op min=0 max=100 value=60></div>
  <div class=row>
   <label>annotation note (click image to place)</label>
   <textarea id=notes placeholder="e.g. pred misses faint sheet here"></textarea>
   <div style=margin-top:4px>
     <button onclick="tag='GOOD'">tag GOOD</button>
     <button onclick="tag='BAD'">tag BAD</button>
     <button onclick="tag='FAINT'">tag FAINT</button>
     <span id=taglab class=k>BAD</span>
   </div>
  </div>
  <div class=row>
   <button onclick=exportJSON()>export annotations</button>
   <button onclick=clearAnns()>clear</button>
  </div>
  <div id=anns></div>
  <div class=row style=color:#666;font-size:11px id=metabox></div>
 </div>
</div>
<script>
const P=__PAYLOAD__;
let mode='over', tag='BAD', z=0, thr=0.40, op=0.6;
let anns=[];  // {z, x, y, tag, note}  x,y in display px
const cv=document.getElementById('cv'), cx=cv.getContext('2d');
cv.width=P.W; cv.height=P.H;
const ctImg=new Image(), prImg=new Image();
document.getElementById('z').max=P.D-1;
document.getElementById('metabox').textContent=P.meta;
const scale=P.sx/P.W;  // full-res voxels per display px (for reporting real coords)

function load(){ ctImg.src=P.ct[z]; prImg.src=P.prob[z]; }
ctImg.onload=prImg.onload=draw;
function draw(){
  cx.clearRect(0,0,cv.width,cv.height);
  if(mode==='ct'){ cx.drawImage(ctImg,0,0); }
  else if(mode==='pred'){ cx.drawImage(prImg,0,0); }
  else {
    cx.drawImage(ctImg,0,0);
    // prob overlay: threshold + hot tint via temp canvas
    const t=document.createElement('canvas'); t.width=P.W;t.height=P.H;
    const tc=t.getContext('2d'); tc.drawImage(prImg,0,0);
    const d=tc.getImageData(0,0,P.W,P.H), a=d.data;
    for(let i=0;i<a.length;i+=4){
      let v=a[i]/255;                       // prob
      if(v<thr){a[i+3]=0;continue;}
      let s=(v-thr)/(1-thr);
      a[i]=Math.min(255,s*3*255); a[i+1]=Math.max(0,Math.min(255,(s*3-1)*255));
      a[i+2]=Math.max(0,Math.min(255,(s*3-2)*255)); a[i+3]=op*255;
    }
    tc.putImageData(d,0,0); cx.drawImage(t,0,0);
  }
  // markers on this z
  for(const m of anns.filter(a=>a.z===z)){
    cx.strokeStyle=m.tag==='GOOD'?'#3f6':m.tag==='FAINT'?'#fd3':'#f44';
    cx.lineWidth=2; cx.beginPath(); cx.arc(m.x,m.y,8,0,7); cx.stroke();
  }
}
document.getElementById('z').oninput=e=>{z=+e.target.value;
  document.getElementById('zlab').textContent=(P.org[0]+z)+' (i='+z+')'; load();};
document.getElementById('thr').oninput=e=>{thr=e.target.value/100;
  document.getElementById('tlab').textContent=thr.toFixed(2); draw();};
document.getElementById('op').oninput=e=>{op=e.target.value/100;
  document.getElementById('olab').textContent=op.toFixed(2); draw();};
setInterval(()=>{document.getElementById('taglab').textContent=tag;},200);
cv.onclick=e=>{
  const r=cv.getBoundingClientRect();
  const x=(e.clientX-r.left)*cv.width/r.width, y=(e.clientY-r.top)*cv.height/r.height;
  const note=document.getElementById('notes').value;
  const vz=P.org[0]+z, vy=Math.round(P.org[1]+y*scale), vx=Math.round(P.org[2]+x*scale);
  anns.push({z,x,y,tag,note,vz,vy,vx}); renderAnns(); draw();
};
function renderAnns(){
  document.getElementById('anns').innerHTML=anns.map((a,i)=>
   `<div class=a onclick="z=${a.z};document.getElementById('z').value=${a.z};document.getElementById('z').oninput({target:{value:${a.z}}})">`
   +`[${a.tag}] z=${a.vz} y=${a.vy} x=${a.vx} ${a.note?('- '+a.note):''}</div>`).join('');
}
function exportJSON(){
  const b=new Blob([JSON.stringify({org:P.org,slab:[P.D,P.sx,P.sx],anns},null,2)],{type:'application/json'});
  const u=URL.createObjectURL(b), a=document.createElement('a');
  a.href=u; a.download='annotations.json'; a.click();
}
function clearAnns(){anns=[];renderAnns();draw();}
document.getElementById('z').oninput({target:{value:0}});
</script></body></html>"""


if __name__ == "__main__":
    main()
