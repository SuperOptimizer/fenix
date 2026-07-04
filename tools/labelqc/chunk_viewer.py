#!/usr/bin/env python3
"""chunk_viewer.py — self-contained 3D chunk viewer for label triage (stdlib only).

Takes .qcchunk files (fenix qc-chunk: JSON header line + N^3 CT u8 + N^3 band u8) and
emits ONE html with a hand-written WebGL2 volume raycaster: raw CT as a rotatable volume
render, the mesh band overlaid in red. Drag = rotate, wheel = zoom, sliders = CT window /
band opacity / density, chunk selector to flip between sample points.

Usage: chunk_viewer.py out.html chunk_0.qcchunk [chunk_1.qcchunk ...]
"""
import base64
import json
import sys


def read_chunk(path):
    b = open(path, "rb").read()
    nl = b.index(b"\n")
    meta = json.loads(b[:nl])
    n = meta["size"]
    body = b[nl + 1:]
    assert len(body) == 2 * n ** 3, f"{path}: expected {2*n**3} payload bytes, got {len(body)}"
    return meta, base64.b64encode(body).decode()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>fenix 3D chunk triage</title>
<style>
body{margin:0;background:#0b0b0b;color:#ddd;font:13px system-ui;overflow:hidden}
#ui{position:fixed;top:8px;left:8px;background:#161616cc;padding:10px;border:1px solid #333;border-radius:6px;z-index:2}
#ui label{display:block;margin:4px 0}
#ui input[type=range]{width:160px;vertical-align:middle}
canvas{display:block;width:100vw;height:100vh}
select{background:#222;color:#ddd;border:1px solid #444}
#meta{color:#8ab;font:11px monospace;margin-top:6px;white-space:pre}
</style></head><body>
<div id="ui">
 <select id="chunk"></select>
 <label>CT lo <input id="lo" type="range" min="0" max="255" value="60"></label>
 <label>CT hi <input id="hi" type="range" min="0" max="255" value="160"></label>
 <label>density <input id="den" type="range" min="1" max="100" value="25"></label>
 <label>band <input id="ba" type="range" min="0" max="100" value="70"></label>
 <div id="meta"></div>
 <div style="color:#777">drag rotate · wheel zoom · b toggles band</div>
</div>
<canvas id="c"></canvas>
<script>
const CHUNKS = __CHUNKS__;
const cv = document.getElementById("c");
const gl = cv.getContext("webgl2");
if (!gl) document.body.innerHTML = "WebGL2 required";
const VS = `#version 300 es
void main(){ vec2 p = vec2((gl_VertexID<<1&2)-1, (gl_VertexID&2)-1); gl_Position = vec4(p,0,1); }`;
const FS2 = `#version 300 es
precision highp float; precision highp sampler3D;
uniform sampler3D uCT, uBand;
uniform mat3 uRt;            // view->object rotation (transpose of object->view)
uniform vec2 uRes;
uniform float uScale, uLo, uHi, uDen, uBa;
out vec4 o;
vec2 boxHit(vec3 ro, vec3 rd){
  vec3 inv = 1.0/rd;
  vec3 t0 = (vec3(-0.5)-ro)*inv, t1 = (vec3(0.5)-ro)*inv;
  vec3 tmin = min(t0,t1), tmax = max(t0,t1);
  return vec2(max(max(tmin.x,tmin.y),tmin.z), min(min(tmax.x,tmax.y),tmax.z));
}
void main(){
  vec2 ndc = (gl_FragCoord.xy*2.0 - uRes)/min(uRes.x,uRes.y);
  vec3 ro = uRt * vec3(ndc/uScale, 2.0);
  vec3 rd = normalize(uRt * vec3(0.0,0.0,-1.0));
  vec2 t = boxHit(ro, rd);
  if (t.x > t.y){ o = vec4(0.04,0.04,0.05,1); return; }
  float steps = 256.0;
  float dt = (t.y - max(t.x,0.0))/steps;
  vec3 acc = vec3(0.0); float aAcc = 0.0;
  for (float i=0.0; i<steps; i+=1.0){
    vec3 pos = ro + rd*(max(t.x,0.0) + (i+0.5)*dt);
    vec3 tc = pos + 0.5;
    float ct = texture(uCT, tc).r;
    float a = uDen * dt * smoothstep(uLo, uHi, ct);
    vec3 col = vec3(ct);
    float band = texture(uBand, tc).r;
    if (band > 0.5){
      float ab = uBa * dt * 18.0;
      col = mix(col, vec3(1.0,0.15,0.1), 0.85);
      a = max(a, ab);
    }
    acc += (1.0-aAcc)*a*col;
    aAcc += (1.0-aAcc)*a;
    if (aAcc > 0.97) break;
  }
  o = vec4(acc + (1.0-aAcc)*vec3(0.04,0.04,0.05), 1.0);
}`;
function shader(src, type){ const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s); return s; }
const prog = gl.createProgram();
gl.attachShader(prog, shader(VS, gl.VERTEX_SHADER));
gl.attachShader(prog, shader(FS2, gl.FRAGMENT_SHADER));
gl.linkProgram(prog);
if(!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw gl.getProgramInfoLog(prog);
gl.useProgram(prog);
const U = n => gl.getUniformLocation(prog, n);

function tex3d(unit, data, n){
  const t = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_3D, t);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, n, n, n, 0, gl.RED, gl.UNSIGNED_BYTE, data);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  return t;
}
function loadChunk(i){
  const c = CHUNKS[i];
  const raw = Uint8Array.from(atob(c.b64), ch => ch.charCodeAt(0));
  const n = c.meta.size, t = n*n*n;
  tex3d(0, raw.subarray(0, t), n);
  tex3d(1, raw.subarray(t, 2*t), n);
  gl.uniform1i(U("uCT"), 0); gl.uniform1i(U("uBand"), 1);
  document.getElementById("meta").textContent =
    `${c.name}\\norigin z${c.meta.origin[0]} y${c.meta.origin[1]} x${c.meta.origin[2]}  ${n}^3`;
  draw();
}
let yaw = 0.6, pitch = 0.4, scale = 1.4, bandOn = 1;
function rotT(){
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  // object->view = Rx(pitch)*Ry(yaw); we need its transpose, column-major
  const m = [cy, sy*sp, -sy*cp,  0, cp, sp,  sy, -cy*sp, cy*cp];
  return new Float32Array(m);
}
function draw(){
  cv.width = innerWidth; cv.height = innerHeight;
  gl.viewport(0,0,cv.width,cv.height);
  gl.uniform2f(U("uRes"), cv.width, cv.height);
  gl.uniformMatrix3fv(U("uRt"), false, rotT());
  gl.uniform1f(U("uScale"), scale);
  gl.uniform1f(U("uLo"), lo.value/255);
  gl.uniform1f(U("uHi"), Math.max(+hi.value, +lo.value+1)/255);
  gl.uniform1f(U("uDen"), den.value/10);
  gl.uniform1f(U("uBa"), bandOn * ba.value/100);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}
let drag = null;
cv.onmousedown = e => drag = [e.clientX, e.clientY];
window.onmouseup = () => drag = null;
window.onmousemove = e => { if(!drag) return;
  yaw += (e.clientX-drag[0])*0.008; pitch += (e.clientY-drag[1])*0.008;
  pitch = Math.max(-1.55, Math.min(1.55, pitch)); drag = [e.clientX, e.clientY]; draw(); };
cv.onwheel = e => { scale *= e.deltaY < 0 ? 1.1 : 0.9; scale = Math.max(0.3, Math.min(8, scale)); draw(); e.preventDefault(); };
window.onkeydown = e => { if(e.key === "b"){ bandOn = 1-bandOn; draw(); } };
window.onresize = draw;
for (const el of ["lo","hi","den","ba"]) document.getElementById(el).oninput = draw;
const sel = document.getElementById("chunk");
CHUNKS.forEach((c,i)=>{ const o = document.createElement("option"); o.value=i; o.textContent=c.name; sel.appendChild(o); });
sel.onchange = () => loadChunk(+sel.value);
loadChunk(0);
</script></body></html>"""


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    out, files = sys.argv[1], sys.argv[2:]
    chunks = []
    for p in files:
        meta, b64 = read_chunk(p)
        name = p.split("/")[-1].replace(".qcchunk", "")
        chunks.append({"name": name, "meta": meta, "b64": b64})
    with open(out, "w") as f:
        f.write(PAGE.replace("__CHUNKS__", json.dumps(chunks)))
    print(f"chunk-viewer: {len(chunks)} chunks -> {out}")


if __name__ == "__main__":
    main()
