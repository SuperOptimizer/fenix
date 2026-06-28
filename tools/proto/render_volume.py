"""Visualize full-volume tracing: overlay each traced sheet (distinct colour) on an axial CT slice
of the 2048^3 cube — shows the concentric wraps segmented. Usage: render_volume.py [z0]"""
import numpy as np, os, sys
import PIL.Image as I

Z0 = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
d = "data/fenix_vol"
npts, nsheets = map(int, open(d + "/meta.txt").read().split())
pts = np.fromfile(d + "/pts.f32", np.float32).reshape(npts, 3)   # z,y,x
ids = np.fromfile(d + "/ids.u16", np.uint16)

# CT slice z=Z0 from big_ct.zarr (load only the needed cz chunk-layer)
C = 128; cz = Z0 // C; zi = Z0 % C
ct = np.zeros((2048, 2048), np.uint8)
for cy in range(16):
    for cx in range(16):
        p = f"data/big_ct.zarr/0/{cz}/{cy}/{cx}"
        if os.path.exists(p):
            blk = np.fromfile(p, np.uint8).reshape(C, C, C)
            ct[cy*C:(cy+1)*C, cx*C:(cx+1)*C] = blk[zi]
g = np.clip(ct.astype(np.float32) / 255, 0, 1)
rgb = np.stack([g, g, g], -1)

# overlay sheet points near the slice, coloured by sheet id (golden-ratio hue)
import colorsys
near = np.abs(pts[:, 0] - Z0) < 2.0
py = pts[near, 1].astype(int); px = pts[near, 2].astype(int); pid = ids[near]
def color(i):
    h = (i * 0.61803398875) % 1.0
    return np.array(colorsys.hsv_to_rgb(h, 0.9, 1.0))
for sid in np.unique(pid):
    m = pid == sid
    yy = np.clip(py[m], 0, 2047); xx = np.clip(px[m], 0, 2047)
    col = color(int(sid))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            rgb[np.clip(yy+dy, 0, 2047), np.clip(xx+dx, 0, 2047)] = col
I.fromarray((rgb * 255).astype(np.uint8)).save("data/fenix_vol_slice.jpg", quality=92)
print(f"z={Z0}: {near.sum()} sheet points from {len(np.unique(pid))} sheets -> data/fenix_vol_slice.jpg")
