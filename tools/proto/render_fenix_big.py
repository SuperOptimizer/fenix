import numpy as np, os
from scipy.ndimage import zoom, map_coordinates
import PIL.Image as I

outdir = "data/fenix_trace_big"
G = int(open(outdir + "/meta.txt").read().split()[0])
X = np.fromfile(outdir + "/x.f32", np.float32).reshape(G, G)
Y = np.fromfile(outdir + "/y.f32", np.float32).reshape(G, G)
Z = np.fromfile(outdir + "/z.f32", np.float32).reshape(G, G)
M = np.fromfile(outdir + "/valid.u8", np.uint8).reshape(G, G)
vs = np.argwhere(M > 0)
(u0, v0), (u1, v1) = vs.min(0), vs.max(0) + 1
X, Y, Z, M = [a[u0:u1, v0:v1] for a in (X, Y, Z, M)]
print("valid grid bbox", X.shape, "valid frac", round(float((M > 0).mean()), 3))

from scipy.ndimage import binary_fill_holes, binary_closing
UP = 2
# fill small interior holes (enclosed gaps + a morphological close), then diffuse coords into them
Mf = binary_fill_holes(M) | binary_closing(M > 0, iterations=2)
mm = M > 0
hole = Mf & ~mm
for a in (X, Y, Z):
    a[~mm] = 0.0
    for _ in range(40):  # diffuse valid coords into holes (local inpaint, no global streak)
        s = np.zeros_like(a); c = np.zeros_like(a)
        for sh in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s += np.roll(a, sh, axis=(0, 1)); c += np.roll((a != 0).astype(np.float32), sh, axis=(0, 1))
        fillv = np.divide(s, np.maximum(c, 1))
        a[hole & (a == 0) & (c > 0)] = fillv[hole & (a == 0) & (c > 0)]
Xu, Yu, Zu = [zoom(a, UP, order=1) for a in (X, Y, Z)]
Mu = zoom(Mf.astype(np.float32), UP, order=1) > 0.5
H, W = Xu.shape

def load_zarr_u8(root, N=2048, C=128):
    a = np.zeros((N, N, N), np.uint8); nc = N // C
    for cz in range(nc):
        for cy in range(nc):
            for cx in range(nc):
                p = f"{root}/0/{cz}/{cy}/{cx}"
                if os.path.exists(p):
                    a[cz*C:(cz+1)*C, cy*C:(cy+1)*C, cx*C:(cx+1)*C] = np.fromfile(p, np.uint8).reshape(C, C, C)
    return a

print("loading big_ct.zarr ...")
CT = load_zarr_u8("data/big_ct.zarr")
coords = np.stack([Zu.ravel(), Yu.ravel(), Xu.ravel()])
gray = map_coordinates(CT, coords, order=1, mode="constant").reshape(H, W).astype(np.float32)
m = Mu
lo, hi = np.percentile(gray[m], 1), np.percentile(gray[m], 99)
g = np.clip((gray - lo) / max(1e-6, hi - lo), 0, 1); g[~m] = 0
gm = np.hypot(*np.gradient(g))
print("on-sheet CT gradient std:", round(float(gm[m].std()), 4), "img", H, "x", W)
I.fromarray((g * 255).astype(np.uint8)).save("data/fenix_big_sheet_ct.jpg", quality=90)
print("wrote data/fenix_big_sheet_ct.jpg")
