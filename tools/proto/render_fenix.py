import numpy as np, os
from scipy.ndimage import zoom, map_coordinates
import PIL.Image as I

outdir = "data/fenix_trace"
G = int(open(outdir + "/meta.txt").read().split()[0])
X = np.fromfile(outdir + "/x.f32", np.float32).reshape(G, G)
Y = np.fromfile(outdir + "/y.f32", np.float32).reshape(G, G)
Z = np.fromfile(outdir + "/z.f32", np.float32).reshape(G, G)
M = np.fromfile(outdir + "/valid.u8", np.uint8).reshape(G, G)

# crop to valid bbox
vs = np.argwhere(M > 0)
(u0, v0), (u1, v1) = vs.min(0), vs.max(0) + 1
X, Y, Z, M = [a[u0:u1, v0:v1] for a in (X, Y, Z, M)]
print("valid grid bbox", X.shape, "valid frac", round(float((M > 0).mean()), 3))

UP = 3
# nearest-neighbour upsample so we never interpolate across invalid cells (which hold coord 0)
Xu, Yu, Zu = [zoom(a, UP, order=0) for a in (X, Y, Z)]
Mu = zoom(M, UP, order=0) > 0
H, W = Xu.shape

def load_nrrd(p):
    f = open(p, "rb")
    while f.readline().strip() != b"": pass
    return np.frombuffer(f.read(), np.float32).reshape(1024, 1024, 1024)

CT = load_nrrd("data/paris4_mid_1024.nrrd")
coords = np.stack([Zu.ravel(), Yu.ravel(), Xu.ravel()])
gray = map_coordinates(CT, coords, order=1, mode="constant").reshape(H, W)
m = Mu
lo, hi = np.percentile(gray[m], 1), np.percentile(gray[m], 99)
g = np.clip((gray - lo) / max(1e-6, hi - lo), 0, 1); g[~m] = 0
I.fromarray((g * 255).astype(np.uint8)).save("data/fenix_sheet_ct.jpg", quality=92)
# crude texture-coherence metric: std of local gradient magnitude on-sheet (lower=more coherent sheet)
gm = np.hypot(*np.gradient(g))
print("on-sheet CT gradient std:", round(float(gm[m].std()), 4), " -> wrote data/fenix_sheet_ct.jpg", H, "x", W)
