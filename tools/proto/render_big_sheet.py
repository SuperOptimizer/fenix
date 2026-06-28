import numpy as np, tifffile, os
from scipy.ndimage import zoom, map_coordinates

d = "data/vc_trace_2k/auto_grown_20260627190830030"
Z = tifffile.imread(d + "/z.tif"); Y = tifffile.imread(d + "/y.tif"); X = tifffile.imread(d + "/x.tif")
print("grid", X.shape)
UP = 3
Zu = zoom(Z, UP, order=1); Yu = zoom(Y, UP, order=1); Xu = zoom(X, UP, order=1)
mask = (Xu > 0) & (Zu > 0)
H, W = Xu.shape
print("render", H, "x", W, "valid", round(float(mask.mean()), 3))

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
del CT

# percentile stretch for visibility
m = mask
lo, hi = np.percentile(gray[m], 1), np.percentile(gray[m], 99)
g = np.clip((gray - lo) / max(1e-6, hi - lo), 0, 1)
g[~m] = 0
gray8 = (g * 255).astype(np.uint8)

# ink only exists for the central 1024^3 (local [512,1536))
def load_nrrd(p):
    f = open(p, "rb")
    while f.readline().strip() != b"": pass
    return np.frombuffer(f.read(), np.float32).reshape(1024, 1024, 1024)

print("loading ink ...")
INK = load_nrrd("data/paris4_mid_1024_ink.nrrd")
inrange = m & (Zu >= 512) & (Zu < 1536) & (Yu >= 512) & (Yu < 1536) & (Xu >= 512) & (Xu < 1536)
ic = np.stack([(Zu - 512).ravel(), (Yu - 512).ravel(), (Xu - 512).ravel()])
ink = map_coordinates(INK, ic, order=1, mode="constant").reshape(H, W)
ink[~inrange] = 0
del INK

# outputs: CT-only (gray) and ink composite (red overlay)
import PIL.Image as I
I.fromarray(gray8).save("data/vc_big2k_ct.jpg", quality=92)

rgb = np.stack([gray8, gray8, gray8], -1).astype(np.float32)
a = np.clip(ink * 1.4, 0, 1)[..., None]   # ink alpha
red = np.array([255, 40, 40], np.float32)
rgb = rgb * (1 - a) + red * a
comp = np.clip(rgb, 0, 255).astype(np.uint8)
I.fromarray(comp).save("data/vc_big2k_ink.jpg", quality=92)
print("ink coverage on sheet:", round(float((ink[inrange] > 0.3).mean()), 3) if inrange.any() else 0)
print("wrote data/vc_big2k_ct.jpg and data/vc_big2k_ink.jpg")
