import numpy as np, os
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes, binary_erosion, map_coordinates
import PIL.Image as I

d = "data/fenix_flat"
N = int(open(d + "/meta.txt").read().split()[0])
pos = np.fromfile(d + "/pos.f32", np.float32).reshape(N, 3)   # z,y,x
uv = np.fromfile(d + "/uv.f32", np.float32).reshape(N, 2)

def load_nrrd(p):
    f = open(p, "rb")
    while f.readline().strip() != b"": pass
    return np.frombuffer(f.read(), np.float32).reshape(1024, 1024, 1024)
CT = load_nrrd("data/paris4_mid_1024.nrrd")
gray = map_coordinates(CT, [pos[:, 0], pos[:, 1], pos[:, 2]], order=1)

u = uv - uv.min(0); res = 1.0
W = int(u[:, 0].max() / res) + 1; H = int(u[:, 1].max() / res) + 1
gz, gy = np.mgrid[0:W, 0:H]
out = griddata(u / res, gray, (gz, gy), method="linear", fill_value=0)  # fills interior holes
tree = cKDTree(u / res)
dist, _ = tree.query(np.stack([gz.ravel(), gy.ravel()], 1), k=1)
cover = (dist.reshape(W, H) <= 1.6)
domain = binary_erosion(binary_fill_holes(cover), iterations=1)
lo, hi = np.percentile(gray, 1), np.percentile(gray, 99)
img = np.clip((out - lo) / (hi - lo), 0, 1); img[~domain] = 0
I.fromarray((img.T * 255).astype(np.uint8)).save("data/fenix_slim_cpp.jpg", quality=90)
print(f"wrote data/fenix_slim_cpp.jpg {W}x{H}  verts {N}")
