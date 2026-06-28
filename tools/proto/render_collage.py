"""Render each traced sheet's flattened CT face and tile them into a collage. Uses the per-sheet
grids dumped by test_trace (data/fenix_vol/sheet_NN/). Usage: render_collage.py"""
import numpy as np, os, glob
from scipy.ndimage import zoom, binary_fill_holes, binary_erosion, map_coordinates
import PIL.Image as I

def load_ct(root="data/big_ct.zarr", N=2048, C=128):
    a = np.zeros((N, N, N), np.uint8); nc = N // C
    for cz in range(nc):
        for cy in range(nc):
            for cx in range(nc):
                p = f"{root}/0/{cz}/{cy}/{cx}"
                if os.path.exists(p):
                    a[cz*C:(cz+1)*C, cy*C:(cy+1)*C, cx*C:(cx+1)*C] = np.fromfile(p, np.uint8).reshape(C, C, C)
    return a

print("loading big_ct ...")
CT = load_ct()

def face(d, UP=2, tile=520):
    G = int(open(d + "/meta.txt").read().split()[0])
    X = np.fromfile(d + "/x.f32", np.float32).reshape(G, G)
    Y = np.fromfile(d + "/y.f32", np.float32).reshape(G, G)
    Z = np.fromfile(d + "/z.f32", np.float32).reshape(G, G)
    M = np.fromfile(d + "/valid.u8", np.uint8).reshape(G, G).astype(bool)
    if M.sum() < 1000:
        return None
    vs = np.argwhere(M); (a0, b0), (a1, b1) = vs.min(0), vs.max(0) + 1
    X, Y, Z, M = [t[a0:a1, b0:b1] for t in (X, Y, Z, M)]
    mm = M > 0
    for t in (X, Y, Z):
        t[~mm] = t[mm].mean()
    Xu, Yu, Zu = [zoom(t, UP, order=1) for t in (X, Y, Z)]
    Mu = zoom(M.astype(np.float32), UP, order=1) > 0.5
    H, W = Xu.shape
    gray = map_coordinates(CT, np.stack([Zu.ravel(), Yu.ravel(), Xu.ravel()]), order=1).reshape(H, W).astype(np.float32)
    dom = binary_erosion(binary_fill_holes(Mu), iterations=1)
    if dom.sum() < 100:
        return None
    lo, hi = np.percentile(gray[dom], 1), np.percentile(gray[dom], 99)
    img = np.clip((gray - lo) / max(1e-6, hi - lo), 0, 1); img[~dom] = 0
    im = I.fromarray((img * 255).astype(np.uint8))
    im.thumbnail((tile, tile))
    canvas = I.new("L", (tile, tile), 25)
    canvas.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
    return np.array(canvas)

dirs = sorted(glob.glob("data/fenix_vol/sheet_*"))
tiles = []
for d in dirs:
    f = face(d)
    if f is not None:
        tiles.append(f); print("rendered", d, f.shape)
n = len(tiles); cols = 4; rows = (n + cols - 1) // cols
T = tiles[0].shape[0]; pad = 6
collage = np.full((rows * T + (rows + 1) * pad, cols * T + (cols + 1) * pad), 12, np.uint8)
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    y = pad + r * (T + pad); x = pad + c * (T + pad)
    collage[y:y+T, x:x+T] = t
I.fromarray(collage).save("data/fenix_patch_collage.jpg", quality=92)
print(f"wrote data/fenix_patch_collage.jpg  ({n} patches, {rows}x{cols})")
