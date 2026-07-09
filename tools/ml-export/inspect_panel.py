"""Surface inspection panel — CT + overlays across a set of 2D slices.

Take an NxNxN CT chunk and render a panel of 2D slices, each showing the CT
grayscale plus one or more overlays:
  - PREDICTION: the fp16 surface teacher's prob field (hot colormap, alpha by prob)
  - GROUND TRUTH: a tifxyz segment mesh rasterized into the chunk (contour color)
Either or both can be shown, so the same tool inspects live predictions, GT segment
alignment, or pred-vs-GT agreement (the correction workflow).

Slices: 3 mid-planes (XY/XZ/YZ) + 6 cube faces + axial quartiles.

Usage:
  # prediction only
  inspect_panel.py --ct chunk.npy --pred [--out panel.png] [--thr 0.4]
  inspect_panel.py --zarr <url> --level 0 --org z,y,x --size 128 --pred
  # GT segment only (tifxyz prefix -> reads <pref>_{x,y,z}.tif, LOD-0 coords)
  inspect_panel.py --zarr ... --org ... --gt /tmp/gtqc/on24
  # both, for correction
  inspect_panel.py --zarr ... --org ... --pred --gt /tmp/gtqc/on24
"""
import sys, os, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from PIL import Image, ImageDraw
from reference import build_and_load


def load_gt(pref, org, S, band=1.5):
    """Rasterize a tifxyz segment (LOD-0 coords) into a chunk-local [S,S,S] u8 mask.
    band = half-thickness in voxels stamped around each surface cell (so a thin sheet
    shows on slices even between exact cell centers)."""
    import tifffile
    X = tifffile.imread(f"{pref}_x.tif").astype(np.float32)
    Y = tifffile.imread(f"{pref}_y.tif").astype(np.float32)
    Z = tifffile.imread(f"{pref}_z.tif").astype(np.float32)
    v = (X > 0) & (Y > 0) & (Z > 0)
    oz, oy, ox = org
    inb = v & (Z >= oz-2) & (Z < oz+S+2) & (Y >= oy-2) & (Y < oy+S+2) & (X >= ox-2) & (X < ox+S+2)
    mask = np.zeros((S, S, S), np.uint8)
    b = int(np.ceil(band))
    for zz, yy, xx in zip(Z[inb]-oz, Y[inb]-oy, X[inb]-ox):
        z0, y0, x0 = int(round(zz)), int(round(yy)), int(round(xx))
        for dz in range(-b, b+1):
            for dy in range(-b, b+1):
                for dx in range(-b, b+1):
                    z, y, x = z0+dz, y0+dy, x0+dx
                    if 0 <= z < S and 0 <= y < S and 0 <= x < S:
                        mask[z, y, x] = 255
    return mask, int(inb.sum())

FENIX = "/home/forrest/fenix/build-release/fenix"


def load_ct(args):
    if args.ct:
        v = np.load(args.ct)
    else:
        # ingest a chunk from zarr -> fxvol -> npy, via the fenix CLI
        z, y, x = [int(t) for t in args.org.split(",")]
        S = args.size
        fx = "/tmp/gtqc/_panel_ct.fxvol"; npy = "/tmp/gtqc/_panel_ct.npy"
        subprocess.run([FENIX, "ingest-zarr", args.zarr, str(args.level),
                        str(z), str(y), str(x), str(S), str(S), str(S), fx, "q=1"], check=True)
        subprocess.run([FENIX, "export-npy", fx, npy], check=True)
        v = np.load(npy)
    return v.astype(np.float32)


def _gauss_window(P, sigma_frac=0.125):
    """3D Gaussian window (peak 1 at center, ~0 at edges) for feathered tile blending —
    the standard nnU-Net sliding-window weight. Kills seams that flat/max blending leave."""
    axes = []
    for n in P:
        c = (n - 1) / 2.0
        s = max(n * sigma_frac, 1e-3)
        g = np.exp(-0.5 * ((np.arange(n) - c) / s) ** 2)
        axes.append(g.astype(np.float32))
    w = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return np.maximum(w, 1e-4)


import itertools as _it

FLIPS8 = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]  # axis-flip group on [1,1,D,H,W]

# Octahedral-48 group on a CUBIC patch = 6 axis permutations x 8 sign flips.
# Each element: permute spatial axes by `perm` (of {0,1,2} over Z,Y,X) then flip `flips`.
# The 48 transforms are the full symmetry group of the cube (24 rotations x 2 reflection).
_PERMS = list(_it.permutations((0, 1, 2)))            # 6
_FLIPSETS = list(_it.product((False, True), repeat=3))  # 8
OCTA48 = [(perm, flip) for perm in _PERMS for flip in _FLIPSETS]  # 48


@torch.no_grad()
def _forward_tta(net, t, tta):
    """fp16 forward with TTA. tta==1 single; tta<=8 axis-flips; tta==48 full octahedral
    group (requires a cubic patch). Returns averaged prob [D,H,W]."""
    if tta >= 48 and t.shape[2] == t.shape[3] == t.shape[4]:
        acc = None
        for perm, flip in OCTA48:
            fa = tuple(2 + i for i, f in enumerate(flip) if f)         # input flip axes {2,3,4}
            pdims = tuple(2 + p for p in perm)                          # input permute (spatial)
            tf = torch.flip(t, fa) if fa else t
            tf = tf.permute(0, 1, *pdims)
            with torch.autocast("cuda", dtype=torch.float16):
                p = net(tf).float().softmax(1)[:, 1]                    # [1,D',H',W'] axes 1,2,3
            # invert: un-permute (spatial axes 1,2,3), then un-flip
            inv = [0, 0, 0]
            for i, pp in enumerate(perm):
                inv[pp] = i
            p = p.permute(0, *[1 + j for j in inv])
            pfa = tuple(1 + i for i, f in enumerate(flip) if f)
            p = torch.flip(p, pfa) if pfa else p
            acc = p if acc is None else acc + p
        return (acc / len(OCTA48))[0].cpu().numpy()

    flips = FLIPS8[:tta] if tta > 1 else [()]
    acc = None
    for ax in flips:                                    # ax on 5D input [1,1,D,H,W]
        tf = torch.flip(t, ax) if ax else t
        with torch.autocast("cuda", dtype=torch.float16):
            p = net(tf).float().softmax(1)[:, 1]        # [1,D,H,W] -> spatial axes 1,2,3
        pax = tuple(a - 1 for a in ax)                  # map input axes {2,3,4} -> prob {1,2,3}
        pb = torch.flip(p, pax) if pax else p
        acc = pb if acc is None else acc + pb
    return (acc / len(flips))[0].cpu().numpy()


@torch.no_grad()
def predict(ct, ckpt, patch=128, overlap=64, tta=1):
    """Sliding-window fp16 surface prediction with Gaussian-feathered overlap blending
    (weighted accumulate, not max) + optional flip-TTA. The net trains at `patch`^3."""
    dev = "cuda"
    net = build_and_load(ckpt).to(dev).eval()
    D, H, W = ct.shape
    acc = np.zeros((D, H, W), np.float32)
    wsum = np.zeros((D, H, W), np.float32)
    step = patch - overlap

    def starts(n):
        if n <= patch:
            return [0]
        s = list(range(0, n - patch + 1, step))
        if s[-1] != n - patch:
            s.append(n - patch)
        return s

    win = None
    for z0 in starts(D):
        for y0 in starts(H):
            for x0 in starts(W):
                P = min(patch, D), min(patch, H), min(patch, W)
                sub = ct[z0:z0+P[0], y0:y0+P[1], x0:x0+P[2]]
                if win is None or win.shape != sub.shape:
                    win = _gauss_window(sub.shape)
                t = torch.from_numpy(sub).to(dev).reshape(1, 1, *sub.shape)
                m, s = t.mean(), t.std().clamp(min=1e-6)
                t = (t - m) / s
                p = _forward_tta(net, t, tta)
                acc[z0:z0+P[0], y0:y0+P[1], x0:x0+P[2]] += p * win
                wsum[z0:z0+P[0], y0:y0+P[1], x0:x0+P[2]] += win
    return acc / np.maximum(wsum, 1e-6)


def hot(prob):
    """prob [0,1] -> RGB heatmap (black->red->yellow->white)."""
    p = np.clip(prob, 0, 1)
    r = np.clip(p * 3, 0, 1)
    g = np.clip(p * 3 - 1, 0, 1)
    b = np.clip(p * 3 - 2, 0, 1)
    return np.stack([r, g, b], -1)


def tile(ct2d, prob2d, gt2d, thr, label, up=2):
    """One panel tile: CT gray + optional prob overlay (hot, alpha by prob) + optional
    GT segment (cyan). up = nearest-neighbor upscale factor."""
    g = np.clip(ct2d, 0, 255).astype(np.uint8)
    out = np.stack([g, g, g], -1).astype(np.float32) / 255.0
    if prob2d is not None:
        a = np.clip((prob2d - thr) / (1 - thr), 0, 1)[..., None]
        out = (1 - a) * out + a * hot(prob2d)
    if gt2d is not None:
        m = gt2d > 0
        out[m] = [0.0, 1.0, 1.0]  # cyan GT segment
    img = Image.fromarray((out * 255).astype(np.uint8))
    img = img.resize((img.size[0]*up, img.size[1]*up), Image.NEAREST)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.size[0]-1, 14], fill=(0, 0, 0))
    d.text((3, 2), label, fill=(0, 255, 0))
    return img


def _sl(v, kind, i):
    if v is None:
        return None
    return v[i] if kind == "z" else v[:, i] if kind == "y" else v[:, :, i]


def build_panel(ct, prob, gt, thr, out, up):
    D, H, W = ct.shape
    cz, cy, cx = D//2, H//2, W//2
    # (label, axis, index)
    specs = [
        ("XY mid z=%d" % cz, "z", cz), ("XZ mid y=%d" % cy, "y", cy), ("YZ mid x=%d" % cx, "x", cx),
        ("face z=0", "z", 0), ("face z=%d" % (D-1), "z", D-1),
        ("face y=0", "y", 0), ("face y=%d" % (H-1), "y", H-1),
        ("face x=0", "x", 0), ("face x=%d" % (W-1), "x", W-1),
        ("XY z=%d" % (D//4), "z", D//4), ("XY z=%d" % (3*D//4), "z", 3*D//4),
    ]
    imgs = [tile(_sl(ct, k, i), _sl(prob, k, i), _sl(gt, k, i), thr, lbl, up)
            for lbl, k, i in specs]
    cols = 4
    rows = (len(imgs) + cols - 1) // cols
    tw, th = imgs[0].size
    panel = Image.new("RGB", (cols*tw + (cols+1)*6, rows*th + (rows+1)*6), (30, 30, 30))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        panel.paste(im, (6 + c*(tw+6), 6 + r*(th+6)))
    panel.save(out)
    info = f"panel: {len(imgs)} tiles"
    if prob is not None:
        info += f", prob[{prob.min():.2f},{prob.max():.2f}] surf-frac(>{thr}) {(prob>thr).mean():.3f}"
    if gt is not None:
        info += f", GT voxels {(gt>0).sum()}"
    print(f"{info} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct")
    ap.add_argument("--zarr"); ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--org"); ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--ckpt", default="/home/forrest/fenix/models/surface_recto_3dunet/"
                    "checkpoint_inference_ready.pth")
    ap.add_argument("--pred", action="store_true", help="overlay surface prediction")
    ap.add_argument("--overlap", type=int, default=64, help="tile overlap (64 = 50%% at patch 128)")
    ap.add_argument("--tta", type=int, default=8, help="flip-TTA passes (1 or 8)")
    ap.add_argument("--gt", help="tifxyz prefix (reads <pref>_{x,y,z}.tif, LOD-0 coords)")
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--up", type=int, default=2, help="per-tile upscale")
    ap.add_argument("--out", default="/tmp/gtqc/panel.png")
    args = ap.parse_args()
    if not args.pred and not args.gt:
        args.pred = True  # default to prediction
    ct = load_ct(args)
    print(f"CT chunk {ct.shape} mean {ct.mean():.1f}")
    prob = predict(ct, args.ckpt, overlap=args.overlap, tta=args.tta) if args.pred else None
    gt = None
    if args.gt:
        org = tuple(int(t) for t in args.org.split(",")) if args.org else (0, 0, 0)
        gt, ncells = load_gt(args.gt, org, ct.shape[0])
        print(f"GT segment: {ncells} cells in chunk")
    build_panel(ct, prob, gt, args.thr, args.out, args.up)


if __name__ == "__main__":
    main()
