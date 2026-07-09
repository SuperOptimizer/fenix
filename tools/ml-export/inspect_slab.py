"""Canonical-orientation surface-prediction slab inspector.

Predict a wide XY, thin-Z slab (default 1024x1024x128) and emit individual z-slice
PNGs (CT grayscale + surface-prediction overlay) at a chosen stride. In canonical
orientation the scroll spiral reads clearly in XY (sheets ~orthogonal to z near the
umbilicus), so z-slices are the natural annotation surface. One image per kept slice
+ a contact-sheet index.

Usage:
  inspect_slab.py --zarr <url> --level 0 --org z,y,x --sz 128 --sxy 1024 \
      --stride 8 [--tta 8] [--overlap 64] [--thr 0.4] [--outdir DIR]
  inspect_slab.py --ct slab.npy --stride 16 ...
"""
import sys, os, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from PIL import Image, ImageDraw
from reference import build_and_load
from inspect_panel import predict, hot   # reuse Gaussian+TTA tiled inference + colormap

FENIX = "/home/forrest/fenix/build-release/fenix"


def load_slab(args):
    if args.ct:
        return np.load(args.ct).astype(np.float32)
    z, y, x = [int(t) for t in args.org.split(",")]
    D, HW = args.sz, args.sxy
    fx = f"{args.outdir}/_slab.fxvol"; npy = f"{args.outdir}/_slab.npy"
    subprocess.run([FENIX, "ingest-zarr", args.zarr, str(args.level),
                    str(z), str(y), str(x), str(D), str(HW), str(HW), fx, "q=1"], check=True)
    subprocess.run([FENIX, "export-npy", fx, npy], check=True)
    return np.load(npy).astype(np.float32)


def slice_img(ct2d, prob2d, thr, label, up=1):
    g = np.clip(ct2d, 0, 255).astype(np.uint8)
    out = np.stack([g, g, g], -1).astype(np.float32) / 255.0
    a = np.clip((prob2d - thr) / (1 - thr), 0, 1)[..., None]
    out = (1 - a) * out + a * hot(prob2d)
    img = Image.fromarray((out * 255).astype(np.uint8))
    if up != 1:
        img = img.resize((img.size[0]*up, img.size[1]*up), Image.NEAREST)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 120, 16], fill=(0, 0, 0))
    d.text((3, 3), label, fill=(0, 255, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct")
    ap.add_argument("--zarr"); ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--org"); ap.add_argument("--sz", type=int, default=128)
    ap.add_argument("--sxy", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=8, help="emit every Nth z-slice")
    ap.add_argument("--ckpt", default="/home/forrest/fenix/models/surface_recto_3dunet/"
                    "checkpoint_inference_ready.pth")
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--outdir", default="/tmp/gtqc/slab")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ct = load_slab(args)
    D, H, W = ct.shape
    print(f"slab {ct.shape} mean {ct.mean():.1f}  ->  predicting (overlap {args.overlap}, tta {args.tta})")
    prob = predict(ct, args.ckpt, overlap=args.overlap, tta=args.tta)
    print(f"prob range [{prob.min():.3f},{prob.max():.3f}] surf-frac(>{args.thr}) {(prob>args.thr).mean():.3f}")
    # save raw arrays for the HTML scrubber builder
    np.save(f"{args.outdir}/ct.npy", ct.astype(np.uint8))
    np.save(f"{args.outdir}/prob.npy", (np.clip(prob, 0, 1) * 255).astype(np.uint8))
    with open(f"{args.outdir}/meta.txt", "w") as f:
        f.write(f"org={args.org} sz={D} sxy={H} tta={args.tta} overlap={args.overlap}\n")

    oz = int(args.org.split(",")[0]) if args.org else 0
    zsel = list(range(0, D, args.stride))
    thumbs = []
    for zi in zsel:
        img = slice_img(ct[zi], prob[zi], args.thr, f"z={oz+zi} (i={zi})")
        img.save(f"{args.outdir}/z{zi:04d}.png")
        thumbs.append(img.resize((256, 256)))
    # contact sheet index
    cols = 8
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols*258+2, rows*258+2), (20, 20, 20))
    for i, th in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(th, (2 + c*258, 2 + r*258))
    sheet.save(f"{args.outdir}/index.png")
    print(f"wrote {len(zsel)} slices (stride {args.stride}) + index.png -> {args.outdir}")


if __name__ == "__main__":
    main()
