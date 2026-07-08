"""sweep — full-scroll tiled inference: CT zarr -> multi-channel PredStore.

Brick (default 256^3 core + 32 margin) = unit of work/resume/sharding; sliding
128^3 patches at stride 96 blended with a floored-Gaussian importance window
into an f32 GPU accumulator; core cropped, u8-encoded, written to chunk-aligned
zarr; done/culled markers with provenance (resume refuses mismatched markers).

Cull doctrine: prescreen from a COARSE level (dilated by one brick) marks
'culled' — a distinct, auditable state, never a silent done. Patch-level skip
only when the patch is exactly empty (a fractional threshold provably drops
real voxels). A transient read failure marks 'failed', never zeros.

  python sweep.py run --ct staged.zarr --out preds.zarr --ckpt CKPT
      [--lane fp16cl|fp8|int8|fake] [--shard i/n] [--brick 256] [--stride 96]
      [--gpus 0,1] [--max-bricks N] [--force-restart]
  python sweep.py report --out preds.zarr
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from ctio import PredStore, open_u8, read_patch, zscore
from lanes import CHANNEL_ORDER, NCH, load_lane


def importance_window(patch, kind="gaussian"):
    if kind == "flat":
        return np.ones((patch,) * 3, np.float32)
    i = np.arange(patch, dtype=np.float32)
    g = np.exp(-0.5 * ((i - (patch - 1) / 2) / (patch / 8)) ** 2)
    g = np.maximum(g, 0.01)
    return (g[:, None, None] * g[None, :, None] * g[None, None, :]).astype(np.float32)


def patch_positions(dim, patch, stride):
    if dim <= patch:
        return [0]
    pos = list(range(0, dim - patch, stride)) + [dim - patch]
    return sorted(set(pos))


def brick_grid(shape, brick, margin):
    bricks = []
    idx = 0
    for z0 in range(0, shape[0], brick):
        for y0 in range(0, shape[1], brick):
            for x0 in range(0, shape[2], brick):
                core = tuple(slice(o, min(o + brick, shape[i]))
                             for i, o in enumerate((z0, y0, x0)))
                read_o = tuple(c.start - margin for c in core)
                read_s = tuple(c.stop - c.start + 2 * margin for c in core)
                bricks.append({"idx": idx,
                               "bidx": (z0 // brick, y0 // brick, x0 // brick),
                               "core": core, "read_o": read_o, "read_s": read_s})
                idx += 1
    return bricks


def occupancy_prescreen(ct_coarse_path, shape, brick, factor, thresh=1,
                        frac=0.005):
    """bool per-brick bitmap from a staged coarse level; dilated by 1 brick."""
    from scipy.ndimage import binary_dilation
    h = open_u8(ct_coarse_path)
    coarse = np.asarray(h[...].read().result())
    occ_shape = tuple((shape[i] + brick - 1) // brick for i in range(3))
    bm = np.zeros(occ_shape, bool)
    cb = max(brick // factor, 1)
    for bz in range(occ_shape[0]):
        for by in range(occ_shape[1]):
            for bx in range(occ_shape[2]):
                sl = tuple(slice(b * cb, min((b + 1) * cb, coarse.shape[i]))
                           for i, b in enumerate((bz, by, bx)))
                sub = coarse[sl]
                bm[bz, by, bx] = sub.size > 0 and (sub >= thresh).mean() >= frac
    return binary_dilation(bm, iterations=1)


def run_brick(brick, ct, lane, store, win_t, patch, stride, batch, device):
    t0 = time.time()
    data, valid = read_patch(ct, brick["read_o"], brick["read_s"])
    if not (data >= 1).any():
        store.mark(brick["bidx"], "culled", {"reason": "empty read"})
        return "culled"
    dims = data.shape
    acc = torch.zeros((NCH,) + dims, dtype=torch.float32, device=device)
    wsum = torch.zeros(dims, dtype=torch.float32, device=device)
    data_t = torch.from_numpy(data.astype(np.float32)).to(device)
    valid_t = torch.from_numpy(valid).to(device)
    positions = [(z, y, x) for z in patch_positions(dims[0], patch, stride)
                 for y in patch_positions(dims[1], patch, stride)
                 for x in patch_positions(dims[2], patch, stride)]
    n_run = 0
    for i in range(0, len(positions), batch):
        chunk = positions[i:i + batch]
        xs, keep, origs = [], [], []
        for (z, y, x) in chunk:
            sl = (slice(z, z + patch), slice(y, y + patch), slice(x, x + patch))
            cut = data_t[sl]
            if not (cut >= 1).any():          # exactly-empty patches only
                continue
            m, s = cut.mean(), cut.std().clamp(min=1e-6)
            xs.append(((cut - m) / s)[None])
            keep.append(sl)
            origs.append((brick["read_o"][0] + z, brick["read_o"][1] + y,
                          brick["read_o"][2] + x))
        if not xs:
            continue
        x_in = torch.stack(xs).to(device)
        pred = lane(x_in, origs).float().to(device)
        if not torch.isfinite(pred).all():
            raise RuntimeError("non-finite prediction")
        for j, sl in enumerate(keep):
            acc[(slice(None),) + sl] += pred[j] * win_t
            wsum[sl] += win_t
        n_run += len(keep)
    core_cut = tuple(slice(brick["core"][i].start - brick["read_o"][i],
                           brick["core"][i].stop - brick["read_o"][i])
                     for i in range(3))
    ws = wsum[core_cut]
    uncovered = ws == 0
    if uncovered.any():
        raw = data_t[core_cut]
        if (raw[uncovered] >= 1).any():
            raise RuntimeError("uncovered voxels with real data — cull bug")
    field = (acc[(slice(None),) + core_cut]
             / ws.clamp(min=1e-8)).cpu().numpy()
    org = tuple(c.start for c in brick["core"])
    store.write_brick(org, {c: field[k] for k, c in enumerate(CHANNEL_ORDER)})
    store.mark(brick["bidx"], "done",
               {"n_patches": n_run, "seconds": round(time.time() - t0, 2)})
    return "done"


def cmd_run(args):
    device = "cuda" if args.lane != "fake" else "cpu"
    ct = open_u8(args.ct)
    shape = tuple(int(v) for v in ct.shape[-3:])
    prov = {"ckpt": args.ckpt and _sha16(args.ckpt), "lane": args.lane,
            "patch": args.patch, "stride": args.stride, "brick": args.brick}
    if os.path.exists(args.out):
        store = PredStore.open(args.out)
        if store.m["provenance"] != prov and not args.force_restart:
            raise SystemExit("existing store has different provenance; "
                             "--force-restart to accept")
    else:
        store = PredStore.create(args.out, scroll_id=args.scroll,
                                 level=args.level, voxel_um=args.voxel_um,
                                 shape_zyx=shape, chunks=(128,) * 3,
                                 provenance=prov)
    bricks = brick_grid(shape, args.brick, args.margin)
    occupancy = None
    if args.coarse_ct:
        occupancy = occupancy_prescreen(args.coarse_ct, shape, args.brick,
                                        args.coarse_factor)
        print(f"prescreen: {occupancy.mean():.1%} of bricks occupied")
    calib = None
    if args.lane == "int8":
        calib = _calib_patches(ct, bricks, args.patch, n=8)
    lane = load_lane(args.lane, args.ckpt, device, args.patch, args.batch,
                     calib_patches=calib)
    win_t = torch.from_numpy(importance_window(args.patch, args.blend)).to(device)
    shard_i, shard_n = (int(v) for v in args.shard.split("/")) \
        if args.shard else (0, 1)
    ndone = nskip = 0
    t0 = time.time()
    for b in bricks:
        if b["idx"] % shard_n != shard_i:
            continue
        if occupancy is not None and not occupancy[b["bidx"]]:
            if store.status(b["bidx"]) is None:
                store.mark(b["bidx"], "culled", {"reason": "prescreen"})
            nskip += 1
            continue
        if store.check_resume(b["bidx"], force=args.force_restart):
            nskip += 1
            continue
        try:
            r = run_brick(b, ct, lane, store, win_t, args.patch, args.stride,
                          args.batch, device)
        except Exception as e:  # noqa: BLE001 — brick isolation by design
            print(f"brick {b['bidx']} FAILED: {e}")
            continue
        ndone += 1
        if ndone % 10 == 0:
            dt = (time.time() - t0) / ndone
            print(f"  {ndone} bricks done ({nskip} skipped), {dt:.1f}s/brick")
        if args.max_bricks and ndone >= args.max_bricks:
            break
    print(f"sweep: {ndone} bricks done, {nskip} skipped, "
          f"{(time.time()-t0)/60:.1f} min")


def _calib_patches(ct, bricks, patch, n):
    out = []
    for b in bricks:
        data, _ = read_patch(ct, b["read_o"], (patch,) * 3)
        if (data >= 1).mean() > 0.2:
            out.append(torch.from_numpy(zscore(data))[None, None])
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} occupied calibration patches found")
    return out


def _sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()[:16]


def cmd_report(args):
    store = PredStore.open(args.out)
    led = os.path.join(args.out, "ledger")
    done = len(glob.glob(os.path.join(led, "*.done")))
    culled = len(glob.glob(os.path.join(led, "*.culled")))
    print(json.dumps({"done": done, "culled": culled,
                      "meta": store.m}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ct", required=True)
    r.add_argument("--coarse-ct", default="", help="staged coarse level for prescreen")
    r.add_argument("--coarse-factor", type=int, default=32,
                   help="downsample factor between --ct and --coarse-ct")
    r.add_argument("--out", required=True)
    r.add_argument("--ckpt", default="")
    r.add_argument("--lane", default="fp16cl",
                   choices=["fp16cl", "fp8", "int8", "trt", "fake"])
    r.add_argument("--scroll", default="unknown")
    r.add_argument("--level", type=int, default=0)
    r.add_argument("--voxel-um", type=float, default=0.0)
    r.add_argument("--patch", type=int, default=128)
    r.add_argument("--stride", type=int, default=96)
    r.add_argument("--margin", type=int, default=32)
    r.add_argument("--brick", type=int, default=256)
    r.add_argument("--batch", type=int, default=2)
    r.add_argument("--blend", default="gaussian", choices=["gaussian", "flat"])
    r.add_argument("--shard", default="")
    r.add_argument("--max-bricks", type=int, default=0)
    r.add_argument("--force-restart", action="store_true")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    if args.cmd == "run" and args.lane != "fake" and args.patch % 64:
        raise SystemExit("--patch must be divisible by 64 for net lanes")
    args.fn(args)


if __name__ == "__main__":
    main()
