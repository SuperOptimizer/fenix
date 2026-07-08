"""wrap_probe — the end-to-end metric that matters: does the net's w field
unwrap + stitch to correct wrap ids on held-out phantoms (tracer contract).

Reuses unwrap.py verbatim (validated; do not edit); the selftest's block loop
and scoring are reimplemented here as importable functions.
"""
import time

import numpy as np
import torch

from phantom import to_batch
from unwrap import stitch, unwrap_block, wrapdiff


def wrap_acc(U, Wgt, mask):
    c = np.median((Wgt - U)[mask])
    return float((np.round(U + c) == np.round(Wgt))[mask].mean())


def blockwise_unwrap(w, conf, block=64, halo=8, iters=40):
    S = w.shape[0]
    nb = S // block
    blocks, confs = {}, {}
    for bz in range(nb):
        for by in range(nb):
            for bx in range(nb):
                sl = tuple(slice(max(0, b * block - halo),
                                 min(S, (b + 1) * block + halo))
                           for b in (bz, by, bx))
                u = unwrap_block(w[sl].astype(np.float64), conf[sl], iters=iters)
                m = conf[sl] > 0.5
                if m.any():
                    u = u + float(np.median(wrapdiff((w[sl] - u) % 1.0)[m]))
                blocks[(bz, by, bx)] = u
                confs[(bz, by, bx)] = conf[sl]
    return blocks, confs, nb


def assemble(blocks, offs, nb, block, halo, S):
    U = np.zeros((S, S, S), np.float32)
    for k, u in blocks.items():
        src = tuple(slice(halo if b > 0 else 0,
                          u.shape[i] - (halo if b < nb - 1 else 0))
                    for i, b in enumerate(k))
        dst = tuple(slice(b * block, (b + 1) * block) for b in k)
        U[dst] = u[src] + (offs[k] if offs[k] is not None else 0.0)
    return U


@torch.no_grad()
def wrap_probe(net_forward, phantoms, device="cuda", block=64, halo=8,
               iters=40, do_stitch=False):
    from model import decode_w
    accs1, accsS, resid_max = [], [], 0.0
    t0 = time.time()
    for ph in phantoms:
        x, _ = to_batch([ph], device)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device.startswith("cuda")):
            out = net_forward(x)
        w = decode_w(out["wind"].float())[0].cpu().numpy()
        conf = ph["wmask"].astype(np.float32)   # oracle interior: probes w-head only
        pap = ph["papyrus"].astype(bool)
        U = unwrap_block(w.astype(np.float64), conf, iters=iters)
        accs1.append(wrap_acc(U, ph["W"], pap))
        if do_stitch and w.shape[0] % block == 0 and w.shape[0] // block >= 2:
            blocks, confs, nb = blockwise_unwrap(w, conf, block, halo, iters)
            offs, resid = stitch(blocks, (nb,) * 3, halo, confs)
            if resid:
                resid_max = max(resid_max,
                                max(abs(r - round(r)) for r in resid))
            Us = assemble(blocks, offs, nb, block, halo, w.shape[0])
            accsS.append(wrap_acc(Us, ph["W"], pap))
    out = {"wrap_acc_1": float(np.mean(accs1)), "probe_s": time.time() - t0}
    if accsS:
        out["wrap_acc_stitch"] = float(np.mean(accsS))
        out["loop_resid_max"] = resid_max
    return out
