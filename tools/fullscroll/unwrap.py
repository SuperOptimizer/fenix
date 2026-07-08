"""Tracer prototype: papyrus-weighted phase unwrap + integer-offset block stitch.

The tracer contract from docs/design/full-scroll-model.md made executable:

1. Per block: unwrap the net's fractional winding coordinate w in [0,1) into a
   continuous scalar U (U mod 1 = w) by weighted least squares (Ghiglia-Romero
   1994: PCG on the confidence-weighted Laplacian, DCT-Poisson preconditioner).
   Confidence = papyrus probability — background w is garbage and must not vote.
2. Between blocks: one integer offset per adjacent pair = round(median(U_a - U_b))
   on the shared halo (papyrus-masked); spanning tree from block 0 sets each
   block's offset; non-tree edges report loop-closure residuals (should be ~0;
   nonzero = conflict regions for the global solve to arbitrate).
3. Wrap id per voxel = round(U + global const) — level sets of the stitched field.

Self-test (CPU): phantom GT w + noise-through-sin/cos as a stand-in for net output,
garbage w in background, wrap-id accuracy vs the analytic W oracle.

C++ port target: src/winding wrap-label/wrap-fill (block+halo+stitch is already
the out-of-core execution model there).
"""
import numpy as np
from scipy.fft import dctn, idctn


def wrapdiff(a):
    """Map a difference of mod-1 values to (-0.5, 0.5]."""
    return (a + 0.5) % 1.0 - 0.5


def _poisson_dct(rhs):
    """Solve lap(U) = rhs, Neumann BC, via DCT-II diagonalization."""
    D, H, W = rhs.shape
    r = dctn(rhs, type=2, norm="ortho")
    kz = 2 * (np.cos(np.pi * np.arange(D) / D) - 1)[:, None, None]
    ky = 2 * (np.cos(np.pi * np.arange(H) / H) - 1)[None, :, None]
    kx = 2 * (np.cos(np.pi * np.arange(W) / W) - 1)[None, None, :]
    eig = kz + ky + kx
    eig[0, 0, 0] = 1.0
    r /= eig
    r[0, 0, 0] = 0.0
    return idctn(r, type=2, norm="ortho")


def _wgrads(w, conf, floor=1e-3):
    """Wrapped forward diffs of w and per-edge weights (squared min confidence)."""
    gs, wts = [], []
    for ax in range(3):
        g = wrapdiff(np.diff(w, axis=ax))
        c = np.minimum(np.take(conf, range(0, w.shape[ax] - 1), axis=ax),
                       np.take(conf, range(1, w.shape[ax]), axis=ax)) ** 2
        gs.append(g)
        wts.append(np.maximum(c, floor))
    return gs, wts


def _div(fields, shape):
    out = np.zeros(shape, np.float64)
    for ax, f in enumerate(fields):
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[ax] = slice(0, shape[ax] - 1)
        sl_hi[ax] = slice(1, shape[ax])
        out[tuple(sl_lo)] += f
        out[tuple(sl_hi)] -= f
    return out


def unwrap_block(w, conf, iters=40, tol=1e-8, return_resid=False):
    """Weighted LS phase unwrap. Returns U with U mod 1 ~= w (up to a constant);
    with return_resid=True returns (U, final relative residual)."""
    gs, wts = _wgrads(w, conf)
    b = _div([g * wt for g, wt in zip(gs, wts)], w.shape)

    def A(u):
        du = [np.diff(u, axis=ax) for ax in range(3)]
        return _div([d * wt for d, wt in zip(du, wts)], w.shape)

    u = np.zeros(w.shape, np.float64)
    r = b - A(u)
    z = _poisson_dct(r)
    p = z.copy()
    rz = (r * z).sum()
    b2 = (b * b).sum() + 1e-30
    resid = 1.0
    for _ in range(iters):
        Ap = A(p)
        alpha = rz / ((p * Ap).sum() + 1e-30)
        u += alpha * p
        r -= alpha * Ap
        resid = (r * r).sum() / b2
        if resid < tol:
            break
        z = _poisson_dct(r)
        rz_new = (r * z).sum()
        p = z + (rz_new / (rz + 1e-30)) * p
        rz = rz_new
    uf = u.astype(np.float32)
    return (uf, float(resid)) if return_resid else uf


def stitch(blocks, grid, halo, conf_blocks, min_overlap=200):
    """Integer-offset stitch of per-block unwraps.

    blocks/conf_blocks: dict (bz,by,bx) -> array with `halo` overlap on interior
    faces. Returns dict of per-block float offsets (tree-solved) and the list of
    loop-closure residuals from non-tree edges.
    """
    offs = {k: None for k in blocks}
    first = min(blocks)
    offs[first] = 0.0
    edges = []
    for k in blocks:
        for ax in range(3):
            nb = list(k)
            nb[ax] += 1
            nb = tuple(nb)
            if nb in blocks:
                edges.append((k, nb, ax))

    def edge_delta(a, bkey, ax):
        ua, ub = blocks[a], blocks[bkey]
        ca, cb = conf_blocks[a], conf_blocks[bkey]
        sla = [slice(None)] * 3
        slb = [slice(None)] * 3
        sla[ax] = slice(ua.shape[ax] - 2 * halo, ua.shape[ax])
        slb[ax] = slice(0, 2 * halo)
        m = (ca[tuple(sla)] > 0.5) & (cb[tuple(slb)] > 0.5)
        if m.sum() < min_overlap:
            return None
        return float(np.median((ua[tuple(sla)] - ub[tuple(slb)])[m]))

    # BFS spanning tree + loop residuals
    residuals = []
    pending = list(edges)
    progressed = True
    while progressed:
        progressed = False
        rest = []
        for a, bkey, ax in pending:
            d = edge_delta(a, bkey, ax)
            if d is None:
                continue
            if offs[a] is not None and offs[bkey] is None:
                offs[bkey] = offs[a] + round(d)
                progressed = True
            elif offs[bkey] is not None and offs[a] is None:
                offs[a] = offs[bkey] - round(d)
                progressed = True
            elif offs[a] is not None and offs[bkey] is not None:
                residuals.append((offs[a] + d) - offs[bkey])
            else:
                rest.append((a, bkey, ax))
        pending = rest
    return offs, residuals


def _selftest():
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from phantom import make_phantom
    rng = np.random.default_rng(0)
    ph = make_phantom(128, seed=5)
    pap = ph["papyrus"].astype(bool)
    Wgt = ph["W"]

    # simulate net output: noise through sin/cos everywhere the field is
    # supervised (scroll interior incl. gaps — the Eulerian view), garbage
    # outside the supervised region
    interior = ph["wmask"].astype(bool)
    ang = 2 * np.pi * ph["w"]
    s = np.sin(ang) + rng.normal(0, 0.15, ang.shape)
    c = np.cos(ang) + rng.normal(0, 0.15, ang.shape)
    w = (np.arctan2(s, c) / (2 * np.pi)) % 1.0
    w[~interior] = rng.uniform(0, 1, (~interior).sum())
    conf = interior.astype(np.float32)

    def wrap_acc(U, mask):
        cshift = np.median((Wgt - U)[mask])
        return (np.round(U + cshift) == np.round(Wgt))[mask].mean(), cshift

    # single-block oracle path
    U = unwrap_block(w.astype(np.float64), conf)
    acc, _ = wrap_acc(U, pap)
    print(f"single-block: wrap-id accuracy on papyrus = {acc:.4f}")

    # 2x2x2 blocks of 64+2*halo, independent unwraps, integer stitch
    halo = 8
    B = 64
    blocks, confs = {}, {}
    for bz in range(2):
        for by in range(2):
            for bx in range(2):
                sl = tuple(slice(max(0, b * B - halo), min(128, (b + 1) * B + halo))
                           for b in (bz, by, bx))
                blocks[(bz, by, bx)] = unwrap_block(
                    w[sl].astype(np.float64), conf[sl])
                confs[(bz, by, bx)] = conf[sl]
    # per-block: align each unwrap's fractional part to w (constant mod 1), so
    # inter-block deltas are near-integers
    for k in blocks:
        sl = tuple(slice(max(0, b * B - halo), min(128, (b + 1) * B + halo))
                   for b in k)
        m = confs[k] > 0.5
        d = wrapdiff((w[sl] - blocks[k]) % 1.0)[m]
        blocks[k] += float(np.median(d))
    offs, residuals = stitch(blocks, (2, 2, 2), halo, confs)
    print(f"stitch offsets: { {k: round(v, 2) for k, v in offs.items()} }")
    print(f"loop residuals (should be ~integer-free): "
          f"{[round(r, 3) for r in residuals]}")

    # assemble stitched field (interior crops) and score
    Us = np.zeros((128, 128, 128), np.float32)
    for k, u in blocks.items():
        bz, by, bx = k
        src = tuple(slice(halo if b > 0 else 0,
                          u.shape[i] - (halo if b < 1 else 0))
                    for i, b in enumerate(k))
        dst = tuple(slice(b * B, (b + 1) * B) for b in k)
        Us[dst] = u[src] + offs[k]
    acc_s, _ = wrap_acc(Us, pap)
    print(f"stitched 2x2x2: wrap-id accuracy on papyrus = {acc_s:.4f}")
    ok = acc > 0.99 and acc_s > 0.99 and all(abs(r - round(r)) < 0.3
                                             for r in residuals)
    print("SELFTEST", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    _selftest()
