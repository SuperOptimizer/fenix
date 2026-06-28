import numpy as np, sys, tifffile, os

def load_fenix(d):
    G = int(open(d + "/meta.txt").read().split()[0])
    X = np.fromfile(d + "/x.f32", np.float32).reshape(G, G)
    Y = np.fromfile(d + "/y.f32", np.float32).reshape(G, G)
    Z = np.fromfile(d + "/z.f32", np.float32).reshape(G, G)
    M = np.fromfile(d + "/valid.u8", np.uint8).reshape(G, G).astype(bool)
    return X, Y, Z, M

def load_tifxyz(d):
    X = tifffile.imread(d + "/x.tif"); Y = tifffile.imread(d + "/y.tif"); Z = tifffile.imread(d + "/z.tif")
    M = (X > 0) | (Y > 0) | (Z > 0)
    return X, Y, Z, M

def metrics(X, Y, Z, M, name):
    G0, G1 = X.shape
    P = np.stack([Z, Y, X], -1).astype(np.float64)
    # edge lengths (axis neighbours, both valid)
    eh = np.linalg.norm(P[1:, :] - P[:-1, :], axis=-1)[M[1:, :] & M[:-1, :]]
    ev = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)[M[:, 1:] & M[:, :-1]]
    E = np.concatenate([eh, ev]); med = np.median(E)
    cv = E.std() / E.mean()
    out_of_range = np.mean((E < 0.5 * med) | (E > 1.5 * med))
    # per-vertex normal (central tangents) for fold detection
    nv = np.zeros((G0, G1, 3))
    tu = P[2:, 1:-1] - P[:-2, 1:-1]; tv = P[1:-1, 2:] - P[1:-1, :-2]
    n = np.cross(tu, tv); ln = np.linalg.norm(n, axis=-1, keepdims=True)
    nv[1:-1, 1:-1] = n / np.maximum(ln, 1e-9)
    core = M[1:-1, 1:-1] & M[2:, 1:-1] & M[:-2, 1:-1] & M[1:-1, 2:] & M[1:-1, :-2]
    # fold rate = adjacent core normals pointing opposite ways
    cc = np.zeros((G0, G1), bool); cc[1:-1, 1:-1] = core
    flips = 0; tot = 0
    for du, dv in ((1, 0), (0, 1)):
        a = cc[1:-1, 1:-1] & cc[1 + du:G0 - 1 + du, 1 + dv:G1 - 1 + dv]
        d = np.sum(nv[1:-1, 1:-1] * nv[1 + du:G0 - 1 + du, 1 + dv:G1 - 1 + dv], -1)
        flips += np.sum(a & (d < 0)); tot += np.sum(a)
    fold = flips / max(tot, 1)
    # self-overlap: 3D 2-voxel bins shared by uv-distant cells
    uu, vv = np.nonzero(M); pts = P[uu, vv]
    key = np.floor(pts / 2.0).astype(np.int64)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    ks = key[order]; us = uu[order]; vs = vv[order]
    same = np.all(ks[1:] == ks[:-1], 1); bnd = np.nonzero(~same)[0] + 1
    groups = np.split(np.arange(len(ks)), bnd)
    multi = [g for g in groups if len(g) > 1]
    farfold = 0
    for g in multi:
        if max(us[g].max() - us[g].min(), vs[g].max() - vs[g].min()) > 5: farfold += len(g)
    overlap_pts = sum(len(g) for g in multi)
    area = float(np.sum(eh[:, None].size and (np.linalg.norm(P[1:, :-1] - P[:-1, :-1], axis=-1) *
                  np.linalg.norm(P[:-1, 1:] - P[:-1, :-1], axis=-1))[M[1:, :-1] & M[:-1, 1:] & M[:-1, :-1]]))
    print(f"=== {name} ===")
    print(f"  valid pts            {M.sum():,}   area ~{area:,.0f} vx^2")
    print(f"  spacing: median {med:.2f}  CV {cv:.3f}  out-of-[.5,1.5]x {100*out_of_range:.1f}%   (want CV low, oor~0)")
    print(f"  fold rate (normal flips across edges) {100*fold:.2f}%   (want ~0)")
    print(f"  self-overlap pts {100*overlap_pts/len(pts):.1f}%  | distant-fold pts {100*farfold/len(pts):.1f}%   (want ~0)")
    print(f"  coverage efficiency unique-bins/pts {len(groups)/len(pts):.3f}   (1.0 = injective)")

for arg in sys.argv[1:]:
    name, d = arg.split("=")
    X, Y, Z, M = (load_tifxyz(d) if os.path.exists(d + "/x.tif") else load_fenix(d))
    metrics(X, Y, Z, M, name)
