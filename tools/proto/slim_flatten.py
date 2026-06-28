"""True global flattening of the grown surface: ARAP/SLIM local-global parameterization
(Liu et al. 2008 'A Local/Global Approach to Mesh Parameterization' + symmetric-Dirichlet
energy reporting, the SLIM family). Fixes the 3D surface, solves fresh low-distortion UVs
over the whole mesh at once (cotangent-Laplacian global solve, prefactored), then renders the
texture in UV space. Prototype (to port to C++ flatten/)."""
import numpy as np, os
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import map_coordinates
import PIL.Image as I

D = "data/fenix_trace"
G = int(open(D + "/meta.txt").read().split()[0])
X = np.fromfile(D + "/x.f32", np.float32).reshape(G, G)
Y = np.fromfile(D + "/y.f32", np.float32).reshape(G, G)
Z = np.fromfile(D + "/z.f32", np.float32).reshape(G, G)
M = np.fromfile(D + "/valid.u8", np.uint8).reshape(G, G).astype(bool)

# --- index valid vertices, build triangles from valid quads (flatten the REAL surface only;
#     holes are filled later in 2D so they don't distort the parameterization) ---
vid = -np.ones((G, G), np.int64)
uu, vv = np.nonzero(M)
vid[uu, vv] = np.arange(len(uu))
P3 = np.stack([Z[uu, vv], Y[uu, vv], X[uu, vv]], 1).astype(np.float64)  # 3D positions
gridUV = np.stack([uu, vv], 1).astype(np.float64)                       # initial UV = grid indices
N = len(uu)

tris = []
Vm = M
A = vid[:-1, :-1]; B = vid[1:, :-1]; C = vid[:-1, 1:]; Dd = vid[1:, 1:]
q = Vm[:-1, :-1] & Vm[1:, :-1] & Vm[:-1, 1:] & Vm[1:, 1:]
# two triangles per fully-valid quad
t1 = np.stack([A[q], B[q], C[q]], 1)
t2 = np.stack([B[q], Dd[q], C[q]], 1)
F = np.concatenate([t1, t2], 0)
print(f"verts {N:,}  tris {len(F):,}")

# keep the largest connected component (holes split the mesh -> multiple nullspaces)
import scipy.sparse.csgraph as csg
e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]], 0)
adj = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(N, N))
adj = adj + adj.T
_, lab = csg.connected_components(adj, directed=False)
big = np.argmax(np.bincount(lab))
keep = lab == big
ni = -np.ones(N, np.int64); ni[keep] = np.arange(int(keep.sum()))
fm = keep[F[:, 0]] & keep[F[:, 1]] & keep[F[:, 2]]
F = ni[F[fm]]; P3 = P3[keep]; gridUV = gridUV[keep]; uu, vv = uu[keep], vv[keep]; N = int(keep.sum())
print(f"largest component: verts {N:,}  tris {len(F):,}")

# --- per-triangle isometric 2D rest coords from 3D edge lengths ---
p0, p1, p2 = P3[F[:, 0]], P3[F[:, 1]], P3[F[:, 2]]
e01 = np.linalg.norm(p1 - p0, axis=1); e02 = np.linalg.norm(p2 - p0, axis=1)
d12 = p2 - p1
e12 = np.linalg.norm(d12, axis=1)
# place v0=(0,0), v1=(e01,0), v2=(x,y)
cosA = np.clip((e01**2 + e02**2 - e12**2) / (2 * e01 * e02 + 1e-12), -1, 1)
x2 = e02 * cosA; y2 = e02 * np.sqrt(np.clip(1 - cosA**2, 0, 1))
R0 = np.stack([np.zeros(len(F)), np.zeros(len(F))], 1)
R1 = np.stack([e01, np.zeros(len(F))], 1)
R2 = np.stack([x2, y2], 1)
rest = np.stack([R0, R1, R2], 1)  # (T,3,2)
area = 0.5 * np.abs(e01 * y2) + 1e-12

# rest edge matrix Dm = [r1-r0, r2-r0] (2x2), inverse for Jacobian
Dm = np.stack([rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]], 2)  # (T,2,2)
Dm_inv = np.linalg.inv(Dm)

# --- cotangent weights (from rest geometry) for the global Laplacian ---
def cot(a, b, c):  # cot of angle at vertex with opposite edge, from squared lengths
    return None
# edge cotans: for triangle, w for edge (i,j) = cot(angle at opposite vertex k)
L01 = e01**2; L12 = e12**2; L02 = e02**2
# angle at v2 opposite e01; etc. cot = (sum of adj squared - opp squared)/(4*area)
w01 = (L02 + L12 - L01) / (4 * area)  # opposite v2 -> weight on edge (0,1)
w12 = (L01 + L02 - L12) / (4 * area)  # opposite v0 -> edge (1,2)
w02 = (L01 + L12 - L02) / (4 * area)  # opposite v1 -> edge (0,2)

def build_L():
    rows, cols, vals = [], [], []
    for (a, b, w) in ((F[:, 0], F[:, 1], w01), (F[:, 1], F[:, 2], w12), (F[:, 0], F[:, 2], w02)):
        for i, j in ((a, b), (b, a)):
            rows.append(i); cols.append(j); vals.append(-w)
            rows.append(i); cols.append(i); vals.append(w)
    Lm = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    return Lm
L = build_L()
# pin one vertex to remove the nullspace (translation); keep rotation fixed via the local step
pin = 0
L = L.tolil(); L[pin, :] = 0; L[pin, pin] = 1; L = L.tocsc()
solve = spla.factorized(L)

def jacobians(UV):
    u0, u1, u2 = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    Ds = np.stack([u1 - u0, u2 - u0], 2)  # (T,2,2)
    return Ds @ Dm_inv  # J maps rest2D -> UV

def sym_dirichlet(UV):
    J = jacobians(UV)
    s = np.linalg.svd(J, compute_uv=False)  # (T,2)
    s = np.clip(s, 1e-6, None)
    e = 0.5 * (s[:, 0]**2 + s[:, 1]**2 + 1.0 / s[:, 0]**2 + 1.0 / s[:, 1]**2)
    return float(np.average(e, weights=area)), s

UV = gridUV.copy().astype(np.float64)
e0, s0 = sym_dirichlet(UV)
print(f"init  sym-Dirichlet {e0:.3f}  (ideal 2.0)  stretch sigma p99 {np.percentile(s0,99):.2f}")

for it in range(20):
    J = jacobians(UV)
    U, S, Vt = np.linalg.svd(J)
    Rt = U @ Vt
    det = np.linalg.det(Rt)
    U[det < 0, :, -1] *= -1
    Rt = U @ Vt  # closest rotation per triangle (ARAP local step)
    # global RHS: b_i = sum_t sum_j w_ij^t R_t (rest_i - rest_j)
    b = np.zeros((N, 2))
    for (ia, ib, w, ra, rb) in ((0, 1, w01, rest[:, 0], rest[:, 1]),
                                 (1, 2, w12, rest[:, 1], rest[:, 2]),
                                 (0, 2, w02, rest[:, 0], rest[:, 2])):
        d = np.einsum('tij,tj->ti', Rt, (ra - rb))  # R*(rest_a-rest_b)
        contrib = (w[:, None] * d)
        np.add.at(b, F[:, ia], contrib); np.add.at(b, F[:, ib], -contrib)
    b[pin] = UV[pin]
    UV = np.stack([solve(b[:, 0]), solve(b[:, 1])], 1)
e1, s1 = sym_dirichlet(UV)
print(f"final sym-Dirichlet {e1:.3f}  ({100*(e0-e1)/(e0-2+1e-9):.0f}% of the way to ideal)  stretch p99 {np.percentile(s1,99):.2f}")

# --- render: sample CT per vertex, lay out in UV space (grid layout vs SLIM layout) ---
def load_nrrd(p):
    f = open(p, "rb")
    while f.readline().strip() != b"": pass
    return np.frombuffer(f.read(), np.float32).reshape(1024, 1024, 1024)
CT = load_nrrd("data/paris4_mid_1024.nrrd")
gray = map_coordinates(CT, [P3[:, 0], P3[:, 1], P3[:, 2]], order=1)

from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes, binary_erosion
def rasterize(UVc, name):
    u = UVc - UVc.min(0); res = 1.0  # ~1 px per voxel-step
    W = int(u[:, 0].max() / res) + 1; H = int(u[:, 1].max() / res) + 1
    gz, gy = np.mgrid[0:W, 0:H]
    pts = u / res
    out = griddata(pts, gray, (gz, gy), method="linear", fill_value=0)  # fills interior holes
    # domain = coverage with INTERIOR holes filled in; mask only the true exterior boundary
    tree = cKDTree(pts)
    dist, _ = tree.query(np.stack([gz.ravel(), gy.ravel()], 1), k=1)
    cover = (dist.reshape(W, H) <= 1.6)
    domain = binary_erosion(binary_fill_holes(cover), iterations=1)  # holes filled; trim ragged edge
    lo, hi = np.percentile(gray, 1), np.percentile(gray, 99)
    img = np.clip((out - lo) / (hi - lo), 0, 1); img[~domain] = 0
    I.fromarray((img.T * 255).astype(np.uint8)).save(f"data/{name}.jpg", quality=90)
    print(f"  wrote data/{name}.jpg {W}x{H}  (interior holes inpainted)")
rasterize(UV, "fenix_slim_ct")
print("done")
