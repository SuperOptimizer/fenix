"""Meshes/PPMs -> winding constraint point sets for fitfield.py.

Meshes are NEVER voxelized into labels here (design doctrine: the fit is the
label generator). Output per mesh component: constraints_<id>.npz with
pos_zyx f32 [N,3] (target-grid vox), theta f64 [N] (per-component unwrapped,
radians — carries an arbitrary per-component 2*pi*k the fit absorbs),
normal_zyx f32 [N,3] (unit, outward = toward increasing W), weight f32 [N],
meta json (source, transform, quality stats, grid).

XYZ appears only inside the OBJ/PPM parsers; converted at the parse boundary.
"""
import argparse
import glob
import json
import os
from collections import defaultdict, deque

import numpy as np


def xyz_to_zyx(a):
    return np.ascontiguousarray(np.asarray(a)[..., ::-1])


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                for i in range(1, len(idx) - 1):     # fan-triangulate
                    faces.append([idx[0] - 1, idx[i] - 1, idx[i + 1] - 1])
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def load_ppm(path, step=8):
    """VC .ppm: text header then H*W records of 6 float64 (x,y,z,nx,ny,nz)."""
    hdr = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line == "<>":
                break
            if ":" in line:
                k, v = line.split(":", 1)
                hdr[k.strip()] = v.strip()
        H, W = int(hdr["height"]), int(hdr["width"])
        assert hdr.get("type", "double") == "double" and int(hdr.get("dim", 6)) == 6
        data = np.fromfile(f, np.float64, H * W * 6).reshape(H, W, 6)
    data = data[::step, ::step]
    pos, nrm = data[..., :3], data[..., 3:]
    valid = ~(np.abs(data).sum(-1) == 0)
    return {"pos_xyz": pos, "normal_xyz": nrm, "valid": valid}


def load_umbilicus(path):
    pts = []
    with open(path) as f:
        for line in f:
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    u = np.asarray(pts, np.float32)
    return u[np.argsort(u[:, 0])]


def umbilicus_at(umb, z):
    z = np.asarray(z, np.float64)
    cy = np.interp(z, umb[:, 0], umb[:, 1])   # clamped extrapolation
    cx = np.interp(z, umb[:, 0], umb[:, 2])
    return np.stack([cy, cx], axis=-1)


def _components(n_verts, faces):
    adj = defaultdict(list)
    for a, b, c in faces:
        adj[a] += [b, c]
        adj[b] += [a, c]
        adj[c] += [a, b]
    comp = np.full(n_verts, -1, np.int64)
    cid = 0
    for s in range(n_verts):
        if comp[s] >= 0 or s not in adj:
            continue
        comp[s] = cid
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if comp[v] < 0:
                    comp[v] = cid
                    q.append(v)
        cid += 1
    return comp, cid, adj


def assign_theta(verts_zyx, faces, umb):
    """Unwrapped theta* per vertex (per-component BFS, nearest-branch continuation).
    Returns (theta_star f64 [N], comp i64 [N], ncomp, edge_violation_rate)."""
    c = umbilicus_at(umb, verts_zyx[:, 0])
    theta = np.arctan2(verts_zyx[:, 1] - c[:, 0], verts_zyx[:, 2] - c[:, 1])
    comp, ncomp, adj = _components(len(verts_zyx), faces)
    ts = np.full(len(verts_zyx), np.nan)
    for cid in range(ncomp):
        seeds = np.nonzero(comp == cid)[0]
        s = seeds[0]
        ts[s] = theta[s]
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if np.isnan(ts[v]):
                    ts[v] = theta[v] + 2 * np.pi * np.round((ts[u] - theta[v]) / (2 * np.pi))
                    q.append(v)
    viol = 0
    tot = 0
    for a, b, c3 in faces[:: max(1, len(faces) // 200000)]:
        for u, v in ((a, b), (b, c3)):
            tot += 1
            if abs(ts[u] - ts[v]) > np.pi / 2:
                viol += 1
    return ts, comp, ncomp, viol / max(tot, 1)


def vertex_normals(verts_zyx, faces):
    n = np.zeros_like(verts_zyx)
    a, b, c = (verts_zyx[faces[:, i]] for i in range(3))
    fn = np.cross(b - a, c - a)         # area-weighted (unnormalized cross)
    for i in range(3):
        np.add.at(n, faces[:, i], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def orient_outward(normals, verts_zyx, comp, ncomp, umb):
    """Flip per-component so mean(dot(n, r_hat)) > 0 (outward = increasing W).
    Returns per-component mean radial dot AFTER flip (quality stat)."""
    c = umbilicus_at(umb, verts_zyx[:, 0])
    r = np.stack([np.zeros(len(verts_zyx)), verts_zyx[:, 1] - c[:, 0],
                  verts_zyx[:, 2] - c[:, 1]], axis=1)
    r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
    dots = (normals * r).sum(1)
    quality = []
    for cid in range(ncomp):
        m = comp == cid
        d = dots[m].mean()
        if d < 0:
            normals[m] *= -1
            d = -d
        quality.append(float(d))
    return quality


def voronoi_weights(verts_zyx, faces):
    w = np.zeros(len(verts_zyx))
    a, b, c = (verts_zyx[faces[:, i]] for i in range(3))
    area3 = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 6.0
    for i in range(3):
        np.add.at(w, faces[:, i], area3)
    return w.astype(np.float32)


def process_mesh(path, umb, scale, offset, volume_shape, out_dir, strict,
                 min_verts=100, meta_extra=None):
    v_xyz, faces = load_obj(path)
    v = xyz_to_zyx(v_xyz) * scale + np.asarray(offset, np.float64)
    mesh_id = os.path.splitext(os.path.basename(path))[0]
    if volume_shape is not None:
        inside = ((v >= 0) & (v < np.asarray(volume_shape))).all(1).mean()
        if inside < 0.99:
            msg = f"{mesh_id}: only {inside:.1%} of vertices inside volume — wrong grid?"
            if strict:
                raise RuntimeError(msg)
            print("SKIP", msg)
            return []
    c = umbilicus_at(umb, v[:, 0])
    r = np.hypot(v[:, 1] - c[:, 0], v[:, 2] - c[:, 1])
    keep = r >= 2.0                         # theta undefined at the axis
    if keep.sum() < min_verts:
        print(f"SKIP {mesh_id}: {keep.sum()} valid vertices")
        return []
    remap = np.cumsum(keep) - 1
    fkeep = keep[faces].all(1)
    faces = remap[faces[fkeep]]
    v = v[keep]
    ts, comp, ncomp, viol = assign_theta(v, faces, umb)
    if viol > 1e-3:
        msg = f"{mesh_id}: theta edge-violation rate {viol:.2%}"
        if strict:
            raise RuntimeError(msg)
        print("SKIP", msg)
        return []
    normals = vertex_normals(v, faces)
    quality = orient_outward(normals, v, comp, ncomp, umb)
    weights = voronoi_weights(v, faces)
    outs = []
    for cid in range(ncomp):
        m = (comp == cid) & np.isfinite(ts)
        if m.sum() < min_verts:
            continue
        name = f"constraints_{mesh_id}" + (f"_c{cid}" if ncomp > 1 else "") + ".npz"
        out = os.path.join(out_dir, name)
        meta = {"source": path, "mesh_id": mesh_id, "component": cid,
                "scale": scale, "offset": list(offset),
                "edge_violation_rate": viol, "mean_radial_dot": quality[cid],
                **(meta_extra or {})}
        tmp = out + f".tmp.{os.getpid()}.npz"
        np.savez_compressed(tmp, pos_zyx=v[m].astype(np.float32), theta=ts[m],
                            normal_zyx=normals[m].astype(np.float32),
                            weight=weights[m], meta=json.dumps(meta))
        os.replace(tmp, out)
        outs.append(out)
    return outs


def process_ppm(path, umb, scale, offset, out_dir, step=8, teacher_conf_meta=None):
    d = load_ppm(path, step=step)
    val = d["valid"]
    pos = xyz_to_zyx(d["pos_xyz"][val]) * scale + np.asarray(offset, np.float64)
    nrm = xyz_to_zyx(d["normal_xyz"][val])
    # 4-connected grid adjacency as pseudo-faces for the unwrap
    H, W = val.shape
    idx = np.full((H, W), -1, np.int64)
    idx[val] = np.arange(val.sum())
    edges = []
    for dz, dx in ((0, 1), (1, 0)):
        a = idx[: H - dz, : W - dx]
        b = idx[dz:, dx:]
        m = (a >= 0) & (b >= 0)
        edges.append(np.stack([a[m], b[m], a[m]], axis=1))
    faces = np.concatenate(edges) if edges else np.zeros((0, 3), np.int64)
    ts, comp, ncomp, viol = assign_theta(pos, faces, umb)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-9)
    quality = orient_outward(nrm, pos, comp, ncomp, umb)
    mesh_id = os.path.basename(os.path.dirname(path)) or \
        os.path.splitext(os.path.basename(path))[0]
    outs = []
    for cid in range(ncomp):
        m = (comp == cid) & np.isfinite(ts)
        if m.sum() < 100:
            continue
        out = os.path.join(out_dir, f"constraints_ppm_{mesh_id}_c{cid}.npz")
        meta = {"source": path, "mesh_id": f"ppm_{mesh_id}", "component": cid,
                "scale": scale, "offset": list(offset), "step": step,
                "edge_violation_rate": viol, "mean_radial_dot": quality[cid],
                **(teacher_conf_meta or {})}
        tmp = out + f".tmp.{os.getpid()}.npz"
        np.savez_compressed(tmp, pos_zyx=pos[m].astype(np.float32), theta=ts[m],
                            normal_zyx=nrm[m].astype(np.float32),
                            weight=np.full(m.sum(), float(step * step), np.float32),
                            meta=json.dumps(meta))
        os.replace(tmp, out)
        outs.append(out)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", default="")
    ap.add_argument("--ppms", default="")
    ap.add_argument("--umbilicus", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--offset", type=float, nargs=3, default=[0, 0, 0])
    ap.add_argument("--volume-shape", type=int, nargs=3, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    umb = load_umbilicus(args.umbilicus)
    n = 0
    for path in sorted(glob.glob(args.meshes)) if args.meshes else []:
        n += len(process_mesh(path, umb, args.scale, args.offset,
                              args.volume_shape, args.out, args.strict))
    for path in sorted(glob.glob(args.ppms)) if args.ppms else []:
        n += len(process_ppm(path, umb, args.scale, args.offset, args.out,
                             step=args.step))
    print(f"wrote {n} constraint files -> {args.out}")


if __name__ == "__main__":
    main()
