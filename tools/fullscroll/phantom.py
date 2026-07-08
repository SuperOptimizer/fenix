"""Synthetic scroll phantom with dense ground truth for every FullScrollNet head.

Archimedean spiral sheet stack around a wandering umbilicus, ZYX. The generating
scalar is the total winding coordinate

    W(z,y,x) = r/pitch - theta/(2*pi) + smooth perturbation

so sheet k lives at W ~ k, the fractional winding coordinate is w = W mod 1, and
the sheet normal is grad(W) normalized (points toward increasing W = outward).
Recto = outward face of each sheet, verso = inward face. Ink = low-frequency
blobs on the recto shell with a faint density bump in the CT rendering. Damage
holes knock out papyrus (and its labels) so the net sees broken sheets.

Everything the tracer contract needs is exact by construction — this validates
heads + unwrap/stitch end-to-end before any real wrap labels exist.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

DTYPE = np.float32


def _smooth_noise(rng, shape, coarse, sigma_out=0.0):
    """Low-frequency field: coarse randn trilinearly upsampled to `shape`."""
    import torch
    import torch.nn.functional as F
    g = torch.from_numpy(rng.standard_normal((1, 1) + coarse).astype(DTYPE))
    up = F.interpolate(g, size=shape, mode="trilinear", align_corners=True)
    out = up[0, 0].numpy()
    if sigma_out:
        out = gaussian_filter(out, sigma_out)
    return out


def _tile_patches(rng, shape, sampler, patch=32, blend=8):
    """Tile random bank patches (random flips) over `shape` with soft-blend."""
    out = np.zeros(shape, DTYPE)
    wgt = np.zeros(shape, DTYPE) + 1e-6
    ramp1 = np.minimum(np.arange(patch) + 1, blend) / blend
    ramp = np.minimum.reduce(np.meshgrid(*([np.minimum(ramp1, ramp1[::-1])] * 3),
                                         indexing="ij"))
    step = patch - blend
    for z0 in range(0, shape[0], step):
        for y0 in range(0, shape[1], step):
            for x0 in range(0, shape[2], step):
                p = sampler(rng)
                for ax in range(3):
                    if rng.random() < 0.5:
                        p = np.flip(p, axis=ax)
                sl = tuple(slice(o, min(o + patch, shape[i]))
                           for i, o in enumerate((z0, y0, x0)))
                cut = tuple(slice(0, s.stop - s.start) for s in sl)
                out[sl] += p[cut] * ramp[cut]
                wgt[sl] += ramp[cut]
    return out / wgt


def _render_bank(rng, shape, Wf, papyrus, ink, bank):
    """CT from harvested real statistics; labels untouched."""
    st = bank.stats
    fdist = Wf - np.round(Wf)
    half = st.get("profile_halflen_wraps", 0.75)
    Ls = bank.profiles.shape[1]
    nmix = int(rng.integers(2, 4))
    profs = [bank.sample_profile(rng) for _ in range(nmix)]
    lam = np.stack([_smooth_noise(rng, shape, (4, 4, 4)) for _ in range(nmix)])
    lam = np.exp(lam) / np.exp(lam).sum(0)
    pos = np.clip((fdist + half) / (2 * half), 0.0, 1.0) * (Ls - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, Ls - 1)
    fr = (pos - lo).astype(DTYPE)
    ct = np.zeros(shape, DTYPE)
    for i, p in enumerate(profs):
        ct += lam[i] * (p[lo] * (1 - fr) + p[hi] * fr)
    outside = np.abs(fdist) >= half
    ct = np.where(outside, st["air_mean"], ct)
    ct += _tile_patches(rng, shape, bank.sample_pap_residual) * papyrus \
        * rng.uniform(0.7, 1.3)
    airtex = _tile_patches(rng, shape, bank.sample_air)
    ct = np.where(papyrus > 0, ct, airtex)
    ct += rng.uniform(0.05, 0.15) * (st["pap_mean"] - st["air_mean"]) * ink
    lo_n = st["air_mean"] - 2 * st["air_std"]
    hi_n = st["pap_mean"] + 2 * st["pap_std"]
    norm = np.clip((ct - lo_n) / max(hi_n - lo_n, 1e-3), 0, 1)
    norm = norm ** rng.uniform(0.85, 1.18)
    ct = lo_n + norm * (hi_n - lo_n)
    ct = gaussian_filter(ct.astype(DTYPE), 0.7)
    ct += rng.standard_normal(shape).astype(DTYPE) * st["air_std"] \
        * rng.uniform(0.5, 1.2)
    return ct.astype(DTYPE)


def make_phantom(size=128, pitch=14.0, thickness=5.0, seed=0, bank=None,
                 bank_p=1.0):
    """Returns a dict of (size^3) arrays: ct f32, papyrus/recto/verso/ink u8,
    w f32 in [0,1), normal (3,D,H,W) f32, wmask u8 (where w/normal are supervised).

    bank: optional texture.TextureBank — replaces the parametric CT rendering
    with harvested real-CT profiles/residuals/air (labels stay EXACT and
    identical for the same seed regardless of bank).
    """
    rng = np.random.default_rng(seed)
    D = H = Wd = size
    z, y, x = np.meshgrid(np.arange(D, dtype=DTYPE), np.arange(H, dtype=DTYPE),
                          np.arange(Wd, dtype=DTYPE), indexing="ij")

    # wandering umbilicus: smooth per-z center + patch offset so the core is
    # usually outside the patch (a mid-scroll patch), sometimes inside
    off = rng.uniform(-1.5, 1.5, 2) * size
    zz = np.arange(D, dtype=DTYPE)
    amp = size * 0.03
    cy = (H / 2 + off[0] + amp * np.sin(2 * np.pi * zz / D * rng.uniform(0.5, 2)
                                        + rng.uniform(0, 6)))[:, None, None]
    cx = (Wd / 2 + off[1] + amp * np.cos(2 * np.pi * zz / D * rng.uniform(0.5, 2)
                                         + rng.uniform(0, 6)))[:, None, None]
    ry, rx = y - cy, x - cx
    r = np.sqrt(ry * ry + rx * rx) + 1e-6
    theta = np.arctan2(ry, rx)

    Wf = r / pitch - theta / (2 * np.pi)
    Wf = Wf + 0.35 * _smooth_noise(rng, (D, H, Wd), (5, 5, 5))  # sheet waviness
    Wf = Wf.astype(DTYPE)

    gz, gy, gx = np.gradient(Wf)
    # theta branch cut: atan2 jumps 2*pi -> W jumps 1 across one voxel; the true
    # field is continuous there, so repair gradient voxels near the cut
    for g in (gz, gy, gx):
        bad = np.abs(g) > 0.25
        g[bad] = 0.0
    nrm = np.sqrt(gz * gz + gy * gy + gx * gx) + 1e-8
    normal = np.stack([gz / nrm, gy / nrm, gx / nrm]).astype(DTYPE)

    fdist = Wf - np.round(Wf)                       # signed distance in W units
    t_half = thickness / (2 * pitch)
    papyrus = np.abs(fdist) < t_half
    band = min(0.35 * t_half, 1.0 / pitch)          # ~1 voxel surface shells
    recto = papyrus & (fdist > t_half - band)       # outward face (increasing W)
    verso = papyrus & (fdist < -t_half + band)
    core = r < pitch * 0.75                          # umbilicus core = background
    papyrus &= ~core
    recto &= ~core
    verso &= ~core

    # damage: ellipsoidal holes
    for _ in range(rng.integers(1, 5)):
        c = rng.uniform(0, size, 3)
        ax = rng.uniform(size * 0.04, size * 0.14, 3)
        hole = (((z - c[0]) / ax[0]) ** 2 + ((y - c[1]) / ax[1]) ** 2
                + ((x - c[2]) / ax[2]) ** 2) < 1
        papyrus &= ~hole
        recto &= ~hole
        verso &= ~hole

    ink = recto & (_smooth_noise(rng, (D, H, Wd), (7, 7, 7)) > 0.9)

    if bank is not None and rng.random() < bank_p:
        ct = _render_bank(rng, (D, H, Wd), Wf, papyrus, ink, bank)
    else:
        fiber = _smooth_noise(rng, (D, H, Wd), (size // 4,) * 3) * 0.5 \
            + rng.standard_normal((D, H, Wd)).astype(DTYPE) * 0.5
        ct = 0.15 + 0.55 * papyrus + 0.07 * fiber * papyrus + 0.08 * ink
        ct = gaussian_filter(ct.astype(DTYPE), 0.7)
        ct += rng.standard_normal((D, H, Wd)).astype(DTYPE) * 0.03

    return {
        "ct": ct.astype(DTYPE),
        "papyrus": papyrus.astype(np.uint8),
        "recto": recto.astype(np.uint8),
        "verso": verso.astype(np.uint8),
        "ink": ink.astype(np.uint8),
        "w": (Wf % 1.0).astype(DTYPE),
        "W": Wf,                                    # unwrapped GT (tracer oracle)
        "normal": normal,
        # w/normal are supervised DENSELY over the scroll interior (gaps included)
        # — the Eulerian-winding-field view. Papyrus-only supervision leaves the
        # wraps in a mid-scroll block as disconnected components and block-local
        # unwrap cannot link them (measured: wrap-id acc 0.15 vs ~1.0 dense).
        "wmask": (~core).astype(np.uint8),
        "pitch": np.float32(pitch),
    }


def to_batch(phantoms, device="cuda"):
    """Stack phantom dicts into training tensors: x (B,1,...), targets dict."""
    import torch

    def st(key, dt):
        return torch.from_numpy(np.stack([p[key] for p in phantoms])).to(
            device=device, dtype=dt)

    x = st("ct", torch.float32)[:, None]
    x = (x - x.mean(dim=(1, 2, 3, 4), keepdim=True)) / \
        x.std(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1e-6)
    t = {
        "sem": torch.stack([st("papyrus", torch.float32),
                            st("recto", torch.float32),
                            st("verso", torch.float32)], dim=1),
        "normal": st("normal", torch.float32),
        "w": st("w", torch.float32),
        "ink": st("ink", torch.float32)[:, None],
        "wmask": st("wmask", torch.float32)[:, None],
        "inkmask": torch.ones_like(x),              # phantom: ink labeled everywhere
        "semmask": torch.ones_like(x),
    }
    t["interior"] = t["wmask"].clone()              # dense on phantoms
    t["intmask"] = torch.ones_like(x)
    t["pitch_vox"] = torch.tensor([float(p.get("pitch", float("nan")))
                                   for p in phantoms], device=device)
    return x, t
