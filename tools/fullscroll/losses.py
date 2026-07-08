"""Masked multi-task loss for FullScrollNet.

No voxel has all labels on real data — every head's loss is masked to where its
labels exist (per-sample, per-voxel masks come with the batch). On phantoms the
masks are dense except w/normal (papyrus-only by definition).

  sem     BCE-with-logits on 3 channels, semmask
  normal  1 - cos(pred, gt), wmask (papyrus)      -- orientation is consistent:
                                                      normals point to increasing W
  wind    MSE on (sin 2*pi*w, cos 2*pi*w), wmask  -- periodic-safe regression
  ink     BCE-with-logits, inkmask, pos_weight for the class imbalance
"""
import dataclasses

import torch
import torch.nn.functional as F

WEIGHTS = {"sem": 1.0, "normal": 0.5, "wind": 1.0, "ink": 1.0, "interior": 0.5}


@dataclasses.dataclass
class LossCfg:
    """Extension knobs. full_scroll_loss(out, t) with cfg=None keeps legacy
    behavior (BCE/cos/sincos/BCE + interior) — all new terms OFF."""
    shell_dice_w: float = 0.5      # soft-Dice on recto/verso (thin shells
    papyrus_dice_w: float = 0.25   # collapse under plain BCE — measured)
    phys_align_w: float = 0.0      # self-sup: grad(w) parallel to normal head
    phys_pitch_w: float = 0.0      # |grad w| * pitch_vox ~ 1 (needs pitch)
    ink_pos_weight: float = 20.0


def _masked_mean(v, m):
    s = m.sum()
    return (v * m).sum() / s.clamp(min=1.0), s


def soft_dice(prob, target, weight):
    """prob/target/weight all (B,1,D,H,W). Per-sample soft Dice, weight-masked."""
    assert prob.dim() == 5 and prob.shape == target.shape == weight.shape, \
        (prob.shape, target.shape, weight.shape)
    p, g, w = prob * weight, target * weight, weight
    inter = (p * g).sum(dim=(1, 2, 3, 4))
    denom = (p * p).sum(dim=(1, 2, 3, 4)) + (g * g).sum(dim=(1, 2, 3, 4))
    return (1.0 - (2 * inter + 1.0) / (denom + 1.0)).mean()


def _grad3(f):
    """Central differences of (B,1,D,H,W) -> (B,3,D,H,W), replicate edges."""
    fp = F.pad(f, (1, 1, 1, 1, 1, 1), mode="replicate")
    gz = (fp[:, :, 2:, 1:-1, 1:-1] - fp[:, :, :-2, 1:-1, 1:-1]) * 0.5
    gy = (fp[:, :, 1:-1, 2:, 1:-1] - fp[:, :, 1:-1, :-2, 1:-1]) * 0.5
    gx = (fp[:, :, 1:-1, 1:-1, 2:] - fp[:, :, 1:-1, 1:-1, :-2]) * 0.5
    return torch.cat([gz, gy, gx], dim=1)


def full_scroll_loss(out, t, cfg: "LossCfg|None" = None):
    losses = {}

    bce = F.binary_cross_entropy_with_logits(out["sem"], t["sem"], reduction="none")
    losses["sem"], _ = _masked_mean(bce.mean(1, keepdim=True), t["semmask"])

    n = F.normalize(out["normal"], dim=1, eps=1e-6)
    cos = (n * t["normal"]).sum(1, keepdim=True)
    losses["normal"], _ = _masked_mean(1.0 - cos, t["wmask"])

    ang = 2 * torch.pi * t["w"]
    sc = torch.stack([torch.sin(ang), torch.cos(ang)], dim=1)
    losses["wind"], _ = _masked_mean(
        (out["wind"] - sc).pow(2).mean(1, keepdim=True), t["wmask"])

    pw = torch.as_tensor(cfg.ink_pos_weight if cfg else 20.0,
                         device=out["ink"].device)
    bce_i = F.binary_cross_entropy_with_logits(out["ink"], t["ink"],
                                               pos_weight=pw, reduction="none")
    losses["ink"], _ = _masked_mean(bce_i, t["inkmask"])

    if "interior" in out and "interior" in t:
        bce_int = F.binary_cross_entropy_with_logits(
            out["interior"], t["interior"], reduction="none")
        losses["interior"], _ = _masked_mean(bce_int, t["intmask"])

    if cfg is not None:
        wm = t["wmask"]
        if cfg.shell_dice_w > 0:
            sd = sum(soft_dice(out["sem"][:, i:i + 1].sigmoid(),
                               t["sem"][:, i:i + 1], t["semmask"])
                     for i in (1, 2))
            losses["shell_dice"] = cfg.shell_dice_w * sd
        if cfg.papyrus_dice_w > 0:
            losses["pap_dice"] = cfg.papyrus_dice_w * soft_dice(
                out["sem"][:, 0:1].sigmoid(), t["sem"][:, 0:1], t["semmask"])
        if cfg.phys_align_w > 0 or cfg.phys_pitch_w > 0:
            # grad of w decoded from sin/cos heads: grad w = (c*grad s - s*grad c)
            # / (2*pi*(s^2+c^2)) — smooth, no mod-1 wrap
            s, c = out["wind"][:, 0:1].float(), out["wind"][:, 1:2].float()
            mag2 = (s * s + c * c).clamp(min=1e-4)
            gw = (c * _grad3(s) - s * _grad3(c)) / (2 * torch.pi * mag2)
            gwn = gw.norm(dim=1, keepdim=True).clamp(min=1e-6)
            conf = (out["interior"].detach().sigmoid()
                    if "interior" in out else wm)
            if cfg.phys_align_w > 0:
                n = F.normalize(out["normal"].float(), dim=1, eps=1e-6)
                align = 1.0 - ((gw / gwn) * n).sum(1, keepdim=True).abs()
                losses["phys_align"], _ = _masked_mean(align * cfg.phys_align_w,
                                                       conf)
            if cfg.phys_pitch_w > 0 and "pitch_vox" in t:
                pv = t["pitch_vox"]
                valid = torch.isfinite(pv).float().view(-1, 1, 1, 1, 1)
                pvs = torch.nan_to_num(pv, nan=1.0).view(-1, 1, 1, 1, 1)
                err = F.huber_loss(gwn * pvs, torch.ones_like(gwn),
                                   reduction="none", delta=0.5)
                losses["phys_pitch"], _ = _masked_mean(
                    err * cfg.phys_pitch_w, conf * valid)

    total = sum(WEIGHTS.get(k, 1.0) * v for k, v in losses.items())
    return total, {k: v.detach() for k, v in losses.items()}


@torch.no_grad()
def head_metrics(out, t):
    """Quick quality readout: papyrus Dice, circular-w MAE (vox of pitch-free
    units), normal mean angular error (deg), ink Dice."""
    m = {}
    for i, name in enumerate(("papyrus", "recto", "verso")):
        p = (out["sem"][:, i:i + 1].sigmoid() > 0.5).float()
        g = t["sem"][:, i:i + 1]
        m[f"dice_{name}"] = (2 * (p * g).sum() / (p.sum() + g.sum() + 1e-6)).item()
    wm = t["wmask"].squeeze(1) > 0
    from model import decode_w
    dw = (decode_w(out["wind"]) - t["w"])[wm]
    dw = torch.minimum(dw % 1.0, (-dw) % 1.0)       # circular distance
    m["w_mae"] = dw.mean().item()
    n = torch.nn.functional.normalize(out["normal"], dim=1, eps=1e-6)
    cos = (n * t["normal"]).sum(1)[wm].clamp(-1, 1)
    m["normal_deg"] = torch.rad2deg(torch.arccos(cos)).mean().item()
    p = (out["ink"].sigmoid() > 0.5).float()
    m["dice_ink"] = (2 * (p * t["ink"]).sum()
                     / (p.sum() + t["ink"].sum() + 1e-6)).item()
    return m
