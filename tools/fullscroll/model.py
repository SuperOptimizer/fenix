"""FullScrollNet — one net, per-voxel: semantics + local frame + winding coord + ink.

docs/design/full-scroll-model.md. Shared vesuvius ResEnc-UNet encoder + decoder body
(the exact upstream building blocks from tools/ml-export/reference.py, so the
surface_recto_3dunet checkpoint warm-starts the backbone strict-by-key), plus four
1x1x1 heads:

  sem    3ch  logits: papyrus, recto-surface, verso-surface     (sigmoid each)
  normal 3ch  unit sheet normal, direction of increasing W       (normalized at use)
  wind   2ch  (sin 2*pi*w, cos 2*pi*w) of the fractional winding coordinate
              -- periodic regression; decode w = atan2(s, c)/(2*pi) mod 1
  ink    1ch  logit                                              (papyrus-masked)

Heads are noise next to the backbone: the whole fp8/int8 QAT + separable-student
lane applies unchanged (swap_convs_fp8 sees the same conv population).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ml-export"))

import torch
import torch.nn as nn

from reference import FEATS, DecoderBody, Encoder

HEADS = {"sem": 3, "normal": 3, "wind": 2, "ink": 1, "interior": 1}
# interior = scroll-interior probability (papyrus + air gaps, the Eulerian
# domain of w). It is the solver's confidence channel — the conf/ producer.


class FullScrollNet(nn.Module):
    def __init__(self, se=True):
        super().__init__()
        self.shared_encoder = Encoder(se=se)
        self.shared_decoder = DecoderBody()
        self.task_heads = nn.ModuleDict(
            {k: nn.Conv3d(FEATS[0], c, 1, bias=True) for k, c in HEADS.items()})

    def forward(self, x):
        f = self.shared_decoder(self.shared_encoder(x))
        return {k: h(f) for k, h in self.task_heads.items()}


def decode_w(out_wind):
    """(B,2,D,H,W) sin/cos logits -> w in [0,1)."""
    w = torch.atan2(out_wind[:, 0], out_wind[:, 1]) / (2 * torch.pi)
    return w % 1.0


def warm_start(net, surface_ckpt):
    """Load encoder + decoder-body weights from the surface checkpoint.

    Surface Net keys: shared_encoder.* and task_decoders.surface.{transpconvs,stages,
    seg_layers}.*; DecoderBody shares the transpconvs/stages naming, seg heads drop.
    Returns (n_loaded, n_total) params of this net covered by the checkpoint.
    """
    ck = torch.load(surface_ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    remap = {}
    for k, v in sd.items():
        if k.startswith("shared_encoder."):
            remap[k] = v
        elif k.startswith("task_decoders.surface."):
            sub = k[len("task_decoders.surface."):]
            if not sub.startswith("seg_layers."):
                remap["shared_decoder." + sub] = v
    missing, unexpected = net.load_state_dict(remap, strict=False)
    assert not unexpected, unexpected[:4]
    own = dict(net.named_parameters()) | dict(net.named_buffers())
    loaded = sum(v.numel() for k, v in remap.items() if k in own)
    total = sum(v.numel() for v in own.values())
    return loaded, total
