"""Distilled surface-student builder + L1-slice warm start.

Plan: docs/design/student-distill-plan.md. Same architecture family as the
teacher (reference.Net) — the whole int8/TRT/fxweights stack keys on that
structure — just narrower/shallower. Rungs drop the teacher's 7th stage
(2^3 bottleneck: 33M params, no receptive-field value at 128^3).

Warm start: per-stage channel index sets chosen by L1 importance over the
teacher stage's convs, then every tensor sliced consistently (conv in/out,
norm gamma/beta, SE, transpconvs, cat'd decoder inputs). Student stage i maps
to teacher stage i; decoder stage s maps to teacher decoder stage s+1 (one
fewer level); a stage's first nb blocks are taken.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import reference as R

RUNGS = {
    "A": ([16, 32, 64, 128, 160, 160], [1, 2, 3, 3, 3, 3]),      # 10x smaller
    "C": ([32, 48, 96, 128, 192, 192], [1, 2, 2, 3, 3, 3]),      # 7x
    "B": ([32, 64, 128, 192, 256, 256], [1, 2, 3, 4, 4, 4]),     # 3x
}


def build_student(rung):
    feats, blocks = RUNGS[rung]
    saved = (R.FEATS, R.BLOCKS, R.STRIDES)
    R.FEATS, R.BLOCKS, R.STRIDES = feats, blocks, [1] + [2] * (len(feats) - 1)
    try:
        net = R.Net()
    finally:
        R.FEATS, R.BLOCKS, R.STRIDES = saved
    return net


def _stage_importance(stage):
    """L1 importance per output channel across a teacher encoder stage."""
    imp = None
    for b in stage.blocks:
        w = b.conv2.conv.weight if hasattr(b.conv2, "conv") else b.conv2.weight
        v = w.detach().abs().sum(dim=(1, 2, 3, 4))
        imp = v if imp is None else imp + v
    return imp


def _pick(imp, n):
    return torch.sort(torch.topk(imp, n).indices).values


def _slice_conv(dst, src, oi, ii):
    w = src.weight.detach()
    if oi is not None:
        w = w[oi]
    if ii is not None:
        w = w[:, ii]
    dst.weight.data.copy_(w[:, :dst.weight.shape[1]] if w.shape[1] != dst.weight.shape[1] else w)
    if dst.bias is not None and src.bias is not None:
        b = src.bias.detach()
        dst.bias.data.copy_(b[oi] if oi is not None else b[:dst.bias.shape[0]])


def _slice_norm(dst, src, oi):
    dst.weight.data.copy_(src.weight.detach()[oi])
    dst.bias.data.copy_(src.bias.detach()[oi])


def _slice_block(db, sb, oi, ii):
    _slice_conv(db.conv1.conv, sb.conv1.conv, oi, ii)
    _slice_norm(db.conv1.norm, sb.conv1.norm, oi)
    _slice_conv(db.conv2.conv, sb.conv2.conv, oi, oi)
    _slice_norm(db.conv2.norm, sb.conv2.norm, oi)
    if hasattr(db, "skip") and not isinstance(db.skip, torch.nn.Identity) \
            and hasattr(sb, "skip") and not isinstance(sb.skip, torch.nn.Identity):
        sk_d = db.skip[0] if isinstance(db.skip, torch.nn.Sequential) else db.skip
        sk_s = sb.skip[0] if isinstance(sb.skip, torch.nn.Sequential) else sb.skip
        if hasattr(sk_d, "weight"):
            _slice_conv(sk_d, sk_s, oi, ii)
    if hasattr(db, "squeeze_excitation") and hasattr(sb, "squeeze_excitation"):
        dse, sse = db.squeeze_excitation, sb.squeeze_excitation
        rd = dse.cSE.fc1.weight.shape[0] if hasattr(dse.cSE, "fc1") else None
        # cSE fc1 [r_d, c_d], fc2 [c_d, r_d]; pick reduction rows by fc2 L1
        f1d, f2d = _se_fcs(dse.cSE)
        f1s, f2s = _se_fcs(sse.cSE)
        rimp = f2s.weight.detach().abs().sum(dim=(0, 2, 3, 4))
        ri = _pick(rimp, f1d.weight.shape[0])
        _slice_conv(f1d, f1s, ri, oi)
        _slice_conv(f2d, f2s, oi, ri)
        _slice_conv(dse.sSE.conv, sse.sSE.conv, None, oi)


def _se_fcs(cse):
    convs = [m for m in cse.modules() if isinstance(m, torch.nn.Conv3d)]
    return convs[0], convs[1]


@torch.no_grad()
def l1_slice_init(student, teacher, rung):
    feats, blocks = RUNGS[rung]
    te, se_ = teacher.shared_encoder, student.shared_encoder
    idx = [_pick(_stage_importance(te.stages[s]), feats[s]) for s in range(len(feats))]
    # stem (1 -> feats[0])
    _slice_conv(se_.stem.convs[0].conv, te.stem.convs[0].conv, idx[0], None)
    _slice_norm(se_.stem.convs[0].norm, te.stem.convs[0].norm, idx[0])
    for s in range(len(feats)):
        prev = idx[s - 1] if s > 0 else idx[0]
        for bi in range(blocks[s]):
            _slice_block(se_.stages[s].blocks[bi], te.stages[s].blocks[bi],
                         idx[s], prev if bi == 0 else idx[s])
    td, sd = teacher.task_decoders["surface"], student.task_decoders["surface"]
    ns = len(feats)
    for s in range(ns - 1):
        ts = s + 1                       # teacher decoder stage (one deeper level)
        below_i, skip_i = idx[ns - 1 - s], idx[ns - 2 - s]
        # ConvTranspose3d weight is [in, out, k,k,k]
        wt = td.transpconvs[ts].weight.detach()[below_i][:, skip_i]
        sd.transpconvs[s].weight.data.copy_(wt)
        sd.transpconvs[s].bias.data.copy_(td.transpconvs[ts].bias.detach()[skip_i])
        cat_i = torch.cat([skip_i, skip_i + td.transpconvs[ts].weight.shape[1] * 0
                           + teacher_skip_width(td, ts)])
        _slice_conv(sd.stages[s].convs[0].conv, td.stages[ts].convs[0].conv,
                    skip_i, cat_i)
        _slice_norm(sd.stages[s].convs[0].norm, td.stages[ts].convs[0].norm, skip_i)
        _slice_conv(sd.seg_layers[s], td.seg_layers[ts], None, skip_i)
    return student


def teacher_skip_width(td, ts):
    # teacher decoder stage ts consumes cat(transp_out, skip) each of width W_t:
    # second half of the cat input indexes offset by the teacher's skip width
    return td.transpconvs[ts].weight.shape[1]


if __name__ == "__main__":
    from reference import build_and_load
    t = build_and_load("/home/forrest/fenix/models/surface_recto_3dunet/checkpoint_inference_ready.pth")
    for rung in ("A", "C", "B"):
        s = build_student(rung)
        n = sum(p.numel() for p in s.parameters())
        l1_slice_init(s, t, rung)
        x = torch.randn(1, 1, 64, 64, 64)
        y = s(x)
        print(f"rung {rung}: {n/1e6:.1f}M params ({142.2/(n/1e6):.1f}x), fwd {tuple(y.shape)} OK")
