#!/usr/bin/env python3
"""Unit tests for the Python training tools (2026-07-13 audit P2): the byte-layout and
sign-flip bug class that wastes a GPU-day before eval catches it.

  - feed_reader.FeedRing: synthetic ring file with known slot contents -> exact
    round-trip of ct/gt/meta, FREE-after-read, stripe partitioning
  - EMA update direction: pe.lerp_(pn, 1-ema) must move the EMA TOWARD the live net
    by (1-ema), never away (the sign/argument-order flip is silent)
  - eval_students.ece/auprc_maxdice sanity on analytic inputs

Run: python3 test_tools.py   (or pytest test_tools.py)
"""
import os, struct, sys, tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feed_reader import FeedRing, READY

HDR = struct.Struct("<8sIIQQII")
SLOT_HDR = 64


def make_ring(path, nslots=4, patch=8, channels=2):
    tensor = patch ** 3
    slot_bytes = SLOT_HDR + channels * tensor
    with open(path, "wb") as f:
        hdr = HDR.pack(b"FXRING1\x00", 1, nslots, patch, slot_bytes, channels, 0)
        f.write(hdr + b"\x00" * (4096 - len(hdr)))
        for s in range(nslots):
            f.write(struct.pack("<IIQqqq", READY, 100 + s, 7000 + s, 10 * s, 20 * s, 30 * s))
            f.write(b"\x00" * (SLOT_HDR - 40))
            ct = np.full(tensor, s + 1, np.uint8)
            gt = np.full(tensor, 255 - s, np.uint8)
            f.write(ct.tobytes() + gt.tobytes())
    return slot_bytes


def test_feed_ring_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ring.bin")
        make_ring(p)
        r = FeedRing(p)
        assert r.ready_count() == 4
        b = r.next_batch(4, timeout_s=2.0)
        assert b["ct"].shape == (4, 8, 8, 8) and b["gt"].shape == (4, 8, 8, 8)
        for i in range(4):
            assert (b["ct"][i] == i + 1).all(), "ct bytes misaligned"
            assert (b["gt"][i] == 255 - i).all(), "gt bytes misaligned"
            mesh, draw, oz, oy, ox = b["meta"][i]
            assert (mesh, draw, oz, oy, ox) == (100 + i, 7000 + i, 10 * i, 20 * i, 30 * i), \
                "slot header unpack misaligned"
        assert r.ready_count() == 0, "slots not freed after read"
        r.close()


def test_feed_ring_stripe_partition():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ring.bin")
        make_ring(p)
        r0 = FeedRing(p, stripe_rank=0, stripe_world=2)
        b0 = r0.next_batch(2, timeout_s=2.0)
        assert sorted(m[0] for m in b0["meta"]) == [100, 102], "rank0 stripe wrong slots"
        r1 = FeedRing(p, stripe_rank=1, stripe_world=2)
        b1 = r1.next_batch(2, timeout_s=2.0)
        assert sorted(m[0] for m in b1["meta"]) == [101, 103], "rank1 stripe wrong slots"
        r0.close(); r1.close()


def test_ema_update_direction():
    import torch
    ema_decay = 0.999
    pe = torch.zeros(4)          # EMA state
    pn = torch.ones(4)           # live net
    pe.lerp_(pn, 1 - ema_decay)  # the exact call from train.py:477
    assert torch.allclose(pe, torch.full((4,), 1 - ema_decay)), \
        "EMA must move toward the net by (1-decay)"
    # 100 repeats converge monotonically toward the net, never overshoot/diverge
    pe = torch.zeros(4)
    prev = 0.0
    for _ in range(100):
        pe.lerp_(pn, 1 - ema_decay)
        assert prev <= pe[0].item() <= 1.0
        prev = pe[0].item()


def test_eval_metrics_analytic():
    from eval_students import ece, auprc_maxdice
    # perfectly calibrated + perfectly ranked
    prob = np.array([0.9] * 90 + [0.1] * 10 + [0.1] * 90 + [0.9] * 10, np.float64)
    sheet = np.array([True] * 100 + [False] * 100)
    lab = np.ones(200, bool)
    assert ece(prob, sheet, lab) < 0.01, "well-calibrated input must score ~0 ECE"
    prob2 = np.linspace(0, 1, 200)
    sheet2 = prob2 > 0.5
    ap, md, _ = auprc_maxdice(prob2, sheet2, lab)
    assert ap > 0.99 and md > 0.99, "perfect ranking must give ~1 AUPRC/max-dice"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} passed")
