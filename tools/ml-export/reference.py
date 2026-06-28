#!/usr/bin/env python3
"""reference.py — authoritative PyTorch reference for the recto-surface ResEnc-UNet.

Assembles the network from the REAL upstream building blocks (`dynamic_network_architectures`
ConvDropoutNormReLU / StackedConvBlocks / StackedResidualBlocks / BasicBlockD) plus vesuvius's
concurrent scSE, with module/param names matching the checkpoint, then `load_state_dict(strict)`.
A clean strict load proves this IS the upstream architecture. Used to validate fenix's C++
reimplementation: run the same raw input through both and compare logits.

  reference.py gen   <D> <H> <W> <in.raw>                 # write a fixed seeded input
  reference.py run   <ckpt.pth> <in.raw> <D> <H> <W> <ref.raw>
  reference.py cmp   <ref.raw> <cpp.raw>                  # report max-abs-diff / correlation
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks, ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BasicBlockD

NORM = nn.InstanceNorm3d
NORM_KW = {"affine": True, "eps": 1e-5}
ACT = nn.LeakyReLU
ACT_KW = {"inplace": True, "negative_slope": 0.01}
FEATS = [32, 64, 128, 256, 320, 320, 320]
BLOCKS = [1, 3, 4, 6, 6, 6, 6]
STRIDES = [1, 2, 2, 2, 2, 2, 2]
K = 3


def _make_divisible(v, divisor=8):
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class SqueezeExcite(nn.Module):  # cSE
    def __init__(self, c):
        super().__init__()
        rd = _make_divisible(c / 16)
        self.fc1 = nn.Conv3d(c, rd, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(rd, c, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = x.mean((2, 3, 4), keepdim=True)
        return x * self.gate(self.fc2(self.act(self.fc1(s))))


class SpatialSE(nn.Module):  # sSE
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv3d(c, 1, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        return x * self.gate(self.conv(x))


class ChannelSpatialSE(nn.Module):  # scSE = cSE + sSE
    def __init__(self, c):
        super().__init__()
        self.cSE = SqueezeExcite(c)
        self.sSE = SpatialSE(c)

    def forward(self, x):
        return self.cSE(x) + self.sSE(x)


class Encoder(nn.Module):
    def __init__(self, se=True):
        super().__init__()
        self.stem = StackedConvBlocks(1, nn.Conv3d, 1, FEATS[0], K, 1, True, NORM, NORM_KW, None, None, ACT, ACT_KW)
        self.stages = nn.ModuleList()
        prev = FEATS[0]
        for s in range(len(FEATS)):
            blk = StackedResidualBlocks(BLOCKS[s], nn.Conv3d, prev, FEATS[s], K, STRIDES[s], True, NORM, NORM_KW,
                                        None, None, ACT, ACT_KW, block=BasicBlockD,
                                        squeeze_excitation=se, squeeze_excitation_reduction_ratio=1. / 16)
            if se:  # replace cSE-only SE with concurrent scSE (vesuvius squeeze_excitation_type="scse")
                for b in blk.blocks:
                    b.squeeze_excitation = ChannelSpatialSE(b.output_channels)
            self.stages.append(blk)
            prev = FEATS[s]

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for st in self.stages:
            x = st(x)
            skips.append(x)
        return skips


class Decoder(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        n = len(FEATS)
        self.transpconvs = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        for s in range(n - 1):
            below, skipc = FEATS[n - 1 - s], FEATS[n - 2 - s]
            self.transpconvs.append(nn.ConvTranspose3d(below, skipc, 2, 2, bias=True))
            self.stages.append(StackedConvBlocks(1, nn.Conv3d, 2 * skipc, skipc, K, 1, True, NORM, NORM_KW,
                                                 None, None, ACT, ACT_KW))
            self.seg_layers.append(nn.Conv3d(skipc, num_classes, 1, bias=True))

    def forward(self, skips):
        lres = skips[-1]
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            lres = x
        return self.seg_layers[-1](lres)


class DecoderBody(nn.Module):  # shared_decoder for ink: transpconvs + stages, no seg heads
    def __init__(self):
        super().__init__()
        n = len(FEATS)
        self.transpconvs = nn.ModuleList()
        self.stages = nn.ModuleList()
        for s in range(n - 1):
            below, skipc = FEATS[n - 1 - s], FEATS[n - 2 - s]
            self.transpconvs.append(nn.ConvTranspose3d(below, skipc, 2, 2, bias=True))
            self.stages.append(StackedConvBlocks(1, nn.Conv3d, 2 * skipc, skipc, K, 1, True, NORM, NORM_KW,
                                                 None, None, ACT, ACT_KW))

    def forward(self, skips):
        lres = skips[-1]
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            lres = x
        return lres


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = Encoder()
        self.task_decoders = nn.ModuleDict({"surface": Decoder()})

    def forward(self, x):
        return self.task_decoders["surface"](self.shared_encoder(x))


class NetInk(nn.Module):  # shared_encoder + shared_decoder + task_heads.ink (1x1 conv)
    def __init__(self):
        super().__init__()
        self.shared_encoder = Encoder(se=False)  # ink encoder has plain residual blocks
        self.shared_decoder = DecoderBody()
        self.task_heads = nn.ModuleDict({"ink": nn.Conv3d(FEATS[0], 1, 1, bias=True)})

    def forward(self, x):
        return self.task_heads["ink"](self.shared_decoder(self.shared_encoder(x)))


def build_and_load(ckpt, ink=False):
    net = (NetInk() if ink else Net()).eval()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["ema_model"] if ink else ck["model"]
    # `all_modules.*` are aliases sharing storage with conv/norm in upstream ConvDropoutNormReLU,
    # so the full state_dict loads strict against our identically-built net.
    net.load_state_dict(sd, strict=True)
    print(f"reference: state_dict loaded strict=True ({'ink' if ink else 'surface'} architecture confirmed)")
    return net


def main():
    cmd = sys.argv[1]
    if cmd == "gen":
        D, H, W, out = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        g = torch.Generator().manual_seed(27)
        x = torch.randn(D * H * W, generator=g, dtype=torch.float32).numpy()
        x.tofile(out)
        print(f"gen: {out} ({D}x{H}x{W})")
    elif cmd == "run":
        ckpt, inp, D, H, W, out = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
        ink = len(sys.argv) > 8 and sys.argv[8] == "ink"
        net = build_and_load(ckpt, ink=ink)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = net.to(dev)
        x = torch.from_numpy(np.fromfile(inp, dtype=np.float32)).reshape(1, 1, D, H, W).to(dev)
        with torch.no_grad():
            y = net(x).cpu().contiguous().numpy().astype(np.float32)
        y.tofile(out)
        print(f"run: {out} shape {y.shape}")
    elif cmd == "cmp":
        a = np.fromfile(sys.argv[2], dtype=np.float32)
        b = np.fromfile(sys.argv[3], dtype=np.float32)
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
        mad = float(np.max(np.abs(a - b)))
        rel = mad / (float(np.max(np.abs(a))) + 1e-9)
        corr = float(np.corrcoef(a, b)[0, 1])
        print(f"cmp: n={n} max_abs_diff={mad:.4e} rel={rel:.4e} corr={corr:.6f}")
        print("VERDICT:", "MATCH" if corr > 0.9999 and rel < 1e-2 else "MISMATCH")
    else:
        sys.exit("unknown cmd")


if __name__ == "__main__":
    main()
