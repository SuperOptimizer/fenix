# Training pipeline — student distillation on the segment corpus

Status: design + phase-1 implementation (2026-07-02). Owner: ml. Relates:
ml-accel-and-distillation.md (teacher recipe), training-data/README.md (corpus),
ADR 0009 (GPU portability), configs/gpu/ (measured per-card profiles).

## Goal
Train a NEW student surface model from (a) the 336-mesh GT corpus (`data/training/fxsurf`,
196 segments across 5 scrolls) and (b) the existing teacher (`surface_recto_3dunet` at
octahedral **tta=48** — the measured single-model ceiling), such that the student at tta≤8
matches or beats the teacher at tta=48, and trains/infers as fast as the hardware allows.

## Data flow (per training sample)
```
manifest ──▶ sampler: pick (mesh, valid cell, jitter) ──▶ patch origin
                │
CT .fxvol ─────┼─▶ u8 CT patch  ──┐
fxsurf mesh ───┼─▶ rasterize GT   ├─▶ augment (octa/rot_z/elastic/intensity/ct/compress)
teacher .fxvol ┴─▶ soft labels ───┘        │ (geometric ops transform surface coords FIRST,
                                           │  then rasterize — labels never interpolated)
                                           ▼
                              student fwd (bf16 autocast)
                                           ▼
              loss = α·KL(student ∥ teacher-soft, T=2) + β·(Dice+CE)(student, GT band)
```
- **GT labels are rasterized at train time, per patch** (decided 2026-07-02): sheet
  thickness / soft-band width are hyperparameters (and augmentable), storage stays
  meshes-only, and geometric augments transform coords before rasterizing so labels
  stay exact.
- **Teacher soft labels are precomputed offline** (the "bulk KD" stage): running the
  tta=48 teacher in the loop would cost 48 forwards per patch. One-time
  `predict-surface <vol> teacher.fxweights <out> ... 48` per training region → u8
  prob .fxvol at q=32 (the standard prediction artifact). ~x255 u8 soft labels are
  plenty for KD (teacher probs are heavily saturated).
- **Sampler** is occupancy-guided by the MESH (better than CT occupancy: the mesh IS
  the supervision): draw a mesh ∝ valid-cell count, draw a valid cell uniformly,
  jitter the patch center ±patch/4, reject if patch exits the volume. Deterministic:
  sample i = f(seed, i) so runs are reproducible and resumable.

## Stack split: Python learning loop, C++ data plane (decided 2026-07-02)
The QAT/precision ecosystem the plan leans on (**torchao**, **TransformerEngine**,
**TensorRT**) is Python-first; hand-rolling QAT in C++ libtorch would forfeit it. So:
- **Learning loop = Python** (`tools/train/`): torch latest, torchao QAT, bf16/fp8
  autocast, EMA, checkpointing. RunPod caveat: the **host driver is fixed** (580.x on
  the current box) — torch wheels bundle their own CUDA runtime, so upgrades are fine
  as long as the wheel's CUDA needs driver ≤ the host's (580 covers cu12.x/cu13-era
  wheels); `probe_precision.py` after every upgrade tells us what's real.
- **Data plane = fenix C++** (this is where our formats + speed live): a
  `fenix train-feed` subprocess runs sampler → CT fetch → GT rasterize → augment in
  producer threads and streams fixed-size (CT u8, GT u8, teacher u8) patch triples
  through a /dev/shm ring; Python maps it as numpy/torch tensors zero-copy. The
  torch-free substrate (`ml/rasterize.hpp`, `ml/sampler.hpp`) is shared with any
  future in-tree C++ loop.

## Precision strategy (probe-driven; upgrade torch/cuDNN freely — only the driver is fixed)
| format | training use | verdict |
|---|---|---|
| **bf16 autocast** | fwd/bwd compute, fp32 master weights + optimizer | **DEFAULT.** PROBED 2026-07-02 (torch 2.11.0+cu128, cuDNN 9.19, RTX 6000 Pro): conv3d bf16 = 2.0× fp32, fp16 = 2.3×. No grad scaler needed. |
| fp16 | inference (already used), training with grad scaler | inference yes; training: bf16 strictly nicer on this hardware. |
| **fp8 (e4m3/e5m2)** | conv/matmul where the stack supports it | torch 2.8 (the image default) has no fp8 conv3d path — but we are NOT pinned to 2.8: the plan is to **upgrade the box to the latest torch + cuDNN and probe fp8 conv3d empirically** PROBED 2026-07-02 at torch **2.11.0** + cuDNN **9.19**: fp8 conv3d still unsupported (`getCudnnDataTypeFromScalarType() not supported for Float8_e4m3fn`); fp8 `_scaled_mm` works (matmul only). Verdict stands: bf16 for the conv UNet; re-probe each release. Probe: `tools/ml-export/probe_precision.py`. |
| fp4 (nvfp4) | — | Blackwell tensor cores exist, but zero torch conv support. Inference-compiler territory (TensorRT) — **not a training-loop concern now**. |
| **int8 QAT (torchao)** | fake-quant in the LAST ~20% of training | **YES — the real "fast student" lever**: torchao's QAT flow (per-channel weights, per-tensor activations), accuracy protected by training-time quant noise, then export. |
| int4 QAT (torchao) | weights-only | measure after int8; weight-only int4 + bf16 activations plausible for the encoder. |
| TransformerEngine fp8 | Linear/attention blocks | LOW ROI while the student is a pure conv3d UNet (TE accelerates Linear/LN/attention, not conv3d); becomes relevant if/when attention blocks or the DINOv2 backbone enter the student. Keep installed, don't contort the arch for it. |
| **TensorRT (deploy)** | post-QAT engine build: int8/fp8/**fp4** inference | The deployment story for the trained student: build a TRT engine from the QAT'd model and (future ADR) load it behind the ML firewall as an alternative to libtorch inference — this is where Blackwell fp4 actually becomes usable. |

Training compute is therefore **bf16 end-to-end** (fp8 slots in for fwd if the
latest-torch probe clears it), with an int8-QAT phase before the final export. EMA of student weights (the teacher's own recipe) is kept in fp32.

## Loop mechanics
- Optimizer AdamW (f64 state per project policy — core/optimize.hpp for classical fits;
  the torch loop uses torch's AdamW with fp32 state), cosine LR, warmup.
- EMA student (decay 0.999) — eval/export uses EMA weights.
- Checkpoint/resume: full state (model, EMA, optimizer, sampler cursor, RNG) every N
  steps, atomic write-temp-rename, identity-stamped like the inference checkpoints.
- Eval cadence: every N steps score on a HELD-OUT segment set (whole segments held
  out, never random patches — patch-level splits leak through overlap) with the
  official composite (eval/): surface-dice + VOI + topo.
- Data loading: producer threads (sampler+rasterize+augment are CPU) feeding a
  bounded queue, GPU consumer — the proven inference producer/consumer shape.

## Phasing
1. **Torch-free substrate (this commit):** `ml/rasterize.hpp` (mesh → per-patch GT
   band via bilinear surface sampling + distance stamp), `ml/sampler.hpp` (corpus
   index + deterministic patch draws), tests on the Mac.
2. **Bulk KD artifacts:** teacher tta=48 predictions over each training segment's
   CT context (box job, resumable inference already does this).
3. **`fenix train-feed`** (C++): the shm-ring patch server over the substrate.
4. **Train loop (`tools/train/train.py`)**: bf16 KD loop, torchao QAT phase, EMA,
   checkpoint/resume; consumes the ring.
5. **Export + TensorRT engine**, then the eval-set firewall (calibration/validation/
   test) decides if the student replaces the teacher.

## Student architecture note
Start = the teacher's ResEnc-UNet config shrunk (fewer stages/filters — sweep later);
rotation robustness comes from octahedral train-time augmentation (measured: the
teacher's weakness), not from architectural equivariance. The 3D-RoPE DINOv2 backbone
stays out of scope until the CNN student baseline exists.
