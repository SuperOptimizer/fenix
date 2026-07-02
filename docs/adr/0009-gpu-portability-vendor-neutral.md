# ADR 0009 — GPU portability: vendor-neutral kernels, AMD/ROCm as a first-class target

**Status:** Accepted (2026-07-02). Sharpens the root `CLAUDE.md` §2.4 line "GPU **backend interfaces
now, implementation deferred** (single-node, multi-GPU-aware)" into a concrete portability stance: the
deferred GPU implementation, when written, targets **both NVIDIA and AMD** through a thin vendor-neutral
backend, not raw CUDA. No kernels exist yet, so this is cheap to adopt now and expensive to retrofit later
— which is exactly why it is being decided before the first kernel is written.

## Context

fenix has two GPU-relevant halves, and they have very different portability stories:

1. **The ML path (`src/ml/`, firewalled behind `FENIX_ML`)** consumes **libtorch**. Its GPU use is entirely
   through torch tensors + standard ops (conv3d, InstanceNorm, softmax, scSE). Torch already runs on AMD via
   **ROCm/HIP**: the ROCm libtorch build keeps the `.cuda()` / `cuda:0` source API (a HIP shim), so the net
   code (`nets/resenc_unet.hpp`) and the sliding-window inference (`infer.hpp`) do **not** change — only the
   build links a different libtorch + runtime libs. Portability here is a **build/provisioning** problem, not
   a code problem, and it's already latent.

2. **The classical pipeline** (codec DCT, the diffeomorphic winding fit, `geom/` primitives) is **CPU-first
   with GPU stubs** — the interfaces exist, the kernels do **not**. This is the fortunate part: there is
   almost no hand-written CUDA to port, because we haven't written it. But it also means every future GPU
   kernel is a portability decision made at authoring time.

The forcing function is scale + economics. The target workload is **inference and training over whole
scrolls — up to ~40 TB of raw CT** — with heavy TTA and large patch batches (up to ~512 patches). At that
scale the binding constraint is **memory capacity and bandwidth**, not just FLOPs:

- The 5090 (32 GB) caps inference at batch≈3 (measured; batch=4 regressed on VRAM pressure) and forces a 3D
  UNet into tiny-batch training with gradient accumulation + activation checkpointing.
- The **AMD MI300X (192 GB HBM3)** removes that ceiling: large batches (amortizing 3D-conv launch latency
  toward peak), whole-TTA-ensembles in one forward, and training batch sizes simply impossible on 32 GB.
  At ~$2/hr vs the 5090's ~$1/hr, the throughput-per-dollar can win **if** MIOpen's 3D convs are efficient —
  the one empirical unknown, cheap to settle with a 1-hour rental benchmark (conv3d forward + one training
  step across batch sizes vs the 5090 baseline).

So AMD is not a hypothetical: it is a plausibly-cost-effective primary target for the training half of the
project. A raw-CUDA classical pipeline would strand fenix on NVIDIA precisely where AMD's memory helps most.

**Note on 40 TB and out-of-core (vendor-independent):** 40 TB fits in no GPU. Whole-volume work is
fundamentally out-of-core — block + halo + stitch, byte-budgeted streaming — regardless of card. That
streaming harness is fenix's hard rule already (root `CLAUDE.md` §2.4) and is the **dominant** engineering
cost of the 40 TB goal, far larger than CUDA-vs-ROCm. The 192 GB makes each streamed block/batch bigger
(less halo re-read, less per-step overhead) but does not remove the architecture. This ADR is about the
kernel-portability layer *underneath* that harness; the harness itself is a separate, vendor-neutral effort.

## Decision

**1. ML path — ROCm is a supported build, no code fork.** Keep torch as the ML GPU abstraction. Add a ROCm
libtorch build variant (a `install-*.sh` / deps path analogous to the CUDA one) that links ROCm's libtorch +
HIP runtime + MIOpen. The net and inference code stay vendor-agnostic (torch tensor ops). Device selection
(`best_device()` in `torch_env.hpp`) already asks torch for the device — it needs no vendor branch. This is
provisioning + CI-matrix work, not a source split.

**2. Classical GPU kernels — write through a vendor-neutral backend from day one, never raw CUDA.** When we
implement the deferred GPU path (codec, fit hot loops, `geom/`), it goes behind a thin `fenix::gpu` backend
interface with **≥2 implementations sharing one kernel source**. The concrete mechanism is chosen when the
first kernel lands (a follow-up ADR), from:
   - **HIP-that-also-compiles-as-CUDA** — HIP source compiles on both AMD (hipcc) and NVIDIA (hipcc→nvcc);
     the leanest option, closest to the metal, one source tree.
   - **SYCL** (AdaptiveCpp/DPC++) — single-source C++, targets NVIDIA + AMD + CPU; more portable, heavier
     toolchain, and it's a **new toolchain dependency** → needs forrest's sign-off per §2.5.
   - **Kokkos** — performance-portability layer; powerful but a large dependency.
   The **default lean** is HIP-as-both (no new heavyweight dep, matches the clang-only stance since hipcc is
   LLVM-based), with SYCL reconsidered if a kernel needs its ergonomics. **Raw CUDA C is rejected** as a
   primary path: it strands the classical pipeline on NVIDIA and doubles maintenance (a later `hipify` is
   per-kernel, lossy, and keeps two divergent trees).

**3. The backend interface is the firewall.** Mirror the `FENIX_ML`/libtorch firewall (ADR 0008): GPU code
lives behind a small typed interface (`fenix::gpu::Backend` — device query, alloc, H2D/D2H, kernel launch),
with a **CPU reference implementation that is always built** (so the classical pipeline runs with no GPU, per
§2.4 CPU-first), and vendor backends compiled in behind a `FENIX_GPU` opt-in. No vendor headers leak past the
interface — the rest of fenix never sees `hip_runtime.h` / `cuda_runtime.h` / SYCL, exactly as it never sees
`<torch/torch.h>` today.

## Consequences

- **Cost now: ~zero.** No kernels exist, so "write them portable" is a policy, not a rewrite. The only
  immediate work is the ROCm libtorch *build* variant (days, mostly provisioning/version-matching — ROCm ↔
  libtorch ↔ MIOpen ↔ driver is finicky), and it's gated on the 1-hour MI300X benchmark clearing the
  throughput-per-dollar bar.
- **Cost of the alternative later: high.** Retrofitting portability onto a raw-CUDA classical pipeline is
  per-kernel `hipify` + a maintained second tree — the expensive path this ADR exists to avoid.
- **Risk retained: MIOpen 3D-conv efficiency.** Unknown until measured; it gates whether AMD is worth
  targeting at all. If MIOpen falls back to untuned 3D-conv kernels, the MI300X could be slower-per-dollar
  despite the memory. **Mitigation:** the benchmark-first rule — no ROCm build investment until a rented
  MI300X shows >~2× 5090 throughput-per-dollar on conv3d forward *and* a training step at a batch the 5090
  can't hold. Large batches (which the 192 GB enables) also directly mitigate small-batch 3D-conv
  inefficiency by amortizing launch overhead.
- **CI:** ROCm/AMD becomes a build-matrix target for the ML module when adopted; the classical GPU backend
  adds a CPU-reference build (always) + vendor builds (self-hosted runners) when kernels land. Numerics stay
  tolerance-only (§2.4) — cross-vendor bit-reproducibility is explicitly not required, which this multi-
  backend stance depends on.
- **Multi-GPU / single-node** (§2.4) is unchanged and orthogonal; the backend interface is where it will hook.

## Not covered / follow-ups

- The concrete portability mechanism (HIP-as-both vs SYCL vs Kokkos) is a **follow-up ADR at first-kernel
  time** — this ADR fixes only "vendor-neutral, not raw CUDA" + "ROCm libtorch is supported."
- The MI300X benchmark (conv3d forward batch-sweep + one training step vs the 5090) is the go/no-go gate for
  the ROCm build investment; results should be recorded (design note or an amendment here).
- The out-of-core streaming train/infer harness is the larger, vendor-independent effort and gets its own
  design treatment; it is *not* this ADR.
