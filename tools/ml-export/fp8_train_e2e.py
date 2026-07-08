"""END-TO-END fp8 training of the full surface ResEnc-UNet on real CT (sm120).

The complete-training demo: every conv3d in the net (252 of them — 3x3x3 s1/s2 and 1x1x1,
encoder+decoder+SE+seg) runs the fp8 fwd + fp8 bwd path (Fp8Conv3d autograd Function;
dgrad/wgrad Triton kernels, f32 master weights). Norm/act/SE-glue/transpconv stay torch
autograd. Teacher = the same net in fp16 autocast (frozen); student trains against the
teacher's soft surface probabilities on random 96^3 crops of a real Scroll-1 CT volume —
i.e. self-distillation, which isolates the MECHANICS (does fp8 training optimize the real
net on real data?) from label quality.

Reports per-step loss (must fall), step wall-clock for fp8 vs an identical fp16-autocast
student, and the whole-net grad-corr snapshot at step 0.

Usage: python fp8_train_e2e.py [--steps 40] [--patch 96] [--lr 1e-4]
Expects /tmp/ct/patch.npy (256^3 u8; see fp8_forward.py / fenix ingest-zarr).
"""
import argparse
import copy
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from fp8_conv3d_op import dump_tuned, load_tuned
from fp8_train import set_graph_mode, swap_convs_fp8


def crops(vol, patch, n, seed=0, batch=1):
    g = np.random.default_rng(seed)
    D, H, W = vol.shape
    for _ in range(n):
        bs = []
        for _ in range(batch):
            z, y, x = (int(g.integers(0, D - patch)), int(g.integers(0, H - patch)),
                       int(g.integers(0, W - patch)))
            c = torch.from_numpy(vol[z:z + patch, y:y + patch, x:x + patch]).float()
            c = (c - c.mean()) / c.std().clamp(min=1e-6)
            bs.append(c[None, None])
        yield torch.cat(bs).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--patch", type=int, default=128)  # must be divisible by 64 (7 stages)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ct", default="/tmp/ct/patch.npy")
    ap.add_argument("--bsparse", type=float, default=0.0,
                    help="BLOCK-sparsity ratio (e.g. 0.5): prune (tap,cin)-groups; OUR "
                         "kernels skip pruned chunks (fwd + wgrad) — measured speedup, "
                         "accuracy is the experiment")
    ap.add_argument("--sparse24", action="store_true",
                    help="sparse-QAT: fixed 2:4 mask on all fp8/int8 conv weights "
                         "(ASP recipe) — trains toward TRT sparse tensor-core deploy")
    ap.add_argument("--int8qat", action="store_true",
                    help="int8-QAT: conv forwards run the deployment int8 kernel "
                         "(deploy-exact), backward stays fp8")
    ap.add_argument("--no-tailfuse", dest="tailfuse", action="store_false",
                    help="disable the fused BasicBlockD tail (scSE+residual+lrelu)")
    ap.add_argument("--no-normfuse", dest="normfuse", action="store_false",
                    help="disable the layout-native fp8 norm+act")
    ap.add_argument("--cl", action="store_true",
                    help="feed the fp8 student channels_last_3d input — our kernels' "
                         "native layout, so every NCDHW<->[M,C] permute-copy (28% of "
                         "the step) becomes a free view as long as the chain holds")
    ap.add_argument("--fused", action="store_true",
                    help="P2.2 fused CDNR blocks (fp8-resident) instead of per-conv swap")
    ap.add_argument("--batch", type=int, default=1,
                    help="crops per step. P3: fp8's 1-byte saved activations fit ~2x "
                         "the batch of fp16 at 16GB — compare samples/sec, not ms/step")
    ap.add_argument("--bwd-fp8", action="store_true",
                    help="int8 fwd + fp8 bwd (the sm120 recipe: fp16-parity loss)")
    ap.add_argument("--bwd-fp16", action="store_true",
                    help="int8 fwd + fp16 bwd (the AMPERE recipe: no bwd "
                         "quantization at all; sm86 has no fp8)")
    ap.add_argument("--recal-every", type=int, default=250,
                    help="re-freeze int8 activation calibration every K steps "
                         "(frozen scales go STALE as weights drift — measured "
                         "divergence at ~1500 steps without this)")
    ap.add_argument("--teacher-gpu", type=int, default=-1,
                    help="offload the KD teacher to this GPU with a 1-step "
                         "pipeline (targets computed during the student step)")
    ap.add_argument("--peak-mem", action="store_true",
                    help="report per-lane peak CUDA memory after the run")
    ap.add_argument("--perturb", type=float, default=0.03,
                    help="relative weight noise applied identically to BOTH students. "
                         "0 = exact teacher copy — a degenerate fixed point where true "
                         "grad is 0 and fp8's grad noise random-walks the weights "
                         "(loss drifts up while the fp16 twin sits at exactly 0); "
                         "perturbed students measure real signal-driven convergence.")
    ap.add_argument("--graph", action="store_true",
                    help="CUDA-graph the fp8 train step (P2.1): whole fwd+bwd+Adam "
                         "captured once, replayed per step — needs the sync-free "
                         "tensor-scale path (P1.3)")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    from reference import build_and_load
    ckpt = ("/home/forrest/fenix/models/surface_recto_3dunet/"
            "checkpoint_inference_ready.pth")
    teacher = build_and_load(ckpt).to(dev).eval()
    student8 = copy.deepcopy(teacher)
    if args.perturb:
        g = torch.Generator(device=dev).manual_seed(7)
        with torch.no_grad():
            for p in student8.parameters():
                if p.dim() >= 2:
                    p.add_(torch.randn(p.shape, generator=g, device=dev)
                           * (args.perturb * p.std().clamp(min=1e-8)))
    student16 = copy.deepcopy(student8)     # identical perturbed init for both lanes
    if args.fused:
        from fp8_train_fused import swap_cdnr_fp8
        nf, ns, nk = swap_cdnr_fp8(student8)
        print(f"student: {nf} fused CDNR + {ns} convs fp8 (kept {nk}); "
              f"perturb={args.perturb}")
    else:
        ns, nk = swap_convs_fp8(student8)
        nn_ = nt = 0
        if args.normfuse:
            from fp8_train import swap_norms_fp8
            nn_ = swap_norms_fp8(student8)
        if args.tailfuse:
            from fp8_train import swap_tails_fp8
            nt = swap_tails_fp8(student8)
        if args.int8qat:
            from fp8_train import set_int8_qat
            print(f"int8-QAT forwards on {set_int8_qat(student8)} convs")
            if args.bwd_fp16:
                from fp8_train import set_int8_bwd_fp8
                print(f"fp16 BACKWARD on {set_int8_bwd_fp8(student8, 'fp16')} convs "
                      "(Ampere recipe: zero bwd quantization)")
            elif args.bwd_fp8:
                from fp8_train import set_int8_bwd_fp8
                print(f"fp8 BACKWARD on {set_int8_bwd_fp8(student8)} convs "
                      "(sm120 recipe)")
        if args.sparse24:
            from fp8_train import set_sparse24
            nl, sp = set_sparse24(student8)
            print(f"2:4 sparse-QAT on {nl} convs ({sp:.1%} weights pruned)")
        if args.bsparse > 0:
            from fp8_train import set_blocksparse
            nl, sp = set_blocksparse(student8, ratio=args.bsparse)
            print(f"block-sparse on {nl} convs ({sp:.1%} weights pruned, "
                  f"kernels SKIP pruned chunks)")
        print(f"student: {ns} convs -> fp8 (kept {nk}), {nn_} norm+act fused, "
              f"{nt} tails fused; perturb={args.perturb}")
    student8.train()
    student16.train()
    # AFTER the deepcopies (a pre-copy convert leaks CL weights into the fp16 twin and
    # slows ITS cuDNN path 266->329ms — found the hard way): teacher-only channels_last,
    # measured ~199 -> ~113ms per teach() — pure wall-clock, targets unchanged
    teacher = teacher.to(memory_format=torch.channels_last_3d)

    tuned = os.path.expanduser("~/.cache/fenix-fp8-tuned-train.json")
    if os.path.exists(tuned):
        print(f"pinned autotune configs: {load_tuned(tuned)} (skip first-step sweep)")

    vol = np.load(args.ct)
    if args.int8qat and args.recal_every >= 0:
        # STATIC calibration for TRAINING too: dynamic affine costs one aminmax
        # host sync per conv per step (~85 ms across 76 convs at 128^3); frozen
        # (scale, zp) is sync-free AND deploy-exact (Step-0 finding, 2026-07-07).
        # --recal-every -1 = dynamic-affine control (never calibrate; slow).
        from fp8_train import int8_calibrate
        int8_calibrate(student8, list(crops(vol, args.patch, 4, seed=99)))
        print("int8 static calibration frozen (sync-free training fwd)")
    # capturable: Adam keeps step counters on-device so opt.step() can live in the graph
    opt8 = torch.optim.Adam(student8.parameters(), lr=args.lr, capturable=args.graph)
    opt16 = torch.optim.Adam(student16.parameters(), lr=args.lr)

    tdev = f"cuda:{args.teacher_gpu}" if args.teacher_gpu >= 0 else "cuda:0"
    if args.teacher_gpu > 0:
        # Tier-1 teacher offload: teacher lives on GPU1 and computes the NEXT
        # iter's target while the student steps on GPU0 (1-step pipeline). The
        # capped GPU1 forward must hide under the ~207 ms student step.
        teacher = teacher.to(tdev)
        print(f"teacher offloaded to {tdev} (1-step pipelined targets)")

    tgraph = {}

    def teach(x):
        # NOTE (offload): returns the target ON THE TEACHER DEVICE. Copying back
        # at launch time enqueues a dep-on-GPU1 copy ahead of the student kernels
        # on GPU0's default stream — serializing the whole step behind the
        # teacher (measured: wall 605 vs 587). Consumer copies after sync(1).
        # The teacher is CUDA-GRAPHED on the offload device: eager launch of ~150
        # modules costs ~50+ ms of HOST time that cannot overlap (measured wall
        # 557 with eager launches vs 113 ms of teacher GPU time).
        if args.teacher_gpu > 0:
            if "g" not in tgraph:
                with torch.cuda.device(args.teacher_gpu):
                    xs = x.to(tdev).to(memory_format=torch.channels_last_3d)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        for _ in range(2):
                            teacher(xs)          # warmup
                    torch.cuda.synchronize(args.teacher_gpu)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                            ys = teacher(xs).float().softmax(1)
                    tgraph.update(g=g, xs=xs, ys=ys)
            with torch.cuda.device(args.teacher_gpu):
                # copy on GPU1's stream so the replay (same stream) orders after it
                tgraph["xs"].copy_(x.to(memory_format=torch.channels_last_3d),
                                   non_blocking=True)
                tgraph["g"].replay()
            return tgraph["ys"]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return teacher(x.to(memory_format=torch.channels_last_3d)).float().softmax(1)

    # per-VOXEL mean KL: batchmean at batch=1 is the SUM over ~2M voxels -> grads 1e6x
    # normal scale -> both lanes exploded to NaN (found the hard way). The fp16 twin
    # additionally needs a GradScaler; the fp8 lane does NOT — its dynamic per-tensor dy
    # quantization IS a built-in loss scale (dy is renormalized to e4m3 range each layer).
    def kl(logits, t):
        nvox = logits.shape[2] * logits.shape[3] * logits.shape[4]
        return F.kl_div(logits.float().log_softmax(1), t, reduction="sum") / nvox

    scaler = torch.amp.GradScaler("cuda")

    def step(student, opt, x, t, fp16=False):
        opt.zero_grad(set_to_none=True)
        if fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = student(x)
            loss = kl(logits, t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            from fp8_train import join_wgrad_stream
            loss = kl(student(x), t)
            loss.backward()
            join_wgrad_stream()     # side-stream wgrads must land before Adam reads
            opt.step()
        return loss.item()

    graph = graph_loss = x_s = t_s = None
    if args.graph:
        set_graph_mode(student8, True)

    def graph_capture(x, t):
        # whole-iteration capture: fwd + bwd + Adam step + grad-zero, replayed per step.
        # Requires the sync-free tensor-scale path (P1.3) and in-graph weight packing.
        xs, ts = x.clone(), t.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                 # canonical warmup on a side stream
            for _ in range(2):
                opt8.zero_grad(set_to_none=True)
                kl(student8(xs), ts).backward()
                opt8.step()
        torch.cuda.current_stream().wait_stream(side)
        opt8.zero_grad(set_to_none=False)             # grads must exist as static buffers
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            loss_s = kl(student8(xs), ts)
            loss_s.backward()
            opt8.step()
            opt8.zero_grad(set_to_none=False)         # zeroed in-graph for the next replay
        return g, loss_s, xs, ts

    print(f"\n== training {args.steps} steps @ {args.patch}^3 batch={args.batch} "
          f"(self-distillation KL{', CUDA-graph fp8 step' if args.graph else ''}) ==")
    t8s, t16s, walls = [], [], []
    mem8 = mem16 = 0
    gen = crops(vol, args.patch, args.steps, batch=args.batch)
    x = next(gen)
    x_next = next(gen, None)          # lazy 1-ahead window (a full list OOMs)
    pending = teach(x) if args.teacher_gpu > 0 else None
    for i in range(args.steps):
        w0 = time.perf_counter()
        if args.teacher_gpu > 0:
            torch.cuda.synchronize(args.teacher_gpu)   # teacher fwd for i is done
            t = pending.to("cuda:0", non_blocking=True)  # cheap now, no false dep
            if x_next is not None:
                pending = teach(x_next)                # overlaps the student step
        else:
            t = teach(x)
        if args.graph and i == 3 and graph is None:   # capture after autotune/warmup steps
            graph, graph_loss, x_s, t_s = graph_capture(x, t)
        if i == 3 and not args.graph:
            from fp8_train import set_wgrad_stream
            set_wgrad_stream(True)    # only after autotune is warm (tuner-pollution)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if args.peak_mem and i == 3:
            torch.cuda.reset_peak_memory_stats()
        if graph is not None:
            x_s.copy_(x)
            t_s.copy_(t)
            graph.replay()
            l8 = graph_loss.item()
        else:
            x8in = (x.to(memory_format=torch.channels_last_3d) if args.cl else x)
            l8 = step(student8, opt8, x8in, t)
        torch.cuda.synchronize()
        t8 = time.perf_counter() - t0
        if args.peak_mem and i == 3:
            mem8 = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        l16 = step(student16, opt16, x, t, fp16=True)
        torch.cuda.synchronize()
        t16 = time.perf_counter() - t0
        if args.peak_mem and i == 3:
            mem16 = torch.cuda.max_memory_allocated()
        if i >= 3:  # skip autotune/warmup steps in timing stats
            t8s.append(t8)
            t16s.append(t16)
            walls.append(time.perf_counter() - w0)
        if i % max(1, args.steps // 8) == 0 or i == args.steps - 1:
            print(f"  step {i:>3}: fp8 loss {l8:.5f} ({t8*1e3:6.0f}ms) | "
                  f"fp16 loss {l16:.5f} ({t16*1e3:6.0f}ms)")
        if args.int8qat and args.recal_every > 0 and (i + 1) % args.recal_every == 0:
            from fp8_train import int8_calibrate
            int8_calibrate(student8, [x])   # refreeze on the current crop
        x = x_next
        x_next = next(gen, None) if x_next is not None else None
        if x is None:
            break

    m8, m16 = statistics.median(t8s), statistics.median(t16s)
    print(f"\nstep time (median, post-warmup): fp8 {m8*1e3:.0f}ms | fp16-autocast "
          f"{m16*1e3:.0f}ms | speedup {m16/m8:.2f}x | "
          f"{args.batch/m8:.2f} vs {args.batch/m16:.2f} samples/s | "
          f"wall {statistics.median(walls)*1e3:.0f}ms/iter (teach+both steps)")
    if args.peak_mem:
        # same persistent baseline (all 3 models + opt states) in both peaks; the
        # difference is the per-lane step transient (activations + grads)
        print(f"peak mem during step: fp8 {mem8/2**30:.2f} GiB | fp16 {mem16/2**30:.2f} GiB")
    print(f"pinned {dump_tuned(tuned)} tuned configs -> {tuned}")


if __name__ == "__main__":
    main()
