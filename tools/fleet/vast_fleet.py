#!/usr/bin/env python3
"""vast_fleet.py — interruptible-capacity fleet controller for the teacher sweep on Vast.ai.

Strategy (forrest, 2026-07-08): bulk piecewise inference runs on the cheapest interruptible
real-GPU capacity available, heterogeneous by design. Each instance is a stateless-ish
shard worker (teacher_sweep.sh: deterministic line-index sharding, .done markers, per-tile
resumable predicts) — an interruption costs only the in-flight tile. Vast interruptible
semantics: when outbid the instance PAUSES (disk persists, auto-resumes when the bid clears
the market again); work is lost only on destroy/host-failure, and even then a respawned
shard re-runs only the items missing .done markers (if outputs were synced) or redoes the
shard (if not — size shards accordingly).

  vast_fleet.py rank [n]               top-n interruptible offers by OUR est. ¢/1024³-block
  vast_fleet.py up <n> <sweep-url>     rent the n best offers, one shard each (0..n-1)
  vast_fleet.py status                 fleet table: state, shard, $/hr, progress if reachable
  vast_fleet.py respawn <sweep-url>    re-rent dead/destroyed shards on current best offers
  vast_fleet.py destroy [id|all]       tear down

Perf model: measured anchor = RTX 3090 + TRT10 .plan at 166 s per occupied 1024³ tta=8
(4x3090 box, 2026-07-06). Other archs scaled by Vast's dlperf relative to the 3090's
(dlperf tracks our conv3d fp16 workload well enough for RANKING; the first tile on a new
arch is the real benchmark — engines are arch-locked and built on-box by pod_bootstrap).
Bids are placed at min_bid * (1 + BID_MARGIN); a losing bid = a pause, not a loss.

Requires: pip install vastai; VAST_API_KEY env or ~/.config/vastai/vast_api_key.
"""
import json
import subprocess
import sys
import time

ANCHOR_GPU = "RTX 3090"
ANCHOR_SEC_PER_BLOCK = 166.0   # s per 1024^3 tta=8, TRT10 .plan, measured
ANCHOR_DLPERF = 45.0           # vast dlperf of the anchor 3090s (their scale)
BID_MARGIN = 0.25              # bid 25% over current min → survive small market moves
MIN_VRAM_GB = 24               # b1 TRT engine + TTA workspace
MIN_DOWN_MBPS = 500            # S3 band-ingest must not starve the GPU
MIN_DISK_GB = 100
MIN_RELIABILITY = 0.95
QUERY = (f"rentable=true num_gpus=1 gpu_ram>={MIN_VRAM_GB} "
         f"reliability>{MIN_RELIABILITY} inet_down>={MIN_DOWN_MBPS} "
         f"disk_space>={MIN_DISK_GB} cuda_max_good>=12.0")
IMAGE = "ghcr.io/superoptimizer/fenix-vast:latest"  # prebaked worker (Dockerfile.vast)
LABEL = "fenix-teacher"

# onstart re-fires on every (re)start — including resume-after-outbid — and vast_onstart.sh
# is idempotent (arch-locked TRT engine built once, sweep items skip .done markers).
BOOTSTRAP = ("nohup sh /opt/fenix/tools/fleet/vast_onstart.sh >> /workspace/onstart.log 2>&1 &")
ENV_FMT = "-e SWEEP_URL={sweep_url} -e SHARD={shard} -e NSHARDS={nshards}"


def vast(*args, raw=True):
    cmd = ["vastai", *args] + (["--raw"] if raw else [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"vastai {' '.join(args)} failed: {out.stderr.strip()}")
    return json.loads(out.stdout) if raw else out.stdout


def est_cents_per_block(offer):
    # dlperf-relative scaling off the measured 3090 anchor
    sec = ANCHOR_SEC_PER_BLOCK * ANCHOR_DLPERF / max(offer["dlperf"], 1.0)
    return sec / 3600.0 * offer["min_bid"] * (1 + BID_MARGIN) * 100.0, sec


def ranked_offers():
    offers = vast("search", "offers", "--type=bid", QUERY, "-o", "dlperf_per_dphtotal-")
    for o in offers:
        o["cents_per_block"], o["est_sec_per_block"] = est_cents_per_block(o)
    return sorted(offers, key=lambda o: o["cents_per_block"])


def cmd_rank(n=15):
    print(f"{'gpu':<22}{'bid$/hr':>9}{'est s/blk':>10}{'¢/blk':>7}{'vram':>6}{'down':>7}{'rel':>7}  offer_id")
    for o in ranked_offers()[:n]:
        bid = o["min_bid"] * (1 + BID_MARGIN)
        print(f"{o['gpu_name']:<22}{bid:>9.3f}{o['est_sec_per_block']:>10.0f}"
              f"{o['cents_per_block']:>7.2f}{o['gpu_ram'] / 1024:>6.0f}{o['inet_down']:>7.0f}"
              f"{o['reliability2']:>7.3f}  {o['id']}")


def rent(offer, shard, nshards, sweep_url):
    bid = round(offer["min_bid"] * (1 + BID_MARGIN), 4)
    env = ENV_FMT.format(sweep_url=sweep_url, shard=shard, nshards=nshards)
    r = vast("create", "instance", str(offer["id"]), "--image", IMAGE,
             "--disk", str(MIN_DISK_GB), "--bid", str(bid), "--env", env,
             "--label", f"{LABEL}-s{shard}of{nshards}", "--onstart-cmd", BOOTSTRAP)
    print(f"shard {shard}: {offer['gpu_name']} @ ${bid}/hr "
          f"(~{offer['cents_per_block']:.2f}¢/blk) -> instance {r.get('new_contract')}")
    return r


def fleet():
    return [i for i in vast("show", "instances") if str(i.get("label", "")).startswith(LABEL)]


def cmd_up(n, sweep_url):
    offers, used_machines = ranked_offers(), set()
    shard = 0
    for o in offers:
        if shard >= n:
            break
        if o["machine_id"] in used_machines:  # one shard per physical machine
            continue
        used_machines.add(o["machine_id"])
        rent(o, shard, n, sweep_url)
        shard += 1
        time.sleep(1)


def cmd_status():
    print(f"{'id':<10}{'label':<26}{'gpu':<20}{'state':<12}{'$/hr':>7}")
    for i in fleet():
        print(f"{i['id']:<10}{i.get('label', ''):<26}{i.get('gpu_name', '?'):<20}"
              f"{i.get('actual_status', '?'):<12}{i.get('dph_total', 0):>7.3f}")


def cmd_respawn(sweep_url):
    alive, total = {}, 0
    for i in fleet():
        label = i.get("label", "")   # fenix-teacher-s<K>of<N>
        k, n = label.split("-s")[1].split("of")
        total = int(n)
        if i.get("actual_status") in ("running", "loading", "created"):
            alive[int(k)] = i
    offers = [o for o in ranked_offers()]
    oi = 0
    for shard in range(total):
        if shard in alive:
            continue
        rent(offers[oi], shard, total, sweep_url)
        oi += 1


def cmd_destroy(target):
    for i in fleet():
        if target == "all" or str(i["id"]) == target:
            vast("destroy", "instance", str(i["id"]))
            print(f"destroyed {i['id']} ({i.get('label')})")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "rank":
        cmd_rank(int(a[1]) if len(a) > 1 else 15)
    elif a[0] == "up":
        cmd_up(int(a[1]), a[2])
    elif a[0] == "status":
        cmd_status()
    elif a[0] == "respawn":
        cmd_respawn(a[1])
    elif a[0] == "destroy":
        cmd_destroy(a[1] if len(a) > 1 else "all")
    else:
        sys.exit(__doc__)
