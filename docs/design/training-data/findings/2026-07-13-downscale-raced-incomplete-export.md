# Downscale pyramid built against an in-flight L0 export (PHercParis4 1.129µm)

**Date:** 2026-07-13 · **Volume:** `PHercParis4/20260608103018-1.129um-0.2m-78keV-masked.zarr`

## What happened

The sneakernet downscale driver on forlindesk2 walks the volume queue bottom-up and
picked this volume first (2026-07-12 19:39 UTC). Its L0 was still being exported by
the last rented fleet box (box2) at the time — 73.4% uploaded (49,871 / 67,968 shards,
z-rows 0–47 of 59) as of 2026-07-13 14:00 UTC.

`dct3d_downscale` deliberately treats a missing source shard as air (the export tool
always writes a shard, even all-air, so absent == past-extent). Against a *partial*
export that assumption is wrong: the not-yet-uploaded region (L0 z ≥ 48 at build
start, plus the moving upload frontier during the 7h38m build) was silently
downsampled as air. Roughly L1 z-rows ≥ 24 (~20% of the pyramid) plus scattered
frontier shards are affected; L2–5 inherit the corruption.

## Repair (pending box2 completion; watcher: box2 `~/export_watch.log`)

1. Wait for the L0 export to reach 67,968 shards.
2. List L0 shards with server mtime > 2026-07-12 19:22 UTC (first driver start —
   conservative); map each (z,y,x) → L1 shard (z/2, y/2, x/2), dedupe.
3. Delete those L1 files under `/shards/exports/.../1/c/` on forlindesk2.
4. Re-run L1 with `--resume` (only deleted shards rebuild, ~2 h), then delete local
   L2–5 entirely and re-run (~72 min — sources are NVMe-local).

## Lesson / guard

A downscale run must not start until its source level is **complete**. "Any shards
present at L(N-1)" is not sufficient; completeness means the full expected shard
count for the level's shape. The driver should compare the expected shard-grid count
(from zarr.json shape) against the actual file count before claiming a volume — or
the export pipeline should write a `complete` marker per level that downstream
consumers require.
