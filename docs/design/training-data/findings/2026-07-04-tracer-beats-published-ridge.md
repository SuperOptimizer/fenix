# 2026-07-04 — trace-eval baseline + CT-ridge arbitration: our tracer sits on the papyrus, the published meshes don't

## The benchmark (`fenix trace-eval`)
Block b_11776_14848_10752 (512³, PHercParis4 2.4µm), v3best surface prediction + CT →
`trace_volume_tiled` → scored vs the 5 published VAL meshes clipped to the block
(1,351 GT cells). Second block b_29184_11776_17408 (826 GT cells) for variance.

## Sweep (recall vs published mesh, block 1)

| lever | recall@2 | recall@4 | sheets/cells |
|---|---|---|---|
| bridge=0 | 0.213 | 0.476 | 815 / 1.49M |
| bridge=2 | 0.269 | 0.536 | 993 / 1.93M |
| bridge=4 | 0.281 | 0.577 | 1077 / 2.12M |
| bridge=6 | 0.284 | 0.574 | 1122 / 2.26M |
| bridge=8 | 0.305 | 0.586 | 1157 / 2.36M |
| thresh 0.06/0.15 | ±0.003 | ±0.01 | — |

- **Weak-field bridging is the live lever** (+0.11 recall@4 from 0→8, flattening past 4–8).
  `thresh` is flat. Block 2 (bridge=4): recall@4 **0.643** — story replicates.
- Precision vs this GT is unreadable by design (5 val meshes label a sliver of the
  block's real papyrus; 2.3M traced cells vs 1.3k GT cells).
- **Systematic ~3 vox offset**: recall@2 ≈ half of recall@4, median distance ~3.3 vox,
  consistent across blocks.

## The arbitration (who owns the 3-voxel offset?)
Identical `surf-qc` delta (CT@surface − CT@±12 along normals, n=400, same cache):

| surface | delta |
|---|---|
| our traced sheets t1..t9 | **+8.9 … +46.0** (most +14…+31) |
| our largest fragment t0 | +0.7 (over-merged sprawl — inspect) |
| published val meshes (5) | +2.2 … +5.6 (two BELOW the +5 label gate) |

**Verdict: the offset is the published meshes' error.** Our tracer snaps to the actual
CT ridge; the published GT floats ~3 vox off it (consistent with the whole
mesh-misregistration findings series). Caveat: published deltas average each mesh's full
extent, not just this block — directionally unambiguous anyway (4–8× contrast gap).

## Implications
1. **recall@4 is the tracer target metric** ("did we find the sheet"); recall@2 against
   this GT penalizes being MORE accurate than the reference.
2. **The tracer is now a label-improvement engine**: traced sheets have 4–8× better
   ridge alignment than the corpus meshes used as training labels. Feeding
   tracer-refined surfaces (or surf-repair snapped along them) back into training data
   directly attacks the label ceiling the multi-gate verdict identified.
3. Next tracer levers: soft_gate (crack-spanning), fragment stitch quality (t0's weak
   delta says the largest-merge path needs the CT-valley consensus gate), then recall@4
   toward the 0.65+ the second block shows is reachable.
